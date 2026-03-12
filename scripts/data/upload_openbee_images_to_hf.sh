#!/usr/bin/env bash
# 将 24 个 openbee_judged_v2 对应的图片文件夹上传到 HF OV2-VideoQA/captions
# 仓库内路径: openbee_images/<子目录名>/
# 可选: START_FROM=3 从第3个目录开始(0-based 为 2); USE_NO_SSL_VERIFY=1 使用 --no-ssl-verify

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_ROOT="/ov2/dataset_mid_source/Open-Bee/Honey-Data-15M_images"
PATH_IN_REPO_PREFIX="openbee_images"
START_FROM="${START_FROM:-0}"
EXTRA_ARGS=()
[[ -n "${USE_NO_SSL_VERIFY}" ]] && EXTRA_ARGS+=(--no-ssl-verify)
[[ -n "${USE_LARGE_FOLDER}" ]] && EXTRA_ARGS+=(--large-folder)

SKIP_NAMES=()
if [[ -n "${RESUME}" ]]; then
  echo "RESUME=1: 检查 HF 上已完整上传的目录（本地数=HF数才跳过）..."
  while IFS= read -r line; do
    [[ -n "$line" ]] && SKIP_NAMES+=("$line")
  done < <(cd "$REPO_ROOT" && python3 scripts/data/check_openbee_complete_on_hf.py 2>/dev/null)
  echo "  已完整 ${#SKIP_NAMES[@]} 个，将跳过；其余会重新/继续上传"
fi

NAMES=(
  AI2D_InternVL
  CMM-Math
  CoSyn_Chemical
  CoSyn_Circuit
  CoSyn_Graphic
  CoSyn_Math
  CoSyn_nutrition
  Docmatix
  Geometry3K
  GeomVerse
  "GeoQA+"
  InterGPS
  MathV360K_TQA
  MAVIS_Function
  MAVIS-Geo
  MAVIS-Metagen
  MMChem
  PathVQA
  PMC-VQA
  ScienceQA
  TabMWP
  TQA
  UniGeo
  VQA-RAD
)

cd "$REPO_ROOT"
echo "共 ${#NAMES[@]} 个图片文件夹，从第 $((START_FROM+1)) 个开始上传 -> ${PATH_IN_REPO_PREFIX}/<name>/"
[[ -n "${USE_NO_SSL_VERIFY}" ]] && echo "已启用 --no-ssl-verify"
[[ -n "${USE_LARGE_FOLDER}" ]] && echo "已启用 --large-folder（大目录分批 commit，防 504）"
echo ""

for i in "${!NAMES[@]}"; do
  [[ "$i" -lt "$START_FROM" ]] && continue
  name="${NAMES[$i]}"
  if [[ ${#SKIP_NAMES[@]} -gt 0 ]]; then
    skip=0
    for s in "${SKIP_NAMES[@]}"; do [[ "$s" = "$name" ]] && skip=1 && break; done
    [[ $skip -eq 1 ]] && echo "[$((i+1))/${#NAMES[@]}] 跳过（HF 已完整）: $name" && continue
  fi
  src="$IMAGE_ROOT/$name"
  if [[ ! -d "$src" ]]; then
    echo "[$((i+1))/${#NAMES[@]}] 跳过（目录不存在）: $name"
    continue
  fi
  echo "[$((i+1))/${#NAMES[@]}] 上传: $name -> ${PATH_IN_REPO_PREFIX}/${name}/"
  python scripts/data/hf_dataset_sync.py upload \
    --dataset-dir "$src" \
    --path-in-repo "${PATH_IN_REPO_PREFIX}/${name}" \
    "${EXTRA_ARGS[@]}"
  echo ""
done

echo "全部完成. 数据集: https://huggingface.co/datasets/OV2-VideoQA/captions"
