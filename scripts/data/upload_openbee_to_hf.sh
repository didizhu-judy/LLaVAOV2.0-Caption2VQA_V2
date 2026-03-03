#!/bin/bash
# 将 total_count<100000 的 25 个 OpenBee JSONL 及对应图片从原始路径直接上传到 HF
# 仓库 OV2-VideoQA/captions；路径 original/openbee/*.jsonl，original/openbee/images/<数据集同名>/
# 使用前：conda activate base && hf auth login

set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# 确保使用 base 环境（hf CLI 在该环境下可用）
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate base

OPENBEE_JSONL="${OPENBEE_JSONL:-/ov2/dataset_jsonl/openbee}"
OPENBEE_IMAGES="${OPENBEE_IMAGES:-/ov2/dataset_mid_source/Open-Bee/Honey-Data-15M_images}"
REPO_ID="${REPO_ID:-OV2-VideoQA/captions}"

# 25 个 jsonl（仅此处维护名单）
FILES=(
  MAVIS_Function.jsonl "GeoQA+.jsonl" MAVIS-Geo.jsonl ScienceQA.jsonl
  MAVIS-Metagen.jsonl Geometry3K.jsonl GeomVerse.jsonl UniGeo.jsonl
  CMM-Math.jsonl TQA.jsonl CoSyn_Graphic.jsonl CoSyn_Math.jsonl
  PMC-VQA.jsonl MMChem.jsonl MathV360K_TQA.jsonl InterGPS.jsonl
  AI2D_InternVL.jsonl CoSyn_Chemical.jsonl TabMWP.jsonl
  CoSyn_nutrition.jsonl CoSyn_Circuit.jsonl Docmatix.jsonl
  VQA-RAD.jsonl PathVQA.jsonl
)

# 只上传这 25 个 jsonl → original/openbee/（已上传则注释掉）
# ALLOW_JSONL=()
# for f in "${FILES[@]}"; do ALLOW_JSONL+=(--allow-patterns "$f"); done
# python scripts/data/hf_dataset_sync.py upload \
#   --repo-id "$REPO_ID" \
#   --dataset-dir "$OPENBEE_JSONL" \
#   --path-in-repo original/openbee \
#   "${ALLOW_JSONL[@]}"

# 只上传这 25 个数据集子目录的图片 → original/openbee/images/<stem>/
# 25 个图片目录路径（与 FILES 一一对应），直接上传不复制
IMAGE_DIRS=(
  "$OPENBEE_IMAGES/MAVIS_Function"
  "$OPENBEE_IMAGES/GeoQA+"
  "$OPENBEE_IMAGES/MAVIS-Geo"
  "$OPENBEE_IMAGES/ScienceQA"
  "$OPENBEE_IMAGES/MAVIS-Metagen"
  "$OPENBEE_IMAGES/Geometry3K"
  "$OPENBEE_IMAGES/GeomVerse"
  "$OPENBEE_IMAGES/UniGeo"
  "$OPENBEE_IMAGES/CMM-Math"
  "$OPENBEE_IMAGES/TQA"
  "$OPENBEE_IMAGES/CoSyn_Graphic"
  "$OPENBEE_IMAGES/CoSyn_Math"
  "$OPENBEE_IMAGES/PMC-VQA"
  "$OPENBEE_IMAGES/MMChem"
  "$OPENBEE_IMAGES/MathV360K_TQA"
  "$OPENBEE_IMAGES/InterGPS"
  "$OPENBEE_IMAGES/AI2D_InternVL"
  "$OPENBEE_IMAGES/CoSyn_Chemical"
  "$OPENBEE_IMAGES/TabMWP"
  "$OPENBEE_IMAGES/CoSyn_nutrition"
  "$OPENBEE_IMAGES/CoSyn_Circuit"
  "$OPENBEE_IMAGES/Docmatix"
  "$OPENBEE_IMAGES/VQA-RAD"
  "$OPENBEE_IMAGES/PathVQA"
)
total="${#IMAGE_DIRS[@]}"
n=0
for dir in "${IMAGE_DIRS[@]}"; do
  stem="$(basename "$dir")"
  if [ ! -d "$dir" ]; then echo "跳过（不存在）: $dir"; continue; fi
  n=$((n+1))
  echo ">>> [$n/$total] 上传 $stem …"
  hf upload "$REPO_ID" "$dir" "original/openbee/images/$stem" --repo-type dataset
done
echo "图片上传完成（共 $n 个目录）."

echo "Done. https://huggingface.co/datasets/${REPO_ID}"
