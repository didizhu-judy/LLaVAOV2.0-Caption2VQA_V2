#!/bin/bash
# Clean MM/OpenBee QA: single file (INPUT=path) or directory (INPUT_DIR=path).
# Directory mode: process *.jsonl in INPUT_DIR, exclude other_STEM / *_clean / *_dirty, sort by file size (small first).
# Outputs: JUDGED_DIR/{base}_judged.jsonl, OUTPUT_CLEAN_DIR/{base}_clean.jsonl, OUTPUT_DIRTY_DIR/{base}_dirty.jsonl.
# Defaults align with run_clean_openbee_with_sglang.sh (env overrides).
#
# 仅处理 upload_openbee_to_hf.sh 中那 25 个 jsonl 并输出到 openbee_clean_new / openbee_dirty_new:
#   OPENBEE_25_ONLY=1 INPUT_DIR=/ov2/dataset_jsonl/openbee bash scripts/tasks/run_clean_pipeline.sh
#
# 使用本地 SGLang（远程 8 口不可用时）:
#   BACKEND_PROFILE=local_multi ...  # 会请求 http://127.0.0.1:10025
#
# 试跑（少量样本、低并发）:
#   MAX_RECORDS=20 INPUT=... WORKERS=2 WORKER_ASYNC_CONCURRENCY=2 FETCH_BATCH_SIZE=4 bash scripts/tasks/run_clean_pipeline.sh
# 拉满速度（本地 8 实例）: 使用默认 WORKERS=8 WORKER_ASYNC_CONCURRENCY=16，并确保 BACKEND_PROFILE=local_multi 且存在 runtime/endpoints.local.json
# 可调参数: MAX_RECORDS, WORKERS, WORKER_ASYNC_CONCURRENCY(每 worker 并发), FETCH_BATCH_SIZE, BACKEND_PROFILE
# 默认 BACKEND_PROFILE=local_multi：优先用 runtime/endpoints.local.json（8 口），否则用 scripts/env/endpoints.json 的 local_multi（现为 8 口）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

INPUT="${INPUT:-}"
INPUT_DIR="${INPUT_DIR:-}"
# 仅处理 OpenBee 25 个 jsonl 且输出到 openbee_clean_new / openbee_dirty_new：OPENBEE_25_ONLY=1
OPENBEE_25_ONLY="${OPENBEE_25_ONLY:-0}"
# 25 个 jsonl 的 stem（与 scripts/data/upload_openbee_to_hf.sh 的 FILES 一致）
OPENBEE_25_STEMS=(
  MAVIS_Function "GeoQA+" MAVIS-Geo ScienceQA MAVIS-Metagen Geometry3K GeomVerse UniGeo
  CMM-Math TQA CoSyn_Graphic CoSyn_Math PMC-VQA MMChem MathV360K_TQA InterGPS
  AI2D_InternVL CoSyn_Chemical TabMWP CoSyn_nutrition CoSyn_Circuit Docmatix VQA-RAD PathVQA
)

# Default paths (same as original OpenBee script)
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output}"
JUDGED_DIR="${JUDGED_DIR:-$ROOT/output/openbee_judged}"
if [[ "$OPENBEE_25_ONLY" == "1" || "$OPENBEE_25_ONLY" == "true" || "$OPENBEE_25_ONLY" == "yes" ]]; then
  OUTPUT_CLEAN_DIR="${OUTPUT_CLEAN_DIR:-/ov2/dataset_jsonl/openbee_clean_new}"
  OUTPUT_DIRTY_DIR="${OUTPUT_DIRTY_DIR:-/ov2/dataset_jsonl/openbee_dirty_new}"
else
  OUTPUT_CLEAN_DIR="${OUTPUT_CLEAN_DIR:-/ov2/dataset_jsonl/openbee_clean}"
  OUTPUT_DIRTY_DIR="${OUTPUT_DIRTY_DIR:-/ov2/dataset_jsonl/openbee_dirty}"
fi

BACKEND_PROFILE="${BACKEND_PROFILE:-local_multi}"
MAX_RECORDS="${MAX_RECORDS:-0}"
MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
IMAGE_DETAIL="${IMAGE_DETAIL:-low}"

WORKERS="${WORKERS:-8}"
# 每 worker 并发数；8 实例时 8×8=64 总并发（过高会压垮 SGLang，反而更慢）
WORKER_ASYNC_CONCURRENCY="${WORKER_ASYNC_CONCURRENCY:-8}"
FETCH_BATCH_SIZE="${FETCH_BATCH_SIZE:-32}"
DUMP_EVERY_N="${DUMP_EVERY_N:-200}"
DUMP_INTERVAL_SEC="${DUMP_INTERVAL_SEC:-30}"

mkdir -p "$OUTPUT_DIR" "$JUDGED_DIR" "$OUTPUT_CLEAN_DIR" "$OUTPUT_DIRTY_DIR"

ENDPOINT_REGISTRY="${ENDPOINT_REGISTRY:-$ROOT/scripts/env/endpoints.json}"
if [[ "$BACKEND_PROFILE" == "local_multi" && -f "$ROOT/runtime/endpoints.local.json" ]]; then
  ENDPOINT_REGISTRY="$ROOT/runtime/endpoints.local.json"
fi

export PIPELINE_NUM_WORKERS="$WORKERS"
export PIPELINE_WORKER_ASYNC_CONCURRENCY="$WORKER_ASYNC_CONCURRENCY"
export PIPELINE_FETCH_BATCH_SIZE="$FETCH_BATCH_SIZE"
export PIPELINE_DUMP_EVERY_N="$DUMP_EVERY_N"
export PIPELINE_DUMP_INTERVAL_SEC="$DUMP_INTERVAL_SEC"
export PIPELINE_RESUME="${RESUME:-1}"

run_one() {
  local input_jsonl="$1"
  local base="$2"
  local judged_jsonl="$JUDGED_DIR/${base}_judged.jsonl"
  local error_jsonl="$JUDGED_DIR/${base}_clean_errors.jsonl"
  local clean_jsonl="$OUTPUT_CLEAN_DIR/${base}_clean.jsonl"
  local dirty_jsonl="$OUTPUT_DIRTY_DIR/${base}_dirty.jsonl"

  export PIPELINE_OUTPUT_JSONL="$judged_jsonl"
  export PIPELINE_ERROR_JSONL="$error_jsonl"

  local task_config_json
  task_config_json=$(python3 - <<PY
import json
import os
d = {
  "input_jsonl": "$input_jsonl",
  "max_records": int("$MAX_RECORDS"),
  "model": "$MODEL",
  "image_root": "$IMAGE_ROOT",
  "image_detail": "$IMAGE_DETAIL",
  "clean_output_jsonl": "$clean_jsonl",
  "dirty_output_jsonl": "$dirty_jsonl"
}
if os.environ.get("MAX_PREFILL_TOKENS"):
  d["max_prefill_tokens"] = int(os.environ["MAX_PREFILL_TOKENS"])
v = os.environ.get("INCLUDE_ANSWER_IN_JUDGE", "").strip().lower()
if v in ("0", "false", "no", "off"):
  d["include_answer_in_judge"] = False
elif v in ("1", "true", "yes", "on"):
  d["include_answer_in_judge"] = True
print(json.dumps(d, ensure_ascii=False))
PY
)

  python -m pipeline.core.main \
    --config "$ROOT/configs/clean_mm_qa.yaml" \
    --task clean_mm_qa \
    --endpoint-group "$BACKEND_PROFILE" \
    --endpoint-registry-file "$ENDPOINT_REGISTRY" \
    --task-config-json "$task_config_json"
}

if [[ -n "$INPUT_DIR" ]]; then
  # Directory mode: one process, one Ray session; process all files (by size asc) without restarting Ray per file
  SIZE_LIST=$(mktemp)
  trap 'rm -f "$SIZE_LIST"' EXIT
  for f in "$INPUT_DIR"/*.jsonl; do
    [[ -f "$f" ]] || continue
    base="$(basename "$f" .jsonl)"
    if [[ "$base" == "other_STEM" ]]; then
      echo "Skip (excluded): $f"
      continue
    fi
    case "$base" in
      *_clean|*_dirty) echo "Skip (result file): $f"; continue ;;
    esac
    if [[ "$OPENBEE_25_ONLY" == "1" || "$OPENBEE_25_ONLY" == "true" || "$OPENBEE_25_ONLY" == "yes" ]]; then
      allowed=0
      for s in "${OPENBEE_25_STEMS[@]}"; do
        if [[ "$base" == "$s" ]]; then allowed=1; break; fi
      done
      if [[ "$allowed" -eq 0 ]]; then
        echo "Skip (not in OPENBEE_25 list): $f"
        continue
      fi
    fi
    stat -c $'%s\t%n' "$f" >> "$SIZE_LIST"
  done
  sort -n < "$SIZE_LIST" | cut -f2- > "${SIZE_LIST}.paths"
  mv "${SIZE_LIST}.paths" "$SIZE_LIST"
  total=$(wc -l < "$SIZE_LIST" 2>/dev/null || echo 0)
  echo "=== Directory mode: $INPUT_DIR => $total files (by size asc, single Ray session) ==="
  if [[ "$OPENBEE_25_ONLY" == "1" || "$OPENBEE_25_ONLY" == "true" || "$OPENBEE_25_ONLY" == "yes" ]]; then
    echo "  (OPENBEE_25_ONLY: only the 25 jsonl from upload_openbee_to_hf.sh)"
  fi
  echo "  Judged: $JUDGED_DIR"
  echo "  Clean:  $OUTPUT_CLEAN_DIR"
  echo "  Dirty:  $OUTPUT_DIRTY_DIR"
  if [[ -n "${MAX_RECORDS:-}" && "${MAX_RECORDS:-0}" -gt 0 ]]; then
    echo "  (MAX_RECORDS=$MAX_RECORDS, 每文件最多处理条数，用于试跑)"
  fi
  export INPUT_LIST="$SIZE_LIST"
  export JUDGED_DIR
  export OUTPUT_CLEAN_DIR
  export OUTPUT_DIRTY_DIR
  DIR_TASK_JSON=$(python3 -c "
import json, os
d = {\"max_records\": $MAX_RECORDS}
v = os.environ.get(\"INCLUDE_ANSWER_IN_JUDGE\", \"\").strip().lower()
if v in (\"0\", \"false\", \"no\", \"off\"): d[\"include_answer_in_judge\"] = False
elif v in (\"1\", \"true\", \"yes\", \"on\"): d[\"include_answer_in_judge\"] = True
print(json.dumps(d))
")
  if [[ -n "${MAX_RECORDS:-}" && "${MAX_RECORDS:-0}" -gt 0 ]]; then
    python -m pipeline.core.main \
      --config "$ROOT/configs/clean_mm_qa.yaml" \
      --task clean_mm_qa \
      --endpoint-group "$BACKEND_PROFILE" \
      --endpoint-registry-file "$ENDPOINT_REGISTRY" \
      --task-config-json "$DIR_TASK_JSON"
  else
    python -m pipeline.core.main \
      --config "$ROOT/configs/clean_mm_qa.yaml" \
      --task clean_mm_qa \
      --endpoint-group "$BACKEND_PROFILE" \
      --endpoint-registry-file "$ENDPOINT_REGISTRY" \
      --task-config-json "$DIR_TASK_JSON"
  fi
  echo "=== Done ($total files) ==="
  exit 0
fi

if [[ -z "$INPUT" ]]; then
  echo "Either INPUT or INPUT_DIR is required."
  echo "  Single file: INPUT=/path/to/file.jsonl bash $0"
  echo "  Directory:   INPUT_DIR=/ov2/dataset_jsonl/openbee bash $0"
  exit 1
fi

# Single-file mode
INPUT_STEM="$(basename "${INPUT%.*}")"
TS="$(date +%Y%m%d_%H%M%S)"
JUDGED_JSONL="${JUDGED_JSONL:-$OUTPUT_DIR/${INPUT_STEM}_judged_${TS}.jsonl}"
ERROR_JSONL="${ERROR_JSONL:-$OUTPUT_DIR/${INPUT_STEM}_clean_errors_${TS}.jsonl}"
CLEAN_JSONL="${CLEAN_JSONL:-$OUTPUT_DIR/${INPUT_STEM}_clean_${TS}.jsonl}"
DIRTY_JSONL="${DIRTY_JSONL:-$OUTPUT_DIR/${INPUT_STEM}_dirty_${TS}.jsonl}"

export PIPELINE_OUTPUT_JSONL="$JUDGED_JSONL"
export PIPELINE_ERROR_JSONL="$ERROR_JSONL"

TASK_CONFIG_JSON=$(python3 - <<PY
import json
import os
d = {
  "input_jsonl": "$INPUT",
  "max_records": int("$MAX_RECORDS"),
  "model": "$MODEL",
  "image_root": "$IMAGE_ROOT",
  "image_detail": "$IMAGE_DETAIL",
  "clean_output_jsonl": "$CLEAN_JSONL",
  "dirty_output_jsonl": "$DIRTY_JSONL"
}
if os.environ.get("MAX_PREFILL_TOKENS"):
  d["max_prefill_tokens"] = int(os.environ["MAX_PREFILL_TOKENS"])
v = os.environ.get("INCLUDE_ANSWER_IN_JUDGE", "").strip().lower()
if v in ("0", "false", "no", "off"):
  d["include_answer_in_judge"] = False
elif v in ("1", "true", "yes", "on"):
  d["include_answer_in_judge"] = True
print(json.dumps(d, ensure_ascii=False))
PY
)

python -m pipeline.core.main \
  --config "$ROOT/configs/clean_mm_qa.yaml" \
  --task clean_mm_qa \
  --endpoint-group "$BACKEND_PROFILE" \
  --endpoint-registry-file "$ENDPOINT_REGISTRY" \
  --task-config-json "$TASK_CONFIG_JSON"
