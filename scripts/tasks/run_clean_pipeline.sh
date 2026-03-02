#!/bin/bash
# Clean MM/OpenBee QA: single file (INPUT=path) or directory (INPUT_DIR=path).
# Directory mode: process *.jsonl in INPUT_DIR, exclude other_STEM / *_clean / *_dirty, sort by file size (small first).
# Outputs: JUDGED_DIR/{base}_judged.jsonl, OUTPUT_CLEAN_DIR/{base}_clean.jsonl, OUTPUT_DIRTY_DIR/{base}_dirty.jsonl.
# Defaults align with run_clean_openbee_with_sglang.sh (env overrides).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

INPUT="${INPUT:-}"
INPUT_DIR="${INPUT_DIR:-}"
# Default paths (same as original OpenBee script)
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output}"
JUDGED_DIR="${JUDGED_DIR:-$ROOT/output/openbee_judged}"
OUTPUT_CLEAN_DIR="${OUTPUT_CLEAN_DIR:-/ov2/dataset_jsonl/openbee_clean}"
OUTPUT_DIRTY_DIR="${OUTPUT_DIRTY_DIR:-/ov2/dataset_jsonl/openbee_dirty}"

BACKEND_PROFILE="${BACKEND_PROFILE:-sglang_openbee_8ports}"
MAX_RECORDS="${MAX_RECORDS:-0}"
MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
IMAGE_DETAIL="${IMAGE_DETAIL:-low}"

WORKERS="${WORKERS:-8}"
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
print(json.dumps({
  "input_jsonl": "$input_jsonl",
  "max_records": int("$MAX_RECORDS"),
  "model": "$MODEL",
  "image_root": "$IMAGE_ROOT",
  "image_detail": "$IMAGE_DETAIL",
  "clean_output_jsonl": "$clean_jsonl",
  "dirty_output_jsonl": "$dirty_jsonl"
}, ensure_ascii=False))
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
    stat -c $'%s\t%n' "$f" >> "$SIZE_LIST"
  done
  sort -n < "$SIZE_LIST" | cut -f2- > "${SIZE_LIST}.paths"
  mv "${SIZE_LIST}.paths" "$SIZE_LIST"
  total=$(wc -l < "$SIZE_LIST" 2>/dev/null || echo 0)
  echo "=== Directory mode: $INPUT_DIR => $total files (by size asc, single Ray session) ==="
  echo "  Judged: $JUDGED_DIR"
  echo "  Clean:  $OUTPUT_CLEAN_DIR"
  echo "  Dirty:  $OUTPUT_DIRTY_DIR"
  export INPUT_LIST="$SIZE_LIST"
  export JUDGED_DIR
  export OUTPUT_CLEAN_DIR
  export OUTPUT_DIRTY_DIR
  python -m pipeline.core.main \
    --config "$ROOT/configs/clean_mm_qa.yaml" \
    --task clean_mm_qa \
    --endpoint-group "$BACKEND_PROFILE" \
    --endpoint-registry-file "$ENDPOINT_REGISTRY"
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
print(json.dumps({
  "input_jsonl": "$INPUT",
  "max_records": int("$MAX_RECORDS"),
  "model": "$MODEL",
  "image_root": "$IMAGE_ROOT",
  "image_detail": "$IMAGE_DETAIL",
  "clean_output_jsonl": "$CLEAN_JSONL",
  "dirty_output_jsonl": "$DIRTY_JSONL"
}, ensure_ascii=False))
PY
)

python -m pipeline.core.main \
  --config "$ROOT/configs/clean_mm_qa.yaml" \
  --task clean_mm_qa \
  --endpoint-group "$BACKEND_PROFILE" \
  --endpoint-registry-file "$ENDPOINT_REGISTRY" \
  --task-config-json "$TASK_CONFIG_JSON"
