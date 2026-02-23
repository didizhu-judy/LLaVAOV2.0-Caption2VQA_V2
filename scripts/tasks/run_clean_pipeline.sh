#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

INPUT="${INPUT:-}"
if [[ -z "$INPUT" ]]; then
  echo "INPUT is required"
  echo "Example: INPUT=/path/to/mmqa.jsonl bash scripts/tasks/run_clean_pipeline.sh"
  exit 1
fi

BACKEND_PROFILE="${BACKEND_PROFILE:-azure_multi}"
MAX_RECORDS="${MAX_RECORDS:-0}"
MODEL="${MODEL:-gpt-4o}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
IMAGE_DETAIL="${IMAGE_DETAIL:-low}"

WORKERS="${WORKERS:-4}"
WORKER_ASYNC_CONCURRENCY="${WORKER_ASYNC_CONCURRENCY:-32}"
FETCH_BATCH_SIZE="${FETCH_BATCH_SIZE:-32}"
DUMP_EVERY_N="${DUMP_EVERY_N:-200}"
DUMP_INTERVAL_SEC="${DUMP_INTERVAL_SEC:-30}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output}"
mkdir -p "$OUTPUT_DIR"
INPUT_STEM="$(basename "${INPUT%.*}")"
TS="$(date +%Y%m%d_%H%M%S)"
JUDGED_JSONL="${JUDGED_JSONL:-$OUTPUT_DIR/${INPUT_STEM}_judged_${TS}.jsonl}"
ERROR_JSONL="${ERROR_JSONL:-$OUTPUT_DIR/${INPUT_STEM}_clean_errors_${TS}.jsonl}"
CLEAN_JSONL="${CLEAN_JSONL:-$OUTPUT_DIR/${INPUT_STEM}_clean_${TS}.jsonl}"
DIRTY_JSONL="${DIRTY_JSONL:-$OUTPUT_DIR/${INPUT_STEM}_dirty_${TS}.jsonl}"

ENDPOINT_REGISTRY="${ENDPOINT_REGISTRY:-$ROOT/scripts/env/endpoints.json}"
if [[ "$BACKEND_PROFILE" == "local_multi" && -f "$ROOT/runtime/endpoints.local.json" ]]; then
  ENDPOINT_REGISTRY="$ROOT/runtime/endpoints.local.json"
fi

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

export PIPELINE_NUM_WORKERS="$WORKERS"
export PIPELINE_WORKER_ASYNC_CONCURRENCY="$WORKER_ASYNC_CONCURRENCY"
export PIPELINE_FETCH_BATCH_SIZE="$FETCH_BATCH_SIZE"
export PIPELINE_DUMP_EVERY_N="$DUMP_EVERY_N"
export PIPELINE_DUMP_INTERVAL_SEC="$DUMP_INTERVAL_SEC"
export PIPELINE_OUTPUT_JSONL="$JUDGED_JSONL"
export PIPELINE_ERROR_JSONL="$ERROR_JSONL"
export PIPELINE_RESUME="${RESUME:-1}"

python -m pipeline.core.main \
  --config "$ROOT/configs/clean_mm_qa.yaml" \
  --task clean_mm_qa \
  --endpoint-group "$BACKEND_PROFILE" \
  --endpoint-registry-file "$ENDPOINT_REGISTRY" \
  --task-config-json "$TASK_CONFIG_JSON"
