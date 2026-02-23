#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

INPUT="${INPUT:-}"
if [[ -z "$INPUT" ]]; then
  echo "INPUT is required"
  echo "Example: INPUT=/path/to/captions.jsonl bash scripts/tasks/run_caption_to_vqa.sh"
  exit 1
fi

BACKEND_PROFILE="${BACKEND_PROFILE:-local_multi}"
MAX_RECORDS="${MAX_RECORDS:-0}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MAX_UNDERSTANDING="${MAX_UNDERSTANDING:-5}"

WORKERS="${WORKERS:-4}"
WORKER_ASYNC_CONCURRENCY="${WORKER_ASYNC_CONCURRENCY:-16}"
FETCH_BATCH_SIZE="${FETCH_BATCH_SIZE:-32}"
DUMP_EVERY_N="${DUMP_EVERY_N:-200}"
DUMP_INTERVAL_SEC="${DUMP_INTERVAL_SEC:-30}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output}"
mkdir -p "$OUTPUT_DIR"
INPUT_STEM="$(basename "${INPUT%.*}")"
TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_JSONL="${OUTPUT_JSONL:-$OUTPUT_DIR/${INPUT_STEM}_caption_vqa_${TS}.jsonl}"
ERROR_JSONL="${ERROR_JSONL:-$OUTPUT_DIR/${INPUT_STEM}_caption_vqa_errors_${TS}.jsonl}"

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
  "max_understanding": int("$MAX_UNDERSTANDING")
}, ensure_ascii=False))
PY
)

export PIPELINE_NUM_WORKERS="$WORKERS"
export PIPELINE_WORKER_ASYNC_CONCURRENCY="$WORKER_ASYNC_CONCURRENCY"
export PIPELINE_FETCH_BATCH_SIZE="$FETCH_BATCH_SIZE"
export PIPELINE_DUMP_EVERY_N="$DUMP_EVERY_N"
export PIPELINE_DUMP_INTERVAL_SEC="$DUMP_INTERVAL_SEC"
export PIPELINE_OUTPUT_JSONL="$OUTPUT_JSONL"
export PIPELINE_ERROR_JSONL="$ERROR_JSONL"
export PIPELINE_RESUME="${RESUME:-1}"

python -m pipeline.core.main \
  --config "$ROOT/configs/caption_to_vqa.yaml" \
  --task caption_to_vqa \
  --endpoint-group "$BACKEND_PROFILE" \
  --endpoint-registry-file "$ENDPOINT_REGISTRY" \
  --task-config-json "$TASK_CONFIG_JSON"
