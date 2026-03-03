#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/.venv/bin/activate"
export OPENBEE_25_ONLY=1
export INPUT_DIR=/ov2/dataset_jsonl/openbee
export MODEL=Qwen/Qwen3-VL-32B-Instruct
export BACKEND_PROFILE=local_multi
exec bash "$ROOT/scripts/tasks/run_clean_pipeline.sh"
