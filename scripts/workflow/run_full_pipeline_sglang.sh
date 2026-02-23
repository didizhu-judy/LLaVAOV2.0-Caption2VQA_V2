#!/bin/bash
#SBATCH --job-name=caption2vqa-sglang
#SBATCH --output=logs/slurm_%j.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h200:8
#SBATCH --partition=lrc-xlong

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

INPUT="${INPUT:-}"
if [[ -z "$INPUT" ]]; then
  echo "INPUT is required"
  exit 1
fi

SGLANG_PORT="${SGLANG_PORT:-10025}"
START_LOCAL_SERVER="${START_LOCAL_SERVER:-1}"
BACKEND_PROFILE="${BACKEND_PROFILE:-local_multi}"

API_PID=""
cleanup() {
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$START_LOCAL_SERVER" == "1" && "$BACKEND_PROFILE" == "local_multi" ]]; then
  bash scripts/api/start_api_server.sh --backend sglang --port "$SGLANG_PORT" &
  API_PID=$!

  for _ in $(seq 1 120); do
    if [[ -f "$ROOT/runtime/endpoints.local.json" ]]; then
      break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
      echo "API server exited early"
      exit 1
    fi
    sleep 5
  done
fi

INPUT="$INPUT" \
BACKEND_PROFILE="$BACKEND_PROFILE" \
MAX_RECORDS="${MAX_RECORDS:-0}" \
WORKERS="${WORKERS:-4}" \
WORKER_ASYNC_CONCURRENCY="${WORKER_ASYNC_CONCURRENCY:-16}" \
bash scripts/tasks/run_caption_to_vqa.sh
