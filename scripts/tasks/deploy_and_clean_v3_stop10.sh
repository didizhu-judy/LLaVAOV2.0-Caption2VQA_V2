#!/bin/bash
# One-click: ensure local SGLang endpoints are healthy, then run OpenBee clean pipeline in v3 mode.
# Differences from deploy_and_clean_v2_full.sh:
# - only OpenBee fixed stem list (24 jsonl via OPENBEE_25_ONLY=1 in runner)
# - output to *_v3 paths
# - stop early per file when dirty reaches STOP_AFTER_DIRTY (default 10)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Configurable via env
SGLANG_BASE_PORT="${SGLANG_BASE_PORT:-10025}"
NUM_SGLANG_INSTANCES="${NUM_SGLANG_INSTANCES:-4}"
HEALTH_TIMEOUT_SEC="${SGLANG_DEPLOY_TIMEOUT_SEC:-1800}"
HEALTH_INTERVAL_SEC="${SGLANG_DEPLOY_POLL_INTERVAL:-10}"
SKIP_DEPLOY="${SKIP_DEPLOY:-0}"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CLEAN_LOG="${CLEAN_LOG:-$ROOT/output/clean_v3_stop10dirty.log}"

mkdir -p "$ROOT/logs" "$ROOT/runtime" "$(dirname "$CLEAN_LOG")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

check_sglang_healthy() {
  for ((i=0; i<NUM_SGLANG_INSTANCES; i++)); do
    local port=$((SGLANG_BASE_PORT + i))
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/health" 2>/dev/null || echo "000")
    if [[ "$code" != "200" ]]; then
      return 1
    fi
  done
  return 0
}

if [[ "$SKIP_DEPLOY" == "1" || "$SKIP_DEPLOY" == "true" || "$SKIP_DEPLOY" == "yes" ]]; then
  log "SKIP_DEPLOY=1, skip SGLang deployment"
elif check_sglang_healthy; then
  log "SGLang already healthy on ports ${SGLANG_BASE_PORT}-$((SGLANG_BASE_PORT + NUM_SGLANG_INSTANCES - 1)), skip deployment"
else
  if [[ "$SKIP_CLEANUP" != "1" && "$SKIP_CLEANUP" != "true" && "$SKIP_CLEANUP" != "yes" ]]; then
    log "Cleaning old SGLang/model processes..."
    pkill -f "run_local_sglang.sh|start_api_server_multi.sh" 2>/dev/null || true
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep 5
  fi

  log "Starting SGLang (${NUM_SGLANG_INSTANCES} instances)..."
  SGLANG_LOG="$ROOT/logs/sglang_multi_$(date +%Y%m%d_%H%M%S).log"
  NUM_SGLANG_INSTANCES="$NUM_SGLANG_INSTANCES" SGLANG_BASE_PORT="$SGLANG_BASE_PORT" \
    nohup bash "$ROOT/scripts/tasks/run_local_sglang.sh" > "$SGLANG_LOG" 2>&1 &
  SGLANG_PID=$!
  log "SGLang pid=$SGLANG_PID log=$SGLANG_LOG"

  log "Waiting for health..."
  deadline=$(($(date +%s) + HEALTH_TIMEOUT_SEC))
  while true; do
    if check_sglang_healthy; then
      log "All SGLang instances are healthy"
      break
    fi
    if [[ $(date +%s) -ge "$deadline" ]]; then
      log "Timeout waiting for SGLang health"
      kill "$SGLANG_PID" 2>/dev/null || true
      exit 1
    fi
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
      log "SGLang process exited early, check $SGLANG_LOG"
      exit 1
    fi
    sleep "$HEALTH_INTERVAL_SEC"
  done
fi

log "Starting clean task (v3, stop_after_dirty per file), log $CLEAN_LOG"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  source "$ROOT/.venv/bin/activate"
fi

export OPENBEE_25_ONLY=1
unset OPENBEE_V2
export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-_v3}"
export STOP_AFTER_DIRTY="${STOP_AFTER_DIRTY:-10}"

export INPUT_DIR="${INPUT_DIR:-/ov2/dataset_jsonl/openbee}"
export BACKEND_PROFILE="${BACKEND_PROFILE:-local_multi}"
export INCLUDE_ANSWER_IN_JUDGE="${INCLUDE_ANSWER_IN_JUDGE:-0}"
export MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"

export JUDGED_DIR="${JUDGED_DIR:-$ROOT/output/openbee_judged_v3}"
export OUTPUT_CLEAN_DIR="${OUTPUT_CLEAN_DIR:-/ov2/dataset_jsonl/openbee_clean_v3}"
export OUTPUT_DIRTY_DIR="${OUTPUT_DIRTY_DIR:-/ov2/dataset_jsonl/openbee_dirty_v3}"

echo "=== $(date -Iseconds) start clean v3 stop_after_dirty=${STOP_AFTER_DIRTY} ===" >> "$CLEAN_LOG"
exec bash "$ROOT/scripts/tasks/run_clean_pipeline.sh" 2>&1 | tee -a "$CLEAN_LOG"
