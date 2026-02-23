#!/bin/bash
# Submit API job, wait for RUNNING, then setup port forwarding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

MODE=""
JOB_ID=""
NO_WAIT=false
WAIT_POLL_SEC="${WAIT_POLL_SEC:-15}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-3600}"

usage() {
  echo "Usage: $0 --mode single|multi [--job-id ID] [--no-wait]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --job-id) JOB_ID="$2"; shift 2 ;;
    --no-wait) NO_WAIT=true; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

if [[ "$MODE" != "single" && "$MODE" != "multi" ]]; then
  usage
fi

source "$ROOT/scripts/env/vllm_model.env" 2>/dev/null || true
BASE_PORT="${SGLANG_BASE_PORT:-10025}"
if [[ "$MODE" == "single" ]]; then
  START_PORT="${API_PORT:-$BASE_PORT}"
  NUM_PORTS=1
else
  START_PORT="$BASE_PORT"
  NUM_PORTS="${NUM_SGLANG_INSTANCES:-8}"
fi

get_first_node() {
  local jid="$1"
  local nodelist
  nodelist=$(scontrol show job "$jid" 2>/dev/null | sed -n 's/^.* NodeList=\([^ ]*\).*/\1/p')
  if [[ -z "$nodelist" ]]; then
    return 1
  fi
  scontrol show hostnames "$nodelist" 2>/dev/null | head -1
}

wait_for_running() {
  local jid="$1"
  local start_ts end_ts
  start_ts=$(date +%s)
  while true; do
    local state
    state=$(squeue -j "$jid" -h -o "%T" 2>/dev/null || true)
    if [[ -z "$state" ]]; then
      echo "Job $jid no longer exists"
      return 1
    fi
    if [[ "$state" == "RUNNING" ]]; then
      return 0
    fi
    end_ts=$(date +%s)
    if [[ $((end_ts - start_ts)) -ge "$WAIT_TIMEOUT_SEC" ]]; then
      echo "Timeout waiting for RUNNING"
      return 1
    fi
    sleep "$WAIT_POLL_SEC"
  done
}

if [[ -z "$JOB_ID" ]]; then
  if [[ "$MODE" == "single" ]]; then
    JOB_ID=$(sbatch --parsable "$ROOT/scripts/api/start_api_server.sh")
  else
    JOB_ID=$(sbatch --parsable "$ROOT/scripts/api/start_api_server_multi.sh")
  fi
  echo "Submitted job: $JOB_ID"
fi

if [[ "$NO_WAIT" == "true" ]]; then
  echo "Run later: bash scripts/workflow/workflow_start_server_and_forward.sh --mode $MODE --job-id $JOB_ID"
  exit 0
fi

wait_for_running "$JOB_ID"
FIRST_NODE=$(get_first_node "$JOB_ID")
if [[ -z "$FIRST_NODE" ]]; then
  echo "Failed to resolve node for job $JOB_ID"
  exit 1
fi

sleep "${FORWARD_DELAY_SEC:-30}"
bash "$ROOT/scripts/api/setup_multi_port_forward.sh" "$FIRST_NODE" "$START_PORT" "$NUM_PORTS"

echo "Done. Stop forwarding with: bash scripts/api/stop_multi_port_forward.sh $START_PORT $NUM_PORTS"
