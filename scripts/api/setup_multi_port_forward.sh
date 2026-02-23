#!/bin/bash
# Forward a range of ports from a compute node to the login node via socat.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <compute_node> <start_port> [end_port_or_count]"
  exit 1
fi

COMPUTE_NODE="$1"
START_PORT="$2"
END_OR_COUNT="${3:-}"

if [[ -z "$END_OR_COUNT" ]]; then
  END_PORT="$START_PORT"
  NUM_PORTS=1
elif [[ "$END_OR_COUNT" -lt "$START_PORT" ]]; then
  NUM_PORTS="$END_OR_COUNT"
  END_PORT=$((START_PORT + NUM_PORTS - 1))
else
  END_PORT="$END_OR_COUNT"
  NUM_PORTS=$((END_PORT - START_PORT + 1))
fi

if ! command -v socat >/dev/null 2>&1; then
  echo "socat not found; please install it first."
  exit 1
fi

PID_DIR="$HOME/.socat_forwards"
mkdir -p "$PID_DIR"

PIDS=()
FAILED=()
for PORT in $(seq "$START_PORT" "$END_PORT"); do
  PID_FILE="$PID_DIR/socat_${PORT}.pid"

  if lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    FAILED+=("$PORT")
    continue
  fi

  socat TCP-LISTEN:"$PORT",fork,reuseaddr TCP:"$COMPUTE_NODE":"$PORT" >/dev/null 2>&1 &
  SOCAT_PID=$!
  echo "$SOCAT_PID" > "$PID_FILE"
  sleep 0.2

  if ! kill -0 "$SOCAT_PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    FAILED+=("$PORT")
  else
    PIDS+=("$SOCAT_PID")
  fi
done

ALL_PIDS_FILE="$PID_DIR/socat_all_${START_PORT}_${END_PORT}.pids"
printf '%s\n' "${PIDS[@]}" > "$ALL_PIDS_FILE"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed ports: ${FAILED[*]}"
fi

echo "Forwarded $START_PORT-$END_PORT ($NUM_PORTS ports)"
if [[ ${#PIDS[@]} -gt 0 ]]; then
  echo "Stop with: bash scripts/api/stop_multi_port_forward.sh $START_PORT $NUM_PORTS"
fi
