#!/bin/bash
# Stop a range of socat forwarding processes.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <start_port> [end_port_or_count]"
  exit 1
fi

START_PORT="$1"
END_OR_COUNT="${2:-}"

if [[ ! "$START_PORT" =~ ^[0-9]+$ ]]; then
  echo "start_port must be numeric"
  exit 1
fi

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

PID_DIR="$HOME/.socat_forwards"
ALL_PIDS_FILE="$PID_DIR/socat_all_${START_PORT}_${END_PORT}.pids"

if [[ -f "$ALL_PIDS_FILE" ]]; then
  while read -r PID; do
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
      kill "$PID" 2>/dev/null || true
    fi
  done < "$ALL_PIDS_FILE"
  rm -f "$ALL_PIDS_FILE"
fi

STOPPED=0
for PORT in $(seq "$START_PORT" "$END_PORT"); do
  PID_FILE="$PID_DIR/socat_${PORT}.pid"
  if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID" 2>/dev/null || true
      STOPPED=$((STOPPED + 1))
    fi
    rm -f "$PID_FILE"
  fi

  for EXISTING_PID in $(lsof -ti :"$PORT" 2>/dev/null || true); do
    if ps -p "$EXISTING_PID" -o comm= 2>/dev/null | grep -q '^socat$'; then
      kill "$EXISTING_PID" 2>/dev/null || true
      STOPPED=$((STOPPED + 1))
    fi
  done
done

echo "Stopped forwarding processes: $STOPPED"
