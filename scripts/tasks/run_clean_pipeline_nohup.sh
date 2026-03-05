#!/bin/bash
# 后台跑清洗：25 个 jsonl、输出带 _v2、每文件累计 10 条 dirty 即停（先看质量）
# 日志: output/clean_v2_stop10dirty.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  source "$ROOT/.venv/bin/activate"
fi

export OPENBEE_25_ONLY=1
export OPENBEE_V2=1
export STOP_AFTER_DIRTY=10
export INPUT_DIR=/ov2/dataset_jsonl/openbee
export MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
export BACKEND_PROFILE="${BACKEND_PROFILE:-local_multi}"

LOG="${LOG:-$ROOT/output/clean_v2_stop10dirty.log}"
mkdir -p "$(dirname "$LOG")"
echo "=== $(date -Iseconds) 启动清洗 v2（每文件 10 dirty 即停）===" >> "$LOG"
exec bash "$ROOT/scripts/tasks/run_clean_pipeline.sh" >> "$LOG" 2>&1
