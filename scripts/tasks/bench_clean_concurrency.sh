#!/bin/bash
# 用同一批数据试多档并发，打表看哪档 rec/s 最高（一次跑完，不用手动一个个试）。
# 用法:
#   INPUT=/ov2/dataset_jsonl/openbee/xxx.jsonl MAX_RECORDS=100 bash scripts/tasks/bench_clean_concurrency.sh
#   MAX_RECORDS=50 bash scripts/tasks/bench_clean_concurrency.sh   # 需已设置 INPUT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

INPUT="${INPUT:-}"
MAX_RECORDS="${MAX_RECORDS:-100}"
# 要试的并发档位（每个 worker 的 async 并发数）
CONCURRENCIES="${CONCURRENCIES:-4 8 16 24 32}"
WORKERS="${WORKERS:-8}"

if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
  echo "Usage: INPUT=/path/to/one.jsonl [MAX_RECORDS=100] [CONCURRENCIES='4 8 16 32'] bash $0"
  echo "  Uses same input slice (MAX_RECORDS) for each concurrency, reports rec/s."
  exit 1
fi

BENCH_DIR="${ROOT}/output/bench_clean"
mkdir -p "$BENCH_DIR"
INPUT_STEM="$(basename "${INPUT%.*}")"

echo "Input: $INPUT (max_records=$MAX_RECORDS)"
echo "Workers: $WORKERS | Concurrency slots to try: $CONCURRENCIES"
echo "---"

for C in $CONCURRENCIES; do
  JUDGED_JSONL="$BENCH_DIR/${INPUT_STEM}_c${C}_judged.jsonl"
  CLEAN_JSONL="$BENCH_DIR/${INPUT_STEM}_c${C}_clean.jsonl"
  DIRTY_JSONL="$BENCH_DIR/${INPUT_STEM}_c${C}_dirty.jsonl"
  ERROR_JSONL="$BENCH_DIR/${INPUT_STEM}_c${C}_errors.jsonl"
  # 每次用独立输出 + RESUME=0，保证都真实跑
  t0=$(date +%s.%N)
  RESUME=0 WORKERS="$WORKERS" MAX_RECORDS="$MAX_RECORDS" \
  JUDGED_JSONL="$JUDGED_JSONL" \
  CLEAN_JSONL="$CLEAN_JSONL" \
  DIRTY_JSONL="$DIRTY_JSONL" \
  PIPELINE_NUM_WORKERS="$WORKERS" \
  PIPELINE_WORKER_ASYNC_CONCURRENCY="$C" \
  PIPELINE_OUTPUT_JSONL="$JUDGED_JSONL" \
  PIPELINE_ERROR_JSONL="$ERROR_JSONL" \
  INPUT="$INPUT" \
  bash scripts/tasks/run_clean_pipeline.sh >/dev/null 2>&1
  t1=$(date +%s.%N)
  elapsed=$(python3 -c "print(round($t1 - $t0, 2))")
  n=$(wc -l < "$JUDGED_JSONL" 2>/dev/null || echo 0)
  n_err=0
  [[ -f "$ERROR_JSONL" ]] && n_err=$(wc -l < "$ERROR_JSONL" 2>/dev/null || echo 0)
  if [[ "$n" -gt 0 ]]; then
    rec_s=$(python3 -c "print(round($n / ($t1 - $t0), 2))")
    echo "concurrency=$C  records=$n  errors=$n_err  time=${elapsed}s  rec/s=$rec_s"
  else
    echo "concurrency=$C  records=0  errors=$n_err  time=${elapsed}s  rec/s=N/A"
  fi
done

echo "---"
echo "Done. Best rec/s above is the concurrency to use (or higher if backend allows)."
