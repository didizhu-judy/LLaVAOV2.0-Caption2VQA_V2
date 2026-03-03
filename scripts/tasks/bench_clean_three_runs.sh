#!/bin/bash
# 同一批样本跑三组参数，对比 rec/s 与 400 错误数，找到并发/截断的瓶颈。
# 用法:
#   INPUT=/ov2/dataset_jsonl/openbee/MAVIS_Function.jsonl MAX_RECORDS=100 bash scripts/tasks/bench_clean_three_runs.sh
# 可选: BACKEND_PROFILE=local_multi（本地 8 实例时用 runtime/endpoints.local.json）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

INPUT="${INPUT:-}"
MAX_RECORDS="${MAX_RECORDS:-100}"
BACKEND_PROFILE="${BACKEND_PROFILE:-local_multi}"

# 三组参数: "workers:concurrency:max_prefill_tokens"（max_prefill 空则用 yaml 默认）
# 组1 保守: 低并发 + 小 prefill，少 400
# 组2 中等: 默认档
# 组3 激进: 高并发，测瓶颈
GROUP1="${BENCH_GROUP1:-4:4:4096}"
GROUP2="${BENCH_GROUP2:-8:8:8192}"
GROUP3="${BENCH_GROUP3:-8:16:8192}"

if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
  echo "Usage: INPUT=/path/to/file.jsonl [MAX_RECORDS=100] [BENCH_GROUP1='4:4:4096'] [BENCH_GROUP2='8:8:8192'] [BENCH_GROUP3='8:16:8192'] bash $0"
  echo "  BENCH_GROUPn = workers:concurrency:max_prefill_tokens (empty token = use config default)"
  exit 1
fi

BENCH_DIR="${ROOT}/output/bench_clean_three"
mkdir -p "$BENCH_DIR"
INPUT_STEM="$(basename "${INPUT%.*}")"

run_one_group() {
  local id="$1"
  local label="$2"
  local workers="$3"
  local concurrency="$4"
  local max_prefill="$5"

  local JUDGED_JSONL="$BENCH_DIR/${INPUT_STEM}_run${id}_judged.jsonl"
  local CLEAN_JSONL="$BENCH_DIR/${INPUT_STEM}_run${id}_clean.jsonl"
  local DIRTY_JSONL="$BENCH_DIR/${INPUT_STEM}_run${id}_dirty.jsonl"
  local ERROR_JSONL="$BENCH_DIR/${INPUT_STEM}_run${id}_errors.jsonl"

  export RESUME=0
  export WORKERS="$workers"
  export MAX_RECORDS="$MAX_RECORDS"
  export JUDGED_JSONL
  export CLEAN_JSONL
  export DIRTY_JSONL
  export PIPELINE_NUM_WORKERS="$workers"
  export PIPELINE_WORKER_ASYNC_CONCURRENCY="$concurrency"
  export PIPELINE_OUTPUT_JSONL="$JUDGED_JSONL"
  export PIPELINE_ERROR_JSONL="$ERROR_JSONL"
  export INPUT
  export BACKEND_PROFILE
  if [[ -n "$max_prefill" ]]; then
    export MAX_PREFILL_TOKENS="$max_prefill"
  else
    unset MAX_PREFILL_TOKENS
  fi

  echo "[Run $id] $label (W=$workers C=$concurrency M=${max_prefill:-default}) ..." >&2
  t0=$(date +%s.%N)
  bash scripts/tasks/run_clean_pipeline.sh >/dev/null 2>&1
  t1=$(date +%s.%N)

  local n_judged=$(wc -l < "$JUDGED_JSONL" 2>/dev/null || echo 0)
  local n_err=0
  [[ -f "$ERROR_JSONL" ]] && n_err=$(wc -l < "$ERROR_JSONL" 2>/dev/null || echo 0)
  local elapsed
  elapsed=$(python3 -c "print(round($t1 - $t0, 2))")
  local rec_s="N/A"
  if [[ "$n_judged" -gt 0 ]]; then
    rec_s=$(python3 -c "print(round($n_judged / ($t1 - $t0), 2))")
  fi
  echo "run$id|$label|$workers|$concurrency|${max_prefill:-default}|$n_judged|$n_err|$elapsed|$rec_s"
}

echo "Input: $INPUT (max_records=$MAX_RECORDS) BACKEND_PROFILE=$BACKEND_PROFILE"
echo "Groups: 1=$GROUP1  2=$GROUP2  3=$GROUP3"
echo "---"

RESULTS=$(mktemp)
trap 'rm -f "$RESULTS"' EXIT

# Group 1
IFS=: read -r w c m <<< "$GROUP1"
run_one_group 1 "conservative" "$w" "$c" "$m" >> "$RESULTS"

# Group 2
IFS=: read -r w c m <<< "$GROUP2"
run_one_group 2 "medium" "$w" "$c" "$m" >> "$RESULTS"

# Group 3
IFS=: read -r w c m <<< "$GROUP3"
run_one_group 3 "aggressive" "$w" "$c" "$m" >> "$RESULTS"

echo "---"
echo "Summary (same sample, three param sets):"
printf "%-6s %-12s %3s %4s %8s %8s %6s %7s %6s\n" "Run" "Label" "W" "C" "M_prefill" "records" "errors" "time_s" "rec/s"
echo "------ ------------ --- ---- -------- -------- ------ ------- ------"
while IFS='|' read -r run label workers concurrency mprefill records errors elapsed rec_s; do
  printf "%-6s %-12s %3s %4s %8s %8s %6s %7s %6s\n" "$run" "$label" "$workers" "$concurrency" "$mprefill" "$records" "$errors" "$elapsed" "$rec_s"
done < "$RESULTS"
echo "---"
echo "Conclusion: pick the run with highest rec/s and acceptable errors; if errors>0 try lower M_prefill or next run's settings."
echo "Outputs: $BENCH_DIR"
