#!/bin/bash
# 一键：先部署 SGLang 模型，等 4 个实例全部健康后再跑清洗任务（v2、每文件 10 dirty 即停）。
# 用法：
#   bash scripts/tasks/deploy_and_clean_v2.sh           # 前台跑清洗，日志同时写 output/clean_v2_stop10dirty.log
#   nohup bash scripts/tasks/deploy_and_clean_v2.sh &   # 整条链路后台跑
#
# 若本机 10025–10028 已就绪则跳过部署直接跑清洗。可通过 SKIP_DEPLOY=1 强制跳过部署。
# 部署前会先结束已有 SGLang 相关进程以释放 GPU，避免 OOM；设 SKIP_CLEANUP=1 可跳过清理。
# 使用后 4 张卡（4,5,6,7）跑 SGLang，前 4 张可留给其他任务；可通过 CUDA_VISIBLE_DEVICES 覆盖。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# 配置（可环境变量覆盖）
SGLANG_BASE_PORT="${SGLANG_BASE_PORT:-10025}"
NUM_SGLANG_INSTANCES="${NUM_SGLANG_INSTANCES:-4}"
HEALTH_TIMEOUT_SEC="${SGLANG_DEPLOY_TIMEOUT_SEC:-1800}"
HEALTH_INTERVAL_SEC="${SGLANG_DEPLOY_POLL_INTERVAL:-10}"
SKIP_DEPLOY="${SKIP_DEPLOY:-0}"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"
# 默认后 4 张卡；可覆盖，如 CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
CLEAN_LOG="${CLEAN_LOG:-$ROOT/output/clean_v2_stop10dirty.log}"

mkdir -p "$ROOT/logs" "$ROOT/runtime" "$(dirname "$CLEAN_LOG")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# 检查 4 个端口是否都已健康
check_sglang_healthy() {
  local ok=0
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

# ---------- 1. 部署 SGLang（若未就绪） ----------
if [[ "$SKIP_DEPLOY" == "1" || "$SKIP_DEPLOY" == "true" || "$SKIP_DEPLOY" == "yes" ]]; then
  log "SKIP_DEPLOY=1，跳过 SGLang 部署"
elif check_sglang_healthy; then
  log "SGLang 已就绪（端口 ${SGLANG_BASE_PORT}-$((SGLANG_BASE_PORT + NUM_SGLANG_INSTANCES - 1))），跳过部署"
else
  # 部署前清理已有 SGLang 相关进程，释放 GPU，避免 CUDA OOM
  if [[ "$SKIP_CLEANUP" != "1" && "$SKIP_CLEANUP" != "true" && "$SKIP_CLEANUP" != "yes" ]]; then
    log "清理已有 SGLang/模型进程以释放 GPU..."
    pkill -f "run_local_sglang.sh|start_api_server_multi.sh" 2>/dev/null || true
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep 5
    log "清理完成"
  fi

  log "启动 SGLang（${NUM_SGLANG_INSTANCES} 实例，端口 ${SGLANG_BASE_PORT}-$((SGLANG_BASE_PORT + NUM_SGLANG_INSTANCES - 1))，GPU $CUDA_VISIBLE_DEVICES）..."
  SGLANG_LOG="$ROOT/logs/sglang_multi_$(date +%Y%m%d_%H%M%S).log"
  NUM_SGLANG_INSTANCES="$NUM_SGLANG_INSTANCES" SGLANG_BASE_PORT="$SGLANG_BASE_PORT" \
    nohup bash "$ROOT/scripts/tasks/run_local_sglang.sh" > "$SGLANG_LOG" 2>&1 &
  SGLANG_PID=$!
  log "SGLang 已后台启动 PID=$SGLANG_PID，日志 $SGLANG_LOG"

  log "等待实例健康（超时 ${HEALTH_TIMEOUT_SEC}s，每 ${HEALTH_INTERVAL_SEC}s 检查）..."
  deadline=$(($(date +%s) + HEALTH_TIMEOUT_SEC))
  while true; do
    if check_sglang_healthy; then
      log "SGLang 全部实例已健康"
      break
    fi
    if [[ $(date +%s) -ge "$deadline" ]]; then
      log "错误：超过 ${HEALTH_TIMEOUT_SEC}s 仍未就绪，退出"
      kill "$SGLANG_PID" 2>/dev/null || true
      exit 1
    fi
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
      log "错误：SGLang 进程已退出，请查看 $SGLANG_LOG"
      exit 1
    fi
    sleep "$HEALTH_INTERVAL_SEC"
  done
fi

# ---------- 2. 跑清洗任务 ----------
log "开始清洗任务（v2、每文件 10 dirty 即停），日志 $CLEAN_LOG"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  source "$ROOT/.venv/bin/activate"
fi

export OPENBEE_25_ONLY=1
export OPENBEE_V2=1
export STOP_AFTER_DIRTY=10
export INPUT_DIR="${INPUT_DIR:-/ov2/dataset_jsonl/openbee}"
export MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
export BACKEND_PROFILE="${BACKEND_PROFILE:-local_multi}"

echo "=== $(date -Iseconds) 启动清洗 v2（每文件 10 dirty 即停）===" >> "$CLEAN_LOG"
exec bash "$ROOT/scripts/tasks/run_clean_pipeline.sh" 2>&1 | tee -a "$CLEAN_LOG"
