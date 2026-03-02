#!/bin/bash
# 本地 4 卡部署 Qwen3-VL-32B（SGLang）：每卡 1 个 SGLang 实例（共 4 实例，端口 10025–10028）。
# 使用项目默认代理拉取模型；可通过 HTTP_PROXY/HTTPS_PROXY 覆盖。
# 用法：在项目根目录执行
#   bash scripts/tasks/run_local_qwen3vl_4gpu.sh
# 或先设置代理再执行：
#   export HTTP_PROXY=http://172.16.5.79:18000
#   bash scripts/tasks/run_local_qwen3vl_4gpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# 代理（未设置时由 start_api_server_multi.sh 使用脚本内默认）
export HTTP_PROXY="${HTTP_PROXY:-http://172.16.5.79:18000}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export HF_HUB_DISABLE_SSL_VERIFY="${HF_HUB_DISABLE_SSL_VERIFY:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 4 卡，每卡 1 个模型实例（DP=1, TP=1 由 multi 脚本默认）
export NUM_SGLANG_INSTANCES=4
export SGLANG_MODEL="${SGLANG_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
export SGLANG_BASE_PORT="${SGLANG_BASE_PORT:-10025}"
# 仅使用前 4 张 GPU（若机器多于 4 卡可改 CUDA_VISIBLE_DEVICES）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=== 本地 4 卡 Qwen3-VL-32B（SGLang）==="
echo "  后端:   SGLang (sglang.launch_server)"
echo "  实例数: $NUM_SGLANG_INSTANCES（1 卡 1 实例）"
echo "  模型:   $SGLANG_MODEL"
echo "  端口:   $SGLANG_BASE_PORT - $((SGLANG_BASE_PORT + NUM_SGLANG_INSTANCES - 1))"
echo "  代理:   $HTTP_PROXY"
echo "  HF:    $HF_ENDPOINT"
echo "==============================="

exec bash "$ROOT/scripts/api/start_api_server_multi.sh"
