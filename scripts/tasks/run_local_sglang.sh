#!/bin/bash
# 本地多卡部署 VLM（SGLang）：每卡 1 个 SGLang 实例。
# 使用项目默认代理拉取模型；可通过 HTTP_PROXY/HTTPS_PROXY 覆盖。
# 用法：在项目根目录执行
#   bash scripts/tasks/run_local_sglang.sh
# 或先设置代理再执行：
#   export HTTP_PROXY=http://172.16.5.79:18000
#   bash scripts/tasks/run_local_sglang.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# 代理（未设置时由 start_api_server_multi.sh 使用脚本内默认）
export HTTP_PROXY="${HTTP_PROXY:-http://172.16.5.79:18000}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
# 本机 127.0.0.1 不走代理，避免 warmup/health 请求被 Squid 拦截 403
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-$NO_PROXY}"
export HF_HUB_DISABLE_SSL_VERIFY="${HF_HUB_DISABLE_SSL_VERIFY:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 仅在本机使用时，默认只监听回环地址，避免对外网卡暴露端口
export SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"

# 8 卡，每卡 1 个模型实例（DP=1, TP=1 由 multi 脚本默认）
export NUM_SGLANG_INSTANCES=8
export SGLANG_MODEL="${SGLANG_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
export SGLANG_BASE_PORT="${SGLANG_BASE_PORT:-10025}"
# 使用前 8 张 GPU（若机器多于 8 卡可改 CUDA_VISIBLE_DEVICES）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

echo "=== 本地 SGLang 多实例部署 ==="
echo "  后端:   SGLang (sglang.launch_server)"
echo "  实例数: $NUM_SGLANG_INSTANCES（1 卡 1 实例）"
echo "  模型:   $SGLANG_MODEL"
echo "  端口:   $SGLANG_BASE_PORT - $((SGLANG_BASE_PORT + NUM_SGLANG_INSTANCES - 1))"
echo "  监听:   $SGLANG_HOST"
echo "  代理:   $HTTP_PROXY"
echo "  HF:    $HF_ENDPOINT"
echo "==============================="

exec bash "$ROOT/scripts/api/start_api_server_multi.sh"
