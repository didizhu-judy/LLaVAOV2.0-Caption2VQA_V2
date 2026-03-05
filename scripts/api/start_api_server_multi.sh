#!/bin/bash
#SBATCH --job-name=sglang-multi-models
#SBATCH --output=logs/slurm_%j.out
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h200:8
#SBATCH --partition=lrc-xlong
#SBATCH --reservation=3dvit

set -euo pipefail

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

setup_venv_cudnn_libpath() {
  local cudnn_lib_dir=""
  cudnn_lib_dir="$(python - <<'PY'
import site
from pathlib import Path

paths = []
try:
    paths.extend(site.getsitepackages())
except Exception:
    pass

try:
    user_site = site.getusersitepackages()
except Exception:
    user_site = None

if user_site:
    paths.append(user_site)

for base in paths:
    candidate = Path(base) / "nvidia" / "cudnn" / "lib"
    if candidate.exists():
        print(candidate)
        break
PY
)"

  if [[ -n "${cudnn_lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${cudnn_lib_dir}:${LD_LIBRARY_PATH:-}"
    log "Prepended cuDNN library path: ${cudnn_lib_dir}"
  else
    log "No venv cuDNN library path detected via Python site-packages"
  fi
}

prefetch_model_artifacts() {
  if [[ "${SGLANG_PREFETCH_MODEL:-1}" != "1" ]]; then
    log "Skip model prefetch (SGLANG_PREFETCH_MODEL=${SGLANG_PREFETCH_MODEL:-1})"
    return
  fi

  declare -A seen_models=()
  for model in "${MODEL_ARRAY[@]}"; do
    if [[ -n "${seen_models[$model]+x}" ]]; then
      continue
    fi
    seen_models[$model]=1
    log "Prefetching model artifacts: ${model}"
    python - "$model" <<'PY'
import os
import sys

# 代理 + 可选禁用 SSL 校验（走企业 TLS 拦截代理时设 HF_HUB_DISABLE_SSL_VERIFY=1）
proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or os.environ.get("http_proxy")
disable_verify = os.environ.get("HF_HUB_DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes")
if proxy or disable_verify:
    try:
        import httpx
        from huggingface_hub import set_client_factory

        def _client_factory():
            kw = {"follow_redirects": True, "timeout": 60.0}
            if proxy:
                kw["proxy"] = proxy
            if disable_verify:
                kw["verify"] = False
            return httpx.Client(**kw)

        set_client_factory(_client_factory)
    except Exception:
        pass

from huggingface_hub import snapshot_download

model_id = sys.argv[1]
cache_dir = os.environ.get("HF_HUB_CACHE")

allow_patterns = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.json",
    "*.txt",
    "*.model",
    "*.spm",
    "*.py",
    "tokenizer*",
    "vocab*",
    "merges.txt",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
]

local_path = snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    resume_download=True,
    allow_patterns=allow_patterns,
)
print(f"[prefetch] ready: {model_id} -> {local_path}")
PY
  done
}

preflight_cudnn_check() {
  python - <<'PY'
import sys
import torch

torch_ver = torch.__version__
try:
    cudnn_raw = torch.backends.cudnn.version()
except Exception:
    cudnn_raw = None

if cudnn_raw is None:
    print(f"[preflight] torch={torch_ver}, cudnn=unknown")
    sys.exit(0)

major_minor = float(str(cudnn_raw)[:3]) / 100.0
print(f"[preflight] torch={torch_ver}, cudnn={major_minor:.2f} (raw={cudnn_raw})")

if torch_ver.startswith("2.9.1") and major_minor < 9.15:
    print(
        "[preflight] ERROR: torch 2.9.1 requires cuDNN >= 9.15 for stable Conv3d.\n"
        "Please run: uv pip install nvidia-cudnn-cu12==9.16.0.29"
    )
    sys.exit(2)
PY
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$SLURM_SUBMIT_DIR"
  cd "$ROOT"
else
  ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
  cd "$ROOT"
fi

mkdir -p "$ROOT/logs" "$ROOT/runtime"
LOG_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Use a writable project-local HF cache by default to avoid shared-path lock issues.
# Allow explicit override via SGLANG_HF_* env vars when needed.
export HF_HOME="${SGLANG_HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${SGLANG_HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="${SGLANG_TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

# 代理：HuggingFace 拉取与网络访问（可选；未设置时使用项目默认代理）
export http_proxy="${HTTP_PROXY:-${http_proxy:-http://172.16.5.79:18000}}"
export https_proxy="${HTTPS_PROXY:-${https_proxy:-$http_proxy}}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
log "Using proxy: http_proxy=$http_proxy"

log "start_api_server_multi.sh started"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  log "Slurm job: $SLURM_JOB_ID, node list: ${SLURM_NODELIST:-unknown}"
fi
log "HF cache: HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  source "$ROOT/.venv/bin/activate"
  log "Activated virtualenv: $ROOT/.venv"
else
  log "Virtualenv not found at $ROOT/.venv; using system Python"
fi
setup_venv_cudnn_libpath
preflight_cudnn_check

source "$ROOT/scripts/env/vllm_model.env" 2>/dev/null || true
MODEL_FROM_ENV="${VLLM_MODEL:-${SGLANG_MODEL:-${VLLM_MODEL_DEFAULT:-}}}"

if [[ -z "${SGLANG_MODELS:-}" ]]; then
  if [[ -z "$MODEL_FROM_ENV" ]]; then
    echo "SGLANG_MODELS and default model are both empty"
    exit 1
  fi
  N="${NUM_SGLANG_INSTANCES:-8}"
  SGLANG_MODELS=""
  for ((i=0;i<N;i++)); do
    [[ -n "$SGLANG_MODELS" ]] && SGLANG_MODELS+=","
    SGLANG_MODELS+="$MODEL_FROM_ENV"
  done
fi

IFS=',' read -ra MODEL_ARRAY <<< "$SGLANG_MODELS"
NUM_MODELS=${#MODEL_ARRAY[@]}

if [[ -n "${SGLANG_DP_PER_MODEL:-}" ]]; then
  IFS=',' read -ra DP_ARRAY <<< "$SGLANG_DP_PER_MODEL"
else
  DP_ARRAY=()
  for _ in $(seq 1 "$NUM_MODELS"); do DP_ARRAY+=("1"); done
fi

if [[ -n "${SGLANG_TP_PER_MODEL:-}" ]]; then
  IFS=',' read -ra TP_ARRAY <<< "$SGLANG_TP_PER_MODEL"
else
  TP_ARRAY=()
  for _ in $(seq 1 "$NUM_MODELS"); do TP_ARRAY+=("1"); done
fi
IFS=$' \t\n'

if [[ ${#DP_ARRAY[@]} -ne "$NUM_MODELS" ]] || [[ ${#TP_ARRAY[@]} -ne "$NUM_MODELS" ]]; then
  echo "DP/TP size mismatch with model count"
  exit 1
fi

BASE_PORT="${SGLANG_BASE_PORT:-10025}"
# 上下文越长显存占用越大；超长样本会被服务端截断（可接受则用默认 20K 省显存）。长答案时设 SGLANG_MAX_MODEL_LEN=32768
MAX_LEN="${SGLANG_MAX_MODEL_LEN:-20480}"
MEM_FRAC="${SGLANG_MEM_FRACTION:-0.85}"
HOST="${SGLANG_HOST:-0.0.0.0}"
HEALTH_TIMEOUT_SEC="${SGLANG_HEALTH_TIMEOUT_SEC:-1800}"
HEALTH_CHECK_INTERVAL_SEC="${SGLANG_HEALTH_CHECK_INTERVAL_SEC:-5}"
HEALTH_MAX_ATTEMPTS=$(( (HEALTH_TIMEOUT_SEC + HEALTH_CHECK_INTERVAL_SEC - 1) / HEALTH_CHECK_INTERVAL_SEC ))

GPU_OFFSET=0
TOTAL_NEEDED=0
declare -a GPU_START_ARRAY GPU_END_ARRAY
for i in $(seq 0 $((NUM_MODELS - 1))); do
  DP=${DP_ARRAY[$i]}
  TP=${TP_ARRAY[$i]}
  NEED=$((DP * TP))
  GPU_START_ARRAY[$i]=$GPU_OFFSET
  GPU_END_ARRAY[$i]=$((GPU_OFFSET + NEED - 1))
  GPU_OFFSET=$((GPU_OFFSET + NEED))
  TOTAL_NEEDED=$((TOTAL_NEEDED + NEED))
done

# 若父进程设置了 CUDA_VISIBLE_DEVICES（如 4,5,6,7），子进程须使用其中的物理 ID
GPU_LIST=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
fi

ALLOCATED_GPUS="${SLURM_GPUS_ON_NODE:-8}"
if [[ -n "${SLURM_JOB_ID:-}" && "$TOTAL_NEEDED" -gt "$ALLOCATED_GPUS" ]]; then
  echo "Need $TOTAL_NEEDED GPUs but only $ALLOCATED_GPUS allocated"
  exit 1
fi

log "Preparing $NUM_MODELS instance(s), base port: $BASE_PORT, host: $HOST"
log "GPU needed: $TOTAL_NEEDED, allocated: $ALLOCATED_GPUS, max_len: $MAX_LEN, mem_fraction: $MEM_FRAC"
log "Health check timeout: ${HEALTH_TIMEOUT_SEC}s (attempts=${HEALTH_MAX_ATTEMPTS}, interval=${HEALTH_CHECK_INTERVAL_SEC}s)"

PIDS=()
cleanup() {
  log "Cleanup triggered; stopping model processes if still alive"
  for PID in "${PIDS[@]:-}"; do
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

prefetch_model_artifacts

for i in $(seq 0 $((NUM_MODELS - 1))); do
  MODEL=${MODEL_ARRAY[$i]}
  DP=${DP_ARRAY[$i]}
  TP=${TP_ARRAY[$i]}
  PORT=$((BASE_PORT + i))
  GPU_START=${GPU_START_ARRAY[$i]}
  GPU_END=${GPU_END_ARRAY[$i]}

  GPUS=""
  if [[ ${#GPU_LIST[@]} -gt 0 ]]; then
    for g in $(seq "$GPU_START" "$GPU_END"); do
      [[ -n "$GPUS" ]] && GPUS+=","
      GPUS+="${GPU_LIST[$g]}"
    done
  else
    for g in $(seq "$GPU_START" "$GPU_END"); do
      [[ -n "$GPUS" ]] && GPUS+=","
      GPUS+="$g"
    done
  fi

  LOG="$ROOT/logs/sglang_model_${i}_${LOG_ID}.log"
  log "Launching instance=$i port=$PORT dp=$DP tp=$TP gpus=$GPUS model=$MODEL log=$LOG"
  CHUNKED_PREFILL="${SGLANG_CHUNKED_PREFILL_SIZE:-10240}"

  CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT=$((29500 + i)) python -m sglang.launch_server \
    --model-path "$MODEL" \
    --dp "$DP" \
    --tp "$TP" \
    --host "$HOST" \
    --port "$PORT" \
    --context-length "$MAX_LEN" \
    --mem-fraction-static "$MEM_FRAC" \
    --trust-remote-code \
    $([[ -n "$CHUNKED_PREFILL" && "$CHUNKED_PREFILL" != "-" ]] && echo "--chunked-prefill-size $CHUNKED_PREFILL") \
    --prefill-attention-backend triton \
    --decode-attention-backend triton \
    --disable-cuda-graph \
    > "$LOG" 2>&1 &

  PIDS+=("$!")
  log "Instance $i started with PID ${PIDS[$i]}"
done

log "Waiting for all instances to become healthy (timeout: ${HEALTH_TIMEOUT_SEC}s)"
for attempt in $(seq 1 "$HEALTH_MAX_ATTEMPTS"); do
  ALL_READY=true
  NOT_READY_PORTS=()
  for i in $(seq 0 $((NUM_MODELS - 1))); do
    PORT=$((BASE_PORT + i))
    if ! curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" | grep -q 200; then
      ALL_READY=false
      NOT_READY_PORTS+=("$PORT")
    fi
  done

  if [[ "$ALL_READY" == "true" ]]; then
    log "All instances are healthy after attempt $attempt"
    break
  fi

  if [[ "$attempt" -le 3 || $((attempt % 6)) -eq 0 ]]; then
    log "Health check attempt $attempt/${HEALTH_MAX_ATTEMPTS} not ready ports: ${NOT_READY_PORTS[*]}"
  fi

  for PID in "${PIDS[@]}"; do
    if ! kill -0 "$PID" 2>/dev/null; then
      log "A model process exited early"
      for i in $(seq 0 $((NUM_MODELS - 1))); do
        LOG="$ROOT/logs/sglang_model_${i}_${LOG_ID}.log"
        if [[ -f "$LOG" ]]; then
          log "Tail of $LOG:"
          tail -n 20 "$LOG" || true
        fi
      done
      exit 1
    fi
  done
  sleep "$HEALTH_CHECK_INTERVAL_SEC"
done

if [[ "$ALL_READY" != "true" ]]; then
  log "Not all endpoints became ready in time"
  for i in $(seq 0 $((NUM_MODELS - 1))); do
    LOG="$ROOT/logs/sglang_model_${i}_${LOG_ID}.log"
    if [[ -f "$LOG" ]]; then
      log "Tail of $LOG:"
      tail -n 20 "$LOG" || true
    fi
  done
  exit 1
fi

MANIFEST="$ROOT/runtime/endpoints.local.json"
{
  echo "{"
  echo "  \"groups\": {"
  echo "    \"local_multi\": {"
  echo "      \"generated_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "      \"endpoints\": ["
  for i in $(seq 0 $((NUM_MODELS - 1))); do
    PORT=$((BASE_PORT + i))
    MODEL=${MODEL_ARRAY[$i]}
    SEP=","
    if [[ "$i" -eq $((NUM_MODELS - 1)) ]]; then
      SEP=""
    fi
    cat <<JSON
        {
          "name": "local-${i}",
          "provider": "openai_compatible",
          "url": "http://127.0.0.1:${PORT}/v1/chat/completions",
          "model": "${MODEL}",
          "auth_type": "none",
          "max_concurrent": ${LOCAL_MAX_CONCURRENT:-256},
          "weight": 1.0
        }${SEP}
JSON
  done
  echo "      ]"
  echo "    }"
  echo "  }"
  echo "}"
} > "$MANIFEST"

log "Wrote runtime endpoint registry: $MANIFEST"
log "Keeping job alive and waiting on all model processes"

wait
