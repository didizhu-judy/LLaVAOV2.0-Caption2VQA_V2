#!/bin/bash
#SBATCH --job-name=llm-api-server
#SBATCH --output=logs/slurm_%j.out
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h200:8
#SBATCH --partition=lrc-xlong

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
  log "Prefetching model artifacts: ${MODEL}"
  python - "$MODEL" <<'PY'
import os
import sys

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

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  source "$ROOT/.venv/bin/activate"
  log "Activated virtualenv: $ROOT/.venv"
else
  log "Virtualenv not found at $ROOT/.venv; using system Python"
fi
setup_venv_cudnn_libpath

source "$ROOT/scripts/env/vllm_model.env" 2>/dev/null || true

BACKEND="${BACKEND:-sglang}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10025}"
MODEL="${MODEL:-${VLLM_MODEL:-${SGLANG_MODEL:-${VLLM_MODEL_DEFAULT:-}}}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "MODEL is empty. Set MODEL or scripts/env/vllm_model.env"
  exit 1
fi

prefetch_model_artifacts

if lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "Port $PORT is already in use"
  exit 1
fi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HOME="${SGLANG_HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${SGLANG_HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="${SGLANG_TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
log "HF cache: HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE"
preflight_cudnn_check

LOG_FILE="$ROOT/logs/api_${BACKEND}_${LOG_ID}.log"

if [[ "$BACKEND" == "sglang" ]]; then
  DP="${SGLANG_DP:-8}"
  TP="${SGLANG_TP:-1}"
  MAX_LEN="${SGLANG_MAX_MODEL_LEN:-20480}"
  MEM_FRAC="${SGLANG_MEM_FRACTION:-0.85}"
  # 多模态 prefill 过短会报 400（prompt too long）；由 estimate_openbee_prefill_tokens.py 统计建议 10240
  CHUNKED_PREFILL="${SGLANG_CHUNKED_PREFILL_SIZE:-10240}"

  python -m sglang.launch_server \
    --model-path "$MODEL" \
    --dp "$DP" \
    --tp "$TP" \
    --host "$HOST" \
    --port "$PORT" \
    --context-length "$MAX_LEN" \
    --mem-fraction-static "$MEM_FRAC" \
    --trust-remote-code \
    $([[ -n "$CHUNKED_PREFILL" && "$CHUNKED_PREFILL" != "-" ]] && echo "--chunked-prefill-size $CHUNKED_PREFILL") \
    > "$LOG_FILE" 2>&1 &
else
  TP="${VLLM_TP:-8}"
  MAX_LEN="${VLLM_MAX_MODEL_LEN:-20000}"
  GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"

  uv run vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --trust-remote-code \
    > "$LOG_FILE" 2>&1 &
fi

SERVER_PID=$!
echo "$SERVER_PID" > "$ROOT/runtime/api_server.pid"

echo "Started $BACKEND server pid=$SERVER_PID log=$LOG_FILE"

READY=false
HEALTH_TIMEOUT_SEC="${SGLANG_HEALTH_TIMEOUT_SEC:-1800}"
HEALTH_CHECK_INTERVAL_SEC="${SGLANG_HEALTH_CHECK_INTERVAL_SEC:-5}"
HEALTH_MAX_ATTEMPTS=$(( (HEALTH_TIMEOUT_SEC + HEALTH_CHECK_INTERVAL_SEC - 1) / HEALTH_CHECK_INTERVAL_SEC ))
log "Health check timeout: ${HEALTH_TIMEOUT_SEC}s (attempts=${HEALTH_MAX_ATTEMPTS}, interval=${HEALTH_CHECK_INTERVAL_SEC}s)"

for _ in $(seq 1 "$HEALTH_MAX_ATTEMPTS"); do
  if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" | grep -q 200; then
    READY=true
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server exited early; check $LOG_FILE"
    exit 1
  fi
  sleep "$HEALTH_CHECK_INTERVAL_SEC"
done

if [[ "$READY" != "true" ]]; then
  echo "Server did not become ready in time"
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

cat > "$ROOT/runtime/endpoints.local.json" <<JSON
{
  "groups": {
    "local_multi": {
      "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
      "endpoints": [
        {
          "name": "local-0",
          "provider": "openai_compatible",
          "url": "http://127.0.0.1:${PORT}/v1/chat/completions",
          "model": "${MODEL}",
          "auth_type": "none",
          "max_concurrent": ${LOCAL_MAX_CONCURRENT:-256},
          "weight": 1.0
        }
      ]
    }
  }
}
JSON

echo "Wrote runtime endpoint registry: runtime/endpoints.local.json"

wait "$SERVER_PID"
