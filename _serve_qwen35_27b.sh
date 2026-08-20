#!/usr/bin/env bash
# Host Qwen3.5-27B at the model max context (262144).
# TP must divide hidden size 10240, so TP=2 (not 3). Default: GPUs 4,5 port 8201.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GPUS="${GPUS:-4,5}"
PORT="${PORT:-8201}"
MAX_LEN="${MAX_LEN:-262144}"
UTIL="${UTIL:-0.90}"
LOG_DIR="${LOG_DIR:-$PWD/_rl_runs/vllm_logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/qwen35_27b_${PORT}.log"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
TP="${#GPU_ARR[@]}"

export PATH="${MERA_VENV:-/shared_home/yuhang.yao/MERA-Evolve/.venv_qwen35}/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}/shared_home/yuhang.yao/MERA-Evolve/experiments/tau-2/compat/qwen35_torch_fallback"
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES="$GPUS"

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "[serve] port $PORT already listening — leaving it up"
  curl -sf "http://127.0.0.1:${PORT}/v1/models" | head -c 400; echo
  exit 0
fi

echo "[serve] starting Qwen/Qwen3.5-27B  GPUs=$GPUS tp=$TP max_len=$MAX_LEN port=$PORT"
nohup vllm serve Qwen/Qwen3.5-27B \
  --served-model-name Qwen/Qwen3.5-27B \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAX_LEN" \
  --dtype bfloat16 \
  --trust-remote-code \
  --language-model-only \
  --gdn-prefill-backend triton \
  --no-async-scheduling \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  >"$LOG" 2>&1 &
echo "[serve] pid=$! log=$LOG"
echo "[serve] wait for http://127.0.0.1:${PORT}/v1/models"
