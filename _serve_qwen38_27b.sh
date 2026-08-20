#!/usr/bin/env bash
# Host Qwen3.8-27B (native VLM; we serve text-only).
# Architecture is Qwen3_5ForConditionalGeneration; hidden=5120 so TP=2 is valid.
# Default: GPUs 4,5 port 8201, max_len=262144.
# Optional LoRA: ENABLE_LORA=1 LORA_NAME=visapp-grpo-r8 LORA_PATH=... MAX_LORA_RANK=16
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GPUS="${GPUS:-4,5}"
PORT="${PORT:-8201}"
HOST="${HOST:-0.0.0.0}"
MAX_LEN="${MAX_LEN:-262144}"
UTIL="${UTIL:-0.90}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
ENABLE_LORA="${ENABLE_LORA:-0}"
LORA_NAME="${LORA_NAME:-visapp-grpo-r8}"
LORA_PATH="${LORA_PATH:-}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
LOG_DIR="${LOG_DIR:-$PWD/_rl_runs/vllm_logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/qwen38_27b_${PORT}.log"
PIDF="$LOG_DIR/qwen38_27b_${PORT}.pid"

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

LORA_ARGS=()
if [[ "$ENABLE_LORA" == "1" ]]; then
  [[ -n "$LORA_PATH" && -d "$LORA_PATH" ]] || { echo "ERROR: LORA_PATH missing: $LORA_PATH" >&2; exit 2; }
  LORA_ARGS+=(--enable-lora --max-lora-rank "$MAX_LORA_RANK"
              --lora-modules "${LORA_NAME}=${LORA_PATH}")
  echo "[serve] LoRA $LORA_NAME <- $LORA_PATH rank<=$MAX_LORA_RANK"
fi

echo "[serve] starting $MODEL  GPUs=$GPUS tp=$TP max_len=$MAX_LEN host=$HOST port=$PORT"
nohup vllm serve "$MODEL" \
  --served-model-name "$MODEL" \
  --host "$HOST" --port "$PORT" \
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
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  "${LORA_ARGS[@]}" \
  >"$LOG" 2>&1 &
echo $! > "$PIDF"
echo "[serve] pid=$(cat "$PIDF") log=$LOG"
echo "[serve] wait for http://127.0.0.1:${PORT}/v1/models (also ${HOST}:${PORT} if not loopback)"
