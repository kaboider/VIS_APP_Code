#!/usr/bin/env bash
# Skip vLLM rollout. Replay 5_travel-booking x8 with reward=1, one 27B LoRA step.
set -euo pipefail

VIS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERA="${MERA_ROOT:-/shared_home/yuhang.yao/MERA-Evolve}"
PYTHON="${PYTHON:-$MERA/.venv_qwen35/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$MERA/venv/bin/python"
OUT="${OUTPUT_DIR:-$VIS/_rl_runs/grpo_27b_replay}"
LOG="$OUT/train.log"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export FLA_TILELANG="${FLA_TILELANG:-0}"
export QWEN35_ENABLE_VERL_PATCHES="${QWEN35_ENABLE_VERL_PATCHES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${QWEN35_TRAIN_TRITON_OVERLAY:-$MERA/.deps/qwen35-triton33}:$MERA/experiments/tau-2/compat/qwen35_torch_fallback:${PYTHONPATH:-}"

echo "[replay] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python=$PYTHON"
echo "[replay] log=$LOG"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

set +e
"$PYTHON" -u "$VIS/rl/train_grpo_replay_27b.py" \
  --model "${MODEL_PATH:-Qwen/Qwen3.8-27B}" \
  --out-dir "$OUT" \
  --n-copies "${N_COPIES:-8}" \
  --reward "${REWARD:-1.0}" \
  --max-prompt "${MAX_PROMPT_LENGTH:-1536}" \
  --max-response "${MAX_RESPONSE_LENGTH:-512}" \
  --lr "${ACTOR_LR:-1e-6}" \
  2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
echo "[replay] exit=$status log=$LOG"
exit "$status"
