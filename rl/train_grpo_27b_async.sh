#!/usr/bin/env bash
# 27B LoRA GRPO, one-step-off async: 1 GPU actor + 2 GPU vLLM TP=2.
# Occupies only CUDA_VISIBLE_DEVICES (default 4,5,6). Do not touch other cards.
set -euo pipefail

VIS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERA="${MERA_ROOT:-/shared_home/yuhang.yao/MERA-Evolve}"
PYTHON="${PYTHON:-$MERA/.venv_qwen35/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$MERA/venv/bin/python"

OUT="${OUTPUT_DIR:-$VIS/_rl_runs/grpo_27b_async}"
DATA="${DATA_DIR:-$VIS/_rl_runs/data_27b}"
LOG="$OUT/train.log"
UTIL="$OUT/gpu_util.csv"
mkdir -p "$OUT" "$DATA"
# Ray AF_UNIX sockets cap at 107 bytes; keep this path short.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/visapp27b_ray}"
mkdir -p "$RAY_TMPDIR"
# WorkerDict imports torch/CUDA from NFS; default 60s register timeout
# retries 5 times then ActorUnschedulableError. Prior 32-step runs recovered
# when a later worker finished in time; this keeps the first start alive.
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-180}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"
export PYTHON
export FLA_TILELANG="${FLA_TILELANG:-0}"
# vLLM sleep-mode CuMemAllocator rejects expandable_segments.
unset PYTORCH_CUDA_ALLOC_CONF

if [[ ! -f "$DATA/train.parquet" ]]; then
  "$PYTHON" "$VIS/rl/prepare_visapp_data.py" --out-dir "$DATA" --max-chars "${MAX_CHARS:-8000}" --repeat "${DATA_REPEAT:-4}"
fi

echo "[27b-async] gpus=$CUDA_VISIBLE_DEVICES out=$OUT register_timeout=${RAY_worker_register_timeout_seconds}s"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
# Warm NFS + CUDA so the first WorkerDict handshake is not a cold import.
_FIRST_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
echo "[27b-async] prewarm torch/CUDA on GPU ${_FIRST_GPU}"
CUDA_VISIBLE_DEVICES="${_FIRST_GPU}" "$PYTHON" -c "import torch; torch.cuda.init(); print('[27b-async] cuda', torch.cuda.get_device_name(0), flush=True)"

# Sample GPU util while training (physical indices).
(
  echo "ts,index,util,mem_mib"
  while true; do
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
      | awk -v ts="$ts" -F', ' '{gsub(/ /,"",$2); gsub(/ /,"",$3); print ts","$1","$2","$3}'
    sleep 15
  done
) >>"$UTIL" &
UTIL_PID=$!
trap 'kill "$UTIL_PID" 2>/dev/null || true' EXIT

# 1 GPU actor LoRA (stay resident on GPU 4) + 2 GPU vLLM TP=2.
# Do not param-offload the actor: reloading 27B after offload OOMs because
# Optimizer still offloads. LoRA is sent as vLLM add_lora (not NIXL full-param).
# Set VISAPP_SKIP_ROLLOUT_LORA=1 to restore the frozen-base generate path.
export VISAPP_SKIP_ROLLOUT_LORA="${VISAPP_SKIP_ROLLOUT_LORA:-0}"
export VISAPP_LORA_PEFT_JSON="${VISAPP_LORA_PEFT_JSON:-$RAY_TMPDIR/lora_peft_config.json}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
"$PYTHON" - <<PY
import json, os
from pathlib import Path
path = Path(os.environ["VISAPP_LORA_PEFT_JSON"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "r": int(os.environ.get("LORA_RANK", "16")),
    "lora_alpha": int(os.environ.get("LORA_ALPHA", "32")),
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.0,
    "bias": "none",
}), encoding="utf-8")
print(f"[27b-async] wrote peft json {path}")
PY
# Do not inherit TOTAL_STEPS=8 from older shells.
VISAPP_TOTAL_STEPS="${VISAPP_TOTAL_STEPS:-32}"
set +e
N_GPUS=1 \
ROLLOUT_N_GPUS=2 \
ROLLOUT_TP=2 \
SEPARATE_ROLLOUT=1 \
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.8-27B}" \
TRAIN_FILE="$DATA/train.parquet" \
VAL_FILE="$DATA/val.parquet" \
OUTPUT_DIR="$OUT" \
PROJECT_NAME=visapp_camel_rl \
EXPERIMENT_NAME=grpo_27b_async \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}" \
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
N_GENERATIONS="${N_GENERATIONS:-8}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}" \
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-4096}" \
MAX_TURNS="${MAX_TURNS:-8}" \
TOTAL_STEPS="$VISAPP_TOTAL_STEPS" \
SAVE_FREQ="${SAVE_FREQ:-32}" \
LORA_RANK="${LORA_RANK:-16}" \
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-8}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}" \
CHECKPOINT_BACKEND="${CHECKPOINT_BACKEND:-nixl_tau2}" \
CHECKPOINT_CUSTOM_BACKEND_MODULE="${CHECKPOINT_CUSTOM_BACKEND_MODULE:-tau2_evolve.nixl_checkpoint}" \
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}" \
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}" \
TOOL_FORMAT="${TOOL_FORMAT:-qwen3_coder}" \
ACTOR_LR="${ACTOR_LR:-2e-6}" \
TRUNCATION="${TRUNCATION:-left}" \
FSDP_MODEL_DTYPE="${FSDP_MODEL_DTYPE:-bf16}" \
FSDP_PARAM_OFFLOAD="${FSDP_PARAM_OFFLOAD:-False}" \
FSDP_OPTIMIZER_OFFLOAD="${FSDP_OPTIMIZER_OFFLOAD:-True}" \
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-False}" \
REQUIRE_ADAPTER="${REQUIRE_ADAPTER:-0}" \
bash "$VIS/rl/train_grpo.sh" \
  +actor_rollout_ref.rollout.enable_sleep_mode=False \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  trainer.resume_mode=disable \
  trainer.total_training_steps="$VISAPP_TOTAL_STEPS" \
  actor_rollout_ref.actor.use_kl_loss=False \
  2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
kill "$UTIL_PID" 2>/dev/null || true
trap - EXIT
echo "[27b-async] exit=$status log=$LOG util=$UTIL"
exit "$status"
