#!/usr/bin/env bash
# VIS_APP camel-style GRPO (verl LoRA + multi-turn VisappAgentLoop).
# No docker. Token-level PRM / LLM-as-judge are out of scope.
set -euo pipefail

VIS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERA="${MERA_ROOT:-/shared_home/yuhang.yao/MERA-Evolve}"
PYTHON="${PYTHON:-$MERA/venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "[train_grpo] python not executable: $PYTHON" >&2; exit 127; }

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-$VIS/_rl_runs/data}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$VIS/_rl_runs/grpo}"
REWARD_FILE="${REWARD_FILE:-$VIS/rl/reward.py}"
TOOL_CONFIG="${TOOL_CONFIG:-$VIS/rl/config/tools.yaml}"
AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-$VIS/rl/config/agent_loop.yaml}"

N_GPUS="${N_GPUS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-2}"
SAVE_FREQ="${SAVE_FREQ:-$TOTAL_STEPS}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
N_GENERATIONS="${N_GENERATIONS:-2}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
MAX_TURNS="${MAX_TURNS:-8}"
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-1}"
REWARD_WORKERS="${REWARD_WORKERS:-1}"
TOOL_FORMAT="${TOOL_FORMAT:-hermes}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-4096}"
MAX_PARALLEL_CALLS="${MAX_PARALLEL_CALLS:-1}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"
PROJECT_NAME="${PROJECT_NAME:-visapp_camel_rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo}"
TRAINING_SEED="${TRAINING_SEED:-1}"
REQUIRE_ADAPTER="${REQUIRE_ADAPTER:-0}"
SEPARATE_ROLLOUT="${SEPARATE_ROLLOUT:-0}"
ROLLOUT_N_GPUS="${ROLLOUT_N_GPUS:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-24}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-False}"
CHECKPOINT_BUCKET_MB="${CHECKPOINT_BUCKET_MB:-256}"
CHECKPOINT_BACKEND="${CHECKPOINT_BACKEND:-nccl}"
CHECKPOINT_CUSTOM_BACKEND_MODULE="${CHECKPOINT_CUSTOM_BACKEND_MODULE:-}"
RAY_WORKER_SETUP_HOOK="${RAY_WORKER_SETUP_HOOK:-}"
MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-}"
VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-}"
VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-}"
ENABLE_THINKING="${ENABLE_THINKING:-}"
TRUNCATION="${TRUNCATION:-error}"
FSDP_MODEL_DTYPE="${FSDP_MODEL_DTYPE:-}"
FSDP_PARAM_OFFLOAD="${FSDP_PARAM_OFFLOAD:-}"
FSDP_OPTIMIZER_OFFLOAD="${FSDP_OPTIMIZER_OFFLOAD:-}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-}"

if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "[train_grpo] missing parquet; generating via prepare_visapp_data.py" >&2
  "$PYTHON" "$VIS/rl/prepare_visapp_data.py" --out-dir "$(dirname "$TRAIN_FILE")"
fi

export PYTHONPATH="$VIS:$MERA:${PYTHONPATH:-}"
cd "$MERA"

IS_QWEN35=0
case "$MODEL_PATH" in
  *Qwen3.5*|*Qwen3.8*|*Qwen3_5*) IS_QWEN35=1 ;;
esac
if [[ "$IS_QWEN35" == "1" ]]; then
  export QWEN35_ENABLE_VERL_PATCHES="${QWEN35_ENABLE_VERL_PATCHES:-1}"
  export FLA_TILELANG="${FLA_TILELANG:-0}"
  export QWEN35_TRAIN_TRITON_OVERLAY="${QWEN35_TRAIN_TRITON_OVERLAY:-$MERA/.deps/qwen35-triton33}"
  export MERA_ROOT="${MERA_ROOT:-$MERA}"
  # tau-2 + torch_fallback for MERA sitecustomize / Qwen35TrainingRayWorkerGroup.
  # Do NOT put Triton 3.3 overlay or rl/compat sitecustomize on the driver
  # PYTHONPATH: vLLM needs Triton 3.6, and a heavy sitecustomize stalls Ray
  # dashboard/metrics agents (raylet waits for metrics_agent_port).
  export PYTHONPATH="$MERA/experiments/tau-2:$MERA/experiments/tau-2/compat/qwen35_torch_fallback:$PYTHONPATH"
  : "${MODEL_ATTN_IMPLEMENTATION:=sdpa}"
  # GDN / Qwen3.5-3.8 training is unstable with flash-attn unpad.
  USE_REMOVE_PADDING="${QWEN35_USE_REMOVE_PADDING:-False}"
  : "${VLLM_LANGUAGE_MODEL_ONLY:=True}"
  : "${VLLM_GDN_PREFILL_BACKEND:=triton}"
  : "${ENABLE_THINKING:=false}"
  if [[ "$SEPARATE_ROLLOUT" == "1" ]]; then
    # Light attention patch only. Do not import one_step_off main_ppo in the
    # global Ray worker hook: that stalls dashboard/metrics agents and raylet.
    : "${RAY_WORKER_SETUP_HOOK:=tau2_evolve.qwen35_worker_setup.install_qwen35_worker_patches}"
    [[ -f "$QWEN35_TRAIN_TRITON_OVERLAY/triton/__init__.py" ]] || {
      echo "[train_grpo] invalid Qwen3.5 Triton overlay: $QWEN35_TRAIN_TRITON_OVERLAY" >&2
      exit 2
    }
  else
    export PYTHONPATH="$QWEN35_TRAIN_TRITON_OVERLAY:$PYTHONPATH"
  fi
fi

LORA_ARGS=()
MODEL_ARGS=()
VLLM_ENGINE_ARGS=()
CHAT_TEMPLATE_ARGS=()
SEPARATION_ARGS=()
if [[ "$LORA_RANK" -gt 0 ]]; then
  LORA_ARGS=(
    actor_rollout_ref.model.lora_rank="$LORA_RANK"
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
    actor_rollout_ref.model.target_modules="$LORA_TARGET_MODULES"
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
  )
  [[ -n "$LORA_ADAPTER_PATH" ]] && LORA_ARGS+=(actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH")
fi
[[ -n "$MODEL_ATTN_IMPLEMENTATION" ]] && \
  MODEL_ARGS+=(+actor_rollout_ref.model.override_config.attn_implementation="$MODEL_ATTN_IMPLEMENTATION")
[[ -n "$VLLM_GDN_PREFILL_BACKEND" ]] && \
  VLLM_ENGINE_ARGS+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend="$VLLM_GDN_PREFILL_BACKEND")
[[ -n "$VLLM_LANGUAGE_MODEL_ONLY" ]] && \
  VLLM_ENGINE_ARGS+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only="$VLLM_LANGUAGE_MODEL_ONLY")
[[ -n "$ENABLE_THINKING" ]] && \
  CHAT_TEMPLATE_ARGS+=(+data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING")
[[ -n "$FSDP_MODEL_DTYPE" ]] && \
  MODEL_ARGS+=(
    actor_rollout_ref.actor.fsdp_config.model_dtype="$FSDP_MODEL_DTYPE"
    actor_rollout_ref.ref.fsdp_config.model_dtype="$FSDP_MODEL_DTYPE"
  )
[[ -n "$FSDP_PARAM_OFFLOAD" ]] && \
  MODEL_ARGS+=(
    actor_rollout_ref.actor.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD"
    actor_rollout_ref.ref.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD"
  )
[[ -n "$FSDP_OPTIMIZER_OFFLOAD" ]] && \
  MODEL_ARGS+=(actor_rollout_ref.actor.fsdp_config.optimizer_offload="$FSDP_OPTIMIZER_OFFLOAD")
[[ -n "$USE_TORCH_COMPILE" ]] && \
  MODEL_ARGS+=(actor_rollout_ref.actor.fsdp_config.use_torch_compile="$USE_TORCH_COMPILE")

TRAIN_ENTRYPOINT="verl.trainer.main_ppo"
if [[ "$SEPARATE_ROLLOUT" == "1" ]]; then
  if [[ "$IS_QWEN35" == "1" ]]; then
    TRAIN_ENTRYPOINT="rl.train_one_step_off"
  else
    TRAIN_ENTRYPOINT="verl.experimental.one_step_off_policy.main_ppo"
  fi
  VERL_TRAINER_CONFIG="${VERL_TRAINER_CONFIG:-$("$PYTHON" -c \
    'from pathlib import Path; import verl; print(Path(verl.__file__).resolve().parent / "trainer" / "config")')}"
  SEPARATION_ARGS=(
    "hydra.searchpath=[file://$VERL_TRAINER_CONFIG]"
    actor_rollout_ref.hybrid_engine=False
    rollout.nnodes=1
    rollout.n_gpus_per_node="$ROLLOUT_N_GPUS"
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.checkpoint_engine.backend="$CHECKPOINT_BACKEND"
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="$CHECKPOINT_BUCKET_MB"
    ray_kwargs.ray_init.num_cpus="$RAY_NUM_CPUS"
    +ray_kwargs.ray_init.include_dashboard="$RAY_INCLUDE_DASHBOARD"
  )
  [[ -n "$CHECKPOINT_CUSTOM_BACKEND_MODULE" ]] && \
    SEPARATION_ARGS+=(actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module="$CHECKPOINT_CUSTOM_BACKEND_MODULE")
  [[ -n "$RAY_WORKER_SETUP_HOOK" ]] && \
    SEPARATION_ARGS+=(+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook="$RAY_WORKER_SETUP_HOOK")
fi

set -x
"$PYTHON" -m "$TRAIN_ENTRYPOINT" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation="$TRUNCATION" \
  data.shuffle=True \
  data.seed="$TRAINING_SEED" \
  data.return_raw_chat=True \
  data.dataloader_num_workers=0 \
  custom_reward_function.path="$REWARD_FILE" \
  custom_reward_function.name=compute_score \
  reward.custom_reward_function.path="$REWARD_FILE" \
  reward.custom_reward_function.name=compute_score \
  reward.num_workers="$REWARD_WORKERS" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.data_loader_seed="$TRAINING_SEED" \
  actor_rollout_ref.actor.fsdp_config.seed="$TRAINING_SEED" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LENGTH" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$N_GENERATIONS" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
  actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.format="$TOOL_FORMAT" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls="$MAX_PARALLEL_CALLS" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
  actor_rollout_ref.rollout.agent.default_agent_loop=visapp_agent \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_LOOP_WORKERS" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.seed="$TRAINING_SEED" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.critic_warmup=0 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  'trainer.logger=["console"]' \
  "${LORA_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${VLLM_ENGINE_ARGS[@]}" \
  "${CHAT_TEMPLATE_ARGS[@]}" \
  "${SEPARATION_ARGS[@]}" \
  "$@"
set +x

if [[ "$LORA_RANK" -gt 0 && -n "$OUTPUT_DIR" ]]; then
  ADAPTER_PATH="$(find "$OUTPUT_DIR" -mindepth 3 -maxdepth 3 -type d -path '*/actor/lora_adapter' -print 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "$ADAPTER_PATH" && -s "$ADAPTER_PATH/adapter_model.safetensors" ]]; then
    printf '%s\n' "$ADAPTER_PATH" > "$OUTPUT_DIR/final_adapter_path.txt"
    echo "[train_grpo] final adapter: $ADAPTER_PATH"
  else
    echo "[train_grpo] warning: no LoRA adapter checkpoint under $OUTPUT_DIR" >&2
    if [[ "$REQUIRE_ADAPTER" == "1" ]]; then
      exit 1
    fi
  fi
fi
