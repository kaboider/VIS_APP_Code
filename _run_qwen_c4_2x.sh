#!/usr/bin/env bash
# Build the c4 suite with Qwen Code driving a self-hosted Qwen3.8-27B (local vLLM).
#
# Two concurrent lanes on FIXED, DISJOINT ports — run_eval.sh defaults every task
# to 38000/38001, so lanes must be told apart explicitly or their compose projects
# fight over the same host ports. Each lane tears down only its OWN compose project
# when a task finishes, so containers never accumulate and the sibling lane (plus
# anything else on this shared docker daemon) is left alone.
#
# Prereq: vLLM already serving, e.g.
#   CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/workspace/junjiaguo/hf_cache \
#   VLLM_USE_FLASHINFER_SAMPLER=0 \
#     vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 2 --max-model-len 262144 \
#       --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
#
# Usage:
#   ./_run_qwen_c4_2x.sh                 # 9 remaining tasks, then score everything
#   TASKS="6_chat 8_ecommerce" ./_run_qwen_c4_2x.sh
#   RUN_EVAL=0 ./_run_qwen_c4_2x.sh      # build only
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export RUNS_ROOT="${RUNS_ROOT:-$PWD/_runs}"
export PATH="$HOME/miniconda3/envs/codexcli/bin:$PATH"     # qwen CLI needs Node >= 22
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
# Parallel lanes: even DOCKER_TEARDOWN=own would kill the sibling lane's app
# mid-build, so sweep nothing globally and clean up per-run instead (below).
export DOCKER_TEARDOWN=none

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
RUN_EVAL="${RUN_EVAL:-1}"

# 4_forum is deliberately absent — the smoke run already produced it.
LANE_A=(1_newsletter 3_job-board 5_travel-booking 7_cloud-storage 9_project-management)
LANE_B=(2_real-estate 6_chat 8_ecommerce 10_streaming_music-streaming)
if [[ -n "${TASKS:-}" ]]; then
  read -r -a _all <<< "$TASKS"
  LANE_A=(); LANE_B=()
  for i in "${!_all[@]}"; do
    (( i % 2 == 0 )) && LANE_A+=("${_all[$i]}") || LANE_B+=("${_all[$i]}")
  done
fi

mkdir -p "$RUNS_ROOT/batch_logs"

# --- pre-flight: the endpoint must actually be serving, or all 9 fail identically
if ! curl -s -m 10 "$OPENAI_BASE_URL/models" | grep -q "$MODEL"; then
  echo "ERROR: $OPENAI_BASE_URL is not serving '$MODEL' — start vLLM first (see header)." >&2
  exit 1
fi
echo "[qwen-2x] endpoint OK: $OPENAI_BASE_URL  model=$MODEL"

cleanup_run_docker() {
  local run_dir="$1" project
  [[ -f "$run_dir/meta.json" ]] || return 0
  project="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("compose_project",""))' "$run_dir/meta.json" 2>/dev/null || true)"
  if [[ -n "$project" && -d "$run_dir/workspace" ]]; then
    (cd "$run_dir/workspace" && docker compose -p "$project" down --remove-orphans -v >/dev/null 2>&1) || true
    sleep 2
  fi
}

run_one() {
  local lane="$1" task="$2" frontend="$3" backend="$4"
  local run_id="${task}_c4_qwen"
  local run_dir="$RUNS_ROOT/$run_id"
  local log="$RUNS_ROOT/batch_logs/${run_id}.log"

  # Resume-safe: only a run that actually DELIVERED is left untouched. is_error
  # alone is not enough — a stream-idle abort mid-build still reports
  # is_error=false, so a truncated workspace would be skipped forever. Require
  # the compose file the task is graded on as the real completion marker.
  if [[ -f "$run_dir/logs/summary.json" && -f "$run_dir/workspace/docker-compose.yml" ]] && \
     python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["summary"]; raise SystemExit(0 if s.get("is_error") is False else 1)' \
       "$run_dir/logs/summary.json" 2>/dev/null; then
    echo "[qwen-2x] RESUME-SKIP lane=$lane task=$task"
    cleanup_run_docker "$run_dir"
    return 0
  fi

  # run_eval.sh refuses to reuse an existing run dir, so retire any partial one.
  if [[ -d "$run_dir" ]]; then
    mkdir -p "$RUNS_ROOT/_incomplete"
    mv "$run_dir" "$RUNS_ROOT/_incomplete/${run_id}.$(date +%H%M%S)"
    echo "[qwen-2x] retired incomplete run for $task → _incomplete/"
  fi

  echo "[qwen-2x] START lane=$lane task=$task ports=$frontend/$backend @ $(date '+%F %T')"
  FRONTEND_PORT="$frontend" BACKEND_PORT="$backend" \
    bash ./run_eval.sh --run-id "$run_id" --task "$task" --variant c4 \
      --cli qwen --model "$MODEL" </dev/null >"$log" 2>&1
  local rc=$?
  cleanup_run_docker "$run_dir"
  echo "[qwen-2x] END   lane=$lane task=$task rc=$rc @ $(date '+%F %T')"
  return "$rc"
}

lane_loop() {
  local lane="$1" frontend="$2" backend="$3"; shift 3
  local task
  for task in "$@"; do
    run_one "$lane" "$task" "$frontend" "$backend" || \
      echo "[qwen-2x] FAILURE-PRESERVED lane=$lane task=$task"
  done
}

echo "[qwen-2x] lane A: ${LANE_A[*]:-(none)}"
echo "[qwen-2x] lane B: ${LANE_B[*]:-(none)}"

lane_loop A 38000 38001 "${LANE_A[@]}" & lane_a_pid=$!
lane_loop B 39000 39001 "${LANE_B[@]}" & lane_b_pid=$!
wait "$lane_a_pid"; a_rc=$?
wait "$lane_b_pid"; b_rc=$?
echo "[qwen-2x] builds done: laneA rc=$a_rc laneB rc=$b_rc @ $(date '+%F %T')"

# Scoring is sequential and re-uses each run's recorded ports, so it must not
# overlap the build phase (both would bind the same host ports).
if [[ "$RUN_EVAL" == "1" ]]; then
  echo "[qwen-2x] scoring all *_c4_qwen runs"
  DOCKER_TEARDOWN=own FILTER='*_c4_qwen' ./eval_all_runs.sh "$RUNS_ROOT"
fi
