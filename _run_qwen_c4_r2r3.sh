#!/usr/bin/env bash
# Repetitions 2 and 3 of the c4 suite with Qwen3.8-27B, pinned to TWO GPUs.
#
# Config is deliberately IDENTICAL to r1 (BF16, TP=2, no MTP, 262144 ctx, same
# parsers, both stream-timeout guards raised). A repetition study is only
# meaningful if the only thing that changed is the random seed of the run.
#
# Two lanes on ONE server (GPU0+1). Measured on this box: 2 concurrent lanes cost
# essentially nothing per lane (76-92 turns/h vs 72 solo), while 4 lanes drop each
# lane to ~45 turns/h. With two GPUs, two lanes is the sweet spot.
#
# Lanes are split by TASK, never by repetition: run_eval.sh derives
# COMPOSE_PROJECT from variant+task+cli with NO run-id component, so r2 and r3 of
# the same task share a docker project name and would fight over containers,
# networks and volumes if they ever overlapped. Each lane owns a disjoint set of
# tasks and walks r2 then r3 within it, so a given task is never live twice.
#
# Usage:  ./_run_qwen_c4_r2r3.sh            # 20 runs (10 tasks x 2 reps)
#         REPS="2" ./_run_qwen_c4_r2r3.sh   # r2 only
#         RUN_EVAL=0 ./_run_qwen_c4_r2r3.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export RUNS_ROOT="${RUNS_ROOT:-$PWD/_runs_qwen3.8-27b_c4_r2r3}"
export PATH="$HOME/miniconda3/envs/codexcli/bin:$PATH"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export DOCKER_TEARDOWN=none          # two lanes in flight; each cleans only its own
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
RUN_EVAL="${RUN_EVAL:-1}"
read -r -a REPS <<< "${REPS:-2 3}"

LANE1=(1_newsletter 3_job-board 5_travel-booking 7_cloud-storage 9_project-management)
LANE2=(2_real-estate 4_forum 6_chat 8_ecommerce 10_streaming_music-streaming)

mkdir -p "$RUNS_ROOT/batch_logs"

curl -s -m 10 "$OPENAI_BASE_URL/models" | grep -q "$MODEL" \
  || { echo "ERROR: $OPENAI_BASE_URL is not serving $MODEL — start vLLM first." >&2; exit 1; }
echo "[r2r3] endpoint OK: $OPENAI_BASE_URL  reps=${REPS[*]}  root=$RUNS_ROOT"

# The graded artifact, not the exit code, decides success: qwen-code reports
# is_error=false and exit 0 even when a stream cap truncated the build.
is_done() {
  local rd="$1"
  [[ -f "$rd/logs/summary.json" && -f "$rd/workspace/docker-compose.yml" ]] || return 1
  python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["summary"]; raise SystemExit(0 if s.get("is_error") is False else 1)' \
    "$rd/logs/summary.json" 2>/dev/null
}

cleanup_run_docker() {
  local rd="$1" project
  [[ -f "$rd/meta.json" ]] || return 0
  project="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("compose_project",""))' "$rd/meta.json" 2>/dev/null || true)"
  if [[ -n "$project" && -d "$rd/workspace" ]]; then
    (cd "$rd/workspace" && docker compose -p "$project" down --remove-orphans -v >/dev/null 2>&1) || true
    sleep 2
  fi
}

run_one() {
  local lane="$1" fe="$2" be="$3" rep="$4" task="$5"
  local run_id="r${rep}_${task}_c4_qwen" rd="$RUNS_ROOT/r${rep}_${task}_c4_qwen"
  local log="$RUNS_ROOT/batch_logs/${run_id}.log"

  if is_done "$rd"; then
    echo "[r2r3] SKIP-DONE lane=$lane $run_id"; cleanup_run_docker "$rd"; return 0
  fi
  # One retry: the failures seen in r1 (stream caps, a squatted port) were
  # transient, and a silent partial build is worse than spending another slot.
  local attempt
  for attempt in 1 2; do
    if [[ -d "$rd" ]]; then
      mkdir -p "$RUNS_ROOT/_incomplete"
      mv "$rd" "$RUNS_ROOT/_incomplete/${run_id}.$(date +%H%M%S)"
    fi
    echo "[r2r3] START lane=$lane $run_id attempt=$attempt ports=$fe/$be @ $(date '+%F %T')"
    FRONTEND_PORT="$fe" BACKEND_PORT="$be" \
      bash ./run_eval.sh --run-id "$run_id" --task "$task" --variant c4 \
        --cli qwen --model "$MODEL" </dev/null >"$log" 2>&1
    local rc=$?
    cleanup_run_docker "$rd"
    if is_done "$rd"; then
      echo "[r2r3] END   lane=$lane $run_id rc=$rc OK @ $(date '+%F %T')"; return 0
    fi
    echo "[r2r3] END   lane=$lane $run_id rc=$rc INCOMPLETE (attempt $attempt) @ $(date '+%F %T')"
  done
  echo "[r2r3] GIVE-UP lane=$lane $run_id after 2 attempts"
  return 1
}

# rep is the OUTER loop so a complete r2 lands before r3 starts — a usable
# result set exists at the halfway point rather than only at the very end.
lane_loop() {
  local lane="$1" fe="$2" be="$3"; shift 3
  local tasks=("$@") rep task
  for rep in "${REPS[@]}"; do
    for task in "${tasks[@]}"; do
      run_one "$lane" "$fe" "$be" "$rep" "$task" || true
    done
  done
}

lane_loop 1 40000 40001 "${LANE1[@]}" & p1=$!
lane_loop 2 41000 41001 "${LANE2[@]}" & p2=$!
wait "$p1"; wait "$p2"
echo "[r2r3] builds done @ $(date '+%F %T')"

if [[ "$RUN_EVAL" == "1" ]]; then
  echo "[r2r3] scoring $RUNS_ROOT"
  DOCKER_TEARDOWN=own ./eval_all_runs.sh "$RUNS_ROOT"
fi
