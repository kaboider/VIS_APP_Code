#!/usr/bin/env bash
# Build every c4 task twice with two concurrent CAMEL workers, then evaluate
# every completed run sequentially. Run IDs are deterministic and resumable.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

set -a
# shellcheck disable=SC1091
source .env
set +a

export OPENAI_API_KEY="$CAMEL_API_KEY"
export OPENAI_API_BASE_URL="$CAMEL_API_URL"
export RUNS_ROOT="${RUNS_ROOT:-$PWD/_runs_camel_gpt-5.6-sol_high_parallel_2x}"
export CAMEL_SINGLE=1
export CAMEL_REASONING_EFFORT=high
export CAMEL_PROMPT_CACHE_RETENTION=24h
export CAMEL_KEEP_TOOLKIT_ARTIFACTS=1
export CAMEL_MAX_STEPS="${CAMEL_MAX_STEPS:-14}"
export CAMEL_TIMEOUT="${CAMEL_TIMEOUT:-3600}"
export SKIP_ANCHOR_STAGE=1
export PRESERVE_OTHER_DOCKER=1

BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUNS_ROOT/batch_logs"
printf '%s\n' "$BATCH_TAG" > "$RUNS_ROOT/batch_tag.txt"

ODD_TASKS=(1_newsletter 3_job-board 5_travel-booking 7_cloud-storage 9_project-management)
EVEN_TASKS=(2_real-estate 4_forum 6_chat 8_ecommerce 10_streaming_music-streaming)

stage_anchor() {
  local task="$1" run_dir="$2" src="$PWD/$task/${task}_anchors.json"
  if [[ -f "$src" ]]; then
    cp "$src" "$run_dir/anchors.json"
    echo "[parallel-2x] staged hidden anchors after build: $run_dir/anchors.json"
  else
    echo "[parallel-2x] WARNING: anchor source missing: $src" >&2
  fi
}

run_one() {
  local worker="$1" rep="$2" task="$3" frontend="$4" backend="$5"
  local run_id="${BATCH_TAG}_r${rep}_w${worker}_${task}_c4_camel"
  local run_dir="$RUNS_ROOT/$run_id"
  local log="$RUNS_ROOT/batch_logs/${run_id}.log"

  if [[ -f "$run_dir/logs/summary.json" ]]; then
    echo "[parallel-2x] resume skip: $run_id"
    stage_anchor "$task" "$run_dir"
    return 0
  fi

  echo "[parallel-2x] START worker=$worker rep=$rep task=$task @ $(date '+%F %T')"
  FRONTEND_PORT="$frontend" BACKEND_PORT="$backend" \
    bash ./run_eval.sh --run-id "$run_id" --task "$task" --variant c4 \
      --cli camel --model gpt-5.6-sol </dev/null >"$log" 2>&1
  local rc=$?
  # An agent may launch its compose stack while checking its work. Tear down
  # only this run's project before the worker reuses its two assigned ports.
  if [[ -f "$run_dir/meta.json" ]]; then
    local project
    project="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("compose_project", ""))' "$run_dir/meta.json" 2>/dev/null || true)"
    if [[ -n "$project" ]]; then
      (cd "$run_dir/workspace" && docker compose -p "$project" down --remove-orphans >/dev/null 2>&1) || true
      sleep 2
    fi
  fi
  stage_anchor "$task" "$run_dir"
  echo "[parallel-2x] END worker=$worker rep=$rep task=$task rc=$rc @ $(date '+%F %T')"
  return "$rc"
}

worker_loop() {
  local worker="$1" frontend="$2" backend="$3"
  shift 3
  local tasks=("$@") rep task
  for rep in 1 2; do
    for task in "${tasks[@]}"; do
      run_one "$worker" "$rep" "$task" "$frontend" "$backend" || \
        echo "[parallel-2x] worker=$worker rep=$rep task=$task non-zero; artifacts preserved"
    done
  done
}

echo "[parallel-2x] batch=$BATCH_TAG root=$RUNS_ROOT concurrency=2"
worker_loop 1 38100 38101 "${ODD_TASKS[@]}" &
worker1_pid=$!
worker_loop 2 38200 38201 "${EVEN_TASKS[@]}" &
worker2_pid=$!

wait "$worker1_pid"; worker1_rc=$?
wait "$worker2_pid"; worker2_rc=$?
echo "[parallel-2x] BUILDS DONE worker1_rc=$worker1_rc worker2_rc=$worker2_rc"

python3 tools/camel_usage_report.py "$RUNS_ROOT"

# Evaluation is intentionally sequential. Restore the normal clean-Docker
# preflight before starting it.
unset PRESERVE_OTHER_DOCKER
FILTER='*_c4_camel' bash ./eval_all_runs.sh "$RUNS_ROOT"
eval_rc=$?
python3 tools/camel_usage_report.py "$RUNS_ROOT"
echo "[parallel-2x] DONE eval_rc=$eval_rc @ $(date '+%F %T')"
exit "$eval_rc"
