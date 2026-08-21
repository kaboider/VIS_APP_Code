#!/usr/bin/env bash
# Finish the c4 suite on all four GPUs: four concurrent lanes across TWO vLLM
# servers (GPU0+1 -> :8000, GPU2+3 -> :8001), both TP=2 with identical flags so
# a run's numbers do not depend on which server happened to serve it.
#
# Why two TP=2 servers rather than TP=4 or 4x TP=1 — all three were measured on
# this box (single-stream, idle, same prompts):
#     TP=1  27.3 tok/s      TP=2  50.3 tok/s   (1.84x, i.e. 92% scaling)
# TP scales well even over PCIe/PHB here, so 2x TP=2 beats 4x TP=1 on aggregate
# throughput, and two independent servers avoid TP=4's extra all-reduce hops.
#
# Ports: every lane needs its own pair — run_eval.sh defaults all tasks to
# 38000/38001. 38xxx/39xxx belong to the earlier 2-lane batch, so start at 40000.
#
# Usage:  ./_run_qwen_c4_4x.sh            # all not-yet-done c4 tasks
#         TASKS="6_chat 8_ecommerce" ./_run_qwen_c4_4x.sh
#         RUN_EVAL=0 ./_run_qwen_c4_4x.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export RUNS_ROOT="${RUNS_ROOT:-$PWD/_runs}"
export PATH="$HOME/miniconda3/envs/codexcli/bin:$PATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export DOCKER_TEARDOWN=none        # 4 lanes in flight; each cleans only its own
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
RUN_EVAL="${RUN_EVAL:-1}"

SERVER_A="http://127.0.0.1:8000/v1"
SERVER_B="http://127.0.0.1:8001/v1"

# Tasks still to do. 4_forum (smoke run) plus anything already complete is
# filtered out by the claim/skip logic below, so listing all 10 is safe.
ALL=(1_newsletter 2_real-estate 3_job-board 5_travel-booking 6_chat \
     7_cloud-storage 8_ecommerce 9_project-management 10_streaming_music-streaming)
[[ -n "${TASKS:-}" ]] && read -r -a ALL <<< "$TASKS"

CLAIMS="$RUNS_ROOT/_claims"
mkdir -p "$RUNS_ROOT/batch_logs" "$CLAIMS"

for s in "$SERVER_A" "$SERVER_B"; do
  curl -s -m 10 "$s/models" | grep -q "$MODEL" || { echo "ERROR: $s not serving $MODEL" >&2; exit 1; }
  echo "[qwen-4x] endpoint OK: $s"
done

# A task is DONE when it produced the artifact it is graded on. is_error alone is
# not enough: a stream-idle abort mid-build still reports is_error=false.
is_done() {
  local rd="$RUNS_ROOT/$1_c4_qwen"
  [[ -f "$rd/logs/summary.json" && -f "$rd/workspace/docker-compose.yml" ]] || return 1
  python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["summary"]; raise SystemExit(0 if s.get("is_error") is False else 1)' \
    "$rd/logs/summary.json" 2>/dev/null
}

# Atomic claim so no two lanes — including the older 2-lane batch, if it is still
# draining — can start the same task. `set -o noclobber` makes `>` fail if the
# file exists, and that test-and-create is atomic on a local filesystem.
claim() {
  local t="$1"
  if ( set -o noclobber; echo "$$ $(date '+%F %T')" > "$CLAIMS/$t" ) 2>/dev/null; then
    return 0
  fi
  # Stale claim from a dead process? Reclaim it; otherwise someone owns it.
  local owner; owner="$(cut -d' ' -f1 "$CLAIMS/$t" 2>/dev/null || echo)"
  if [[ -n "$owner" ]] && ! kill -0 "$owner" 2>/dev/null; then
    echo "$$ $(date '+%F %T')" > "$CLAIMS/$t"; return 0
  fi
  return 1
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
  local lane="$1" endpoint="$2" fe="$3" be="$4" task="$5"
  local run_id="${task}_c4_qwen" rd="$RUNS_ROOT/${task}_c4_qwen"
  local log="$RUNS_ROOT/batch_logs/${run_id}.log"

  if is_done "$task"; then
    echo "[qwen-4x] SKIP-DONE lane=$lane task=$task"; cleanup_run_docker "$rd"; return 0
  fi
  if ! claim "$task"; then
    echo "[qwen-4x] SKIP-CLAIMED lane=$lane task=$task (another lane owns it)"; return 0
  fi
  # A partial dir would make run_eval.sh refuse to start; retire it with evidence.
  if [[ -d "$rd" ]]; then
    mkdir -p "$RUNS_ROOT/_incomplete"
    mv "$rd" "$RUNS_ROOT/_incomplete/${run_id}.$(date +%H%M%S)"
    echo "[qwen-4x] retired incomplete run for $task"
  fi

  echo "[qwen-4x] START lane=$lane task=$task ep=${endpoint##*:} ports=$fe/$be @ $(date '+%F %T')"
  OPENAI_BASE_URL="$endpoint" FRONTEND_PORT="$fe" BACKEND_PORT="$be" \
    bash ./run_eval.sh --run-id "$run_id" --task "$task" --variant c4 \
      --cli qwen --model "$MODEL" </dev/null >"$log" 2>&1
  local rc=$?
  cleanup_run_docker "$rd"
  # Release the claim unless the task actually produced its graded artifact, so a
  # crashed/aborted task stays retryable. Holding the claim after a failure would
  # silently drop that task from the whole suite.
  is_done "$task" || rm -f "$CLAIMS/$task"
  echo "[qwen-4x] END   lane=$lane task=$task rc=$rc @ $(date '+%F %T')"
  return "$rc"
}

# Lanes pull from one shared list, so a lane that finishes early picks up the
# next unclaimed task instead of idling while another lane still has a queue.
lane_loop() {
  local lane="$1" endpoint="$2" fe="$3" be="$4"
  local task
  for task in "${ALL[@]}"; do
    run_one "$lane" "$endpoint" "$fe" "$be" "$task" || \
      echo "[qwen-4x] FAILURE-PRESERVED lane=$lane task=$task"
  done
}

echo "[qwen-4x] queue: ${ALL[*]}"
lane_loop A "$SERVER_A" 40000 40001 & pA=$!
lane_loop B "$SERVER_A" 41000 41001 & pB=$!
lane_loop C "$SERVER_B" 42000 42001 & pC=$!
lane_loop D "$SERVER_B" 43000 43001 & pD=$!
for p in $pA $pB $pC $pD; do wait "$p"; done
echo "[qwen-4x] builds done @ $(date '+%F %T')"

if [[ "$RUN_EVAL" == "1" ]]; then
  echo "[qwen-4x] scoring all *_c4_qwen runs"
  DOCKER_TEARDOWN=own FILTER='*_c4_qwen' ./eval_all_runs.sh "$RUNS_ROOT"
fi
