#!/usr/bin/env bash
# CAMEL single-agent harness against a locally hosted Qwen3.5-27B (vLLM).
# Default: GPU-hosted OpenAI-compatible server at http://127.0.0.1:8200/v1
#
#   ./_run_camel_qwen35_27b.sh              # all 10 c4 tasks
#   TASKS="1_newsletter 4_forum" ./_run_camel_qwen35_27b.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
export CAMEL_API_URL="${CAMEL_API_URL:-http://127.0.0.1:8201/v1}"
export CAMEL_API_KEY="${CAMEL_API_KEY:-EMPTY}"
export OPENAI_API_KEY="$CAMEL_API_KEY"
export OPENAI_API_BASE_URL="$CAMEL_API_URL"
export CAMEL_API_MODE="${CAMEL_API_MODE:-chat}"
export CAMEL_SINGLE=1
export CAMEL_MAX_STEPS="${CAMEL_MAX_STEPS:-40}"
export CAMEL_TIMEOUT="${CAMEL_TIMEOUT:-7200}"
export CAMEL_MAX_TOKENS="${CAMEL_MAX_TOKENS:-16384}"
export CAMEL_MAX_MODEL_LEN="${CAMEL_MAX_MODEL_LEN:-262144}"
export CAMEL_SHELL_TIMEOUT="${CAMEL_SHELL_TIMEOUT:-300}"
export MODEL_TIMEOUT="${MODEL_TIMEOUT:-600}"
export ALLOW_UNSANDBOXED_EVAL="${ALLOW_UNSANDBOXED_EVAL:-1}"
export CAMEL_PY="${CAMEL_PY:-$PWD/.camel-venv/bin/python}"
RUN_TIMEOUT="${RUN_TIMEOUT:-7500}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
SKIP_EVAL="${SKIP_EVAL:-1}"

KEY_FILE="$PWD/_rl_runs/dummy_openai_key.txt"
mkdir -p "$(dirname "$KEY_FILE")"
printf '%s\n' "$CAMEL_API_KEY" > "$KEY_FILE"
export CAMEL_KEY_FILE="$KEY_FILE"

export RUNS_ROOT="${RUNS_ROOT:-$PWD/_runs_camel_qwen35_27b_256k}"
mkdir -p "$RUNS_ROOT"

if [[ ! -x "$CAMEL_PY" ]]; then
  echo "ERROR: camel venv python not found at $CAMEL_PY" >&2
  exit 127
fi

echo "[camel-qwen] waiting for $CAMEL_API_URL/models (model=$MODEL)"
ready=0
for _ in $(seq 1 180); do
  if curl -sf "$CAMEL_API_URL/models" 2>/dev/null | grep -q "$MODEL"; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "$ready" != "1" ]]; then
  echo "ERROR: local server not ready at $CAMEL_API_URL" >&2
  exit 3
fi

if [[ -n "${TASKS:-}" ]]; then
  # shellcheck disable=SC2206
  TASKS_LIST=($TASKS)
else
  TASKS_LIST=(1_newsletter 2_real-estate 3_job-board 4_forum 5_travel-booking \
              6_chat 7_cloud-storage 8_ecommerce 9_project-management \
              10_streaming_music-streaming)
fi

is_clean() {
  [ -f "$1/workspace/docker-compose.yml" ] || return 1
  [ "$(python3 -c "import json;print(json.load(open('$1/logs/summary.json'))['summary'].get('is_error'))" 2>/dev/null)" = "False" ]
}

run_task() {
  local TASK="$1" attempt d
  for d in "$RUNS_ROOT"/*_${TASK}_c4_camel/; do
    is_clean "$d" && { echo "[camel-qwen] $TASK already clean — skipping"; return 0; }
  done
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "[camel-qwen] === $MODEL · $TASK attempt $attempt/$MAX_ATTEMPTS @ $(date '+%m-%d %H:%M') ==="
    timeout "$RUN_TIMEOUT" ./run_eval.sh --task "$TASK" --variant c4 --cli camel --model "$MODEL" </dev/null \
      || echo "[camel-qwen] $TASK run_eval rc=$?"
    d=$(ls -td "$RUNS_ROOT"/*_${TASK}_c4_camel/ 2>/dev/null | head -1)
    if [ -n "${d:-}" ] && is_clean "$d"; then
      echo "[camel-qwen] OK $TASK clean — $(basename "$d")"; return 0
    fi
    echo "[camel-qwen] $TASK not clean"
    [ "$attempt" -lt "$MAX_ATTEMPTS" ] && [ -n "${d:-}" ] && { chmod -R u+w "$d" 2>/dev/null; rm -rf "$d" 2>/dev/null; }
    [ "$attempt" -lt "$MAX_ATTEMPTS" ] && sleep 5
  done
  echo "[camel-qwen] GIVEUP $TASK"; return 1
}

echo "[camel-qwen] START model=$MODEL url=$CAMEL_API_URL runs=$RUNS_ROOT @ $(date '+%m-%d %H:%M')"
done_n=0; fail=()
for TASK in "${TASKS_LIST[@]}"; do
  run_task "$TASK" && done_n=$((done_n+1)) || fail+=("$TASK")
done
echo "[camel-qwen] ALL BUILDS DONE @ $(date '+%m-%d %H:%M') — clean $done_n/${#TASKS_LIST[@]}; failed: ${fail[*]:-none}"
if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "[camel-qwen] SKIP_EVAL=1 — not scoring (move the runs dir to a docker machine later)"
else
  echo "[camel-qwen] scoring…"
  FILTER='*camel*' ./eval_all_runs.sh "$RUNS_ROOT" || echo "[camel-qwen] eval rc=$?"
fi
echo "[camel-qwen] DONE @ $(date '+%m-%d %H:%M')"
