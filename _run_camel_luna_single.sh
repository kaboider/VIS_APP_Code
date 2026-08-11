#!/usr/bin/env bash
# CAMEL SINGLE-agent (gpt-5.6-luna) over all 10 c4 tasks -> _runs_camel_luna_single.
# One ChatAgent using CAMEL's OFFICIAL toolkits (TerminalToolkit/FileToolkit/
# NoteTakingToolkit, eigent.py style) via CAMEL_SINGLE=1. This is the single-agent A/B
# arm against the multi-agent Workforce (_runs_camel_luna). Skip-guard keeps clean
# builds; retry not-clean up to 3x; then score with eval_all_runs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export RUNS_ROOT="${RUNS_ROOT:-$PWD/_runs_camel_luna_single}"
mkdir -p "$RUNS_ROOT"
MODEL=gpt-5.6-luna
export CAMEL_SINGLE=1                       # <- single-agent, official toolkits
export CAMEL_MAX_STEPS="${CAMEL_MAX_STEPS:-14}"
export CAMEL_TIMEOUT="${CAMEL_TIMEOUT:-3600}"
RUN_TIMEOUT="${RUN_TIMEOUT:-4200}"

TASKS_LIST=(1_newsletter 2_real-estate 3_job-board 4_forum 5_travel-booking \
            6_chat 7_cloud-storage 8_ecommerce 9_project-management \
            10_streaming_music-streaming)

is_clean() {  # $1 = run dir
  [ -f "$1/workspace/docker-compose.yml" ] || return 1
  [ "$(python3 -c "import json;print(json.load(open('$1/logs/summary.json'))['summary'].get('is_error'))" 2>/dev/null)" = "False" ]
}

run_task() {
  local TASK="$1" attempt d
  for d in "$RUNS_ROOT"/*_${TASK}_c4_camel/; do
    is_clean "$d" && { echo "[camel1-luna] $TASK already clean — skipping"; return 0; }
  done
  for attempt in 1 2 3; do
    echo "[camel1-luna] === $TASK attempt $attempt/3 @ $(date '+%m-%d %H:%M') ==="
    timeout "$RUN_TIMEOUT" ./run_eval.sh --task "$TASK" --variant c4 --cli camel --model "$MODEL" </dev/null \
      || echo "[camel1-luna] $TASK run_eval rc=$?"
    d=$(ls -td "$RUNS_ROOT"/*_${TASK}_c4_camel/ 2>/dev/null | head -1)
    if [ -n "${d:-}" ] && is_clean "$d"; then
      echo "[camel1-luna] OK $TASK clean — $(basename "$d")"; return 0
    fi
    echo "[camel1-luna] $TASK not clean — discard + retry"
    [ -n "${d:-}" ] && { chmod -R u+w "$d" 2>/dev/null; rm -rf "$d" 2>/dev/null; }
    [ "$attempt" -lt 3 ] && sleep 15
  done
  echo "[camel1-luna] GIVEUP $TASK (3 fails)"; return 1
}

done_n=0; fail=()
for TASK in "${TASKS_LIST[@]}"; do
  run_task "$TASK" && done_n=$((done_n+1)) || fail+=("$TASK")
done
echo "[camel1-luna] ALL BUILDS DONE @ $(date '+%m-%d %H:%M') — clean $done_n/10; failed: ${fail[*]:-none}"
echo "[camel1-luna] scoring…"
FILTER='*camel*' ./eval_all_runs.sh "$RUNS_ROOT" || echo "[camel1-luna] eval rc=$?"
echo "[camel1-luna] DONE @ $(date '+%m-%d %H:%M')"
