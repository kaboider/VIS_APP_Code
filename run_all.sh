#!/usr/bin/env bash
#
# run_all.sh — sequentially run agent eval over all 10 tasks for a variant.
#
# Usage:
#   ./run_all.sh                              # variant c3, all tasks
#   ./run_all.sh --variant c0                 # change variant
#   ./run_all.sh --variant c4 --cli codex     # c4 = no-scaffold-hint (model picks)
#   ./run_all.sh --tasks 1_newsletter 4_forum # run a subset only
#   ./run_all.sh --skip-existing              # skip tasks that already have a run today
#   ./run_all.sh --eval-after                 # run tools/eval_run.py after each agent run
#
# Anything else after these flags is forwarded to run_eval.sh, e.g.:
#   ./run_all.sh --variant c2 --cli codex
#   ./run_all.sh --variant c3 --cli gemini --model gemini-2.5-pro
#   ./run_all.sh --variant c4 --cli codex --preload   # ERROR — c4 has no preload
#   ./run_all.sh --variant c3 --cli claude --skill    # tell agent skills are available
#
# Single-docker host: each run_eval.sh shuts down the prior compose project
# before starting, so you can leave this running unattended.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VARIANT="c3"
SKIP_EXISTING=false
EVAL_AFTER=false
TASKS_OVERRIDE=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)        VARIANT="$2"; shift 2 ;;
    --skip-existing)  SKIP_EXISTING=true; shift ;;
    --eval-after)     EVAL_AFTER=true; shift ;;
    --tasks)          shift; while [[ $# -gt 0 && "$1" != --* ]]; do TASKS_OVERRIDE+=("$1"); shift; done ;;
    --)               shift; EXTRA_ARGS+=("$@"); break ;;
    *)                EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# All 10 tasks (in increasing complexity / time roughly)
DEFAULT_TASKS=(
  4_forum                       # smallest, ~5 pages
  1_newsletter
  5_travel-booking
  8_ecommerce
  6_chat
  9_project-management
  10_streaming_music-streaming
  2_real-estate
  3_job-board
  7_cloud-storage               # biggest, 33 pages
)

if [[ ${#TASKS_OVERRIDE[@]} -gt 0 ]]; then
  TASKS=("${TASKS_OVERRIDE[@]}")
else
  TASKS=("${DEFAULT_TASKS[@]}")
fi

# Sanity: every task has a description and anchors JSON
for t in "${TASKS[@]}"; do
  if [[ ! -f "$VARIANT/$t/description.md" ]]; then
    echo "ERROR: $VARIANT/$t/description.md not found" >&2
    exit 1
  fi
done

LOG_DIR="_runs/_batch_logs"
mkdir -p "$LOG_DIR"
BATCH_TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$LOG_DIR/batch_${BATCH_TS}_${VARIANT//\//_}.summary.txt"

echo "==========================================================" | tee -a "$SUMMARY"
echo " Batch run starting at $(date)"                              | tee -a "$SUMMARY"
echo " Variant: $VARIANT"                                          | tee -a "$SUMMARY"
echo " Tasks (${#TASKS[@]}):  ${TASKS[*]}"                           | tee -a "$SUMMARY"
echo " Skip existing: $SKIP_EXISTING   Eval after: $EVAL_AFTER"    | tee -a "$SUMMARY"
echo " Forwarded args: ${EXTRA_ARGS[*]:-(none)}"                    | tee -a "$SUMMARY"
echo " Summary log: $SUMMARY"                                        | tee -a "$SUMMARY"
echo "==========================================================" | tee -a "$SUMMARY"
echo                                                                  | tee -a "$SUMMARY"

ok=0; fail=0; skipped=0
START_ALL=$(date +%s)

for t in "${TASKS[@]}"; do
  echo "----------------------------------------------------------" | tee -a "$SUMMARY"
  echo " [$(date +%H:%M:%S)] Task: $t  (variant=$VARIANT)"          | tee -a "$SUMMARY"

  if [[ "$SKIP_EXISTING" == true ]]; then
    # Skip only if a prior SUCCESSFUL run exists (summary.json is_error=false).
    # Failed / usage-limited runs (is_error=true, e.g. 429 rate-limit) do NOT
    # count, so they get retried. Matches any date and the optional
    # `_codex` / `_gemini` CLI suffix (trailing `*`).
    found_ok=false
    for d in _runs/*_${t}_${VARIANT//\//_}*/; do
      [[ -d "$d" ]] || continue                      # no-match → literal pattern; skip
      s="${d%/}/logs/summary.json"
      [[ -f "$s" ]] || continue
      if python3 -c "import json,sys; sys.exit(0 if json.load(open('$s'))['summary']['is_error'] is False else 1)" 2>/dev/null; then
        found_ok=true; break
      fi
    done
    if [[ "$found_ok" == true ]]; then
      echo "  → already has a SUCCESSFUL run; skipping" | tee -a "$SUMMARY"
      skipped=$((skipped+1)); continue
    fi
  fi

  task_log="$LOG_DIR/${BATCH_TS}_${t}_${VARIANT//\//_}.log"
  start=$(date +%s)
  if ./run_eval.sh --task "$t" --variant "$VARIANT" \
       ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$task_log"; then
    elapsed=$(( $(date +%s) - start ))
    # Trailing `*` matches the optional `_codex` / `_gemini` suffix that
    # run_eval.sh appends for non-claude CLIs. `|| true` so a missing
    # match doesn't trip set -e and abort the whole batch.
    run_dir=$(ls -td _runs/*_${t}_${VARIANT//\//_}* 2>/dev/null | head -1 || true)
    echo "  ✓ done in ${elapsed}s  (run_dir: ${run_dir:-?})"  | tee -a "$SUMMARY"
    ok=$((ok+1))

    if [[ "$EVAL_AFTER" == true && -d "$run_dir" ]]; then
      echo "  running eval_run.py..."                     | tee -a "$SUMMARY"
      if python3 tools/eval_run.py "$run_dir" 2>&1 | tee -a "$task_log" | tail -1 \
           | tee -a "$SUMMARY"; then
        echo "  ✓ eval done"                              | tee -a "$SUMMARY"
      else
        echo "  ⚠ eval failed (continuing)"               | tee -a "$SUMMARY"
      fi
    fi
  else
    elapsed=$(( $(date +%s) - start ))
    echo "  ✗ FAILED after ${elapsed}s  (see $task_log)"  | tee -a "$SUMMARY"
    fail=$((fail+1))
  fi
  echo                                                     | tee -a "$SUMMARY"
done

total=$(( $(date +%s) - START_ALL ))
echo "==========================================================" | tee -a "$SUMMARY"
echo " Batch finished at $(date)"                                  | tee -a "$SUMMARY"
echo " Total time: $((total/60))m $((total%60))s"                  | tee -a "$SUMMARY"
echo " Results: $ok ok / $fail fail / $skipped skipped"            | tee -a "$SUMMARY"
echo "==========================================================" | tee -a "$SUMMARY"
