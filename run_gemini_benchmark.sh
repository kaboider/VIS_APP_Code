#!/usr/bin/env bash
#
# run_gemini_benchmark.sh — Run all Gemini models (via Antigravity CLI) against
# one or more tasks and variants, with clean human-readable run IDs.
#
# Run ID format:  <model_slug>__<task>__<variant>
# Example:        gemini35flash_high__4_forum__c4
#
# Usage:
#   ./run_gemini_benchmark.sh                          # all models × all tasks × c4
#   ./run_gemini_benchmark.sh --tasks "4_forum 1_newsletter"
#   ./run_gemini_benchmark.sh --models "gemini35pro_high gemini35flash_high"
#   ./run_gemini_benchmark.sh --variants "c4 c0"
#   ./run_gemini_benchmark.sh --dry-run               # print commands, don't run
#   ./run_gemini_benchmark.sh --timeout 120m
#   ./run_gemini_benchmark.sh --skip-existing         # skip if run dir already exists
#
# After all runs complete, score them:
#   ./eval_all_runs.sh ./_runs
#
# ── Model slug map ──────────────────────────────────────────────────────────────
# The "display name" passed to --model must exactly match `agy models` output.
# The slug is used in the run ID (filesystem-safe, no spaces/parens).
# ────────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ───────────────────────────────────────────────────────────────────

# All Gemini models available in `agy models` (slug → display name)
declare -A MODEL_MAP
MODEL_MAP["gemini35flash_low"]="Gemini 3.5 Flash (Low)"
MODEL_MAP["gemini35flash_medium"]="Gemini 3.5 Flash (Medium)"
MODEL_MAP["gemini35flash_high"]="Gemini 3.5 Flash (High)"
MODEL_MAP["gemini31pro_low"]="Gemini 3.1 Pro (Low)"
MODEL_MAP["gemini31pro_high"]="Gemini 3.1 Pro (High)"

# Ordered list of slugs (controls run order: cheapest → most expensive)
DEFAULT_MODELS=(
  "gemini35flash_low"
  "gemini35flash_medium"
  "gemini35flash_high"
  "gemini31pro_low"
  "gemini31pro_high"
)

ALL_TASKS=(
  "1_newsletter"
  "2_real-estate"
  "3_job-board"
  "4_forum"
  "5_travel-booking"
  "6_chat"
  "7_cloud-storage"
  "8_ecommerce"
  "9_project-management"
  "10_streaming_music-streaming"
)

DEFAULT_VARIANTS=("c4")
DEFAULT_TIMEOUT="90m"
DRY_RUN=false
SKIP_EXISTING=false

# ── Arg parsing ────────────────────────────────────────────────────────────────

SELECTED_MODELS=()
SELECTED_TASKS=()
SELECTED_VARIANTS=()
TIMEOUT="$DEFAULT_TIMEOUT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      read -ra SELECTED_MODELS <<< "$2"; shift 2 ;;
    --tasks)
      read -ra SELECTED_TASKS  <<< "$2"; shift 2 ;;
    --variants)
      read -ra SELECTED_VARIANTS <<< "$2"; shift 2 ;;
    --timeout)
      TIMEOUT="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    --skip-existing)
      SKIP_EXISTING=true; shift ;;
    -h|--help)
      sed -n '2,20p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Apply defaults
[[ ${#SELECTED_MODELS[@]}   -eq 0 ]] && SELECTED_MODELS=("${DEFAULT_MODELS[@]}")
[[ ${#SELECTED_TASKS[@]}    -eq 0 ]] && SELECTED_TASKS=("${ALL_TASKS[@]}")
[[ ${#SELECTED_VARIANTS[@]} -eq 0 ]] && SELECTED_VARIANTS=("${DEFAULT_VARIANTS[@]}")

# ── Validate model slugs ───────────────────────────────────────────────────────

for slug in "${SELECTED_MODELS[@]}"; do
  if [[ -z "${MODEL_MAP[$slug]+_}" ]]; then
    echo "ERROR: Unknown model slug '$slug'" >&2
    echo "Known slugs: ${!MODEL_MAP[*]}" >&2
    exit 1
  fi
done

# ── Summary ────────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "  Gemini Benchmark Runner"
echo "════════════════════════════════════════════════════════════════"
echo "  Models   (${#SELECTED_MODELS[@]}): ${SELECTED_MODELS[*]}"
echo "  Tasks    (${#SELECTED_TASKS[@]}): ${SELECTED_TASKS[*]}"
echo "  Variants (${#SELECTED_VARIANTS[@]}): ${SELECTED_VARIANTS[*]}"
echo "  Timeout  : $TIMEOUT per run"
echo "  Dry run  : $DRY_RUN"
echo "  Skip existing: $SKIP_EXISTING"
TOTAL=$(( ${#SELECTED_MODELS[@]} * ${#SELECTED_TASKS[@]} * ${#SELECTED_VARIANTS[@]} ))
echo "  Total runs planned: $TOTAL"
echo "════════════════════════════════════════════════════════════════"
echo

RUNS_ROOT="$SCRIPT_DIR/_runs"
PASS=0; FAIL=0; SKIP=0

# ── Main loop ──────────────────────────────────────────────────────────────────

for slug in "${SELECTED_MODELS[@]}"; do
  DISPLAY_NAME="${MODEL_MAP[$slug]}"

  for task in "${SELECTED_TASKS[@]}"; do
    for variant in "${SELECTED_VARIANTS[@]}"; do

      # Build human-readable run ID
      SAFE_VARIANT="${variant//\//_}"
      RUN_ID="${slug}__${task}__${SAFE_VARIANT}"

      echo "────────────────────────────────────────────────────────────────"
      echo "  Run ID  : $RUN_ID"
      echo "  Model   : $DISPLAY_NAME"
      echo "  Task    : $task"
      echo "  Variant : $variant"
      echo "────────────────────────────────────────────────────────────────"

      # Skip if run dir already exists and --skip-existing is set
      if [[ "$SKIP_EXISTING" == "true" && -d "$RUNS_ROOT/$RUN_ID" ]]; then
        echo "  → SKIPPED (run dir exists; use --skip-existing=false to re-run)"
        SKIP=$(( SKIP + 1 ))
        echo
        continue
      fi

      CMD=(
        "$SCRIPT_DIR/run_eval_antigravity.sh"
        --task    "$task"
        --variant "$variant"
        --model   "$DISPLAY_NAME"
        --run-id  "$RUN_ID"
        --print-timeout "$TIMEOUT"
      )

      echo "  Command : ${CMD[*]}"
      echo

      if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] skipping execution"
        echo
        continue
      fi

      # Run — capture exit code without killing the batch on failure
      set +e
      "${CMD[@]}"
      RC=$?
      set -e

      if [[ $RC -eq 0 ]]; then
        echo "  → DONE (exit 0)"
        PASS=$(( PASS + 1 ))
      else
        echo "  → FAILED (exit $RC) — continuing to next run"
        FAIL=$(( FAIL + 1 ))
      fi
      echo
    done
  done
done

# ── Final summary ──────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "  Benchmark complete"
echo "  Passed : $PASS"
echo "  Failed : $FAIL"
echo "  Skipped: $SKIP"
echo "════════════════════════════════════════════════════════════════"
echo
echo "Next step — score all runs:"
echo "  ./eval_all_runs.sh ./_runs"
echo
echo "Or score only these runs by model slug pattern, e.g.:"
echo "  FILTER='gemini35flash_high__*' ./eval_all_runs.sh ./_runs"
echo "════════════════════════════════════════════════════════════════"
