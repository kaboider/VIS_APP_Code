#!/usr/bin/env bash
#
# run_anyclaude_benchmark.sh — Run anyclaude (coder/anyclaude) models against
# one or more tasks and variants, with clean human-readable run IDs.
#
# anyclaude wraps Claude Code CLI with a proxy for alternative LLM providers
# (OpenAI, Google, xAI) via the Vercel AI SDK. Model names use <provider>/<model>
# format.
#
# Run ID format:  <model_slug>__<task>__<variant>
# Example:        gemini25flash__4_forum__c4
#
# Usage:
#   ./run_anyclaude_benchmark.sh                                 # default: gemini-2.5-flash × all tasks × c4
#   ./run_anyclaude_benchmark.sh --tasks "4_forum 1_newsletter"
#   ./run_anyclaude_benchmark.sh --models "gpt5mini gemini35flash"
#   ./run_anyclaude_benchmark.sh --variants "c4 c0"
#   ./run_anyclaude_benchmark.sh --dry-run                       # print commands, don't run
#   ./run_anyclaude_benchmark.sh --timeout 120m
#   ./run_anyclaude_benchmark.sh --skip-existing                 # skip if run dir already exists
#
# After all runs complete, score them:
#   ./eval_all_runs.sh ./_runs
#
# ── Required env vars ────────────────────────────────────────────────────────
#   GOOGLE_GENERATIVE_AI_API_KEY or GOOGLE_API_KEY for google/* models
#   OPENAI_API_KEY     for openai/* models
#   XAI_API_KEY        for xai/* models
#   (plus `claude /login` for Claude Code Max-plan token)
#
# ── Model slug map ──────────────────────────────────────────────────────────
# The "display name" is the --model value passed to anyclaude (<provider>/<model>).
# The slug is used in the run ID (filesystem-safe, no slashes/spaces).
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────

# All anyclaude models we might benchmark (slug → display name)
declare -A MODEL_MAP
# Google models
MODEL_MAP["gemini25flash"]="google/gemini-2.5-flash"
MODEL_MAP["gemini25pro"]="google/gemini-2.5-pro"
MODEL_MAP["gemini35flash"]="google/gemini-3.5-flash"
# OpenAI models
MODEL_MAP["gpt5"]="openai/gpt-5"
MODEL_MAP["gpt5mini"]="openai/gpt-5-mini"
MODEL_MAP["gpt5mini_high"]="openai/gpt-5-mini"        # same model, high reasoning
MODEL_MAP["o4mini"]="openai/o4-mini"
# xAI models
MODEL_MAP["grok3"]="xai/grok-3"

# Per-slug overrides for anyclaude-specific env vars.
# REASONING_EFFORT and SERVICE_TIER are only relevant for OpenAI models.
declare -A SLUG_REASONING_EFFORT
SLUG_REASONING_EFFORT["gpt5mini_high"]="high"

declare -A SLUG_SERVICE_TIER
# (none by default; user can add e.g. SLUG_SERVICE_TIER["gpt5_priority"]="priority")

# Ordered list of slugs (controls run order)
DEFAULT_MODELS=(
  "gemini25flash"
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

# ── Arg parsing ──────────────────────────────────────────────────────────────

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
      sed -n '2,35p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Apply defaults
[[ ${#SELECTED_MODELS[@]}   -eq 0 ]] && SELECTED_MODELS=("${DEFAULT_MODELS[@]}")
[[ ${#SELECTED_TASKS[@]}    -eq 0 ]] && SELECTED_TASKS=("${ALL_TASKS[@]}")
[[ ${#SELECTED_VARIANTS[@]} -eq 0 ]] && SELECTED_VARIANTS=("${DEFAULT_VARIANTS[@]}")

# ── Validate model slugs ────────────────────────────────────────────────────

for slug in "${SELECTED_MODELS[@]}"; do
  if [[ -z "${MODEL_MAP[$slug]+_}" ]]; then
    echo "ERROR: Unknown model slug '$slug'" >&2
    echo "Known slugs: ${!MODEL_MAP[*]}" >&2
    exit 1
  fi
done

# ── Validate API keys ───────────────────────────────────────────────────────

check_api_key() {
  local slug="$1" model="${MODEL_MAP[$1]}"
  local provider="${model%%/*}"
  case "$provider" in
    google)
      if [[ -z "${GOOGLE_GENERATIVE_AI_API_KEY:-}" && -n "${GOOGLE_API_KEY:-}" ]]; then
        export GOOGLE_GENERATIVE_AI_API_KEY="$GOOGLE_API_KEY"
      fi
      if [[ -z "${GOOGLE_GENERATIVE_AI_API_KEY:-}" ]]; then
        echo "ERROR: GOOGLE_GENERATIVE_AI_API_KEY not set (or set GOOGLE_API_KEY as a compatibility alias) for model '$model'" >&2
        return 1
      fi ;;
    openai)
      if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "ERROR: OPENAI_API_KEY not set (required for model '$model')" >&2
        return 1
      fi ;;
    xai)
      if [[ -z "${XAI_API_KEY:-}" ]]; then
        echo "ERROR: XAI_API_KEY not set (required for model '$model')" >&2
        return 1
      fi ;;
  esac
  return 0
}

for slug in "${SELECTED_MODELS[@]}"; do
  check_api_key "$slug" || exit 1
done

# ── Summary ──────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "  anyclaude Benchmark Runner"
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

RUNS_ROOT="${RUNS_ROOT:-$SCRIPT_DIR/_runs}"
PASS=0; FAIL=0; SKIP=0

# ── Main loop ────────────────────────────────────────────────────────────────

for slug in "${SELECTED_MODELS[@]}"; do
  DISPLAY_NAME="${MODEL_MAP[$slug]}"

  # Set per-slug anyclaude env vars
  export ANYCLAUDE_REASONING_EFFORT="${SLUG_REASONING_EFFORT[$slug]:-}"
  export ANYCLAUDE_SERVICE_TIER="${SLUG_SERVICE_TIER[$slug]:-}"

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
      [[ -n "$ANYCLAUDE_REASONING_EFFORT" ]] && echo "  Reasoning effort : $ANYCLAUDE_REASONING_EFFORT"
      [[ -n "$ANYCLAUDE_SERVICE_TIER" ]]     && echo "  Service tier     : $ANYCLAUDE_SERVICE_TIER"
      echo "────────────────────────────────────────────────────────────────"

      # Skip if run dir already exists and --skip-existing is set
      if [[ "$SKIP_EXISTING" == "true" && -d "$RUNS_ROOT/$RUN_ID" ]]; then
        echo "  → SKIPPED (run dir exists; use --skip-existing=false to re-run)"
        SKIP=$(( SKIP + 1 ))
        echo
        continue
      fi

      CMD=(
        "$SCRIPT_DIR/run_eval_anyclaude.sh"
        --task    "$task"
        --variant "$variant"
        --model   "$DISPLAY_NAME"
        --run-id  "$RUN_ID"
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

# Clean up exported vars
unset ANYCLAUDE_REASONING_EFFORT ANYCLAUDE_SERVICE_TIER

# ── Final summary ────────────────────────────────────────────────────────────

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
echo "  FILTER='gemini25flash__*' ./eval_all_runs.sh ./_runs"
echo "════════════════════════════════════════════════════════════════"
