#!/usr/bin/env bash
#
# run_eval_antigravity.sh — thin wrapper around run_eval.sh, fixed to
# --cli antigravity. Same flag surface as run_eval.sh (e.g. --task, --variant,
# --model).
#
# CLI: Google Antigravity, launched as `agy` (default ~/.local/bin/agy; override
#      with AGY_BIN=/abs/path/to/agy).
#
# Auth: run `agy` once to sign in.
#
# Model: --model takes a display-name string (run `agy models` to list), e.g.
#        "Gemini 3.1 Pro (High)", "Claude Sonnet 4.6 (Thinking)". Default is
#        "Gemini 3.1 Pro (High)". An unknown value silently falls back to the
#        CLI default.
#
# Note: `agy --print` streams plain-text agent narration (no stream-json), so
#       there's no per-run event analysis — just the transcript/stderr logs and
#       the built workspace. The print timeout defaults to 90m; override with
#       AGY_PRINT_TIMEOUT (e.g. AGY_PRINT_TIMEOUT=120m).
#
# Examples:
#   ./run_eval_antigravity.sh --task 1_newsletter --variant c0
#   ./run_eval_antigravity.sh --task 4_forum --variant c3 --model "Claude Sonnet 4.6 (Thinking)"
#
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_eval.sh" --cli antigravity "$@"
