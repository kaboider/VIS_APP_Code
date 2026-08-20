#!/usr/bin/env bash
# CAMEL single-agent harness against locally hosted Qwen3.8-27B (vLLM :8201).
# Generation does not need docker. Official Playwright scores do.
# This machine has no docker access, so post-run scoring uses structural fallback.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
export CAMEL_API_URL="${CAMEL_API_URL:-http://127.0.0.1:8201/v1}"
export RUNS_ROOT="${RUNS_ROOT:-$PWD/_runs_camel_qwen38_27b_256k}"
export SKIP_EVAL="${SKIP_EVAL:-0}"
exec ./_run_camel_qwen35_27b.sh "$@"
