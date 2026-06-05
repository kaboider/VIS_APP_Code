#!/usr/bin/env bash
#
# run_eval_gemini.sh — thin wrapper around run_eval.sh, fixed to --cli gemini.
# Same flag surface as run_eval.sh (e.g. --task, --variant, --model).
#
# Auth: run `gemini` once for Google OAuth (free Code Assist tier) — OR set
#       GEMINI_API_KEY (https://aistudio.google.com/apikey, paid tier needed
#       for serious batch runs since free tier RPD is tight).
#
# Requires @google/gemini-cli >= 0.11 (stream-json output format).
#
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_eval.sh" --cli gemini "$@"
