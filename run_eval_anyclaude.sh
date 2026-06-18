#!/usr/bin/env bash
#
# run_eval_anyclaude.sh — thin wrapper around run_eval.sh, fixed to
# --cli anyclaude. Same flag surface as run_eval.sh (e.g. --task, --variant,
# --model).
#
# CLI: coder/anyclaude — wraps Claude Code CLI with a proxy for alternative LLM
#      providers (OpenAI, Google, xAI, Azure) via the Vercel AI SDK.
#
# Install: npm install -g anyclaude
#   OR:    cd /path/to/anyclaude && bun run build && npm install -g .
#
# Auth: requires `claude /login` for Max-plan token (anyclaude wraps claude),
#       PLUS the provider's API key: OPENAI_API_KEY,
#       GOOGLE_GENERATIVE_AI_API_KEY (or GOOGLE_API_KEY), XAI_API_KEY.
#
# Model: --model takes a <provider>/<model> string, e.g.:
#        "openai/gpt-5-mini", "google/gemini-2.5-flash", "xai/grok-3".
#        Default: "google/gemini-2.5-flash".
#
# Extra env vars (optional):
#   ANYCLAUDE_REASONING_EFFORT  — "minimal" | "low" | "medium" | "high" (OpenAI)
#   ANYCLAUDE_SERVICE_TIER      — "flex" | "priority" (OpenAI)
#
# Examples:
#   ./run_eval_anyclaude.sh --task 1_newsletter --variant c4
#   ./run_eval_anyclaude.sh --task 4_forum --variant c4 --model "openai/gpt-5-mini"
#   GOOGLE_GENERATIVE_AI_API_KEY=... ./run_eval_anyclaude.sh --task 4_forum --variant c4 --model "google/gemini-2.5-flash"
#   ANYCLAUDE_REASONING_EFFORT=high ./run_eval_anyclaude.sh --task 4_forum --variant c4 --model "openai/gpt-5"
#
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_eval.sh" --cli anyclaude "$@"
