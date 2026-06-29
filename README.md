# VIS_APP Code

## 1. Run Antigravity (c4 variant)

Run one task with the Antigravity CLI on the `c4` variant:

```bash
./run_eval_antigravity.sh --task 4_forum --variant c4
```

- `--task` — any task folder: `1_newsletter`, `2_real-estate`, `3_job-board`,
  `4_forum`, `5_travel-booking`, `6_chat`, `7_cloud-storage`, `8_ecommerce`,
  `9_project-management`, `10_streaming_music-streaming`
- Default model: `Gemini 3.1 Pro (High)`. Override with
  `--model "Claude Sonnet 4.6 (Thinking)"` (run `agy models` to list).
- Auth: run `agy` once to sign in. Override the binary with `AGY_BIN=/abs/path/to/agy`.

Output lands in `./_runs/<run_id>/`.

Other CLIs use the generic runner — one task with `run_eval.sh`, all 10 with
`run_all.sh`:

```bash
./run_eval.sh --task 4_forum --variant c4 --cli claude --model claude-sonnet-4-6
./run_all.sh  --variant c4 --cli codex  --model gpt-5.5      # all 10 tasks
```

`--cli` is one of `claude`, `codex`, `gemini`, `cursor`, `antigravity`.

## 2. Pin the harness (CLI) version

The agent CLI ("harness") version is a controlled variable — every run records
it in `meta.json` as `cli_version` and `cli_bin`. To lock a run to a specific
version, install that version into an **isolated npm prefix** (your global
install stays untouched) and point the runner at it with the per-CLI `*_BIN`
env var:

```bash
# install a pinned Claude Code, then run against it
npm install --prefix ~/.claude-pinned/2.1.152 @anthropic-ai/claude-code@2.1.152
CLAUDE_BIN=~/.claude-pinned/2.1.152/node_modules/.bin/claude \
  ./run_all.sh --variant c4 --cli claude --model claude-sonnet-4-6
```

Each CLI has its own override var; `run_eval.sh` resolves it first and otherwise
falls back to whatever is on `PATH`:

| CLI | env var | install (npm) |
|-----|---------|---------------|
| claude | `CLAUDE_BIN` | `@anthropic-ai/claude-code@<ver>` → `.../node_modules/.bin/claude` |
| codex | `CODEX_BIN` | `@openai/codex@<ver>` → `.../node_modules/.bin/codex` |
| gemini | `GEMINI_BIN` | `@google/gemini-cli@<ver>` (>= 0.11) |
| cursor | `CURSOR_BIN` | vendor installer — point `CURSOR_BIN` at the chosen `cursor-agent` |
| antigravity | `AGY_BIN` | vendor installer — point `AGY_BIN` at the chosen `agy` |

Notes:
- The npm-prefix trick works for claude / codex / gemini. For cursor and
  antigravity (not npm packages), install the version you want via their own
  installer and point `CURSOR_BIN` / `AGY_BIN` at that binary.
- Verify the version that actually ran: check `cli_version` in the run's
  `meta.json` (printed at launch too).
- Reasoning effort is a separate knob: Claude via `CLAUDE_EFFORT` (e.g.
  `xhigh`); Codex via `model_reasoning_effort` in its `config.toml`
  (`CODEX_HOME`).

## 3. Eval

Score every run under `_runs/`:

```bash
./eval_all_runs.sh ./_runs
```

Writes per-run results and a leaderboard to `./_runs/eval_summary.csv`.

Useful overrides:

```bash
FILTER='*c4*' ./eval_all_runs.sh ./_runs        # only c4 runs
FORCE=1 ./eval_all_runs.sh ./_runs              # re-eval even if already scored
SKIP_DOCKER=1 ./eval_all_runs.sh ./_runs        # services already up
```
