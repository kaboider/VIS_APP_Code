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

## 2. Eval

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
