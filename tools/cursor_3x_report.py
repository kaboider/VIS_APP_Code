#!/usr/bin/env python3
"""Aggregate three Cursor benchmark repetitions into mean/sample-SD reports."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


TASKS = [
    "1_newsletter", "2_real-estate", "3_job-board", "4_forum",
    "5_travel-booking", "6_chat", "7_cloud-storage", "8_ecommerce",
    "9_project-management", "10_streaming_music-streaming",
]

# Cursor does not include billed USD in the CLI trajectory summaries.  These
# rates are therefore an explicit theoretical estimate, using the published
# standard Grok short-context rates (USD per 1M tokens), not a Cursor invoice.
INPUT_USD_PER_MTOK = 2.00
CACHE_USD_PER_MTOK = 0.30
OUTPUT_USD_PER_MTOK = 6.00


def load_run(root: Path, rep: int, task: str) -> dict:
    run = root / f"r{rep}_{task}_c4_cursor"
    summary = json.loads((run / "logs" / "summary.json").read_text())["summary"]
    evaluation = json.loads((run / "logs" / "eval_result.json").read_text())["summary"]
    tokens = summary["tokens"]
    theoretical_cost = (
        int(tokens["input_tokens"]) * INPUT_USD_PER_MTOK
        + int(tokens["cache_read_input_tokens"]) * CACHE_USD_PER_MTOK
        + int(tokens["output_tokens"]) * OUTPUT_USD_PER_MTOK
    ) / 1_000_000
    return {
        "run": run.name,
        "score": float(evaluation["combined_score_critical"]),
        "input": int(tokens["input_tokens"]),
        "cache": int(tokens["cache_read_input_tokens"]),
        "output": int(tokens["output_tokens"]),
        "runtime": float(summary["wall_clock_s"]),
        "theoretical_cost_usd": theoretical_cost,
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: cursor_3x_report.py <runs_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    canonical_path = root / "cursor_grok-4.6-high_c4_3x_summary.csv"
    data = {rep: {task: load_run(root, rep, task) for task in TASKS}
            for rep in (1, 2, 3)}

    run_rows = [
        {"task": task, "replicate": f"r{rep}", **data[rep][task]}
        for task in TASKS for rep in (1, 2, 3)
    ]

    task_rows = []
    for task in TASKS:
        scores = [data[r][task]["score"] for r in (1, 2, 3)]
        inputs = [data[r][task]["input"] for r in (1, 2, 3)]
        caches = [data[r][task]["cache"] for r in (1, 2, 3)]
        outputs = [data[r][task]["output"] for r in (1, 2, 3)]
        runtimes = [data[r][task]["runtime"] for r in (1, 2, 3)]
        costs = [data[r][task]["theoretical_cost_usd"] for r in (1, 2, 3)]
        score_mean, score_sd = mean_sd(scores)
        runtime_mean, runtime_sd = mean_sd(runtimes)
        task_rows.append({
            "task": task, "r1_score": scores[0], "r2_score": scores[1],
            "r3_score": scores[2], "score_mean": score_mean,
            "score_sample_sd": score_sd,
            "input_mean": statistics.mean(inputs),
            "cache_mean": statistics.mean(caches),
            "output_mean": statistics.mean(outputs),
            "runtime_mean_s": runtime_mean, "runtime_sample_sd_s": runtime_sd,
            "theoretical_cost_mean_usd": statistics.mean(costs),
            "theoretical_cost_sample_sd_usd": statistics.stdev(costs),
        })

    rep_means = [statistics.mean([data[r][t]["score"] for t in TASKS])
                 for r in (1, 2, 3)]
    overall_mean, overall_sd = mean_sd(rep_means)

    canonical_fields = [
        "task", "r1_score", "r2_score", "r3_score", "score_mean",
        "score_sample_sd", "input_mean", "cache_mean", "output_mean",
        "runtime_mean_s", "runtime_sample_sd_s",
    ]
    with canonical_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=canonical_fields)
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in canonical_fields} for row in task_rows
        )

    runs_csv_path = root / "cursor_grok-4.6-high_c4_3x_runs.csv"
    with runs_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    csv_path = root / "cursor_grok-4.6-high_c4_3x_summary_with_cost.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)

    md_path = root / "cursor_grok-4.6-high_c4_3x_summary.md"
    lines = [
        "# Cursor Grok 4.6 High — C4, three repetitions",
        "",
        f"- Replicate means: **r1={rep_means[0]:.3f}, r2={rep_means[1]:.3f}, r3={rep_means[2]:.3f}**",
        f"- Overall score: **{overall_mean:.3f} ± {overall_sd:.3f}** (mean ± sample SD across replicate means)",
        "- Every task statistic below uses three independent runs; SD is sample SD (n=3, denominator n−1).",
        ("- Cost is a theoretical token estimate, not Cursor billed cost: "
         f"${INPUT_USD_PER_MTOK:.2f}/M non-cached input + "
         f"${CACHE_USD_PER_MTOK:.2f}/M cached input + "
         f"${OUTPUT_USD_PER_MTOK:.2f}/M output."),
        "",
        "| Task | r1 | r2 | r3 | Score mean ± SD | Runtime mean ± SD | Mean input | Mean cached | Mean output | Est. cost/run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['task']} | {row['r1_score']:.3f} | {row['r2_score']:.3f} | "
            f"{row['r3_score']:.3f} | {row['score_mean']:.3f} ± {row['score_sample_sd']:.3f} | "
            f"{row['runtime_mean_s']:.1f}s ± {row['runtime_sample_sd_s']:.1f}s | "
            f"{row['input_mean']:,.0f} | {row['cache_mean']:,.0f} | {row['output_mean']:,.0f} | "
            f"${row['theoretical_cost_mean_usd']:.2f} |"
        )
    total_cost = sum(row["theoretical_cost_usd"] for row in run_rows)
    lines.extend(["", f"- Estimated total for all 30 runs: **${total_cost:.2f}**."])
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {runs_csv_path}")
    print(f"wrote {canonical_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"overall={overall_mean:.6f} sample_sd={overall_sd:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
