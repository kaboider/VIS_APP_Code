#!/usr/bin/env python3
"""
Aggregate c4 eval results under tasks/_runs* and surface benchmark patterns.

Reads:
  - <run>/logs/eval_result.json
  - <run>/logs/summary.json, when available

Writes to tasks/analysis by default:
  - eval_run_details.csv
  - eval_batch_summary.csv
  - eval_model_summary.csv
  - eval_task_summary.csv
  - eval_type_summary.csv
  - eval_missing.csv
  - eval_note_summary.csv
  - eval_insights.md

Usage:
    python3 tasks/tools/analyze_eval_results.py
    python3 tasks/tools/analyze_eval_results.py --include-legacy-runs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK_RE = re.compile(r"^(\d{8}_\d{6})_(.+)_c(\d+)(?:_|$)")


def mean(values: list[float | int | None]) -> float | None:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return sum(clean) / len(clean) if clean else None


def stdev(values: list[float | int | None]) -> float | None:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return statistics.pstdev(clean) if len(clean) > 1 else 0.0 if clean else None


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def corr(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            xs.append(float(x))
            ys.append(float(y))
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sx = (sum((x - mx) ** 2 for x in xs) / len(xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / len(ys)) ** 0.5
    if not sx or not sy:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs) / sx / sy


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_model(exp: str, summary_model: str | None) -> str:
    if summary_model and not summary_model.startswith("<"):
        return summary_model
    if "cursor" in exp:
        return "Composer 2.5"
    if "5.4-mini" in exp:
        return "gpt-5.4-mini"
    if "5.5" in exp:
        return "gpt-5.5"
    if "claude_fable5" in exp:
        return "claude-fable-5"
    if "claude_4.8" in exp:
        return "claude-opus-4-8"
    if "claude_4.7" in exp:
        return "claude-opus-4-7"
    if "claude_4.6" in exp:
        return "claude-sonnet-4-6"
    if "claude_4.5" in exp:
        return "claude-haiku-4-5"
    return summary_model or "unknown"


def parse_run_dir(run_dir: Path) -> tuple[str, str, str]:
    match = TASK_RE.search(run_dir.name)
    if not match:
        return "", "", ""
    return match.group(1), match.group(2), f"c{match.group(3)}"


def iter_run_dirs(tasks_dir: Path, include_legacy_runs: bool) -> list[Path]:
    runs: list[Path] = []
    for exp_dir in sorted(tasks_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("_runs"):
            continue
        if not include_legacy_runs and exp_dir.name in {"_runs", "_runs-fix"}:
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue
            if "_superseded" in run_dir.parts:
                continue
            if (run_dir / "logs").exists():
                runs.append(run_dir)
    return runs


def collect(tasks_dir: Path, include_legacy_runs: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for run_dir in iter_run_dirs(tasks_dir, include_legacy_runs):
        exp = run_dir.parent.name
        ts, task_from_name, variant_from_name = parse_run_dir(run_dir)
        logs = run_dir / "logs"
        summary_json = read_json(logs / "summary.json")
        model = infer_model(exp, summary_json.get("summary", {}).get("model"))
        eval_path = logs / "eval_result.json"
        base = {
            "experiment": exp,
            "run": run_dir.name,
            "timestamp": ts,
            "task": task_from_name,
            "variant": variant_from_name,
            "model": model,
            "path": str(run_dir),
        }
        if not eval_path.exists():
            missing.append({**base, "reason": "missing_eval_result"})
            continue
        data = read_json(eval_path)
        summary = data.get("summary") or {}
        if not summary:
            missing.append({**base, "reason": "bad_or_empty_eval_result"})
            continue
        n_critical = summary.get("n_critical") or 0
        n_bonus = summary.get("n_bonus") or 0
        found_critical = summary.get("found_critical") or 0
        found_bonus = summary.get("found_bonus") or 0
        row = {
            **base,
            "task": summary.get("task") or task_from_name,
            "variant": summary.get("variant") or variant_from_name,
            "auth_bypass_used": summary.get("auth_bypass_used"),
            "n_critical": n_critical,
            "n_bonus": n_bonus,
            "found_critical": found_critical,
            "found_bonus": found_bonus,
            "found_rate_critical": found_critical / n_critical if n_critical else None,
            "found_rate_bonus": found_bonus / n_bonus if n_bonus else None,
            "avg_localization_critical": summary.get("avg_localization_critical"),
            "avg_behavior_critical": summary.get("avg_behavior_critical"),
            "combined_score_critical": summary.get("combined_score_critical"),
            "turn_count": summary_json.get("summary", {}).get("turn_count"),
            "wall_clock_first_to_last_assistant_s": summary_json.get("summary", {}).get(
                "wall_clock_first_to_last_assistant_s"
            ),
            "results": data.get("results") or [],
        }
        for key, value in (summary.get("tier_distribution") or {}).items():
            row[f"tier_{key}"] = value
        rows.append(row)
    return rows, missing


def dedupe_latest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["experiment"], row["task"])
        if key not in latest or row.get("timestamp", "") > latest[key].get("timestamp", ""):
            latest[key] = row
    return list(latest.values())


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    omit: set[str] | None = None,
    preferred: list[str] | None = None,
) -> None:
    omit = omit or set()
    preferred = preferred or []
    serializable = []
    for row in rows:
        serializable.append({k: v for k, v in row.items() if k not in omit})
    all_keys = sorted({k for row in serializable for k in row.keys()})
    keys = [k for k in preferred if k in all_keys] + [k for k in all_keys if k not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(serializable)


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    out = []
    for value, group in grouped.items():
        out.append(
            {
                key: value,
                "n": len(group),
                "score_mean": mean([r.get("combined_score_critical") for r in group]),
                "score_std": stdev([r.get("combined_score_critical") for r in group]),
                "score_min": min(r.get("combined_score_critical") for r in group),
                "score_max": max(r.get("combined_score_critical") for r in group),
                "localization_mean": mean([r.get("avg_localization_critical") for r in group]),
                "behavior_mean": mean([r.get("avg_behavior_critical") for r in group]),
                "found_rate_mean": mean([r.get("found_rate_critical") for r in group]),
                "critical_items_mean": mean([r.get("n_critical") for r in group]),
            }
        )
    return sorted(out, key=lambda r: (r["score_mean"] is None, -(r["score_mean"] or 0), r[key]))


def missing_summary(missing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in missing:
        grouped[row["experiment"]].append(row)
    out = []
    for exp, group in grouped.items():
        out.append(
            {
                "experiment": exp,
                "missing_count": len(group),
                "tasks": ",".join(sorted(row.get("task") or "" for row in group)),
            }
        )
    return sorted(out, key=lambda r: (-r["missing_count"], r["experiment"]))


def model_task_pivot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = sorted({row["task"] for row in rows}, key=task_sort_key)
    models = sorted({row["model"] for row in rows})
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        score = row.get("combined_score_critical")
        if isinstance(score, (int, float)):
            grouped[(row["task"], row["model"])].append(float(score))
    out = []
    for task in tasks:
        rec: dict[str, Any] = {"task": task}
        for model in models:
            rec[model] = mean(grouped[(task, model)])
        out.append(rec)
    return out


def task_sort_key(task: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)_", task or "")
    return (int(match.group(1)) if match else 999, task or "")


def type_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for item in row["results"]:
            if item.get("tier") == "critical":
                grouped[str(item.get("type") or "unknown")].append(item)
    out = []
    for typ, items in grouped.items():
        out.append(
            {
                "type": typ,
                "n": len(items),
                "found_rate": sum(1 for item in items if item.get("found")) / len(items),
                "localization_mean": mean([item.get("localization") for item in items]),
                "behavior_mean": mean([item.get("behavior") for item in items]),
            }
        )
    return sorted(out, key=lambda r: -r["n"])


def note_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    by_type: Counter[tuple[str, str]] = Counter()
    for row in rows:
        for item in row["results"]:
            if item.get("tier") != "critical" or not item.get("found"):
                continue
            note = item.get("note") or ""
            typ = item.get("type") or "unknown"
            counts[note] += 1
            by_type[(note, typ)] += 1
    out = []
    for note, count in counts.most_common():
        type_counts = {typ: by_type[(note, typ)] for item_note, typ in by_type if item_note == note}
        top = max(type_counts.items(), key=lambda kv: kv[1])[0] if type_counts else ""
        out.append({"note": note, "count": count, "top_type": top})
    return out


def model_task_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    out = []
    for task, group in grouped.items():
        best = max(group, key=lambda r: r.get("combined_score_critical") or -1)
        worst = min(group, key=lambda r: r.get("combined_score_critical") or 999)
        out.append(
            {
                "task": task,
                "best_score": best["combined_score_critical"],
                "best_model": best["model"],
                "best_experiment": best["experiment"],
                "worst_score": worst["combined_score_critical"],
                "worst_model": worst["model"],
                "worst_experiment": worst["experiment"],
            }
        )
    return sorted(out, key=lambda r: r["task"])


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> list[str]:
    use_rows = rows[:limit] if limit else rows
    lines = []
    lines.append("| " + " | ".join(title for title, _ in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in use_rows:
        vals = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                vals.append(fmt(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(
    path: Path,
    all_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    type_rows: list[dict[str, Any]],
    note_rows: list[dict[str, Any]],
    winners: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Eval Insights")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Evaluated run files found: {len(all_rows)}")
    lines.append(f"- Latest-per-experiment-task rows used for headline analysis: {len(rows)}")
    lines.append(f"- Runs missing usable eval_result.json: {len(missing)}")
    lines.append("- Excluded by default: tasks/_runs, tasks/_runs-fix, _superseded directories.")
    lines.append("")
    lines.append("## Headline Patterns")
    lines.append("")
    lines.append(
        f"- Score tracks behavior more strongly than localization: corr(score, behavior)={fmt(corr(rows, 'combined_score_critical', 'avg_behavior_critical'))}, "
        f"corr(score, localization)={fmt(corr(rows, 'combined_score_critical', 'avg_localization_critical'))}, "
        f"corr(score, found_rate)={fmt(corr(rows, 'combined_score_critical', 'found_rate_critical'))}."
    )
    lines.append("- This suggests the benchmark is mostly separating functional interaction depth from static visual coverage.")
    if type_rows:
        worst_type = min(type_rows, key=lambda r: r["behavior_mean"])
        best_type = max(type_rows, key=lambda r: r["behavior_mean"])
        lines.append(
            f"- Interaction type gap is large: best behavior is {best_type['type']} ({fmt(best_type['behavior_mean'])}); "
            f"weakest is {worst_type['type']} ({fmt(worst_type['behavior_mean'])})."
        )
    if task_rows:
        hardest = min(task_rows, key=lambda r: r["score_mean"] or 999)
        easiest = max(task_rows, key=lambda r: r["score_mean"] or -1)
        volatile = max(task_rows, key=lambda r: r["score_std"] or -1)
        lines.append(
            f"- Hardest task by mean score: {hardest['task']} ({fmt(hardest['score_mean'])}); "
            f"easiest: {easiest['task']} ({fmt(easiest['score_mean'])}); "
            f"most volatile: {volatile['task']} (std {fmt(volatile['score_std'])})."
        )
    if model_rows:
        leader = model_rows[0]
        lines.append(
            f"- Top model family by current mean: {leader['model']} ({fmt(leader['score_mean'])}), "
            f"but task winners vary by domain."
        )
    lines.append("")
    lines.append("## Model Summary")
    lines.extend(
        markdown_table(
            model_rows,
            [
                ("model", "model"),
                ("n", "n"),
                ("score", "score_mean"),
                ("loc", "localization_mean"),
                ("beh", "behavior_mean"),
                ("found", "found_rate_mean"),
            ],
        )
    )
    lines.append("")
    lines.append("## Task Difficulty")
    lines.extend(
        markdown_table(
            sorted(task_rows, key=lambda r: r["score_mean"] or 0),
            [
                ("task", "task"),
                ("n", "n"),
                ("score", "score_mean"),
                ("std", "score_std"),
                ("loc", "localization_mean"),
                ("beh", "behavior_mean"),
                ("found", "found_rate_mean"),
            ],
        )
    )
    lines.append("")
    lines.append("## Batch Summary")
    lines.extend(
        markdown_table(
            batch_rows,
            [
                ("experiment", "experiment"),
                ("n", "n"),
                ("score", "score_mean"),
                ("loc", "localization_mean"),
                ("beh", "behavior_mean"),
                ("found", "found_rate_mean"),
            ],
        )
    )
    lines.append("")
    if missing_rows:
        lines.append("## Missing Eval Results By Batch")
        lines.extend(
            markdown_table(
                missing_rows,
                [
                    ("experiment", "experiment"),
                    ("missing", "missing_count"),
                    ("tasks", "tasks"),
                ],
            )
        )
        lines.append("")
    lines.append("## Critical Item Type Summary")
    lines.extend(
        markdown_table(
            type_rows,
            [
                ("type", "type"),
                ("n", "n"),
                ("found", "found_rate"),
                ("loc", "localization_mean"),
                ("beh", "behavior_mean"),
            ],
        )
    )
    lines.append("")
    lines.append("## Best And Worst By Task")
    lines.extend(
        markdown_table(
            winners,
            [
                ("task", "task"),
                ("best", "best_score"),
                ("best model", "best_model"),
                ("worst", "worst_score"),
                ("worst model", "worst_model"),
            ],
        )
    )
    lines.append("")
    lines.append("## Frequent Behavior Notes")
    lines.extend(
        markdown_table(
            note_rows,
            [("count", "count"), ("top type", "top_type"), ("note", "note")],
            limit=20,
        )
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Low click/toggle behavior notes point to missing state transitions, dialogs, popouts, and route changes.")
    lines.append("- High found rates with low behavior scores mean more visual or DOM coverage alone will not move the benchmark much.")
    lines.append("- Real-estate, project-management, streaming, and cloud-storage should be treated as stress tests for routing and multi-page state.")
    lines.append("- Ecommerce and chat are useful sanity checks; they reward models that implement common form and navigation flows correctly.")
    lines.append("- Missing eval_result.json rows should be tracked separately from score, because failure to evaluate can otherwise look like better average performance.")
    lines.append("- For future runs, compare model families with the model-task pivot rather than only the global model mean.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))
    parser.add_argument("--out-dir", type=Path, default=Path("tasks/analysis"))
    parser.add_argument("--include-legacy-runs", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, missing = collect(args.tasks_dir, args.include_legacy_runs)
    rows = dedupe_latest(all_rows)
    batch_rows = group_summary(rows, "experiment")
    model_rows = group_summary(rows, "model")
    task_rows = sorted(group_summary(rows, "task"), key=lambda r: r["score_mean"] or 0)
    type_rows = type_summary(rows)
    note_rows = note_summary(rows)
    winners = model_task_winners(rows)
    missing_rows = missing_summary(missing)
    pivot_rows = model_task_pivot(rows)

    write_csv(args.out_dir / "eval_run_details.csv", rows, omit={"results"})
    write_csv(args.out_dir / "eval_all_run_details.csv", all_rows, omit={"results"})
    write_csv(args.out_dir / "eval_batch_summary.csv", batch_rows)
    write_csv(args.out_dir / "eval_model_summary.csv", model_rows)
    write_csv(args.out_dir / "eval_task_summary.csv", task_rows)
    write_csv(args.out_dir / "eval_type_summary.csv", type_rows)
    write_csv(args.out_dir / "eval_missing.csv", missing)
    write_csv(args.out_dir / "eval_missing_summary.csv", missing_rows)
    write_csv(args.out_dir / "eval_note_summary.csv", note_rows)
    write_csv(args.out_dir / "eval_task_winners.csv", winners)
    write_csv(args.out_dir / "eval_model_task_pivot.csv", pivot_rows, preferred=["task"])
    write_report(
        args.out_dir / "eval_insights.md",
        all_rows,
        rows,
        missing,
        batch_rows,
        model_rows,
        task_rows,
        type_rows,
        note_rows,
        winners,
        missing_rows,
    )
    print(f"Analyzed {len(all_rows)} evaluated runs; {len(rows)} latest rows; {len(missing)} missing evals.")
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
