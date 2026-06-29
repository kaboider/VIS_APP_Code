#!/usr/bin/env python3
"""
Generate dependency-free SVG figures for the refreshed results directory.

Inputs:
  - results/eval_analysis/*.csv
  - results/workflow_analysis/*.csv

Outputs:
  - results/figures/*.svg
  - results/figures/index.md
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


MODEL_COLORS = {
    "claude-fable-5": "#7c3aed",
    "claude-opus-4-8": "#c2410c",
    "claude-opus-4-7": "#ea580c",
    "claude-sonnet-4-6": "#9333ea",
    "claude-haiku-4-5": "#a855f7",
    "gpt-5.5": "#0891b2",
    "gpt-5.4-mini": "#4f46e5",
    "Composer 2.5": "#0f766e",
}


def parse_value(value: str) -> Any:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [{k: parse_value(v) for k, v in row.items()} for row in csv.DictReader(f)]


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; fill: #0f172a; }}
.title {{ font-size: 20px; font-weight: 700; }}
.subtitle {{ font-size: 12px; fill: #64748b; }}
.axis {{ stroke: #cbd5e1; stroke-width: 1; }}
.grid {{ stroke: #e2e8f0; stroke-width: 1; }}
.label {{ font-size: 12px; fill: #334155; }}
.small {{ font-size: 11px; fill: #64748b; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
{body}
</svg>
""",
        encoding="utf-8",
    )


def title_block(title: str, subtitle: str = "") -> str:
    sub = f'<text class="subtitle" x="28" y="50">{esc(subtitle)}</text>' if subtitle else ""
    return f'<text class="title" x="28" y="30">{esc(title)}</text>{sub}'


def bar_chart(
    path: Path,
    rows: list[dict[str, Any]],
    label_key: str,
    value_key: str,
    title: str,
    subtitle: str,
    *,
    color_key: str | None = None,
    width: int = 980,
    row_h: int = 34,
) -> None:
    rows = [r for r in rows if isinstance(r.get(value_key), (int, float))]
    rows = sorted(rows, key=lambda r: float(r[value_key]), reverse=True)
    left, right, top = 250, 40, 76
    chart_w = width - left - right
    height = top + len(rows) * row_h + 48
    max_v = max(float(r[value_key]) for r in rows) if rows else 1
    body = [title_block(title, subtitle)]
    for i in range(6):
        x = left + chart_w * i / 5
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-44}"/>')
        body.append(f'<text class="small" x="{x-10:.1f}" y="{height-24}">{max_v*i/5:.2f}</text>')
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        v = float(row[value_key])
        w = chart_w * v / max_v if max_v else 0
        label = row.get(label_key)
        color = MODEL_COLORS.get(str(row.get(color_key or label_key)), "#2563eb")
        body.append(f'<text class="label" x="28" y="{y+21}">{esc(label)}</text>')
        body.append(f'<rect x="{left}" y="{y+6}" width="{w:.1f}" height="20" rx="3" fill="{color}"/>')
        body.append(f'<text class="small" x="{left+w+8:.1f}" y="{y+21}">{fmt(v)}</text>')
    write_svg(path, width, height, "\n".join(body))


def grouped_bar_chart(path: Path, rows: list[dict[str, Any]], title: str, subtitle: str) -> None:
    metrics = [
        ("fast_prototype_rate", "fast"),
        ("long_planning_rate", "long"),
        ("late_or_no_verify_rate", "late/no verify"),
        ("high_churn_repair_rate", "high churn"),
    ]
    rows = sorted(rows, key=lambda r: float(r.get("score_mean") or -1), reverse=True)
    width, top, left, right = 1120, 88, 190, 50
    row_h = 46
    height = top + len(rows) * row_h + 60
    chart_w = width - left - right
    group_w = chart_w / len(metrics)
    colors = ["#0891b2", "#7c3aed", "#f97316", "#dc2626"]
    body = [title_block(title, subtitle)]
    for i, (_, label) in enumerate(metrics):
        x = left + i * group_w + group_w / 2
        body.append(f'<text class="small" x="{x-34:.1f}" y="70">{esc(label)}</text>')
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        body.append(f'<text class="label" x="28" y="{y+24}">{esc(row.get("model"))}</text>')
        for i, (key, _) in enumerate(metrics):
            v = float(row.get(key) or 0)
            x = left + i * group_w
            bar_w = max(1, (group_w - 24) * v)
            body.append(f'<rect x="{x}" y="{y+9}" width="{bar_w:.1f}" height="18" rx="3" fill="{colors[i]}"/>')
            body.append(f'<text class="small" x="{x+bar_w+5:.1f}" y="{y+23}">{v:.2f}</text>')
    write_svg(path, width, height, "\n".join(body))


def heatmap(path: Path, rows: list[dict[str, Any]], title: str, subtitle: str) -> None:
    if not rows:
        return
    models = [k for k in rows[0].keys() if k != "task"]
    tasks = [str(r["task"]) for r in rows]
    width = 170 + len(models) * 96 + 30
    height = 88 + len(tasks) * 34 + 46
    left, top, cell_w, cell_h = 170, 88, 96, 34
    values = [float(r[m]) for r in rows for m in models if isinstance(r.get(m), (int, float))]
    lo, hi = (min(values), max(values)) if values else (0, 1)

    def color(v: Any) -> str:
        if not isinstance(v, (int, float)):
            return "#f1f5f9"
        t = 0 if hi == lo else (float(v) - lo) / (hi - lo)
        # blue -> yellow -> red
        if t < 0.5:
            p = t / 0.5
            r = int(37 + (250 - 37) * p)
            g = int(99 + (204 - 99) * p)
            b = int(235 + (21 - 235) * p)
        else:
            p = (t - 0.5) / 0.5
            r = int(250 + (220 - 250) * p)
            g = int(204 + (38 - 204) * p)
            b = int(21 + (38 - 21) * p)
        return f"#{r:02x}{g:02x}{b:02x}"

    body = [title_block(title, subtitle)]
    for j, model in enumerate(models):
        x = left + j * cell_w + 6
        body.append(f'<text class="small" x="{x}" y="72" transform="rotate(-28 {x} 72)">{esc(model)}</text>')
    for i, row in enumerate(rows):
        y = top + i * cell_h
        body.append(f'<text class="label" x="24" y="{y+22}">{esc(row.get("task"))}</text>')
        for j, model in enumerate(models):
            x = left + j * cell_w
            v = row.get(model)
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{color(v)}"/>')
            body.append(f'<text class="small" x="{x+28}" y="{y+21}">{fmt(v)}</text>')
    write_svg(path, width, height, "\n".join(body))


def scatter(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, subtitle: str) -> None:
    pts = [r for r in rows if isinstance(r.get(x_key), (int, float)) and isinstance(r.get(y_key), (int, float))]
    width, height = 920, 620
    left, right, top, bottom = 82, 36, 76, 70
    chart_w, chart_h = width - left - right, height - top - bottom
    xs = [float(r[x_key]) for r in pts]
    ys = [float(r[y_key]) for r in pts]
    x_min, x_max = (min(xs), max(xs)) if xs else (0, 1)
    y_min, y_max = (min(ys), max(ys)) if ys else (0, 1)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1

    def px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * chart_w

    def py(y: float) -> float:
        return top + chart_h - (y - y_min) / (y_max - y_min) * chart_h

    body = [title_block(title, subtitle)]
    for i in range(6):
        x = left + chart_w * i / 5
        y = top + chart_h * i / 5
        body.append(f'<line class="grid" x1="{x}" y1="{top}" x2="{x}" y2="{top+chart_h}"/>')
        body.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{left+chart_w}" y2="{y}"/>')
        body.append(f'<text class="small" x="{x-18}" y="{top+chart_h+24}">{x_min+(x_max-x_min)*i/5:.1f}</text>')
        body.append(f'<text class="small" x="28" y="{y+4}">{y_max-(y_max-y_min)*i/5:.2f}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}"/>')
    body.append(f'<text class="label" x="{left+chart_w/2-40}" y="{height-22}">{esc(x_key)}</text>')
    body.append(f'<text class="label" x="20" y="{top-12}">{esc(y_key)}</text>')
    for row in pts:
        color = MODEL_COLORS.get(str(row.get("model")), "#334155")
        body.append(f'<circle cx="{px(float(row[x_key])):.1f}" cy="{py(float(row[y_key])):.1f}" r="4.3" fill="{color}" opacity="0.72"><title>{esc(row.get("model"))} | {esc(row.get("task"))} | {fmt(row.get(y_key))}</title></circle>')
    write_svg(path, width, height, "\n".join(body))


def archetype_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = [r for r in rows if isinstance(r.get("score_mean"), (int, float))]
    groups = defaultdict(list)
    for row in rows:
        groups[row["dimension"]].append(row)
    width, top, left = 1080, 78, 250
    row_h = 30
    ordered = []
    for dim in ["start_style", "verify_style", "repair_style", "edit_style"]:
        ordered.extend(sorted(groups[dim], key=lambda r: float(r["score_mean"]), reverse=True))
        ordered.append({"separator": True, "dimension": dim})
    height = top + len(ordered) * row_h + 42
    max_v = max(float(r.get("score_mean") or 0) for r in rows) or 1
    body = [title_block("Workflow Archetypes vs Score", "Mean critical score by workflow category")]
    y = top
    current_dim = None
    for row in ordered:
        if row.get("separator"):
            y += 8
            continue
        if row["dimension"] != current_dim:
            current_dim = row["dimension"]
            body.append(f'<text class="small" x="28" y="{y+18}" font-weight="700">{esc(current_dim)}</text>')
        label = str(row["value"])
        v = float(row["score_mean"])
        w = (width - left - 50) * v / max_v
        body.append(f'<text class="label" x="120" y="{y+18}">{esc(label)}</text>')
        body.append(f'<rect x="{left}" y="{y+4}" width="{w:.1f}" height="18" rx="3" fill="#2563eb"/>')
        body.append(f'<text class="small" x="{left+w+8:.1f}" y="{y+18}">{v:.3f} / n={int(row["n"])}</text>')
        y += row_h
    write_svg(path, width, height, "\n".join(body))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    out = args.results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    eval_dir = args.results_dir / "eval_analysis"
    wf_dir = args.results_dir / "workflow_analysis"
    eval_models = read_csv(eval_dir / "eval_model_summary.csv")
    eval_tasks = read_csv(eval_dir / "eval_task_summary.csv")
    eval_pivot = read_csv(eval_dir / "eval_model_task_pivot.csv")
    wf_models = read_csv(wf_dir / "workflow_model_summary.csv")
    wf_arch = read_csv(wf_dir / "workflow_archetype_summary.csv")
    wf_corr = read_csv(wf_dir / "workflow_correlations.csv")
    wf_features = read_csv(wf_dir / "workflow_features.csv")

    bar_chart(out / "eval_model_scores.svg", eval_models, "model", "score_mean", "Eval Score by Model", "Mean combined critical score", color_key="model")
    bar_chart(out / "eval_task_difficulty.svg", sorted(eval_tasks, key=lambda r: float(r.get("score_mean") or 0)), "task", "score_mean", "Task Difficulty", "Lower bars are harder tasks")
    heatmap(out / "eval_model_task_heatmap.svg", eval_pivot, "Model x Task Score Heatmap", "Mean score per task/model; blank cells mean missing eval")
    grouped_bar_chart(out / "workflow_model_patterns.svg", wf_models, "Workflow Pattern Rates by Model", "Rates of fast start, long planning, late/no verify, high-churn repair")
    archetype_chart(out / "workflow_archetype_scores.svg", wf_arch)
    bar_chart(out / "workflow_correlations.svg", wf_corr[:20], "feature", "spearman", "Strongest Workflow Correlations", "Absolute Spearman correlations from workflow report")
    scatter(out / "workflow_first_write_vs_score.svg", wf_features, "time_to_first_write_s", "score", "First Write Time vs Score", "Each point is one latest run; color indicates model")
    scatter(out / "workflow_fail_count_vs_behavior.svg", wf_features, "fail_count", "avg_behavior_critical", "Failures vs Behavior Score", "More failures generally correlate with lower behavior score")
    scatter(out / "workflow_edit_files_vs_score.svg", wf_features, "edit_file_count", "score", "Files Edited vs Score", "Broader implementation work tends to correlate with score")

    index = [
        "# Figures",
        "",
        "Generated from refreshed `results/eval_analysis` and `results/workflow_analysis`.",
        "",
    ]
    for svg in sorted(out.glob("*.svg")):
        index.append(f"- [{svg.name}]({svg.name})")
    (out / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"Wrote {len(list(out.glob('*.svg')))} SVG figures to {out}")


if __name__ == "__main__":
    main()
