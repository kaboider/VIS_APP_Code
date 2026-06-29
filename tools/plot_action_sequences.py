#!/usr/bin/env python3
"""
Visualize model-specific action order patterns from tasks/_runs* logs.

Outputs:
  - results/workflow_analysis/workflow_actions.csv
  - results/figures/action_sequence_raster.svg
  - results/figures/action_mix_by_progress.svg
  - results/figures/action_transition_heatmaps.svg
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_task_workflows import (  # noqa: E402
    infer_model,
    iter_run_dirs,
    parse_actions,
    parse_run_name,
    read_json,
    task_sort_key,
)


ACTION_ORDER = [
    "inspect",
    "write",
    "deps",
    "docker_up_build",
    "build",
    "server",
    "test_lint",
    "probe",
    "message",
    "thinking",
    "todo",
    "search",
    "error",
    "other_cmd",
    "fs_ops",
    "git",
    "other_tool",
]

COLORS = {
    "inspect": "#64748b",
    "write": "#2563eb",
    "deps": "#7c3aed",
    "docker_up_build": "#0d9488",
    "build": "#16a34a",
    "server": "#22c55e",
    "test_lint": "#84cc16",
    "probe": "#eab308",
    "message": "#cbd5e1",
    "thinking": "#e2e8f0",
    "todo": "#f9a8d4",
    "search": "#a855f7",
    "error": "#dc2626",
    "other_cmd": "#f97316",
    "fs_ops": "#fb923c",
    "git": "#334155",
    "other_tool": "#94a3b8",
    "unknown": "#475569",
}

MODEL_ORDER = [
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "gpt-5.5",
    "Composer 2.5",
    "gpt-5.4-mini",
    "claude-haiku-4-5",
]


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; fill: #0f172a; }}
.title {{ font-size: 20px; font-weight: 700; }}
.subtitle {{ font-size: 12px; fill: #64748b; }}
.label {{ font-size: 11px; fill: #334155; }}
.small {{ font-size: 10px; fill: #64748b; }}
.grid {{ stroke: #e2e8f0; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="#fff"/>
{body}
</svg>
""",
        encoding="utf-8",
    )


def title(title_text: str, subtitle: str = "") -> str:
    sub = f'<text class="subtitle" x="28" y="50">{esc(subtitle)}</text>' if subtitle else ""
    return f'<text class="title" x="28" y="30">{esc(title_text)}</text>{sub}'


def normalize_category(cat: str) -> str:
    if cat.startswith("rate_"):
        return "other_tool"
    if cat in ACTION_ORDER:
        return cat
    return "other_tool"


def model_sort_key(model: str) -> tuple[int, str]:
    return (MODEL_ORDER.index(model) if model in MODEL_ORDER else 999, model)


def collect(tasks_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    actions_out: list[dict[str, Any]] = []
    latest: dict[tuple[str, str], Path] = {}
    for run_dir in iter_run_dirs(tasks_dir, include_legacy=False):
        parts = parse_run_name(run_dir)
        key = (run_dir.parent.name, parts["task"])
        if key not in latest or parts["timestamp"] > parse_run_name(latest[key])["timestamp"]:
            latest[key] = run_dir

    for run_dir in sorted(latest.values(), key=lambda p: (p.parent.name, p.name)):
        experiment = run_dir.parent.name
        parts = parse_run_name(run_dir)
        summary = read_json(run_dir / "logs" / "summary.json").get("summary") or {}
        model = infer_model(experiment, summary.get("model"))
        actions, source = parse_actions(run_dir, experiment)
        usable = [a for a in actions if normalize_category(a.category) not in {"message", "thinking", "todo"}]
        run_rec = {
            "experiment": experiment,
            "run_id": run_dir.name,
            "timestamp": parts["timestamp"],
            "task": parts["task"],
            "model": model,
            "source": source,
            "action_count": len(usable),
        }
        runs.append({**run_rec, "actions": usable})
        for idx, action in enumerate(usable):
            actions_out.append(
                {
                    **run_rec,
                    "seq": idx + 1,
                    "elapsed_s": action.elapsed_s,
                    "category": normalize_category(action.category),
                    "raw_category": action.category,
                    "is_write": action.is_write,
                    "is_verify": action.is_verify,
                    "is_failure": action.is_failure,
                    "detail": action.detail,
                }
            )
    return runs, actions_out


def write_actions_csv(path: Path, actions: list[dict[str, Any]]) -> None:
    keys = [
        "model",
        "experiment",
        "task",
        "run_id",
        "timestamp",
        "source",
        "action_count",
        "seq",
        "elapsed_s",
        "category",
        "raw_category",
        "is_write",
        "is_verify",
        "is_failure",
        "detail",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(actions)


def legend(x: int, y: int, cats: list[str], columns: int = 6) -> str:
    body = []
    cell_w = 142
    for i, cat in enumerate(cats):
        cx = x + (i % columns) * cell_w
        cy = y + (i // columns) * 20
        body.append(f'<rect x="{cx}" y="{cy-10}" width="11" height="11" fill="{COLORS.get(cat, COLORS["unknown"])}"/>')
        body.append(f'<text class="small" x="{cx+16}" y="{cy}">{esc(cat)}</text>')
    return "\n".join(body)


def plot_raster(path: Path, runs: list[dict[str, Any]]) -> None:
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["model"]].append(run)
    for model in grouped:
        grouped[model].sort(key=lambda r: task_sort_key(r["task"]))
    models = sorted(grouped, key=model_sort_key)
    width = 1280
    left, top, row_h, group_gap = 260, 96, 13, 26
    chart_w = 940
    height = top + sum(len(grouped[m]) * row_h + group_gap for m in models) + 94
    body = [title("Action Sequence Raster by Model", "Each row is one latest run; x-axis is normalized action order after removing message/thinking/todo.")]
    body.append(legend(28, 70, ACTION_ORDER[:14], columns=7))
    y = top
    for model in models:
        body.append(f'<text class="label" x="28" y="{y+10}" font-weight="700">{esc(model)}</text>')
        for run in grouped[model]:
            actions = run["actions"]
            body.append(f'<text class="small" x="138" y="{y+10}">{esc(run["task"])}</text>')
            body.append(f'<line class="grid" x1="{left}" y1="{y+5}" x2="{left+chart_w}" y2="{y+5}"/>')
            n = max(1, len(actions) - 1)
            for idx, action in enumerate(actions):
                cat = normalize_category(action.category)
                x = left + idx / n * chart_w
                body.append(f'<rect x="{x:.1f}" y="{y}" width="3.2" height="10" fill="{COLORS.get(cat, COLORS["unknown"])}"><title>{esc(model)} | {esc(run["task"])} | {idx+1}: {esc(cat)} {esc(action.detail)}</title></rect>')
            y += row_h
        y += group_gap
    write_svg(path, width, height, "\n".join(body))


def plot_realtime_raster(path: Path, runs: list[dict[str, Any]]) -> None:
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["model"]].append(run)
    for model in grouped:
        grouped[model].sort(key=lambda r: task_sort_key(r["task"]))
    models = sorted(grouped, key=model_sort_key)
    run_maxes: list[float] = []
    for run in runs:
        run_max = 0.0
        for action in run["actions"]:
            if action.elapsed_s is not None:
                run_max = max(run_max, float(action.elapsed_s))
        if run_max:
            run_maxes.append(run_max)

    def percentile(values: list[float], pct: float) -> float:
        values = sorted(values)
        if not values:
            return 1.0
        if len(values) == 1:
            return values[0]
        k = (len(values) - 1) * pct / 100
        lo = int(k)
        hi = min(lo + 1, len(values) - 1)
        if lo == hi:
            return values[lo]
        return values[lo] * (hi - k) + values[hi] * (k - lo)

    raw_max_elapsed = max(run_maxes) if run_maxes else 1.0
    max_elapsed = max(1.0, percentile(run_maxes, 95))

    width = 1280
    left, top, row_h, group_gap = 260, 112, 13, 26
    chart_w = 940
    height = top + sum(len(grouped[m]) * row_h + group_gap for m in models) + 112
    body = [
        title(
            "Action Sequence Raster by Real Time",
            f"x-axis is elapsed seconds, clipped at run-duration P95={max_elapsed:.0f}s; red ticks mark runs/actions beyond the cap. Raw max={raw_max_elapsed:.0f}s.",
        )
    ]
    body.append(legend(28, 70, ACTION_ORDER[:14], columns=7))
    # Time grid.
    grid_top = top - 12
    grid_bottom = height - 76
    for i in range(7):
        t = max_elapsed * i / 6
        x = left + chart_w * i / 6
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{grid_top}" x2="{x:.1f}" y2="{grid_bottom}"/>')
        body.append(f'<text class="small" x="{x-16:.1f}" y="{height-50}">{t:.0f}s</text>')

    y = top
    for model in models:
        body.append(f'<text class="label" x="28" y="{y+10}" font-weight="700">{esc(model)}</text>')
        for run in grouped[model]:
            actions = run["actions"]
            run_max = max((float(a.elapsed_s) for a in actions if a.elapsed_s is not None), default=0.0)
            clipped_run_max = min(run_max, max_elapsed)
            run_end_x = left + chart_w * clipped_run_max / max_elapsed
            body.append(f'<text class="small" x="138" y="{y+10}">{esc(run["task"])}</text>')
            body.append(f'<line class="grid" x1="{left}" y1="{y+5}" x2="{run_end_x:.1f}" y2="{y+5}"/>')
            if run_max > max_elapsed:
                body.append(f'<line x1="{left+chart_w+5}" y1="{y-1}" x2="{left+chart_w+5}" y2="{y+11}" stroke="#dc2626" stroke-width="2"><title>{esc(model)} | {esc(run["task"])} exceeds cap: {run_max:.1f}s</title></line>')
            for idx, action in enumerate(actions):
                if action.elapsed_s is None:
                    continue
                cat = normalize_category(action.category)
                elapsed = float(action.elapsed_s)
                x = left + min(elapsed, max_elapsed) / max_elapsed * chart_w
                opacity = "0.55" if elapsed > max_elapsed else "1"
                body.append(f'<rect x="{x:.1f}" y="{y}" width="3.2" height="10" fill="{COLORS.get(cat, COLORS["unknown"])}"><title>{esc(model)} | {esc(run["task"])} | t={float(action.elapsed_s):.1f}s | {idx+1}: {esc(cat)} {esc(action.detail)}</title></rect>')
                if elapsed > max_elapsed:
                    body[-1] = body[-1].replace("<rect ", f'<rect opacity="{opacity}" ')
            y += row_h
        y += group_gap
    write_svg(path, width, height, "\n".join(body))


def plot_mix(path: Path, runs: list[dict[str, Any]]) -> None:
    models = sorted({r["model"] for r in runs}, key=model_sort_key)
    cats = ["inspect", "write", "deps", "docker_up_build", "build", "server", "test_lint", "probe", "error", "other_cmd", "fs_ops", "git", "search", "other_tool"]
    deciles = 10
    width = 1320
    left, top = 160, 92
    panel_h, panel_gap = 104, 32
    chart_w = 1040
    height = top + len(models) * (panel_h + panel_gap) + 96
    body = [title("Action Mix Over Normalized Progress", "Stacked proportions per model and progress decile; narration actions removed.")]
    body.append(legend(28, 70, cats, columns=7))
    for mi, model in enumerate(models):
        y0 = top + mi * (panel_h + panel_gap)
        body.append(f'<text class="label" x="28" y="{y0+48}" font-weight="700">{esc(model)}</text>')
        counts = [[Counter() for _ in range(deciles)]][0]
        for run in runs:
            if run["model"] != model:
                continue
            actions = run["actions"]
            if not actions:
                continue
            n = max(1, len(actions) - 1)
            for idx, action in enumerate(actions):
                bucket = min(deciles - 1, int((idx / n) * deciles))
                counts[bucket][normalize_category(action.category)] += 1
        for b in range(deciles):
            total = sum(counts[b].values()) or 1
            x = left + b * (chart_w / deciles)
            y = y0
            for cat in cats:
                h = panel_h * counts[b][cat] / total
                if h > 0:
                    y2 = y0 + panel_h - (y + h - y0)
                    body.append(f'<rect x="{x}" y="{y2:.1f}" width="{chart_w/deciles-3:.1f}" height="{h:.1f}" fill="{COLORS.get(cat, COLORS["unknown"])}"/>')
                y += h
            body.append(f'<text class="small" x="{x+26:.1f}" y="{y0+panel_h+16}">{b+1}</text>')
        body.append(f'<line class="grid" x1="{left}" y1="{y0+panel_h}" x2="{left+chart_w}" y2="{y0+panel_h}"/>')
    write_svg(path, width, height, "\n".join(body))


def plot_transitions(path: Path, runs: list[dict[str, Any]]) -> None:
    cats = ["inspect", "write", "deps", "docker_up_build", "build", "server", "test_lint", "probe", "error", "other_cmd", "fs_ops", "git", "search", "other_tool"]
    models = sorted({r["model"] for r in runs}, key=model_sort_key)
    cell = 19
    panel_w = 330
    panel_h = 344
    width = 2 * panel_w + 80
    height = 92 + ((len(models) + 1) // 2) * panel_h + 54
    body = [title("Action Transition Heatmaps by Model", "Rows are current action, columns are next action; color is row-normalized transition probability.")]

    def heat(v: float) -> str:
        # white -> blue
        t = max(0.0, min(1.0, v))
        r = int(239 + (37 - 239) * t)
        g = int(246 + (99 - 246) * t)
        b = int(255 + (235 - 255) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    for mi, model in enumerate(models):
        col = mi % 2
        row = mi // 2
        x0 = 42 + col * panel_w
        y0 = 92 + row * panel_h
        trans = {cat: Counter() for cat in cats}
        for run in runs:
            if run["model"] != model:
                continue
            seq = [normalize_category(a.category) for a in run["actions"]]
            seq = [s for s in seq if s in cats]
            for a, b in zip(seq, seq[1:]):
                trans[a][b] += 1
        body.append(f'<text class="label" x="{x0}" y="{y0-16}" font-weight="700">{esc(model)}</text>')
        for j, cat in enumerate(cats):
            body.append(f'<text class="small" x="{x0+88+j*cell}" y="{y0-4}" transform="rotate(-45 {x0+88+j*cell} {y0-4})">{esc(cat[:7])}</text>')
            body.append(f'<text class="small" x="{x0}" y="{y0+18+j*cell}">{esc(cat[:12])}</text>')
        for i, src in enumerate(cats):
            total = sum(trans[src].values()) or 1
            for j, dst in enumerate(cats):
                v = trans[src][dst] / total
                x = x0 + 92 + j * cell
                y = y0 + 4 + i * cell
                body.append(f'<rect x="{x}" y="{y}" width="{cell-1}" height="{cell-1}" fill="{heat(v)}"><title>{esc(model)}: {esc(src)} → {esc(dst)} = {v:.2f} ({trans[src][dst]})</title></rect>')
    write_svg(path, width, height, "\n".join(body))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    fig_dir = args.results_dir / "figures"
    wf_dir = args.results_dir / "workflow_analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)
    wf_dir.mkdir(parents=True, exist_ok=True)

    runs, actions = collect(args.tasks_dir)
    write_actions_csv(wf_dir / "workflow_actions.csv", actions)
    plot_raster(fig_dir / "action_sequence_raster.svg", runs)
    plot_realtime_raster(fig_dir / "action_sequence_realtime.svg", runs)
    plot_mix(fig_dir / "action_mix_by_progress.svg", runs)
    plot_transitions(fig_dir / "action_transition_heatmaps.svg", runs)

    index = fig_dir / "index.md"
    existing = index.read_text(encoding="utf-8") if index.exists() else "# Figures\n\n"
    additions = [
        "- [action_sequence_raster.svg](action_sequence_raster.svg)",
        "- [action_sequence_realtime.svg](action_sequence_realtime.svg)",
        "- [action_mix_by_progress.svg](action_mix_by_progress.svg)",
        "- [action_transition_heatmaps.svg](action_transition_heatmaps.svg)",
    ]
    for line in additions:
        if line not in existing:
            existing += line + "\n"
    index.write_text(existing, encoding="utf-8")
    print(f"Wrote {len(actions)} action rows and 4 action-order figures.")


if __name__ == "__main__":
    main()
