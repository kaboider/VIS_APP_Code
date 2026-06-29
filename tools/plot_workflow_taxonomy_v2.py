#!/usr/bin/env python3
"""Plot taxonomy-v2 workflow, error, planning, and tool figures."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MAIN_ORDER = [
    "inspect",
    "planning",
    "write",
    "setup",
    "run",
    "verify",
    "probe",
    "repair_failure",
    "search_delegate",
    "misc",
]

COLORS = {
    "inspect": "#64748b",
    "planning": "#cbd5e1",
    "write": "#2563eb",
    "setup": "#7c3aed",
    "run": "#0d9488",
    "verify": "#16a34a",
    "probe": "#eab308",
    "repair_failure": "#dc2626",
    "search_delegate": "#a855f7",
    "misc": "#f97316",
    "message": "#94a3b8",
    "thinking": "#e2e8f0",
    "todo": "#f9a8d4",
}

ERROR_COLORS = {
    "tool_error": "#dc2626",
    "syntax_runtime_error": "#ea580c",
    "build_error": "#ca8a04",
    "test_lint_error": "#16a34a",
    "dependency_error": "#7c3aed",
    "docker_error": "#0d9488",
    "server_error": "#0284c7",
    "probe_error": "#eab308",
    "permission_or_path_error": "#be123c",
    "unknown_error": "#64748b",
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


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["seq"] = int(row.get("seq") or 0)
        row["elapsed_s"] = as_float(row.get("elapsed_s"))
        row["is_error"] = str(row.get("is_error")).lower() == "true"
        for key in ["todo_pending", "todo_active", "todo_done", "text_len"]:
            row[key] = int(float(row.get(key) or 0))
    return rows


def model_key(model: str) -> tuple[int, str]:
    return (MODEL_ORDER.index(model) if model in MODEL_ORDER else 999, model)


def run_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["model"], row["task"], row["run_id"])


def sort_runs(run_ids: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    return sorted(run_ids, key=lambda r: (model_key(r[0]), r[1], r[2]))


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; fill: #0f172a; }}
.title {{ font-size: 20px; font-weight: 700; }}
.subtitle {{ font-size: 12px; fill: #64748b; }}
.label {{ font-size: 11px; fill: #334155; }}
.small {{ font-size: 10px; fill: #64748b; }}
.grid {{ stroke: #e2e8f0; stroke-width: 1; }}
.axis {{ stroke: #94a3b8; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="#fff"/>
{body}
</svg>
""",
        encoding="utf-8",
    )


def title(text: str, subtitle: str = "") -> str:
    sub = f'<text class="subtitle" x="28" y="50">{esc(subtitle)}</text>' if subtitle else ""
    return f'<text class="title" x="28" y="30">{esc(text)}</text>{sub}'


def legend(x: int, y: int, cats: list[str], colors: dict[str, str], columns: int = 5) -> str:
    out = []
    cell_w = 170
    for i, cat in enumerate(cats):
        cx = x + (i % columns) * cell_w
        cy = y + (i // columns) * 20
        out.append(f'<rect x="{cx}" y="{cy-10}" width="11" height="11" fill="{colors.get(cat, "#64748b")}"/>')
        out.append(f'<text class="small" x="{cx+16}" y="{cy}">{esc(cat)}</text>')
    return "\n".join(out)


def plot_main_raster(path: Path, rows: list[dict[str, Any]]) -> None:
    by_run: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[run_key(row)].append(row)
    for actions in by_run.values():
        actions.sort(key=lambda r: r["seq"])
    run_ids = sort_runs(list(by_run))
    width = 1320
    left, top, row_h, chart_w = 278, 112, 12, 982
    height = top + row_h * len(run_ids) + 96
    body = [
        title(
            "Taxonomy V2 Main Workflow Sequence",
            "Each row is a latest run; x-axis is normalized action order. Planning is retained as a compact main class.",
        ),
        legend(28, 72, MAIN_ORDER, COLORS, columns=5),
    ]
    last_model = ""
    y = top
    for rid in run_ids:
        model, task, _ = rid
        if model != last_model:
            body.append(f'<text class="label" x="28" y="{y+8}" font-weight="700">{esc(model)}</text>')
            last_model = model
        body.append(f'<text class="small" x="145" y="{y+8}">{esc(task)}</text>')
        body.append(f'<line class="grid" x1="{left}" y1="{y+4}" x2="{left+chart_w}" y2="{y+4}"/>')
        actions = by_run[rid]
        denom = max(1, len(actions) - 1)
        for idx, action in enumerate(actions):
            cat = action["action_main"]
            x = left + idx / denom * chart_w
            body.append(
                f'<rect x="{x:.1f}" y="{y}" width="3" height="9" fill="{COLORS.get(cat, "#64748b")}">'
                f'<title>{esc(model)} | {esc(task)} | {idx+1}: {esc(cat)} / {esc(action["action_subtype"])}</title></rect>'
            )
        y += row_h
    write_svg(path, width, height, "\n".join(body))


def plot_mix_by_progress(path: Path, rows: list[dict[str, Any]]) -> None:
    by_run: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[run_key(row)].append(row)
    counts: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for rid, actions in by_run.items():
        model = rid[0]
        actions.sort(key=lambda r: r["seq"])
        n = len(actions)
        for idx, action in enumerate(actions):
            decile = min(9, int(idx / max(1, n) * 10))
            counts[(model, decile)][action["action_main"]] += 1
    models = sorted({r["model"] for r in rows}, key=model_key)
    width, height = 1220, 120 + len(models) * 82
    left, top, bar_w, gap = 190, 96, 76, 18
    body = [
        title("Action Mix by Progress Decile", "Stacked bars show action_main composition at each normalized phase of the run."),
        legend(28, 70, MAIN_ORDER, COLORS, columns=5),
    ]
    for i in range(10):
        x = left + i * (bar_w + gap)
        body.append(f'<text class="small" x="{x+24}" y="{top-16}">{i+1}</text>')
    for row_idx, model in enumerate(models):
        y = top + row_idx * 82
        body.append(f'<text class="label" x="28" y="{y+44}" font-weight="700">{esc(model)}</text>')
        for d in range(10):
            x = left + d * (bar_w + gap)
            total = sum(counts[(model, d)].values()) or 1
            yy = y + 54
            for cat in MAIN_ORDER:
                h = 54 * counts[(model, d)][cat] / total
                if h <= 0:
                    continue
                yy -= h
                body.append(f'<rect x="{x}" y="{yy:.1f}" width="{bar_w}" height="{h:.1f}" fill="{COLORS[cat]}"><title>{esc(model)} decile {d+1}: {esc(cat)} {counts[(model, d)][cat]}/{total}</title></rect>')
            body.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="54" fill="none" stroke="#e2e8f0"/>')
    write_svg(path, width, height, "\n".join(body))


def plot_transition_heatmaps(path: Path, rows: list[dict[str, Any]]) -> None:
    by_run: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[run_key(row)].append(row)
    models = sorted({r["model"] for r in rows}, key=model_key)
    counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for rid, actions in by_run.items():
        actions.sort(key=lambda r: r["seq"])
        for a, b in zip(actions, actions[1:]):
            counts[rid[0]][(a["action_main"], b["action_main"])] += 1
    cell, panel_w, panel_h = 18, 260, 250
    width = 1120
    height = 94 + math.ceil(len(models) / 4) * panel_h
    body = [title("Action Main Transition Heatmaps", "Rows are previous action; columns are next action. Darker means more frequent within model.")]
    for mi, model in enumerate(models):
        px = 28 + (mi % 4) * panel_w
        py = 76 + (mi // 4) * panel_h
        body.append(f'<text class="label" x="{px}" y="{py}" font-weight="700">{esc(model)}</text>')
        max_v = max(counts[model].values() or [1])
        for i, prev in enumerate(MAIN_ORDER):
            body.append(f'<text class="small" x="{px}" y="{py+32+i*cell}">{esc(prev[:12])}</text>')
            for j, nxt in enumerate(MAIN_ORDER):
                v = counts[model][(prev, nxt)]
                alpha = 0.08 + 0.88 * (v / max_v if max_v else 0)
                x = px + 92 + j * cell
                y = py + 20 + i * cell
                body.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="#2563eb" fill-opacity="{alpha:.2f}"><title>{esc(model)}: {esc(prev)} -> {esc(nxt)} = {v}</title></rect>')
        for j, cat in enumerate(MAIN_ORDER):
            body.append(f'<text class="small" transform="translate({px+102+j*cell},{py+202}) rotate(65)">{esc(cat[:12])}</text>')
    write_svg(path, width, height, "\n".join(body))


def plot_error_by_model(path: Path, rows: list[dict[str, Any]]) -> None:
    errors = [r for r in rows if r["is_error"]]
    models = sorted({r["model"] for r in rows}, key=model_key)
    types = sorted({r["error_type"] for r in errors if r["error_type"]}, key=lambda t: -sum(1 for r in errors if r["error_type"] == t))
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in errors:
        counts[row["model"]][row["error_type"]] += 1
    width, height = 1180, 130 + len(models) * 46
    left, top, bar_w = 190, 92, 820
    max_total = max((sum(counts[m].values()) for m in models), default=1)
    body = [title("Error Type by Model", "Only failed tool/command events. Claude tool_result errors remain separated as tool_error.")]
    body.append(legend(28, 70, types, ERROR_COLORS, columns=5))
    y = top
    for model in models:
        total = sum(counts[model].values())
        body.append(f'<text class="label" x="28" y="{y+16}" font-weight="700">{esc(model)}</text>')
        x = left
        for typ in types:
            w = bar_w * counts[model][typ] / max_total
            if w:
                body.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="24" fill="{ERROR_COLORS.get(typ, "#64748b")}"><title>{esc(model)} {esc(typ)}: {counts[model][typ]}</title></rect>')
            x += w
        body.append(f'<text class="small" x="{left+bar_w+12}" y="{y+16}">{total}</text>')
        y += 46
    write_svg(path, width, height, "\n".join(body))


def plot_error_recovery(path: Path, rows: list[dict[str, Any]]) -> None:
    by_run: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[run_key(row)].append(row)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for rid, actions in by_run.items():
        actions.sort(key=lambda r: r["seq"])
        for idx, row in enumerate(actions[:-1]):
            if row["is_error"]:
                counts[rid[0]][actions[idx + 1]["action_main"]] += 1
    models = sorted(counts, key=model_key)
    width, height = 1120, 112 + len(models) * 48
    left, top, bar_w = 190, 82, 780
    max_total = max((sum(counts[m].values()) for m in models), default=1)
    body = [
        title("Post-Error Next Action", "What the agent does immediately after an error event."),
        legend(28, 60, MAIN_ORDER, COLORS, columns=5),
    ]
    for i, model in enumerate(models):
        y = top + i * 48
        total = sum(counts[model].values())
        body.append(f'<text class="label" x="28" y="{y+16}" font-weight="700">{esc(model)}</text>')
        x = left
        for cat in MAIN_ORDER:
            w = bar_w * counts[model][cat] / max_total
            if w:
                body.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="24" fill="{COLORS[cat]}"><title>{esc(model)} after error: {esc(cat)} {counts[model][cat]}</title></rect>')
            x += w
        body.append(f'<text class="small" x="{left+bar_w+12}" y="{y+16}">{total}</text>')
    write_svg(path, width, height, "\n".join(body))


def plot_error_timeline(path: Path, rows: list[dict[str, Any]]) -> None:
    by_run: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[run_key(row)].append(row)
    run_ids = sort_runs(list(by_run))
    width = 1280
    left, top, row_h, chart_w = 276, 76, 12, 930
    height = top + row_h * len(run_ids) + 48
    body = [title("Error Timeline Raster", "x-axis is normalized sequence position; marks show error_type only.")]
    y = top
    last_model = ""
    for rid in run_ids:
        model, task, _ = rid
        if model != last_model:
            body.append(f'<text class="label" x="28" y="{y+8}" font-weight="700">{esc(model)}</text>')
            last_model = model
        body.append(f'<text class="small" x="145" y="{y+8}">{esc(task)}</text>')
        actions = sorted(by_run[rid], key=lambda r: r["seq"])
        denom = max(1, len(actions) - 1)
        for idx, action in enumerate(actions):
            if not action["is_error"]:
                continue
            typ = action["error_type"] or "unknown_error"
            x = left + idx / denom * chart_w
            body.append(f'<circle cx="{x:.1f}" cy="{y+4}" r="4" fill="{ERROR_COLORS.get(typ, "#64748b")}"><title>{esc(model)} | {esc(task)} | {esc(typ)} | seq {idx+1}</title></circle>')
        y += row_h
    write_svg(path, width, height, "\n".join(body))


def plot_planning_by_model(path: Path, rows: list[dict[str, Any]]) -> None:
    planning = [r for r in rows if r["action_main"] == "planning"]
    models = sorted({r["model"] for r in rows}, key=model_key)
    kinds = ["message", "thinking", "todo"]
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in planning:
        sub = row["action_subtype"] if row["action_subtype"] in kinds else "message"
        counts[row["model"]][sub] += 1
    width, height = 1050, 112 + len(models) * 46
    left, top, bar_w = 190, 82, 720
    max_total = max((sum(counts[m].values()) for m in models), default=1)
    body = [title("Planning / Message / Thinking / Todo by Model", "Planning is split here so main workflow figures stay readable.")]
    body.append(legend(28, 60, kinds, COLORS, columns=3))
    for i, model in enumerate(models):
        y = top + i * 46
        x = left
        total = sum(counts[model].values())
        body.append(f'<text class="label" x="28" y="{y+16}" font-weight="700">{esc(model)}</text>')
        for kind in kinds:
            w = bar_w * counts[model][kind] / max_total
            if w:
                body.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="24" fill="{COLORS[kind]}"><title>{esc(model)} {esc(kind)}: {counts[model][kind]}</title></rect>')
            x += w
        body.append(f'<text class="small" x="{left+bar_w+12}" y="{y+16}">{total}</text>')
    write_svg(path, width, height, "\n".join(body))


def plot_planning_timeline(path: Path, rows: list[dict[str, Any]]) -> None:
    planning = [r for r in rows if r["action_main"] == "planning"]
    by_run: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    all_rows_by_run: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        all_rows_by_run[run_key(row)] += 1
    for row in planning:
        by_run[run_key(row)].append(row)
    run_ids = sort_runs(list(all_rows_by_run))
    width = 1280
    left, top, row_h, chart_w = 276, 94, 12, 930
    height = top + row_h * len(run_ids) + 56
    body = [
        title("Planning Timeline", "message/thinking/todo positions in each run; x-axis is normalized full action sequence."),
        legend(28, 66, ["message", "thinking", "todo"], COLORS, columns=3),
    ]
    y = top
    last_model = ""
    for rid in run_ids:
        model, task, _ = rid
        if model != last_model:
            body.append(f'<text class="label" x="28" y="{y+8}" font-weight="700">{esc(model)}</text>')
            last_model = model
        body.append(f'<text class="small" x="145" y="{y+8}">{esc(task)}</text>')
        denom = max(1, all_rows_by_run[rid] - 1)
        for row in by_run.get(rid, []):
            sub = row["action_subtype"] if row["action_subtype"] in {"message", "thinking", "todo"} else "message"
            x = left + (row["seq"] - 1) / denom * chart_w
            body.append(f'<rect x="{x:.1f}" y="{y}" width="3" height="9" fill="{COLORS[sub]}"><title>{esc(model)} | {esc(task)} | {esc(sub)} | seq {row["seq"]}</title></rect>')
        y += row_h
    write_svg(path, width, height, "\n".join(body))


def plot_todo_updates(path: Path, rows: list[dict[str, Any]]) -> None:
    todos = [r for r in rows if r["action_subtype"] == "todo"]
    models = sorted({r["model"] for r in rows}, key=model_key)
    counts = Counter(r["model"] for r in todos)
    width, height = 980, 110 + len(models) * 42
    left, top, bar_w = 190, 76, 680
    max_v = max(counts.values() or [1])
    body = [title("Todo Update Count by Model", "Counts TodoWrite, todo_list, and updateTodosToolCall events.")]
    for i, model in enumerate(models):
        y = top + i * 42
        w = bar_w * counts[model] / max_v
        body.append(f'<text class="label" x="28" y="{y+16}" font-weight="700">{esc(model)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="24" fill="{COLORS["todo"]}"><title>{esc(model)} todo updates: {counts[model]}</title></rect>')
        body.append(f'<text class="small" x="{left+w+10:.1f}" y="{y+16}">{counts[model]}</text>')
    write_svg(path, width, height, "\n".join(body))


def plot_tool_heatmap(path: Path, rows: list[dict[str, Any]], key: str, title_text: str, subtitle: str, top_n: int) -> None:
    models = sorted({r["model"] for r in rows}, key=model_key)
    totals = Counter(r[key] for r in rows if r.get(key))
    cats = [c for c, _ in totals.most_common(top_n)]
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        val = row.get(key)
        if val in cats:
            counts[row["model"]][val] += 1
    cell_w, cell_h = 62, 30
    left, top = 220, 92
    width = left + cell_w * len(cats) + 80
    height = top + cell_h * len(models) + 170
    max_v = max((counts[m][c] for m in models for c in cats), default=1)
    body = [title(title_text, subtitle)]
    for j, cat in enumerate(cats):
        body.append(f'<text class="small" transform="translate({left+j*cell_w+16},{top-10}) rotate(-50)">{esc(cat[:24])}</text>')
    for i, model in enumerate(models):
        y = top + i * cell_h
        body.append(f'<text class="label" x="28" y="{y+19}" font-weight="700">{esc(model)}</text>')
        for j, cat in enumerate(cats):
            v = counts[model][cat]
            alpha = 0.06 + 0.9 * (v / max_v if max_v else 0)
            x = left + j * cell_w
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" fill="#2563eb" fill-opacity="{alpha:.2f}"><title>{esc(model)} {esc(cat)}: {v}</title></rect>')
            if v:
                body.append(f'<text class="small" x="{x+6}" y="{y+18}">{v}</text>')
    write_svg(path, width, height, "\n".join(body))


def update_index(figures_dir: Path) -> None:
    existing = figures_dir / "index.md"
    text = existing.read_text(encoding="utf-8") if existing.exists() else "# Figures\n"
    block = """\n## Taxonomy V2 Workflow Figures\n\n- [main_workflow/action_main_sequence_raster.svg](main_workflow/action_main_sequence_raster.svg)\n- [main_workflow/action_main_mix_by_progress.svg](main_workflow/action_main_mix_by_progress.svg)\n- [main_workflow/action_main_transition_heatmaps.svg](main_workflow/action_main_transition_heatmaps.svg)\n- [errors/error_type_by_model.svg](errors/error_type_by_model.svg)\n- [errors/error_timeline_raster.svg](errors/error_timeline_raster.svg)\n- [errors/error_recovery_next_action.svg](errors/error_recovery_next_action.svg)\n- [planning/planning_type_by_model.svg](planning/planning_type_by_model.svg)\n- [planning/planning_timeline.svg](planning/planning_timeline.svg)\n- [planning/todo_updates_by_model.svg](planning/todo_updates_by_model.svg)\n- [tools/tool_kind_by_model.svg](tools/tool_kind_by_model.svg)\n- [tools/command_prefix_by_model.svg](tools/command_prefix_by_model.svg)\n"""
    marker = "## Taxonomy V2 Workflow Figures"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n" + block
    else:
        text = text.rstrip() + "\n" + block
    existing.write_text(text, encoding="utf-8")


def update_readme(results_dir: Path) -> None:
    path = results_dir / "README.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Current Results\n"
    additions = [
        "python3 tasks/tools/analyze_workflow_taxonomy_v2.py --tasks-dir tasks --out-dir results/workflow_analysis",
        "python3 tasks/tools/plot_workflow_taxonomy_v2.py --results-dir results",
    ]
    for cmd in additions:
        if cmd not in text:
            text = text.replace("python3 tasks/tools/plot_action_sequences.py --tasks-dir tasks --results-dir results", "python3 tasks/tools/plot_action_sequences.py --tasks-dir tasks --results-dir results\n" + cmd)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    analysis_dir = args.results_dir / "workflow_analysis"
    figures_dir = args.results_dir / "figures"
    rows = read_rows(analysis_dir / "workflow_actions_v2.csv")

    plot_main_raster(figures_dir / "main_workflow" / "action_main_sequence_raster.svg", rows)
    plot_mix_by_progress(figures_dir / "main_workflow" / "action_main_mix_by_progress.svg", rows)
    plot_transition_heatmaps(figures_dir / "main_workflow" / "action_main_transition_heatmaps.svg", rows)
    plot_error_by_model(figures_dir / "errors" / "error_type_by_model.svg", rows)
    plot_error_timeline(figures_dir / "errors" / "error_timeline_raster.svg", rows)
    plot_error_recovery(figures_dir / "errors" / "error_recovery_next_action.svg", rows)
    plot_planning_by_model(figures_dir / "planning" / "planning_type_by_model.svg", rows)
    plot_planning_timeline(figures_dir / "planning" / "planning_timeline.svg", rows)
    plot_todo_updates(figures_dir / "planning" / "todo_updates_by_model.svg", rows)
    plot_tool_heatmap(figures_dir / "tools" / "tool_kind_by_model.svg", rows, "raw_tool_kind", "Raw Tool Kind by Model", "Top raw tool/event kinds, separated from compact action_main taxonomy.", 20)
    command_rows = [r for r in rows if r.get("command_prefix")]
    plot_tool_heatmap(figures_dir / "tools" / "command_prefix_by_model.svg", command_rows, "command_prefix", "Command Prefix by Model", "Top shell command prefixes from Bash/command_execution/shellToolCall events.", 20)
    update_index(figures_dir)
    update_readme(args.results_dir)
    print("Wrote taxonomy-v2 figures.")


if __name__ == "__main__":
    main()
