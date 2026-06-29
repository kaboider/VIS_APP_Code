#!/usr/bin/env python3
"""
Workflow analysis for the current tasks/_runs* benchmark outputs.

This is the c4-era successor to results/scripts/analyze_workflows.py. It reads
new run directories directly from tasks/_runs* and writes a fresh results tree
focused on agent workflow patterns:

  - results/workflow_analysis/workflow_features.csv
  - results/workflow_analysis/workflow_model_summary.csv
  - results/workflow_analysis/workflow_experiment_summary.csv
  - results/workflow_analysis/workflow_task_summary.csv
  - results/workflow_analysis/workflow_archetype_summary.csv
  - results/workflow_analysis/workflow_correlations.csv
  - results/workflow_analysis/workflow_report.md

No external packages are required.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RUN_RE = re.compile(r"^(\d{8}_\d{6})_(.+)_c(\d+)(?:_|$)")
VERIFY_CATEGORIES = {"deps", "docker_up_build", "build", "server", "test_lint", "probe"}
INSPECT_TOOLS = {"Read", "Grep", "Glob", "readToolCall", "globToolCall", "grepToolCall"}
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "editToolCall"}


@dataclass
class Action:
    elapsed_s: float | None
    category: str
    is_write: bool = False
    is_verify: bool = False
    is_failure: bool = False
    batch_files: int = 0
    detail: str = ""


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_num(value: Any) -> float:
    return safe_float(value) or 0.0


def mean(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return statistics.mean(vals) if vals else None


def median(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return statistics.median(vals) if vals else None


def stdev(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0 if vals else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer() and abs(value) >= 1:
        return str(int(value))
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def rank(values: list[float]) -> list[float]:
    indexed = sorted((value, idx) for idx, value in enumerate(values))
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][0] == indexed[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[indexed[k][1]] = avg
        i = j + 1
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 6 or len(xs) != len(ys):
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if not sx or not sy:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def spearman(rows: list[dict[str, Any]], feature: str, target: str) -> float | None:
    pairs = []
    for row in rows:
        x = row.get(feature)
        y = row.get(target)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            pairs.append((float(x), float(y)))
    if len(pairs) < 6:
        return None
    xs, ys = zip(*pairs)
    return pearson(rank(list(xs)), rank(list(ys)))


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def classify_command(command: str | None) -> str:
    c = (command or "").lower()
    if any(t in c for t in ["npm install", "pnpm install", "yarn install", "npm i ", "pip install", "apt-get", "apt install"]):
        return "deps"
    if any(t in c for t in ["docker compose up", "docker-compose up", "docker compose build", "docker-compose build"]):
        return "docker_up_build"
    if any(t in c for t in ["docker compose down", "docker-compose down"]):
        return "docker_down"
    if any(t in c for t in ["npm run build", "pnpm build", "yarn build", "next build", "vite build", "tsc --noemit"]):
        return "build"
    if any(t in c for t in ["npm run dev", "npm start", "node server", "uvicorn", "vite --host", "next dev", "astro dev"]):
        return "server"
    if any(t in c for t in ["npm test", "npm run test", "pytest", "playwright", "vitest", "jest", "eslint", "npm run lint", "npm run check", "svelte-check", "tsc "]):
        return "test_lint"
    if any(t in c for t in ["curl ", "wget ", "nc ", "lsof ", "ss ", "ps ", "docker ps", "docker logs"]):
        return "probe"
    if any(t in c for t in ["sed -n", "cat ", "head ", "tail ", "ls ", "find ", "rg ", "grep ", "pwd", "wc ", "tree ", "jq "]):
        return "inspect"
    if any(t in c for t in ["mkdir ", "touch ", "cp ", "mv ", "chmod "]):
        return "fs_ops"
    if any(t in c for t in ["kill", "pkill"]):
        return "cleanup"
    if "git " in c:
        return "git"
    return "other_cmd"


def parse_run_name(run_dir: Path) -> dict[str, str]:
    match = RUN_RE.match(run_dir.name)
    if not match:
        return {"timestamp": "", "task": "", "variant": ""}
    return {"timestamp": match.group(1), "task": match.group(2), "variant": f"c{match.group(3)}"}


def infer_model(experiment: str, summary_model: str | None) -> str:
    if summary_model and not str(summary_model).startswith("<"):
        return str(summary_model)
    if "cursor" in experiment:
        return "Composer 2.5"
    if "5.4-mini" in experiment:
        return "gpt-5.4-mini"
    if "5.5" in experiment:
        return "gpt-5.5"
    if "claude_fable5" in experiment:
        return "claude-fable-5"
    if "claude_4.8" in experiment:
        return "claude-opus-4-8"
    if "claude_4.7" in experiment:
        return "claude-opus-4-7"
    if "claude_4.6" in experiment:
        return "claude-sonnet-4-6"
    if "claude_4.5" in experiment:
        return "claude-haiku-4-5"
    return summary_model or "unknown"


def iter_run_dirs(tasks_dir: Path, include_legacy: bool) -> list[Path]:
    out = []
    for exp_dir in sorted(tasks_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("_runs"):
            continue
        if not include_legacy and exp_dir.name in {"_runs", "_runs-fix"}:
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if run_dir.is_dir() and not run_dir.name.startswith("_") and (run_dir / "logs").exists():
                if "_superseded" not in run_dir.parts:
                    out.append(run_dir)
    return out


def parse_codex_items(run_dir: Path) -> list[Action]:
    path = run_dir / "logs" / "items.csv"
    if not path.exists():
        return parse_codex_events(run_dir)
    actions: list[Action] = []
    text = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    with io.StringIO(text, newline="") as f:
        for row in csv.DictReader(f):
            elapsed = safe_float(row.get("elapsed_s"))
            item_type = row.get("item_type") or ""
            if item_type == "command_execution":
                category = classify_command(row.get("command"))
                exit_code = row.get("exit_code")
                failure = exit_code not in ("", "0", None)
                actions.append(Action(elapsed, category, is_verify=category in VERIFY_CATEGORIES, is_failure=failure, detail=(row.get("command") or "")[:180]))
            elif item_type == "file_change":
                actions.append(Action(elapsed, "write", is_write=True, batch_files=1))
            elif item_type == "agent_message":
                actions.append(Action(elapsed, "message"))
            elif item_type == "todo_list":
                actions.append(Action(elapsed, "todo"))
            elif item_type:
                actions.append(Action(elapsed, item_type))
    return actions


def parse_codex_events(run_dir: Path) -> list[Action]:
    actions = []
    for event in load_jsonl(run_dir / "logs" / "codex_events.jsonl"):
        elapsed = safe_float(event.get("_elapsed_s"))
        item = event.get("item") or {}
        item_type = item.get("type")
        if event.get("type") == "item.completed" and item_type == "command_execution":
            category = classify_command(item.get("command"))
            actions.append(Action(elapsed, category, is_verify=category in VERIFY_CATEGORIES, is_failure=item.get("exit_code") not in (None, 0), detail=(item.get("command") or "")[:180]))
        elif event.get("type") == "item.completed" and item_type == "file_change":
            changes = item.get("changes") or []
            actions.append(Action(elapsed, "write", is_write=True, batch_files=len(changes)))
        elif event.get("type") == "item.completed" and item_type == "agent_message":
            actions.append(Action(elapsed, "message"))
        elif event.get("type") == "error" or item_type == "error":
            actions.append(Action(elapsed, "error", is_failure=True))
    return actions


def parse_claude_events(run_dir: Path) -> list[Action]:
    actions = []
    for event in load_jsonl(run_dir / "logs" / "events.jsonl"):
        elapsed = safe_float(event.get("_elapsed_s"))
        if event.get("type") == "rate_limit_event":
            status = (event.get("rate_limit_info") or {}).get("status", "unknown")
            actions.append(Action(elapsed, f"rate_{status}", is_failure=status == "rejected"))
            continue
        message = event.get("message") or {}
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for chunk in content:
            if not isinstance(chunk, dict):
                continue
            chunk_type = chunk.get("type")
            if chunk_type == "tool_use":
                name = chunk.get("name") or ""
                tool_input = chunk.get("input") or {}
                if name == "Bash":
                    category = classify_command(tool_input.get("command"))
                elif name in WRITE_TOOLS:
                    category = "write"
                elif name in INSPECT_TOOLS:
                    category = "inspect"
                elif name == "TodoWrite":
                    category = "todo"
                elif name in {"Task", "Agent"}:
                    category = "subagent"
                elif name in {"WebFetch", "WebSearch", "ToolSearch"}:
                    category = "search"
                else:
                    category = "other_tool"
                actions.append(Action(elapsed, category, is_write=category == "write", is_verify=category in VERIFY_CATEGORIES, batch_files=1 if category == "write" else 0, detail=name))
            elif chunk_type == "tool_result" and chunk.get("is_error"):
                actions.append(Action(elapsed, "error", is_failure=True))
            elif chunk_type == "text":
                actions.append(Action(elapsed, "message"))
            elif chunk_type == "thinking":
                actions.append(Action(elapsed, "thinking"))
    return actions


def cursor_tool_kind(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for key, value in tool_call.items():
        if key.endswith("ToolCall") and isinstance(value, dict):
            return key, value
    return "unknownToolCall", {}


def parse_cursor_events(run_dir: Path) -> list[Action]:
    actions = []
    for event in load_jsonl(run_dir / "logs" / "events.jsonl"):
        elapsed = safe_float(event.get("_elapsed_s"))
        event_type = event.get("type")
        if event_type == "assistant":
            actions.append(Action(elapsed, "message"))
        elif event_type == "thinking":
            actions.append(Action(elapsed, "thinking"))
        elif event_type == "tool_call" and event.get("subtype") == "completed":
            kind, body = cursor_tool_kind(event.get("tool_call") or {})
            args = body.get("args") or {}
            result = body.get("result") or {}
            success = "success" in result
            if kind == "editToolCall":
                category = "write"
            elif kind in {"readToolCall", "globToolCall", "grepToolCall"}:
                category = "inspect"
            elif kind in {"terminalToolCall", "runTerminalCommandToolCall"}:
                category = classify_command(args.get("command"))
            elif kind in {"webSearchToolCall", "fetchToolCall"}:
                category = "search"
            else:
                category = "other_tool"
            actions.append(Action(elapsed, category, is_write=category == "write", is_verify=category in VERIFY_CATEGORIES, is_failure=not success, batch_files=1 if category == "write" else 0, detail=kind))
        elif event_type == "error":
            actions.append(Action(elapsed, "error", is_failure=True))
    return actions


def parse_actions(run_dir: Path, experiment: str) -> tuple[list[Action], str]:
    if (run_dir / "logs" / "codex_events.jsonl").exists() or (run_dir / "logs" / "items.csv").exists():
        return parse_codex_items(run_dir), "codex"
    if "cursor" in experiment:
        return parse_cursor_events(run_dir), "cursor"
    return parse_claude_events(run_dir), "claude"


def load_eval(run_dir: Path) -> dict[str, Any]:
    data = read_json(run_dir / "logs" / "eval_result.json")
    summary = data.get("summary") or {}
    ncrit = summary.get("n_critical") or 0
    found = summary.get("found_critical") or 0
    return {
        "has_eval": bool(summary),
        "score": summary.get("combined_score_critical"),
        "avg_localization_critical": summary.get("avg_localization_critical"),
        "avg_behavior_critical": summary.get("avg_behavior_critical"),
        "n_critical": ncrit,
        "found_critical": found,
        "found_rate_critical": found / ncrit if ncrit else None,
    }


def load_edit_stats(run_dir: Path) -> dict[str, Any]:
    rows = load_jsonl(run_dir / "logs" / "edits.jsonl")
    files = {row.get("file_path") for row in rows if row.get("file_path")}
    return {
        "edit_record_count": len(rows),
        "edit_file_count": len(files),
        "edit_added_lines": sum(safe_num(row.get("added_lines")) for row in rows),
        "edit_removed_lines": sum(safe_num(row.get("removed_lines")) for row in rows),
        "edit_change_bytes": sum(safe_num(row.get("change_bytes")) for row in rows),
        "full_rewrite_count": sum(1 for row in rows if row.get("is_full_rewrite")),
        "new_file_count": sum(1 for row in rows if row.get("is_new_file")),
        "median_edit_ratio": median([row.get("ratio") for row in rows]),
    }


def sequence(actions: list[Action], *, end: bool = False, limit: int = 10) -> str:
    cats = [a.category for a in actions if a.category not in {"thinking", "message", "todo", "rate_allowed"}]
    cats = cats[-limit:] if end else cats[:limit]
    return ">".join(cats)


def build_row(run_dir: Path, tasks_dir: Path) -> dict[str, Any]:
    experiment = run_dir.parent.name
    parts = parse_run_name(run_dir)
    summary = read_json(run_dir / "logs" / "summary.json").get("summary") or {}
    model = infer_model(experiment, summary.get("model"))
    actions, source = parse_actions(run_dir, experiment)
    eval_stats = load_eval(run_dir)
    edit_stats = load_edit_stats(run_dir)

    duration = summary.get("wall_clock_first_to_last_assistant_s") or summary.get("wall_clock_first_to_last_turn_s")
    if not duration:
        duration = max((a.elapsed_s for a in actions if a.elapsed_s is not None), default=None)
    first_write = next((a for a in actions if a.is_write), None)
    first_failure = next((a for a in actions if a.is_failure), None)
    first_verify_after_write = next((a for a in actions if first_write and a.is_verify and a.elapsed_s is not None and first_write.elapsed_s is not None and a.elapsed_s >= first_write.elapsed_s), None)
    first_write_after_fail = next((a for a in actions if first_failure and a.is_write and a.elapsed_s is not None and first_failure.elapsed_s is not None and a.elapsed_s > first_failure.elapsed_s), None)

    counts = Counter(a.category for a in actions)
    write_actions = [a for a in actions if a.is_write]
    verify_actions = [a for a in actions if a.is_verify]
    fail_count = sum(1 for a in actions if a.is_failure)
    first_write_s = first_write.elapsed_s if first_write else None
    edit_to_verify_s = (
        first_verify_after_write.elapsed_s - first_write.elapsed_s
        if first_write and first_verify_after_write and first_write.elapsed_s is not None and first_verify_after_write.elapsed_s is not None
        else None
    )
    planning_frac = first_write_s / duration if first_write_s is not None and duration else None
    verify_delay_frac = edit_to_verify_s / duration if edit_to_verify_s is not None and duration else None
    write_after_first_fail = sum(1 for a in write_actions if first_failure and a.elapsed_s is not None and first_failure.elapsed_s is not None and a.elapsed_s > first_failure.elapsed_s)
    verify_after_first_fail = sum(1 for a in verify_actions if first_failure and a.elapsed_s is not None and first_failure.elapsed_s is not None and a.elapsed_s > first_failure.elapsed_s)
    first_fail_to_next_write_s = (
        first_write_after_fail.elapsed_s - first_failure.elapsed_s
        if first_failure and first_write_after_fail and first_failure.elapsed_s is not None and first_write_after_fail.elapsed_s is not None
        else None
    )

    if first_write_s is None:
        start_style = "no_write"
    elif first_write_s <= 120:
        start_style = "fast_prototype"
    elif first_write_s > 300:
        start_style = "long_planning"
    else:
        start_style = "moderate_planning"

    if edit_to_verify_s is None:
        verify_style = "no_verify_after_write"
    elif edit_to_verify_s <= 300:
        verify_style = "early_verify"
    elif edit_to_verify_s > 600:
        verify_style = "late_verify"
    else:
        verify_style = "middle_verify"

    if fail_count == 0:
        repair_style = "no_failure"
    elif write_after_first_fail == 0 and verify_after_first_fail == 0:
        repair_style = "unrepaired_failure"
    elif fail_count >= 6 and write_after_first_fail >= 10:
        repair_style = "high_churn_repair"
    elif write_after_first_fail or verify_after_first_fail:
        repair_style = "repair_loop"
    else:
        repair_style = "failure_without_loop"

    first_batch = write_actions[0].batch_files if write_actions else None
    if edit_stats["edit_record_count"] == 0 and not write_actions:
        edit_style = "no_recorded_edits"
    elif first_batch and first_batch >= 20:
        edit_style = "large_initial_batch"
    elif first_batch and first_batch >= 5:
        edit_style = "medium_initial_batch"
    elif edit_stats["full_rewrite_count"] >= edit_stats["edit_record_count"] * 0.6 and edit_stats["edit_record_count"]:
        edit_style = "rewrite_heavy"
    elif edit_stats["edit_record_count"] >= 30:
        edit_style = "many_incremental_edits"
    else:
        edit_style = "small_or_moderate_edits"

    row = {
        "experiment": experiment,
        "run_id": run_dir.name,
        "timestamp": parts["timestamp"],
        "task": parts["task"],
        "variant": parts["variant"],
        "model": model,
        "source": source,
        "path": str(run_dir.relative_to(tasks_dir.parent)),
        "duration_s": duration,
        "action_count": len(actions),
        "message_count": counts.get("message", 0),
        "thinking_count": counts.get("thinking", 0),
        "inspect_count": counts.get("inspect", 0),
        "write_action_count": len(write_actions),
        "verify_action_count": len(verify_actions),
        "deps_count": counts.get("deps", 0),
        "build_count": counts.get("build", 0),
        "test_lint_count": counts.get("test_lint", 0),
        "probe_count": counts.get("probe", 0),
        "server_count": counts.get("server", 0),
        "docker_up_build_count": counts.get("docker_up_build", 0),
        "fail_count": fail_count,
        "fail_rate_per_action": fail_count / len(actions) if actions else None,
        "time_to_first_write_s": first_write_s,
        "planning_frac": planning_frac,
        "edit_to_first_verify_s": edit_to_verify_s,
        "verify_delay_frac": verify_delay_frac,
        "write_after_first_fail_count": write_after_first_fail,
        "verify_after_first_fail_count": verify_after_first_fail,
        "first_fail_to_next_write_s": first_fail_to_next_write_s,
        "start_style": start_style,
        "verify_style": verify_style,
        "repair_style": repair_style,
        "edit_style": edit_style,
        "start_sequence": sequence(actions),
        "end_sequence": sequence(actions, end=True),
        **eval_stats,
        **edit_stats,
    }
    return row


def dedupe_latest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["experiment"], row["task"])
        if key not in latest or row.get("timestamp", "") > latest[key].get("timestamp", ""):
            latest[key] = row
    return list(latest.values())


def task_sort_key(task: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)_", task or "")
    return (int(match.group(1)) if match else 999, task or "")


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "")].append(row)
    out = []
    for value, group in grouped.items():
        out.append(
            {
                key: value,
                "n": len(group),
                "eval_n": sum(1 for r in group if r.get("has_eval")),
                "score_mean": mean(r.get("score") for r in group),
                "score_median": median(r.get("score") for r in group),
                "behavior_mean": mean(r.get("avg_behavior_critical") for r in group),
                "localization_mean": mean(r.get("avg_localization_critical") for r in group),
                "found_rate_mean": mean(r.get("found_rate_critical") for r in group),
                "duration_median": median(r.get("duration_s") for r in group),
                "time_to_first_write_median": median(r.get("time_to_first_write_s") for r in group),
                "edit_to_first_verify_median": median(r.get("edit_to_first_verify_s") for r in group),
                "inspect_mean": mean(r.get("inspect_count") for r in group),
                "write_mean": mean(r.get("write_action_count") for r in group),
                "verify_mean": mean(r.get("verify_action_count") for r in group),
                "fail_mean": mean(r.get("fail_count") for r in group),
                "edit_records_mean": mean(r.get("edit_record_count") for r in group),
                "fast_prototype_rate": mean(1 if r.get("start_style") == "fast_prototype" else 0 for r in group),
                "long_planning_rate": mean(1 if r.get("start_style") == "long_planning" else 0 for r in group),
                "late_or_no_verify_rate": mean(1 if r.get("verify_style") in {"late_verify", "no_verify_after_write"} else 0 for r in group),
                "high_churn_repair_rate": mean(1 if r.get("repair_style") == "high_churn_repair" else 0 for r in group),
            }
        )
    return sorted(out, key=lambda r: (r["score_mean"] is None, -(r["score_mean"] or 0), str(r.get(key) or "")))


def archetype_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for key in ["start_style", "verify_style", "repair_style", "edit_style"]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(key) or "")].append(row)
        for value, group in grouped.items():
            out.append(
                {
                    "dimension": key,
                    "value": value,
                    "n": len(group),
                    "score_mean": mean(r.get("score") for r in group),
                    "score_median": median(r.get("score") for r in group),
                    "behavior_mean": mean(r.get("avg_behavior_critical") for r in group),
                    "duration_median": median(r.get("duration_s") for r in group),
                    "fail_mean": mean(r.get("fail_count") for r in group),
                }
            )
    return sorted(out, key=lambda r: (r["dimension"], -(r["score_mean"] or 0)))


def correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "duration_s",
        "action_count",
        "inspect_count",
        "write_action_count",
        "verify_action_count",
        "fail_count",
        "fail_rate_per_action",
        "time_to_first_write_s",
        "planning_frac",
        "edit_to_first_verify_s",
        "verify_delay_frac",
        "edit_record_count",
        "edit_file_count",
        "edit_added_lines",
        "full_rewrite_count",
        "new_file_count",
    ]
    out = []
    for target in ["score", "avg_behavior_critical", "avg_localization_critical"]:
        for feature in features:
            value = spearman(rows, feature, target)
            if value is not None:
                out.append({"target": target, "feature": feature, "n": len([r for r in rows if isinstance(r.get(feature), (int, float)) and isinstance(r.get(target), (int, float))]), "spearman": value})
    return sorted(out, key=lambda r: -abs(r["spearman"]))


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str] | None = None) -> None:
    preferred = preferred or []
    keys = sorted({k for row in rows for k in row.keys()})
    keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]], limit: int | None = None) -> list[str]:
    use_rows = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(title for title, _ in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in use_rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for _, key in cols) + " |")
    return lines


def write_report(path: Path, rows: list[dict[str, Any]], model_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]], exp_rows: list[dict[str, Any]], arch_rows: list[dict[str, Any]], corrs: list[dict[str, Any]]) -> None:
    scored = [r for r in rows if isinstance(r.get("score"), (int, float))]
    lines = ["# Workflow Analysis Report", ""]
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Latest-per-experiment-task runs: {len(rows)}")
    lines.append(f"- Runs with eval score: {len(scored)}")
    lines.append("- Source: tasks/_runs* logs; excludes tasks/_runs, tasks/_runs-fix, and _superseded by default.")
    lines.append("")
    lines.append("## Headline Patterns")
    lines.append("")
    if corrs:
        top = corrs[0]
        lines.append(f"- Strongest workflow correlation: {top['feature']} vs {top['target']} has Spearman {fmt(top['spearman'])}.")
    no_verify = next((r for r in arch_rows if r["dimension"] == "verify_style" and r["value"] == "no_verify_after_write"), None)
    early = next((r for r in arch_rows if r["dimension"] == "verify_style" and r["value"] == "early_verify"), None)
    if no_verify and early:
        lines.append(f"- Early verification scores {fmt(early['score_mean'])}; no post-write verification scores {fmt(no_verify['score_mean'])}.")
    fast = next((r for r in arch_rows if r["dimension"] == "start_style" and r["value"] == "fast_prototype"), None)
    longp = next((r for r in arch_rows if r["dimension"] == "start_style" and r["value"] == "long_planning"), None)
    if fast and longp:
        lines.append(f"- Fast prototype starts score {fmt(fast['score_mean'])}; long planning starts score {fmt(longp['score_mean'])}.")
    high_churn = next((r for r in arch_rows if r["dimension"] == "repair_style" and r["value"] == "high_churn_repair"), None)
    if high_churn:
        lines.append(f"- High-churn repair appears in {high_churn['n']} runs and averages {fmt(high_churn['score_mean'])}.")
    lines.append("- Treat these as process signals, not causal proof: task difficulty and model family still confound workflow style.")
    lines.append("")
    lines.append("## Model Summary")
    lines.extend(md_table(model_rows, [("model", "model"), ("runs", "n"), ("eval", "eval_n"), ("score", "score_mean"), ("beh", "behavior_mean"), ("dur med", "duration_median"), ("first write", "time_to_first_write_median"), ("verify delay", "edit_to_first_verify_median"), ("fail", "fail_mean")]))
    lines.append("")
    lines.append("## Task Summary")
    lines.extend(md_table(sorted(task_rows, key=lambda r: task_sort_key(r["task"])), [("task", "task"), ("runs", "n"), ("eval", "eval_n"), ("score", "score_mean"), ("beh", "behavior_mean"), ("dur med", "duration_median"), ("late/no verify", "late_or_no_verify_rate"), ("fail", "fail_mean")]))
    lines.append("")
    lines.append("## Experiment Summary")
    lines.extend(md_table(exp_rows, [("experiment", "experiment"), ("runs", "n"), ("eval", "eval_n"), ("score", "score_mean"), ("dur med", "duration_median"), ("fast", "fast_prototype_rate"), ("long", "long_planning_rate"), ("late/no verify", "late_or_no_verify_rate")]))
    lines.append("")
    lines.append("## Archetype Summary")
    lines.extend(md_table(arch_rows, [("dimension", "dimension"), ("value", "value"), ("n", "n"), ("score", "score_mean"), ("beh", "behavior_mean"), ("dur med", "duration_median"), ("fail", "fail_mean")]))
    lines.append("")
    lines.append("## Strongest Correlations")
    lines.extend(md_table(corrs, [("target", "target"), ("feature", "feature"), ("n", "n"), ("rho", "spearman")], limit=25))
    lines.append("")
    lines.append("## Reading Guide")
    lines.append("")
    lines.append("- `start_style` measures how quickly the first write appears.")
    lines.append("- `verify_style` measures the delay from first write to first build/test/probe/server check.")
    lines.append("- `repair_style` captures whether failures are followed by more edits or verification.")
    lines.append("- `edit_style` is derived from recorded edits and first write batch shape.")
    lines.append("- Cursor/Composer logs expose read/glob/edit actions but not terminal verification commands, so their verification timing is under-observed.")
    lines.append("- Use `workflow_features.csv` for run-level drilldown and `workflow_model_task_pivot.csv` for cross-model task comparisons.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_task_pivot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = sorted({r["model"] for r in rows})
    tasks = sorted({r["task"] for r in rows}, key=task_sort_key)
    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["model"])].append(row.get("score"))
    out = []
    for task in tasks:
        rec: dict[str, Any] = {"task": task}
        for model in models:
            rec[model] = mean(grouped[(task, model)])
        out.append(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/workflow_analysis"))
    parser.add_argument("--include-legacy-runs", action="store_true")
    args = parser.parse_args()

    all_rows = [build_row(run_dir, args.tasks_dir) for run_dir in iter_run_dirs(args.tasks_dir, args.include_legacy_runs)]
    rows = dedupe_latest(all_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_rows = group_summary(rows, "model")
    exp_rows = group_summary(rows, "experiment")
    task_rows = group_summary(rows, "task")
    arch_rows = archetype_summary(rows)
    corrs = correlation_rows([r for r in rows if r.get("has_eval")])
    pivot = model_task_pivot(rows)

    write_csv(args.out_dir / "workflow_features.csv", rows, preferred=["experiment", "model", "task", "run_id"])
    write_csv(args.out_dir / "workflow_all_features.csv", all_rows, preferred=["experiment", "model", "task", "run_id"])
    write_csv(args.out_dir / "workflow_model_summary.csv", model_rows, preferred=["model"])
    write_csv(args.out_dir / "workflow_experiment_summary.csv", exp_rows, preferred=["experiment"])
    write_csv(args.out_dir / "workflow_task_summary.csv", task_rows, preferred=["task"])
    write_csv(args.out_dir / "workflow_archetype_summary.csv", arch_rows, preferred=["dimension", "value"])
    write_csv(args.out_dir / "workflow_correlations.csv", corrs, preferred=["target", "feature"])
    write_csv(args.out_dir / "workflow_model_task_pivot.csv", pivot, preferred=["task"])
    write_report(args.out_dir / "workflow_report.md", rows, model_rows, task_rows, exp_rows, arch_rows, corrs)

    print(f"Analyzed {len(all_rows)} run directories; {len(rows)} latest rows.")
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
