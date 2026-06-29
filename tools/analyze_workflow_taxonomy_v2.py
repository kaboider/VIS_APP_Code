#!/usr/bin/env python3
"""
Build taxonomy-v2 workflow tables from tasks/_runs* logs.

Taxonomy v2 keeps the main workflow compact while preserving raw tool/action
fields for error, planning/todo, and tool-specific analysis.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_task_workflows import infer_model, iter_run_dirs, parse_run_name, read_json  # noqa: E402


INSPECT_TOOLS = {"Read", "Grep", "Glob", "readToolCall", "grepToolCall", "globToolCall"}
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "editToolCall", "deleteToolCall", "file_change"}
TODO_TOOLS = {"TodoWrite", "todo_list", "updateTodosToolCall"}
SEARCH_TOOLS = {"ToolSearch", "WebSearch", "WebFetch", "web_search"}
DELEGATE_TOOLS = {"Agent", "Task", "TaskStop", "subagent"}


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def shell_prefix(command: str) -> str:
    c = (command or "").strip()
    if not c:
        return ""
    parts = c.split()
    if parts[0] in {"/bin/zsh", "/bin/bash", "bash", "zsh"} and len(parts) >= 3:
        inner = " ".join(parts[2:]).strip("'\"")
        return inner.split()[0] if inner.split() else parts[0]
    return parts[0]


def command_text(command: str) -> str:
    c = (command or "").strip()
    parts = c.split()
    if parts and parts[0] in {"/bin/zsh", "/bin/bash", "bash", "zsh"} and len(parts) >= 3:
        return " ".join(parts[2:]).strip("'\"")
    return c


def classify_shell(command: str) -> tuple[str, str]:
    c = command_text(command).lower()
    prefix = shell_prefix(command)
    if any(t in c for t in ["npm install", "pnpm install", "yarn install", "npm i ", "pip install", "apt-get", "apt install", "npm create", "create-next-app", "create vite", "npm init"]):
        return "setup", "deps_or_scaffold"
    if any(t in c for t in ["mkdir ", "touch ", "cp ", "mv ", "chmod ", "rm ", "rsync "]):
        return "setup", "fs_ops"
    if any(t in c for t in ["docker compose up", "docker-compose up", "docker compose build", "docker-compose build"]):
        return "run", "docker_lifecycle"
    if any(t in c for t in ["npm run dev", "npm start", "node server", "uvicorn", "vite --host", "next dev", "astro dev", "npm run start"]):
        return "run", "server"
    if any(t in c for t in ["npm run build", "pnpm build", "yarn build", "next build", "vite build", "tsc ", "svelte-check", "npm run check"]):
        return "verify", "build_check"
    if any(t in c for t in ["npm test", "npm run test", "pytest", "playwright", "vitest", "jest", "eslint", "npm run lint"]):
        return "verify", "test_lint"
    if any(t in c for t in ["curl ", "wget ", "nc ", "lsof ", "ss ", "ps ", "docker ps", "docker logs", "health"]):
        return "probe", "probe"
    if any(t in c for t in ["sed -n", "cat ", "head ", "tail ", "ls ", "find ", "rg ", "grep ", "pwd", "wc ", "tree ", "jq "]):
        return "inspect", "shell_inspect"
    if "git " in c:
        return "misc", "git"
    if any(t in c for t in ["kill", "pkill", "docker compose down", "docker-compose down"]):
        return "misc", "cleanup"
    if prefix in {"python", "python3", "node", "npx"}:
        return "misc", "script_or_runtime"
    return "misc", "other_cmd"


def classify_error(command: str, raw_tool_kind: str, action_main: str, output: str = "") -> str:
    blob = f"{command}\n{output}".lower()
    if raw_tool_kind and raw_tool_kind not in {"command_execution", "Bash", "shellToolCall"}:
        if "tool" in raw_tool_kind.lower() or raw_tool_kind in {"Read", "Write", "Edit"}:
            return "tool_error"
    if any(t in blob for t in ["permission denied", "operation not permitted", "enoent", "no such file", "not found", "cannot find path"]):
        return "permission_or_path_error"
    if any(t in blob for t in ["syntaxerror", "referenceerror", "typeerror", "module not found", "cannot find module", "failed to compile", "typescript", "tsc", "svelte"]):
        return "syntax_runtime_error"
    if action_main == "setup":
        return "dependency_error"
    if "docker" in blob:
        return "docker_error"
    if action_main == "run":
        return "server_error"
    if action_main == "verify":
        if any(t in blob for t in ["lint", "eslint", "test", "playwright", "vitest", "jest", "pytest"]):
            return "test_lint_error"
        return "build_error"
    if action_main == "probe":
        return "probe_error"
    return "unknown_error"


def todo_stats_from_obj(obj: Any) -> tuple[int, int, int]:
    pending = active = done = 0
    if isinstance(obj, dict):
        vals = obj.get("todos") or obj.get("items") or obj.get("todoList") or []
    elif isinstance(obj, list):
        vals = obj
    else:
        vals = []
    if isinstance(vals, list):
        for item in vals:
            if not isinstance(item, dict):
                continue
            if item.get("completed") is True:
                done += 1
                continue
            if item.get("completed") is False:
                pending += 1
                continue
            status = str(item.get("status") or item.get("state") or "").lower()
            if status in {"completed", "done", "checked"}:
                done += 1
            elif status in {"in_progress", "active", "doing"}:
                active += 1
            else:
                pending += 1
    return pending, active, done


def base_run(run_dir: Path) -> dict[str, Any]:
    exp = run_dir.parent.name
    parts = parse_run_name(run_dir)
    summary = read_json(run_dir / "logs" / "summary.json").get("summary") or {}
    return {
        "model": infer_model(exp, summary.get("model")),
        "experiment": exp,
        "run_id": run_dir.name,
        "task": parts["task"],
        "timestamp": parts["timestamp"],
        "source": "",
    }


def collect_codex(run_dir: Path, base: dict[str, Any]) -> list[dict[str, Any]]:
    if (run_dir / "logs" / "codex_events.jsonl").exists():
        return collect_codex_events(run_dir, base)
    path = run_dir / "logs" / "items.csv"
    rows = []
    if not path.exists():
        return rows
    text = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    has_event_todos = any((ev.get("item") or {}).get("type") == "todo_list" for ev in load_jsonl(run_dir / "logs" / "codex_events.jsonl"))
    for item in csv.DictReader(io.StringIO(text)):
        kind = item.get("item_type") or ""
        if kind == "todo_list" and has_event_todos:
            continue
        elapsed = safe_float(item.get("elapsed_s"))
        command = item.get("command") or ""
        output = item.get("output_excerpt") or ""
        exit_code = item.get("exit_code")
        is_error = kind == "command_execution" and exit_code not in ("", "0", None)
        if kind == "command_execution":
            main, sub = classify_shell(command)
        elif kind == "file_change":
            main, sub = "write", "file_change"
        elif kind == "agent_message":
            main, sub = "planning", "message"
        elif kind == "todo_list":
            main, sub = "planning", "todo"
        else:
            main, sub = "misc", kind or "unknown"
        pending = active = done = 0
        if kind == "todo_list":
            pending, active, done = todo_stats_from_obj(item)
        rows.append({**base, "source": "codex", "elapsed_s": elapsed, "raw_tool_kind": kind, "raw_command": command, "command_prefix": shell_prefix(command), "action_main": main, "action_subtype": sub, "is_error": is_error, "error_type": classify_error(command, kind, main, output) if is_error else "", "todo_pending": pending, "todo_active": active, "todo_done": done, "text_len": len(item.get("text") or ""), "detail": command[:240]})
    rows.extend(collect_codex_todo_events(run_dir, base))
    return rows


def collect_codex_events(run_dir: Path, base: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for ev in load_jsonl(run_dir / "logs" / "codex_events.jsonl"):
        typ = ev.get("type")
        item = ev.get("item") or {}
        kind = item.get("type") or ""
        elapsed = safe_float(ev.get("_elapsed_s"))
        if kind == "todo_list" and typ in {"item.started", "item.updated", "item.completed"}:
            pending, active, done = todo_stats_from_obj(item.get("items") or [])
            rows.append({
                **base,
                "source": "codex",
                "elapsed_s": elapsed,
                "raw_tool_kind": "todo_list",
                "raw_command": "",
                "command_prefix": "",
                "action_main": "planning",
                "action_subtype": "todo",
                "is_error": False,
                "error_type": "",
                "todo_pending": pending,
                "todo_active": active,
                "todo_done": done,
                "text_len": sum(len(str(todo.get("text") or "")) for todo in item.get("items", []) if isinstance(todo, dict)),
                "detail": typ or "",
            })
            continue
        if typ != "item.completed":
            continue
        command = item.get("command") or ""
        output = item.get("aggregated_output") or ""
        exit_code = item.get("exit_code")
        if kind == "command_execution":
            main, sub = classify_shell(command)
            is_error = exit_code not in ("", "0", 0, None)
        elif kind == "file_change":
            main, sub, is_error = "write", "file_change", False
        elif kind == "agent_message":
            main, sub, is_error = "planning", "message", False
        elif kind in SEARCH_TOOLS or kind in {"web_search", "mcp_tool_call"}:
            main, sub, is_error = "search_delegate", kind, False
        else:
            main, sub, is_error = "misc", kind or "unknown", False
        text = item.get("text") or ""
        rows.append({
            **base,
            "source": "codex",
            "elapsed_s": elapsed,
            "raw_tool_kind": kind,
            "raw_command": command,
            "command_prefix": shell_prefix(command),
            "action_main": main,
            "action_subtype": sub,
            "is_error": is_error,
            "error_type": classify_error(command, kind, main, output) if is_error else "",
            "todo_pending": 0,
            "todo_active": 0,
            "todo_done": 0,
            "text_len": len(text),
            "detail": (text or command)[:240],
        })
    return rows


def collect_codex_todo_events(run_dir: Path, base: dict[str, Any]) -> list[dict[str, Any]]:
    path = run_dir / "logs" / "codex_events.jsonl"
    rows = []
    if not path.exists():
        return rows
    for ev in load_jsonl(path):
        if ev.get("type") not in {"item.started", "item.updated", "item.completed"}:
            continue
        item = ev.get("item") or {}
        if item.get("type") != "todo_list":
            continue
        pending, active, done = todo_stats_from_obj(item.get("items") or [])
        rows.append({
            **base,
            "source": "codex",
            "elapsed_s": safe_float(ev.get("_elapsed_s")),
            "raw_tool_kind": "todo_list",
            "raw_command": "",
            "command_prefix": "",
            "action_main": "planning",
            "action_subtype": "todo",
            "is_error": False,
            "error_type": "",
            "todo_pending": pending,
            "todo_active": active,
            "todo_done": done,
            "text_len": sum(len(str(todo.get("text") or "")) for todo in item.get("items", []) if isinstance(todo, dict)),
            "detail": ev.get("type", ""),
        })
    return rows


def collect_claude(run_dir: Path, base: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    pending_tool: dict[str, dict[str, Any]] = {}
    for ev in load_jsonl(run_dir / "logs" / "events.jsonl"):
        elapsed = safe_float(ev.get("_elapsed_s"))
        if ev.get("type") == "rate_limit_event":
            continue
        msg = ev.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for chunk in content:
            if not isinstance(chunk, dict):
                continue
            typ = chunk.get("type")
            if typ == "tool_use":
                name = chunk.get("name") or ""
                inp = chunk.get("input") or {}
                command = inp.get("command") or ""
                if name == "Bash" or name == "bash":
                    main, sub = classify_shell(command)
                elif name in WRITE_TOOLS:
                    main, sub = "write", name
                elif name in INSPECT_TOOLS:
                    main, sub = "inspect", name
                elif name in TODO_TOOLS:
                    main, sub = "planning", "todo"
                elif name in SEARCH_TOOLS or name in DELEGATE_TOOLS:
                    main, sub = "search_delegate", name
                else:
                    main, sub = "misc", name or "other_tool"
                if main == "planning":
                    pending, active, done = todo_stats_from_obj(inp)
                else:
                    pending, active, done = (0, 0, 0)
                rec = {**base, "source": "claude", "elapsed_s": elapsed, "raw_tool_kind": name, "raw_command": command, "command_prefix": shell_prefix(command), "action_main": main, "action_subtype": sub, "is_error": False, "error_type": "", "todo_pending": pending, "todo_active": active, "todo_done": done, "text_len": 0, "detail": (name if not command else command[:240])}
                rows.append(rec)
                if chunk.get("id"):
                    pending_tool[chunk["id"]] = rec
            elif typ == "tool_result" and chunk.get("is_error"):
                parent = pending_tool.get(chunk.get("tool_use_id"), {})
                command = parent.get("raw_command", "")
                main = parent.get("action_main", "misc")
                rows.append({**base, "source": "claude", "elapsed_s": elapsed, "raw_tool_kind": "tool_result_error", "raw_command": command, "command_prefix": shell_prefix(command), "action_main": "repair_failure", "action_subtype": "tool_error", "is_error": True, "error_type": classify_error(command, "tool_result_error", main, str(chunk.get("content") or "")), "todo_pending": 0, "todo_active": 0, "todo_done": 0, "text_len": len(str(chunk.get("content") or "")), "detail": str(chunk.get("content") or "")[:240]})
            elif typ == "text":
                rows.append({**base, "source": "claude", "elapsed_s": elapsed, "raw_tool_kind": "message", "raw_command": "", "command_prefix": "", "action_main": "planning", "action_subtype": "message", "is_error": False, "error_type": "", "todo_pending": 0, "todo_active": 0, "todo_done": 0, "text_len": len(chunk.get("text") or ""), "detail": (chunk.get("text") or "")[:240]})
            elif typ == "thinking":
                rows.append({**base, "source": "claude", "elapsed_s": elapsed, "raw_tool_kind": "thinking", "raw_command": "", "command_prefix": "", "action_main": "planning", "action_subtype": "thinking", "is_error": False, "error_type": "", "todo_pending": 0, "todo_active": 0, "todo_done": 0, "text_len": len(chunk.get("thinking") or ""), "detail": ""})
    return rows


def cursor_tool_kind(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for key, value in tool_call.items():
        if key.endswith("ToolCall") and isinstance(value, dict):
            return key, value
    return "unknownToolCall", {}


def collect_cursor(run_dir: Path, base: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for ev in load_jsonl(run_dir / "logs" / "events.jsonl"):
        elapsed = safe_float(ev.get("_elapsed_s"))
        typ = ev.get("type")
        if typ == "assistant":
            text = "".join(ch.get("text", "") for ch in (ev.get("message") or {}).get("content", []) if isinstance(ch, dict))
            rows.append({**base, "source": "cursor", "elapsed_s": elapsed, "raw_tool_kind": "message", "raw_command": "", "command_prefix": "", "action_main": "planning", "action_subtype": "message", "is_error": False, "error_type": "", "todo_pending": 0, "todo_active": 0, "todo_done": 0, "text_len": len(text), "detail": text[:240]})
        elif typ == "thinking":
            rows.append({**base, "source": "cursor", "elapsed_s": elapsed, "raw_tool_kind": "thinking", "raw_command": "", "command_prefix": "", "action_main": "planning", "action_subtype": "thinking", "is_error": False, "error_type": "", "todo_pending": 0, "todo_active": 0, "todo_done": 0, "text_len": len(ev.get("text") or ""), "detail": ""})
        elif typ == "tool_call" and ev.get("subtype") == "completed":
            kind, body = cursor_tool_kind(ev.get("tool_call") or {})
            args = body.get("args") or {}
            result = body.get("result") or {}
            command = args.get("command") or args.get("cmd") or ""
            success = "success" in result
            if kind == "editToolCall" or kind == "deleteToolCall":
                main, sub = "write", kind
            elif kind in INSPECT_TOOLS:
                main, sub = "inspect", kind
            elif kind == "shellToolCall":
                main, sub = classify_shell(command)
            elif kind == "updateTodosToolCall":
                main, sub = "planning", "todo"
            elif kind in {"awaitToolCall"}:
                main, sub = "run", "await"
            else:
                main, sub = "misc", kind
            pending, active, done = todo_stats_from_obj(args) if main == "planning" else (0, 0, 0)
            rows.append({**base, "source": "cursor", "elapsed_s": elapsed, "raw_tool_kind": kind, "raw_command": command, "command_prefix": shell_prefix(command), "action_main": main if success else "repair_failure", "action_subtype": sub, "is_error": not success, "error_type": classify_error(command, kind, main) if not success else "", "todo_pending": pending, "todo_active": active, "todo_done": done, "text_len": 0, "detail": kind if not command else command[:240]})
    return rows


def dedupe_latest(run_dirs: list[Path]) -> list[Path]:
    latest: dict[tuple[str, str], Path] = {}
    for run_dir in run_dirs:
        parts = parse_run_name(run_dir)
        key = (run_dir.parent.name, parts["task"])
        if key not in latest or parts["timestamp"] > parse_run_name(latest[key])["timestamp"]:
            latest[key] = run_dir
    by_run_id: dict[str, Path] = {}
    for run_dir in latest.values():
        current = by_run_id.get(run_dir.name)
        if current is None:
            by_run_id[run_dir.name] = run_dir
            continue
        current_is_log = current.parent.name.endswith("_log")
        candidate_is_log = run_dir.parent.name.endswith("_log")
        if current_is_log and not candidate_is_log:
            by_run_id[run_dir.name] = run_dir
    return sorted(by_run_id.values(), key=lambda p: (p.parent.name, p.name))


def collect(tasks_dir: Path) -> list[dict[str, Any]]:
    out = []
    for run_dir in dedupe_latest(iter_run_dirs(tasks_dir, include_legacy=False)):
        base = base_run(run_dir)
        if (run_dir / "logs" / "codex_events.jsonl").exists() or (run_dir / "logs" / "items.csv").exists():
            out.extend(collect_codex(run_dir, base))
        elif "cursor" in run_dir.parent.name:
            out.extend(collect_cursor(run_dir, base))
        else:
            out.extend(collect_claude(run_dir, base))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in out:
        grouped[row["run_id"]].append(row)
    final = []
    for run_id, rows in grouped.items():
        rows.sort(key=lambda r: (r["elapsed_s"] is None, r["elapsed_s"] or 0))
        last_error = None
        for idx, row in enumerate(rows, 1):
            row["seq"] = idx
            row["phase"] = "after_first_error" if last_error is not None else "normal"
            row["is_after_error"] = last_error is not None
            if row["is_error"]:
                last_error = row
            final.append(row)
    return final


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dist(rows: list[dict[str, Any]], key: str, extra_keys: list[str] | None = None) -> list[dict[str, Any]]:
    extra_keys = extra_keys or []
    counts = Counter(tuple([row.get(key)] + [row.get(k) for k in extra_keys]) for row in rows)
    total = sum(counts.values()) or 1
    out = []
    for vals, count in counts.most_common():
        rec = {key: vals[0], "count": count, "share": count / total}
        for k, v in zip(extra_keys, vals[1:]):
            rec[k] = v
        out.append(rec)
    return out


def write_taxonomy_doc(path: Path) -> None:
    path.write_text(
        """# Action Taxonomy V2

Main workflow categories:

- `inspect`: read/search local task or code context.
- `planning`: message, thinking, TodoWrite, todo_list, updateTodosToolCall.
- `write`: file creation/edit/delete/change.
- `setup`: dependencies, scaffolding, filesystem setup.
- `run`: server startup, docker lifecycle, await/run commands.
- `verify`: build, typecheck, lint, tests.
- `probe`: curl/health/lsof/docker logs/ps and other runtime probes.
- `repair_failure`: failed command/tool result, with `error_type`.
- `search_delegate`: web/tool search and subagent/delegation.
- `misc`: git, cleanup, scripts, uncategorized shell/tool calls.

Error types:

- `build_error`
- `test_lint_error`
- `dependency_error`
- `docker_error`
- `server_error`
- `probe_error`
- `tool_error`
- `permission_or_path_error`
- `syntax_runtime_error`
- `unknown_error`

Use `workflow_actions_v2.csv` for the full sequence, `workflow_errors.csv` for
failure analysis, `workflow_planning_actions.csv` for message/thinking/todo
behavior, and `workflow_tool_distribution.csv` for raw tool usage.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/workflow_analysis"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect(args.tasks_dir)
    fields = [
        "model", "experiment", "run_id", "task", "timestamp", "source", "seq", "elapsed_s",
        "raw_tool_kind", "raw_command", "command_prefix", "action_main", "action_subtype",
        "phase", "is_after_error", "is_error", "error_type", "todo_pending", "todo_active",
        "todo_done", "text_len", "detail",
    ]
    write_csv(args.out_dir / "workflow_actions_v2.csv", rows, fields)
    write_csv(args.out_dir / "workflow_errors.csv", [r for r in rows if r["is_error"]], fields)
    write_csv(args.out_dir / "workflow_planning_actions.csv", [r for r in rows if r["action_main"] == "planning"], fields)

    tool_dist = dist(rows, "raw_tool_kind")
    write_csv(args.out_dir / "workflow_tool_distribution.csv", tool_dist, ["raw_tool_kind", "count", "share"])
    action_dist = dist(rows, "action_main")
    write_csv(args.out_dir / "workflow_action_main_distribution.csv", action_dist, ["action_main", "count", "share"])
    model_action = dist(rows, "action_main", ["model"])
    write_csv(args.out_dir / "workflow_action_main_by_model.csv", model_action, ["action_main", "model", "count", "share"])
    error_dist = dist([r for r in rows if r["is_error"]], "error_type", ["model"])
    write_csv(args.out_dir / "workflow_error_type_by_model.csv", error_dist, ["error_type", "model", "count", "share"])
    cmd_prefix = dist([r for r in rows if r["command_prefix"]], "command_prefix", ["model"])
    write_csv(args.out_dir / "workflow_command_prefix_distribution.csv", cmd_prefix, ["command_prefix", "model", "count", "share"])
    write_taxonomy_doc(args.out_dir / "action_taxonomy_v2.md")
    print(f"Wrote taxonomy-v2 tables for {len(rows)} actions.")


if __name__ == "__main__":
    main()
