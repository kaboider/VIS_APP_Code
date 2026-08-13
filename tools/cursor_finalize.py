#!/usr/bin/env python3
"""cursor_finalize.py — post-process a cursor-agent run's summary.json.

cursor-agent's stream-json differs from Claude's in two ways that break the
shared analyze_run.py:

  1. Token usage is reported ONLY on the terminal `result` event, and in
     camelCase (`inputTokens`, `outputTokens`, `cacheReadTokens`,
     `cacheWriteTokens`) — NOT per-assistant-message in Anthropic snake_case.
     So analyze_run.py's per-message accumulator leaves tokens all-zero.
     We map the result-event usage into summary.json's `tokens` block here.

  2. cursor-agent often gets killed by run_eval's inactivity watchdog AFTER
     finishing the work but BEFORE emitting `result`, leaving is_error=null.
     If the work is clearly present (workspace has docker-compose.yml + at
     least one assistant text message), promote is_error null -> false.

Idempotent. Safe to run repeatedly (used both inline by run_eval.sh and for
backfilling older runs). Leaves genuinely-failed early runs untouched.

Usage: cursor_finalize.py <run_dir>
"""
import glob
import json
import os
import sys


def event_stats(events_path):
    """Derive telemetry that Cursor emits even when its result event is missing."""
    stats = {
        "event_count": 0,
        "elapsed_s": 0.0,
        "reasoning_steps": 0,
        "assistant_messages": 0,
        "tool_calls": 0,
        "tool_call_counts": {},
        "has_result_event": False,
    }
    if not os.path.exists(events_path):
        return stats
    with open(events_path) as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            stats["event_count"] += 1
            elapsed = ev.get("_elapsed_s")
            if isinstance(elapsed, (int, float)):
                stats["elapsed_s"] = max(stats["elapsed_s"], float(elapsed))
            if ev.get("type") == "thinking" and ev.get("subtype") == "completed":
                stats["reasoning_steps"] += 1
            elif ev.get("type") == "assistant":
                stats["assistant_messages"] += 1
            elif ev.get("type") == "result":
                stats["has_result_event"] = True
            elif ev.get("type") == "tool_call" and ev.get("subtype") == "completed":
                stats["tool_calls"] += 1
                payload = ev.get("tool_call") or {}
                name = next(iter(payload), "unknown")
                counts = stats["tool_call_counts"]
                counts[name] = counts.get(name, 0) + 1
    return stats


def last_result_usage(events_path):
    """Return the usage dict from the last `result` event, or None."""
    usage = None
    if not os.path.exists(events_path):
        return None
    with open(events_path) as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "result" and isinstance(ev.get("usage"), dict):
                usage = ev["usage"]
    return usage


def has_assistant_text(events_path):
    if not os.path.exists(events_path):
        return False
    with open(events_path) as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "assistant":
                for blk in ev.get("message", {}).get("content", []):
                    if blk.get("type") == "text" and blk.get("text", "").strip():
                        return True
    return False


def main():
    if len(sys.argv) != 2:
        print("usage: cursor_finalize.py <run_dir>", file=sys.stderr)
        return 2
    run = sys.argv[1]
    sp = os.path.join(run, "logs", "summary.json")
    ev = os.path.join(run, "logs", "events.jsonl")
    try:
        doc = json.load(open(sp))
    except Exception as e:
        print(f"[cursor_finalize] no/invalid summary.json ({e}); skipping")
        return 0
    summ = doc.setdefault("summary", {})
    changed = False

    # Cursor always emits timestamped stream events, even if its terminal result
    # event never arrives. Preserve those reliable wall-clock/tool statistics.
    stats = event_stats(ev)
    derived = {
        "event_count": stats["event_count"],
        "wall_clock_s": round(stats["elapsed_s"], 3),
        "reasoning_steps": stats["reasoning_steps"],
        "assistant_messages": stats["assistant_messages"],
        "tool_calls": stats["tool_calls"],
        "tool_call_counts": stats["tool_call_counts"],
        "has_result_event": stats["has_result_event"],
    }
    for key, value in derived.items():
        if summ.get(key) != value:
            summ[key] = value
            changed = True

    # (1) map result-event usage (camelCase) -> tokens block
    usage = last_result_usage(ev)
    if usage:
        tk = summ.setdefault("tokens", {})
        mapped = {
            "input_tokens": int(usage.get("inputTokens", 0) or 0),
            "output_tokens": int(usage.get("outputTokens", 0) or 0),
            "cache_read_input_tokens": int(usage.get("cacheReadTokens", 0) or 0),
            "cache_creation_input_tokens": int(usage.get("cacheWriteTokens", 0) or 0),
        }
        mapped["total_input_billable"] = (
            mapped["input_tokens"] + mapped["cache_creation_input_tokens"]
        )
        # Only overwrite if analyze_run left them zero (don't clobber real data).
        if any(tk.get(k, 0) for k in mapped):
            pass
        else:
            tk.update(mapped)
            tk["_source"] = "cursor_result_event"
            changed = True
    else:
        tk = summ.setdefault("tokens", {})
        if tk.get("_source") != "unavailable_no_cursor_result_event":
            tk["_source"] = "unavailable_no_cursor_result_event"
            tk["_note"] = (
                "Cursor's intermediate stream has no token counts and the CLI "
                "did not emit its terminal result event before watchdog cleanup."
            )
            changed = True

    # (2) synthetic success when work is present but no result event fired
    if summ.get("is_error") is None:
        has_compose = os.path.exists(
            os.path.join(run, "workspace", "docker-compose.yml")
        ) or bool(
            glob.glob(
                os.path.join(run, "workspace", "**", "docker-compose.y*ml"),
                recursive=True,
            )
        )
        if has_compose and has_assistant_text(ev):
            summ["is_error"] = False
            summ["_cursor_synthetic_success"] = True
            changed = True

    if changed:
        json.dump(doc, open(sp, "w"), indent=2)
        tk = summ.get("tokens", {})
        print(
            f"[cursor_finalize] {os.path.basename(run)}: "
            f"is_error={summ.get('is_error')} "
            f"in={tk.get('input_tokens')} out={tk.get('output_tokens')} "
            f"cache_read={tk.get('cache_read_input_tokens')}"
        )
    else:
        print(f"[cursor_finalize] {os.path.basename(run)}: no change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
