#!/usr/bin/env python3
"""Summarize CAMEL model usage, cache telemetry, timing, and eval scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run_id", "task", "model", "reasoning_effort", "status",
    "build_elapsed_s", "model_calls", "model_elapsed_s",
    "input_tokens", "cached_input_tokens", "uncached_input_tokens",
    "output_tokens", "total_tokens", "actual_cache_rate",
    "theoretical_full_hit_cached_tokens",
    "theoretical_full_hit_uncached_tokens",
    "theoretical_full_hit_rate", "cache_telemetry", "combined_score",
]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def collect(run: Path) -> dict | None:
    trace = run / "logs" / "camel_single_trace.jsonl"
    if not trace.is_file():
        return None
    events = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    calls = [event for event in events if event.get("event") == "model_call"]
    summary = read_json(run / "logs" / "summary.json").get("summary", {})
    meta = read_json(run / "meta.json")
    result = read_json(run / "logs" / "eval_result.json").get("summary", {})
    usages = [call.get("usage") or {} for call in calls]
    inputs = [int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
              for usage in usages]
    cached = [int(usage.get("cached_tokens") or 0) for usage in usages]
    outputs = [int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
               for usage in usages]
    input_total, cached_total, output_total = sum(inputs), sum(cached), sum(outputs)
    # First request must populate the cache. Under the optimistic full-hit model,
    # every later request's complete input is billed/read as cached input.
    theoretical_cached = sum(inputs[1:]) if inputs else 0
    theoretical_uncached = inputs[0] if inputs else 0
    telemetry = bool(usages) and all("cached_tokens" in usage for usage in usages)
    status = "ok" if summary.get("is_error") is False else "error"
    if not summary:
        status = "incomplete"
    return {
        "run_id": run.name,
        "task": meta.get("task", ""),
        "model": summary.get("model") or meta.get("model", ""),
        "reasoning_effort": summary.get("reasoning_effort", ""),
        "status": status,
        "build_elapsed_s": summary.get("elapsed_s", ""),
        "model_calls": len(calls),
        "model_elapsed_s": round(sum(float(call.get("elapsed_s") or 0) for call in calls), 3),
        "input_tokens": input_total,
        "cached_input_tokens": cached_total if telemetry else "unknown",
        "uncached_input_tokens": input_total - cached_total if telemetry else "unknown",
        "output_tokens": output_total,
        "total_tokens": input_total + output_total,
        "actual_cache_rate": round(cached_total / max(input_total, 1), 4) if telemetry else "unknown",
        "theoretical_full_hit_cached_tokens": theoretical_cached,
        "theoretical_full_hit_uncached_tokens": theoretical_uncached,
        "theoretical_full_hit_rate": round(theoretical_cached / max(input_total, 1), 4),
        "cache_telemetry": telemetry,
        "combined_score": result.get("combined_score_critical", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root")
    args = parser.parse_args()
    root = Path(args.runs_root).expanduser().resolve()
    rows = [row for run in sorted(root.iterdir()) if run.is_dir()
            if (row := collect(run)) is not None]
    csv_path = root / "camel_usage_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    md_path = root / "camel_usage_report.md"
    cols = ("task", "status", "build_elapsed_s", "model_calls", "input_tokens",
            "cached_input_tokens", "output_tokens", "actual_cache_rate",
            "theoretical_full_hit_rate", "combined_score")
    lines = ["# CAMEL usage report", "", "| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
