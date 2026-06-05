#!/usr/bin/env python3
"""
Sync `- **Description:**` lines between `tasks/<task>/description.md` (source)
and the variant copies under `c0/`, `c2/`, `c3/`, `c1/pick_A/`, etc.

The single source of truth is `tasks/<task>/description.md`. Variants share
the same per-page Description prose but differ in metadata format and the
Stack section. This script only touches the `- **Description:**` line for
each page; everything else is preserved verbatim.

Modes:
  pull   <task>            — copy variant's Description lines INTO source.
                              Default variant: c0. Use this once when sub-agents
                              already edited variants and we need to lift the
                              changes back to source.
  push   <task>            — copy source's Description lines to ALL variants.
                              Use this after editing source to keep variants
                              in sync.
  pull-all                 — pull for every task that has anchors.
  push-all                 — push for every task.

Examples:
  python3 tools/sync_descriptions.py pull 1_newsletter
  python3 tools/sync_descriptions.py pull-all
  python3 tools/sync_descriptions.py push 8_ecommerce
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TASKS_ROOT = Path(__file__).parent.parent.resolve()
VARIANTS = ["c0", "c2", "c3", "c1/pick_A", "c1/pick_B", "c1/pick_C"]

# Header pattern: "### 1. Home" or "### 1. Single-post (Post detail)"
_HEAD_RE = re.compile(r"^###\s+(\d+)\.\s*(.+?)\s*$")
_DESC_PREFIX = "- **Description:**"


def extract_descriptions(path: Path) -> dict[int, str]:
    """Return {page_idx: description_line} keyed by the `### N.` page index."""
    out: dict[int, str] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _HEAD_RE.match(line)
        if m:
            current = int(m.group(1))
        elif current is not None and line.startswith(_DESC_PREFIX):
            out[current] = line
            current = None  # only first description per page
    return out


def replace_descriptions(path: Path, replacements: dict[int, str]) -> int:
    """Rewrite path's Description lines using replacements. Returns count."""
    text = path.read_text(encoding="utf-8")
    new_lines: list[str] = []
    current = None
    replaced = 0
    for line in text.splitlines():
        m = _HEAD_RE.match(line)
        if m:
            current = int(m.group(1))
            new_lines.append(line)
        elif current is not None and line.startswith(_DESC_PREFIX) and current in replacements:
            new_lines.append(replacements[current])
            replaced += 1
            current = None
        else:
            new_lines.append(line)
    # Preserve trailing newline
    suffix = "\n" if text.endswith("\n") else ""
    path.write_text("\n".join(new_lines) + suffix, encoding="utf-8")
    return replaced


def task_source(task: str) -> Path:
    return TASKS_ROOT / task / "description.md"


def task_variants(task: str) -> list[Path]:
    out = []
    for v in VARIANTS:
        p = TASKS_ROOT / v / task / "description.md"
        if p.exists():
            out.append(p)
    return out


def pull(task: str, prefer_variant: str = "c0") -> None:
    src = task_source(task)
    if not src.exists():
        print(f"[pull] {task}: no source description.md, skip"); return
    pref = TASKS_ROOT / prefer_variant / task / "description.md"
    if not pref.exists():
        # fall back to first available variant
        variants = task_variants(task)
        if not variants:
            print(f"[pull] {task}: no variants found, skip"); return
        pref = variants[0]
    descs = extract_descriptions(pref)
    if not descs:
        print(f"[pull] {task}: variant has no Description lines"); return
    n = replace_descriptions(src, descs)
    print(f"[pull] {task}: pulled {n} description lines from {pref.relative_to(TASKS_ROOT)} → source")


def push(task: str) -> None:
    src = task_source(task)
    if not src.exists():
        print(f"[push] {task}: no source description.md, skip"); return
    descs = extract_descriptions(src)
    if not descs:
        print(f"[push] {task}: source has no Description lines"); return
    targets = task_variants(task)
    if not targets:
        print(f"[push] {task}: no variants found"); return
    total = 0
    for vp in targets:
        n = replace_descriptions(vp, descs)
        rel = vp.relative_to(TASKS_ROOT)
        print(f"[push] {task}: → {rel}  ({n} lines)")
        total += n
    print(f"[push] {task}: total {total} replacements across {len(targets)} variants")


def all_tasks() -> list[str]:
    return sorted(
        d.name for d in TASKS_ROOT.iterdir()
        if d.is_dir() and d.name[:1].isdigit() and (d / "description.md").exists()
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "pull-all":
        for t in all_tasks():
            pull(t)
    elif cmd == "push-all":
        for t in all_tasks():
            push(t)
    elif cmd in ("pull", "push") and len(sys.argv) >= 3:
        fn = pull if cmd == "pull" else push
        fn(sys.argv[2])
    else:
        print(__doc__)
        return 2


if __name__ == "__main__":
    main()
