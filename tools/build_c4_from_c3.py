#!/usr/bin/env python3
"""
build_c4_from_c3.py — generate the c4 variant by mirroring c3 task folders
and stripping the trailing "## Stack: ...\n```bash\n...\n```" section from
each description.md, so the model isn't told which framework to use.

c4 keeps everything else from c3 unchanged: mockup PNGs, Figma structure
JSONs, description text up to the Stack section, the same Bootstrap rule
elsewhere if any. The model picks its own stack based on the mockup +
description's semantic intent.

Usage:
    python3 build_c4_from_c3.py                  # all c3 tasks → c4
    python3 build_c4_from_c3.py --tasks 4_forum  # only this task
    python3 build_c4_from_c3.py --force          # overwrite existing c4 task folders
    python3 build_c4_from_c3.py --diff           # show what would be stripped (dry run)
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
C3 = ROOT / "c3"
C4 = ROOT / "c4"

# Matches "## Stack:" heading line through to end of file. The whole trailing
# block (heading + code fence + anything after) is stripped.
STACK_HEADER_RE = re.compile(r"\n##\s+Stack\b.*", re.DOTALL | re.IGNORECASE)


def strip_stack_section(text: str) -> tuple[str, str | None]:
    """Return (stripped_text, removed_section). removed_section is None if no Stack heading found."""
    m = STACK_HEADER_RE.search(text)
    if not m:
        return text, None
    stripped = text[: m.start()].rstrip() + "\n"
    return stripped, text[m.start():].strip()


def find_c3_tasks() -> list[Path]:
    if not C3.exists():
        return []
    return sorted(t for t in C3.iterdir() if t.is_dir() and (t / "description.md").is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="filter by task name (e.g. 4_forum 1_newsletter)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing c4/<task>/ folders")
    parser.add_argument("--diff", action="store_true",
                        help="dry run: show what would be stripped from each description")
    args = parser.parse_args()

    tasks = find_c3_tasks()
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t.name in wanted]
    if not tasks:
        print("[c4] no matching c3 tasks found", file=sys.stderr)
        return 1

    print(f"[c4] {len(tasks)} task(s) to process (mode={'diff' if args.diff else 'build'})")
    print()

    n_built = n_skipped = n_no_stack = 0
    for c3_task in tasks:
        c4_task = C4 / c3_task.name
        rel_c4 = c4_task.relative_to(ROOT)

        desc_text = (c3_task / "description.md").read_text(encoding="utf-8")
        stripped, removed = strip_stack_section(desc_text)
        if removed is None:
            print(f"[{rel_c4}] no '## Stack' heading found in description.md — copying unchanged")
            n_no_stack += 1
        else:
            removed_short = "  ".join(line for line in removed.splitlines()[:4])
            print(f"[{rel_c4}] strip: {removed_short[:120]}{'…' if len(removed_short) > 120 else ''}")

        if args.diff:
            continue

        if c4_task.exists() and not args.force:
            print(f"  exists — skip (use --force to overwrite)")
            n_skipped += 1
            continue

        if c4_task.exists():
            shutil.rmtree(c4_task)

        # Mirror everything except description.md verbatim
        c4_task.mkdir(parents=True, exist_ok=True)
        for src in c3_task.iterdir():
            tgt = c4_task / src.name
            if src.is_dir():
                shutil.copytree(src, tgt, dirs_exist_ok=True)
            elif src.name != "description.md":
                shutil.copy2(src, tgt)

        # Write stripped description
        (c4_task / "description.md").write_text(stripped, encoding="utf-8")

        n_files = sum(1 for _ in c4_task.rglob("*") if _.is_file())
        print(f"  ok: {n_files} files in {rel_c4}")
        n_built += 1
        print()

    print("=" * 60)
    if args.diff:
        print(f" Diff complete (no files written). {len(tasks) - n_no_stack} would have stack section stripped, {n_no_stack} have no stack section.")
    else:
        print(f" Done. built={n_built}  skipped={n_skipped}  no_stack={n_no_stack}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
