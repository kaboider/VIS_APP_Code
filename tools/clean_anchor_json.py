#!/usr/bin/env python3
"""
Clean cross-task pollution from each task's `<task>_anchors.json`.

The anchor_marker.html had a localStorage-leak bug where switching tasks in
the same browser session retained anchors from previous tasks under their old
page names. Result: every exported JSON contains a mix of multiple tasks'
pages.

This script:
  1. For each `tasks/<task>/<task>_anchors.json`,
  2. reads the list of valid PNG basenames from `tasks/<task>/pages/`,
  3. keeps only anchor entries whose page key matches an actual PNG,
  4. backs up the dirty version to `<task>_anchors.raw.json`,
  5. overwrites `<task>_anchors.json` with the cleaned data.

Usage:
    python3 tools/clean_anchor_json.py            # all tasks
    python3 tools/clean_anchor_json.py 8_ecommerce # one task
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def clean_one(task_dir: Path) -> dict | None:
    f = task_dir / f"{task_dir.name}_anchors.json"
    if not f.exists():
        return None
    raw_backup = task_dir / f"{task_dir.name}_anchors.raw.json"

    data = json.loads(f.read_text(encoding="utf-8"))
    raw_anchors = data.get("anchors") or {}
    own_pages = {p.stem for p in (task_dir / "pages").glob("*.png")}

    # Filter — keep only entries whose page key matches a real PNG, and drop
    # empty arrays.
    cleaned = {p: arr for p, arr in raw_anchors.items() if p in own_pages and arr}

    raw_pages = sum(1 for v in raw_anchors.values() if v)
    raw_total = sum(len(v) for v in raw_anchors.values())
    clean_pages = len(cleaned)
    clean_total = sum(len(v) for v in cleaned.values())
    dropped_pages = sorted(set(raw_anchors.keys()) - set(cleaned.keys()) - {k for k, v in raw_anchors.items() if not v})

    if not raw_backup.exists():
        raw_backup.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out = {"task": data.get("task", task_dir.name), "anchors": cleaned}
    f.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{task_dir.name}")
    print(f"  pages:    {raw_pages:>4} → {clean_pages:>4}  (dropped {raw_pages - clean_pages})")
    print(f"  anchors:  {raw_total:>4} → {clean_total:>4}  (dropped {raw_total - clean_total})")
    if dropped_pages:
        sample = dropped_pages[:5]
        more = f"  …(+{len(dropped_pages)-5})" if len(dropped_pages) > 5 else ""
        print(f"  dropped page keys: {sample}{more}")
    print()
    return out


def main():
    tasks_root = Path(__file__).parent.parent.resolve()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    summary_total = {"tasks": 0, "pages": 0, "anchors": 0}
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name[0:1].isdigit():
            continue
        if only and task_dir.name != only:
            continue
        result = clean_one(task_dir)
        if result:
            summary_total["tasks"] += 1
            summary_total["pages"] += len(result["anchors"])
            summary_total["anchors"] += sum(len(v) for v in result["anchors"].values())
    print("=" * 60)
    print(f"TOTAL: {summary_total['tasks']} tasks, "
          f"{summary_total['pages']} pages with anchors, "
          f"{summary_total['anchors']} anchors")


if __name__ == "__main__":
    main()
