#!/usr/bin/env python3
"""
Reconstruct manifest.json from a task's interaction annotations + pages PNGs.

Some tasks (8_ecommerce, 10_streaming) ship with an incomplete manifest.json
that only lists a subset of pages. This script scans:

  <task>/interaction/*_human_interaction_annotation.json
  <task>/pages/*.png

and emits a fresh manifest.json covering every page actually present.

Usage:
    python3 repair_manifest.py <task_dir>          # writes manifest.json (in place)
    python3 repair_manifest.py <task_dir> --check  # report drift, do NOT write

The annotation file has `page_name`, `figma_meta` (origin_x/y, figma_w/h).
We try to preserve any extra metadata (file_key, node_id) from existing entries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    args = sys.argv[1:]
    check_only = "--check" in args
    args = [a for a in args if a != "--check"]
    if len(args) != 1:
        print("usage: repair_manifest.py <task_dir> [--check]", file=sys.stderr)
        return 2

    task_dir = Path(args[0]).resolve()
    interaction = task_dir / "interaction"
    pages = task_dir / "pages"
    manifest_path = task_dir / "manifest.json"

    if not interaction.is_dir() or not pages.is_dir():
        print(f"missing interaction/ or pages/ in {task_dir}", file=sys.stderr)
        return 1

    # Existing manifest entries by png filename (so we preserve file_key etc.)
    existing: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for e in data:
                    if e.get("png"):
                        existing[e["png"]] = e
        except Exception:
            pass

    # Build entries from annotation files (canonical source of page_name + figma_meta)
    entries = []
    seen_pngs: set[str] = set()
    for f in sorted(interaction.glob("*_human_interaction_annotation.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        page_name = doc.get("page_name") or f.stem.replace("_human_interaction_annotation", "")
        png_name = doc.get("source_png") or f"{page_name}.png"
        seen_pngs.add(png_name)
        # Strip leading "NN_" if it appears in page_name to get a clean human label,
        # but preserve the prefixed form for png consistency.
        idx = None
        if "_" in page_name and page_name.split("_", 1)[0].isdigit():
            idx = int(page_name.split("_", 1)[0])
            human_name = page_name.split("_", 1)[1].replace("_", " ")
        else:
            human_name = page_name.replace("_", " ")
        ex = existing.get(png_name, {})
        entries.append({
            "index": ex.get("index", idx if idx is not None else len(entries) + 1),
            "page_name": ex.get("page_name", human_name),
            "file_key": ex.get("file_key", ""),
            "node_id": ex.get("node_id", ""),
            "png": png_name,
            "structure_json": ex.get("structure_json", png_name.replace(".png", "_structure-only.json")),
            "figma_meta": doc.get("figma_meta") or {},
        })

    # Also include any PNGs in pages/ that don't have an annotation file yet
    pngs_on_disk = {p.name for p in pages.glob("*.png")}
    for png in sorted(pngs_on_disk - seen_pngs):
        page_name = png.replace(".png", "")
        idx = None
        if "_" in page_name and page_name.split("_", 1)[0].isdigit():
            idx = int(page_name.split("_", 1)[0])
            human_name = page_name.split("_", 1)[1].replace("_", " ")
        else:
            human_name = page_name.replace("_", " ")
        ex = existing.get(png, {})
        entries.append({
            "index": ex.get("index", idx if idx is not None else len(entries) + 1),
            "page_name": ex.get("page_name", human_name),
            "file_key": ex.get("file_key", ""),
            "node_id": ex.get("node_id", ""),
            "png": png,
            "structure_json": ex.get("structure_json", png.replace(".png", "_structure-only.json")),
            "figma_meta": ex.get("figma_meta", {}),
            "_note": "no annotation file found for this PNG",
        })

    entries.sort(key=lambda e: (e.get("index") or 999, e.get("png", "")))

    n_existing = len(existing)
    n_new = len(entries)
    delta = n_new - n_existing

    if check_only:
        print(f"[manifest] {task_dir.name}: existing={n_existing} actual={n_new} (delta {delta:+d})")
        if delta != 0 or not existing:
            print(f"  → would rewrite. Pages currently in manifest:")
            for png in sorted(existing.keys()):
                print(f"     {png}")
            print(f"  → Pages we'd add/keep:")
            for e in entries:
                marker = "+ NEW" if e["png"] not in existing else "      "
                print(f"     {marker}  {e['png']}")
        return 0

    manifest_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"[manifest] {task_dir.name}: rewrote {manifest_path} with {n_new} entries (was {n_existing}, delta {delta:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
