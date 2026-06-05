#!/usr/bin/env python3
"""
Figma JSON Structure Extractor
Scans a folder (recursively) for Figma JSON files and produces
a _structure-only version of each, stripping all render fields.

Usage:
    python figma_extract_structure.py <target_folder>
"""

import json
import sys
from pathlib import Path

# ── Fields to KEEP ────────────────────────────────────────────────────────────

KEEP_ALWAYS = {
    "id", "name", "type", "children",
    # Position — y is most important (vertical order), keep full bbox
    "absoluteBoundingBox",
    # Component variant name (raw componentId UUID is meaningless without library)
    "componentProperties",
    # Text content
    "characters",
    # Layout direction only — HORIZONTAL vs VERTICAL is meaningful
    # sizing/alignment/padding details dropped: zero or near-zero entropy in practice
    "layoutMode",
    # Interactions — only kept when non-empty
    "interactions",
}

# TEXT-node only: typographic fields for heading-level detection
KEEP_TEXT_ONLY = {
    "style",   # trimmed to fontSize, fontWeight, fontFamily only
}

# Dropped fields (documentation only — filter is whitelist-based,
# so any unlisted field is automatically excluded):
#
# Zero-entropy (single value across entire file):
#   counterAxisAlignItems, counterAxisSizingMode, primaryAxisSizingMode,
#   layoutWrap, layoutGrow (98% = 0)
#
# No meaning without component library:
#   componentId, componentPropertyReferences
#
# Empty in 80%+ of nodes:
#   overrides (81% empty list)
#
# Not useful for page-structure understanding:
#   layoutAlign, layoutSizingHorizontal, layoutSizingVertical,
#   itemSpacing, paddingTop/Bottom/Left/Right, locked, booleanOperation
#
# Pure render / visual:
#   blendMode, fills, strokes, strokeWeight, strokeAlign,
#   complexStrokeProperties, individualStrokeWeights,
#   effects, styles, background, backgroundColor,
#   absoluteRenderBounds, scrollBehavior,
#   clipsContent, cornerRadius, cornerSmoothing,
#   characterStyleOverrides, styleOverrideTable,
#   lineTypes, lineIndentations, layoutVersion,
#   fillOverrideTable, boundVariables,
#   transitionNodeID, transitionDuration, transitionEasing,
#   uniformScaleFactor, opacity, rotation,
#   exportSettings, layoutGrids, constraints


# ── Core filter ───────────────────────────────────────────────────────────────

def filter_node(node: dict):
    """
    Recursively keep only structural fields from a Figma node.
    Returns None if the node is invisible — caller should drop it.
    """
    # Prune hidden subtrees entirely
    if node.get("visible") is False:
        return None

    is_text = node.get("type") == "TEXT"
    allowed = KEEP_ALWAYS | (KEEP_TEXT_ONLY if is_text else set())

    result = {}
    for key, value in node.items():
        if key not in allowed:
            continue

        if key == "children" and isinstance(value, list):
            filtered = [filter_node(c) for c in value]
            result[key] = [c for c in filtered if c is not None]

        elif key == "interactions" and isinstance(value, list):
            if value:   # drop empty interaction arrays
                result[key] = value

        elif key == "style" and isinstance(value, dict):
            keep = {"fontSize", "fontWeight", "fontFamily"}
            trimmed = {k: v for k, v in value.items() if k in keep}
            if trimmed:
                result[key] = trimmed

        else:
            result[key] = value

    return result


# ── File processing ───────────────────────────────────────────────────────────

def process_file(src: Path) -> tuple[int, int]:
    """
    Load a Figma JSON, filter it, write _structure-only file.
    Returns (original_bytes, new_bytes).
    """
    raw = src.read_bytes()
    data = json.loads(raw)

    # Support both raw node export and { "document": ... } wrapper
    if "document" in data:
        data["document"] = filter_node(data["document"])
    else:
        data = filter_node(data)

    dst = src.with_name(src.stem + "_structure-only" + src.suffix)
    out = json.dumps(data, ensure_ascii=False, indent=2)
    dst.write_text(out, encoding="utf-8")

    return len(raw), len(out.encode("utf-8"))


def scan_and_process(folder: Path):
    json_files = [
        p for p in folder.rglob("*.json")
        if "_structure-only" not in p.stem      # skip already-processed files
    ]

    if not json_files:
        print(f"No JSON files found under {folder}")
        return

    total_before = total_after = 0
    for p in sorted(json_files):
        try:
            before, after = process_file(p)
            total_before += before
            total_after += after
            ratio = (1 - after / before) * 100 if before else 0
            dst_name = p.stem + "_structure-only" + p.suffix
            print(f"  ✓ {p.relative_to(folder)}")
            print(f"      → {dst_name}  "
                  f"{before/1024:.1f} KB → {after/1024:.1f} KB  (-{ratio:.0f}%)")
        except Exception as e:
            print(f"  ✗ {p.relative_to(folder)}  ERROR: {e}")

    overall = (1 - total_after / total_before) * 100 if total_before else 0
    print(f"\nDone. {len(json_files)} file(s) processed.")
    print(f"Total: {total_before/1024:.1f} KB → {total_after/1024:.1f} KB  (-{overall:.0f}%)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python figma_extract_structure.py <target_folder>")
        sys.exit(1)

    target = Path(sys.argv[1]).resolve()
    if not target.is_dir():
        print(f"Error: '{target}' is not a directory.")
        sys.exit(1)

    print(f"Scanning: {target}\n")
    scan_and_process(target)
