#!/usr/bin/env python3
"""
Render a Markdown digest of interaction_index.json that's optimized for an
LLM agent to consume while writing code.

The index is logical-element-keyed; it's machine-friendly but agents implement
code page-by-page. This script produces:

  inputs/interaction_summary.md

with two views:

  1. Cross-page elements (header/footer/nav) — one shared testid each, listed
     with vision-derived descriptions. Agent picks ONE testid name and reuses
     it across pages.

  2. Per-page tables — for each page, every logical_element that touches it
     (cross-page or unique). When the agent is writing pages/<page>/...,
     this is the checklist of testids to include.

Usage:
    python3 build_interaction_summary.py <inputs_dir>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PRIORITY = {
    "navigate": 0,                    # critical
    "input":    0,
    "toggle":   0,
}
CRITICAL_CLICK_SUBTYPES = {
    "click_popout","click_external","click_upload_file","click_social_oauth",
    "click_play_music","click_vol","click_next_misuc","click_pre_misuc",
}


def tier(le: dict) -> str:
    if not le.get("interactable", True):
        return "skip"
    t = le.get("type")
    if t in ("navigate","input","toggle"):
        return "critical"
    if t == "click":
        st = le.get("subtype")
        if st == "click_dead":
            return "skip"
        if st in CRITICAL_CLICK_SUBTYPES:
            return "critical"
        if st == "click_unknown_nav":
            return "bonus"
        # any other click_* → conservative critical
        if st and st.startswith("click_"):
            return "critical"
    return "bonus"


def describe(le: dict) -> str:
    """Return a one-line description of the element, preferring vision_reason."""
    vr = (le.get("vision_reason") or "").strip()
    if vr:
        return vr[:200]
    rr = (le.get("common_reasoning_excerpt") or "").strip()
    if rr:
        return rr[:200]
    return "(no description; check bbox in PNG)"


def format_subtype(le: dict) -> str:
    t = le.get("type") or "?"
    st = le.get("subtype")
    return f"{t}" if not st else f"{t}/{st}"


def format_method(le: dict) -> str:
    m = le.get("grouping_method", "?")
    return {
        "figma_leaf_id":          "leaf",
        "fuzzy_position_match":   "fuzzy",
        "vision_verified_match":  "vision✓",
        "vision_split_singleton": "vision-split",
        "fuzzy_position_match_unclear": "fuzzy?",
        "singleton":              "—",
    }.get(m, m)


def main() -> int:
    # Args: <inputs_dir> [<index_path>]
    # If index_path is omitted, look in <inputs_dir>/interaction_index.json.
    if len(sys.argv) not in (2, 3):
        print("usage: build_interaction_summary.py <inputs_dir> [<index_path>]", file=sys.stderr)
        return 2
    inputs_dir = Path(sys.argv[1]).resolve()
    if len(sys.argv) == 3:
        idx_path = Path(sys.argv[2]).resolve()
    else:
        idx_path = inputs_dir / "interaction_index.json"
    if not idx_path.exists():
        print(f"No interaction_index.json at {idx_path}", file=sys.stderr)
        return 1

    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    elems: list[dict] = idx["logical_elements"]

    # Tag every element with tier; build per-page bucket.
    per_page: dict[str, list[dict]] = defaultdict(list)
    for le in elems:
        for occ in le["occurrences"]:
            per_page[occ["page"]].append({"le": le, "occ_id": occ["id"]})

    cross_page_elems = [le for le in elems if le["page_count"] >= 2 and tier(le) != "skip"]
    cross_page_elems.sort(key=lambda x: (-x["page_count"], x["logical_id"]))

    pages_sorted = sorted(per_page.keys())

    # ---- compose markdown ----
    L: list[str] = []
    L.append("# Interaction summary\n")
    L.append("This is a human-friendly digest of `interaction_index.json` you can consume while writing each page. The JSON is the source of truth; this Markdown only re-organizes it.\n")

    s = idx.get("stats", {})
    L.append(f"- Total annotations: **{s.get('total_annotations','?')}**")
    L.append(f"- Total logical elements: **{s.get('total_logical_elements','?')}**")
    if "vision_verified_summary" in s:
        v = s["vision_verified_summary"]
        L.append(f"- Vision-verified groups: SAME={v.get('SAME',0)}, DIFFERENT={v.get('DIFFERENT',0)}, UNCLEAR={v.get('UNCLEAR',0)}")
    L.append("")

    # Tier counts
    counts = {"critical": 0, "bonus": 0, "skip": 0}
    for le in elems:
        counts[tier(le)] = counts.get(tier(le), 0) + 1
    L.append(f"**Tiers:** critical={counts['critical']} (must implement + must work), bonus={counts['bonus']} (must have testid; behavior unconstrained), skip={counts['skip']} (decorative — `click_dead`).\n")

    # ---- 1. Cross-page elements ----
    L.append("## Cross-page elements")
    L.append("These appear on multiple pages. Use ONE shared testid for each. Pick a name once and reuse it across every page in `pages` below.\n")
    if not cross_page_elems:
        L.append("_(none in this task)_\n")
    else:
        L.append("| logical_id | tier | type/subtype | grouping | pages | description |")
        L.append("|---|---|---|---|---|---|")
        for le in cross_page_elems:
            pages = ", ".join(le["pages"])
            L.append(f"| `{le['logical_id']}` | {tier(le)} | {format_subtype(le)} | {format_method(le)} | {len(le['pages'])}: {pages} | {describe(le)} |")
        L.append("")

    # ---- 2. Per-page tables ----
    L.append("## Per-page checklists")
    L.append("For each page, every logical_element that has an occurrence on it. When you implement that page's component, every `logical_id` row below MUST have its testid present in the rendered HTML.\n")

    for page in pages_sorted:
        bucket = per_page[page]
        # Sort: critical first, then bonus, then skip
        bucket.sort(key=lambda b: (
            {"critical":0, "bonus":1, "skip":2}[tier(b["le"])],
            b["le"]["logical_id"]
        ))
        critical_n = sum(1 for b in bucket if tier(b["le"]) == "critical")
        bonus_n    = sum(1 for b in bucket if tier(b["le"]) == "bonus")
        skip_n     = sum(1 for b in bucket if tier(b["le"]) == "skip")
        L.append(f"### {page} — {critical_n} critical, {bonus_n} bonus, {skip_n} skip")
        L.append("")
        L.append("| ann_id | logical_id | tier | type/subtype | shared? | description |")
        L.append("|---|---|---|---|---|---|")
        for b in bucket:
            le = b["le"]
            shared = "✓" if le["page_count"] > 1 else ""
            L.append(
                f"| #{b['occ_id']} | `{le['logical_id']}` | {tier(le)} | {format_subtype(le)} | {shared} | {describe(le)} |"
            )
        L.append("")

    out_path = inputs_dir / "interaction_summary.md"
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(
        f"[summary] wrote {out_path}  "
        f"({counts['critical']} critical, {counts['bonus']} bonus, {counts['skip']} skip; "
        f"{len(cross_page_elems)} cross-page elements)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
