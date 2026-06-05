#!/usr/bin/env python3
"""
Build a cross-page logical-element index from human interaction annotations.

Input:  <inputs_dir>/interaction/<page>_human_interaction_annotation.json  (one per page)
Output: <inputs_dir>/interaction_index.json

Two annotations are grouped into ONE logical_element when they share the same
Figma source-component leaf id (`node.id` segment after the last `;`). Figma
gives every instance of a component the same source leaf, so this grouping
deterministically identifies "this is the same logical UI element appearing
on N pages" — without any heuristic guessing.

The output gives the agent a flat list of N logical elements (typically much
smaller than total annotations) plus an annotation→logical_id map that the
eval uses to look up which testid each annotation should resolve to.

Usage:
    python3 build_interaction_index.py <inputs_dir>
"""

from __future__ import annotations

import json
import sys
from collections import Counter as collections_counter
from collections import defaultdict
from pathlib import Path
from typing import Any


def leaf_of(node_id: str) -> str:
    if not node_id:
        return ""
    return node_id.rsplit(";", 1)[-1]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: build_interaction_index.py <inputs_dir>", file=sys.stderr)
        return 2
    inputs_dir = Path(sys.argv[1]).resolve()
    interaction_dir = inputs_dir / "interaction"
    out_path = inputs_dir / "interaction_index.json"

    if not interaction_dir.is_dir():
        # Some tasks don't have interaction annotations; emit an empty index so
        # downstream code can read uniformly.
        out_path.write_text(
            json.dumps(
                {
                    "logical_elements": [],
                    "annotation_to_logical": {},
                    "stats": {
                        "total_annotations": 0,
                        "total_logical_elements": 0,
                        "compression_ratio": 0.0,
                        "interaction_dir_present": False,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[indexer] no interaction/ dir; wrote empty index to {out_path}")
        return 0

    # Load every annotation and tag with its (page, id).
    flat: list[dict[str, Any]] = []
    for f in sorted(interaction_dir.glob("*_human_interaction_annotation.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[indexer] WARN: skip {f.name}: {e}", file=sys.stderr)
            continue
        page = doc.get("page_name") or f.stem.replace("_human_interaction_annotation", "")
        for a in doc.get("annotations", []) or []:
            node_id = (a.get("node") or {}).get("id", "") or ""
            flat.append(
                {
                    "page": page,
                    "id": a.get("id"),
                    "type": a.get("type"),
                    "subtype": a.get("subtype"),
                    "interactable": bool(a.get("interactable", True)),
                    "navigateTo": a.get("navigateTo"),
                    "node_id": node_id,
                    "leaf": leaf_of(node_id),
                    "bbox_png": a.get("bbox_png") or {},
                    "reasoning": (a.get("reasoning") or "").strip(),
                    "note": (a.get("note") or "").strip(),
                }
            )

    if not flat:
        print(f"[indexer] WARN: no annotations found under {interaction_dir}", file=sys.stderr)

    # Group by leaf id. Annotations without a leaf become their own group
    # (one logical_id each).
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_leaf_counter = 0
    for a in flat:
        if a["leaf"]:
            key = f"leaf:{a['leaf']}"
        else:
            no_leaf_counter += 1
            key = f"orphan:{a['page']}#{a['id']}"
        groups[key].append(a)

    # Within a group, optionally split if subtypes disagree (heuristic safeguard
    # against Figma reusing one component for two semantically distinct uses).
    refined_groups: list[list[dict[str, Any]]] = []
    for key, members in groups.items():
        # Bucket by subtype (treat None as its own bucket).
        by_subtype: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for m in members:
            by_subtype[m["subtype"]].append(m)
        # If the leaf has only one subtype across pages, keep one group.
        # Otherwise, split — these are likely different logical elements that
        # happen to reuse the same source component.
        if len(by_subtype) == 1:
            refined_groups.append(members)
        else:
            for sub_members in by_subtype.values():
                refined_groups.append(sub_members)

    # ----- Second pass: fuzzy merge for tasks where Figma didn't use Components ---
    # Some tasks (8_ecommerce, 5_travel-booking, 9_project-management, 4_forum)
    # have flat node.ids ('106:524', not 'I...;...;leaf'), meaning the source
    # mockup didn't reuse Components for header/footer/etc. Leaf-id grouping
    # leaves every annotation as a singleton in those tasks.
    #
    # For singleton groups whose member has a flat node_id (no ";"), attempt a
    # fuzzy merge using a coarse fingerprint:
    #   (type, subtype, bbox_x_bucketed, bbox_y_bucketed, navigateTo.name, reasoning_excerpt)
    # bbox bucketing: round x and y to 40-px multiples to absorb minor
    # cross-page misalignment of nominally-the-same element.
    BUCKET_PX = 40

    def fuzzy_key(m: dict[str, Any]) -> tuple:
        bb = m.get("bbox_png") or {}
        x = round(bb.get("x", 0) / BUCKET_PX) * BUCKET_PX
        y = round(bb.get("y", 0) / BUCKET_PX) * BUCKET_PX
        nt = m.get("navigateTo")
        nav_name = nt.get("name") if isinstance(nt, dict) else None
        # Reasoning matters: empty reasonings cluster together by position, but
        # nonempty reasonings discriminate. We truncate to limit noise from
        # tiny formatting variations.
        reasoning_key = (m["reasoning"] or "")[:120]
        return (m["type"], m["subtype"], x, y, nav_name, reasoning_key)

    keepers: list[list[dict[str, Any]]] = []
    fuzzy_pool: list[dict[str, Any]] = []
    for g in refined_groups:
        if len(g) == 1 and ";" not in (g[0]["node_id"] or ""):
            fuzzy_pool.append(g[0])
        else:
            keepers.append(g)

    fuzzy_buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for m in fuzzy_pool:
        fuzzy_buckets[fuzzy_key(m)].append(m)

    fuzzy_merge_count = 0
    for bucket_members in fuzzy_buckets.values():
        # Only count as a "merge" if it crosses pages.
        if len({m["page"] for m in bucket_members}) > 1:
            fuzzy_merge_count += 1
        keepers.append(bucket_members)

    refined_groups = keepers

    # Sort groups by descending fan-out (most cross-page elements first), then
    # by first occurrence's (page, id) for stability.
    refined_groups.sort(
        key=lambda g: (-len({m["page"] for m in g}), g[0]["page"], g[0]["id"] or 0)
    )

    logical_elements: list[dict[str, Any]] = []
    annotation_to_logical: dict[str, str] = {}

    for idx, members in enumerate(refined_groups, start=1):
        logical_id = f"le_{idx:03d}"
        pages_set = sorted({m["page"] for m in members})
        # How was this group formed? Useful for the agent and eval to know
        # whether to trust the grouping fully (leaf) or treat as approximate (fuzzy).
        if any(";" in (m["node_id"] or "") for m in members):
            grouping_method = "figma_leaf_id"
        elif len(members) > 1:
            grouping_method = "fuzzy_position_match"
        else:
            grouping_method = "singleton"
        # Pick the most informative reasoning (longest non-empty) as
        # representative, since some occurrences have empty reasoning.
        reasoning_excerpt = max(
            (m["reasoning"] for m in members if m["reasoning"]),
            key=len,
            default="",
        )[:240]
        # navigateTo may differ across pages (breadcrumb back-link, etc.) —
        # record whether it's consistent and list distinct destinations.
        nav_targets = set()
        for m in members:
            nt = m.get("navigateTo")
            if nt:
                nav_targets.add(json.dumps(nt, sort_keys=True))
        navigate_consistent = len(nav_targets) <= 1
        # Use the first member's bbox as a representative.
        rep_bbox = members[0]["bbox_png"] or {}
        # Use the first member's leaf if any (orphans show empty leaf).
        leaf = members[0]["leaf"]

        logical_elements.append(
            {
                "logical_id": logical_id,
                "leaf_node_id": leaf,
                "grouping_method": grouping_method,
                "type": members[0]["type"],
                "subtype": members[0]["subtype"],
                "interactable": all(m["interactable"] for m in members),
                "occurrences": [
                    {"page": m["page"], "id": m["id"]}
                    for m in sorted(members, key=lambda m: (m["page"], m["id"] or 0))
                ],
                "page_count": len(pages_set),
                "pages": pages_set,
                "navigateTo_consistent": navigate_consistent,
                "navigateTo_targets": [json.loads(t) for t in sorted(nav_targets)],
                "common_reasoning_excerpt": reasoning_excerpt,
                "representative_bbox_png": rep_bbox,
            }
        )
        for m in members:
            annotation_to_logical[f"{m['page']}-{m['id']}"] = logical_id

    by_method = collections_counter(le["grouping_method"] for le in logical_elements)
    output = {
        "logical_elements": logical_elements,
        "annotation_to_logical": annotation_to_logical,
        "stats": {
            "total_annotations": len(flat),
            "total_logical_elements": len(logical_elements),
            "compression_ratio": round(
                len(logical_elements) / max(1, len(flat)), 3
            ),
            "orphans_without_leaf_id": no_leaf_counter,
            "interaction_dir_present": True,
            "groups_by_method": dict(by_method),
            "fuzzy_cross_page_merges": fuzzy_merge_count,
        },
    }
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    s = output["stats"]
    print(
        f"[indexer] {s['total_annotations']} annotations → "
        f"{s['total_logical_elements']} logical elements "
        f"(compression {s['compression_ratio']:.0%}, "
        f"{s['orphans_without_leaf_id']} orphans). "
        f"Wrote {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
