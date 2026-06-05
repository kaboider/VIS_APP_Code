#!/usr/bin/env python3
"""
Vision-verify the fuzzy-grouped logical elements in interaction_index.json.

For tasks whose Figma source did not use Components properly (so leaf-id
grouping found nothing), the indexer falls back to position-based fuzzy
matching. Those `fuzzy_position_match` groups are roughly 95% accurate but
worth double-checking. This script:

  1. Reads <inputs_dir>/interaction_index.json
  2. For every group with grouping_method == "fuzzy_position_match" and
     page_count >= 2, crops the bbox from each occurrence's page PNG.
  3. Stacks the crops side-by-side into one PNG.
  4. Asks Claude (Sonnet, via the CLI) whether they show the SAME logical
     UI element or different ones.
  5. Writes <inputs_dir>/interaction_index.verified.json with:
       - confirmed groups → grouping_method = "vision_verified_match"
       - rejected groups → split back into singletons
       - unverifiable / errored → kept as-is, flagged

Usage:
    python3 vision_verify_index.py <inputs_dir>
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:
    print(
        "ERROR: Pillow is required. Install with: pip install Pillow",
        file=sys.stderr,
    )
    sys.exit(2)


CLAUDE_BIN = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or "claude"
MODEL = os.environ.get("VISION_MODEL", "sonnet")
MARGIN_PX = 20  # pixels of context to include around each bbox crop


def crop_bbox(page_png: Path, bbox: dict, margin: int = MARGIN_PX) -> Image.Image:
    img = Image.open(page_png).convert("RGB")
    x = max(0, int(bbox.get("x", 0)) - margin)
    y = max(0, int(bbox.get("y", 0)) - margin)
    x2 = min(img.width, int(bbox.get("x", 0)) + int(bbox.get("w", 0)) + margin)
    y2 = min(img.height, int(bbox.get("y", 0)) + int(bbox.get("h", 0)) + margin)
    return img.crop((x, y, x2, y2))


def stack_horizontal(images: list[Image.Image], gap: int = 12, bg=(240, 240, 240)) -> Image.Image:
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    cx = 0
    for im in images:
        # Center vertically
        cy = (h - im.height) // 2
        canvas.paste(im, (cx, cy))
        cx += im.width + gap
    return canvas


def build_prompt(group: dict, occurrences: list[dict[str, Any]]) -> str:
    lines = [
        "You are verifying whether several image crops, taken from different pages of the same web-app mockup, show the SAME logical UI element (e.g., header logo, nav link, footer button) or DIFFERENT elements that just happened to land in similar positions.",
        "",
        f"Number of crops to compare: {len(occurrences)}",
        f"Each crop is from a different page; they are stacked side-by-side in the attached image, in this left-to-right order:",
        "",
    ]
    for i, occ in enumerate(occurrences, start=1):
        lines.append(f"  {i}. Page `{occ['page']}` annotation #{occ['id']} — reasoning: {occ.get('reasoning','')[:140]!r}")
    lines += [
        "",
        f"Annotation type: {group.get('type')}",
        f"Subtype: {group.get('subtype')}",
        f"NavigateTo target consistent across pages: {group.get('navigateTo_consistent')}",
        "",
        "Decide: are ALL crops showing the SAME logical UI element (same icon/text/role, just placed on different pages of the same site)?",
        "",
        "Answer in this exact JSON format on one line, no markdown:",
        '{"verdict":"SAME"|"DIFFERENT"|"UNCLEAR","reason":"<one sentence>"}',
    ]
    return "\n".join(lines)


def call_claude_vision(prompt: str, image_path: Path, timeout_s: int = 90) -> dict:
    """Run claude CLI with one image attached, return parsed JSON answer."""
    # Embed the image path inline in the prompt; --add-dir gives read access.
    full_prompt = (
        f"{prompt}\n\nImage to inspect: {image_path}\n"
        "Use the Read tool to view the image, then answer."
    )
    try:
        proc = subprocess.run(
            [
                CLAUDE_BIN,
                "--model", MODEL,
                "--add-dir", str(image_path.parent),
                "--dangerously-skip-permissions",
                "--output-format", "json",
                "-p", full_prompt,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"verdict": "UNCLEAR", "reason": "claude CLI timed out", "_error": "timeout"}
    if proc.returncode != 0:
        return {"verdict": "UNCLEAR", "reason": f"CLI err: {proc.stderr[:200]}", "_error": "cli_failed"}
    try:
        wrapper = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"verdict": "UNCLEAR", "reason": "claude returned non-JSON wrapper", "_error": "wrapper_parse"}
    text = (wrapper.get("result") or "").strip()
    # Try to find a JSON object in the result
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"verdict": "UNCLEAR", "reason": f"no JSON in answer: {text[:120]}", "_error": "answer_parse"}
    try:
        ans = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"verdict": "UNCLEAR", "reason": f"answer not valid JSON: {text[:120]}", "_error": "answer_parse"}
    return ans


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: vision_verify_index.py <inputs_dir>", file=sys.stderr)
        return 2
    inputs_dir = Path(sys.argv[1]).resolve()
    index_path = inputs_dir / "interaction_index.json"
    pages_dir = inputs_dir / "pages"
    if not index_path.exists():
        print(f"No interaction_index.json at {index_path}", file=sys.stderr)
        return 1
    if not pages_dir.is_dir():
        print(f"No pages/ dir at {pages_dir}", file=sys.stderr)
        return 1

    idx = json.loads(index_path.read_text(encoding="utf-8"))
    fuzzy_groups = [
        le for le in idx["logical_elements"]
        if le.get("grouping_method") == "fuzzy_position_match" and le["page_count"] >= 2
    ]
    print(f"[vision] {len(fuzzy_groups)} fuzzy groups to verify (out of {len(idx['logical_elements'])} total).")

    if not fuzzy_groups:
        # Nothing to do; just write a copy.
        out_path = inputs_dir / "interaction_index.verified.json"
        out_path.write_text(index_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[vision] no fuzzy groups — copied to {out_path}")
        return 0

    # Build a quick lookup: (page, ann_id) -> annotation full record (we already have reasoning, bbox in occurrences when stored, but the index occurrences only have {page,id}; reload from interaction/ files).
    interaction_dir = inputs_dir / "interaction"
    ann_lookup: dict[tuple, dict] = {}
    for f in sorted(interaction_dir.glob("*_human_interaction_annotation.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        page = doc.get("page_name") or f.stem.replace("_human_interaction_annotation", "")
        for a in doc.get("annotations", []):
            ann_lookup[(page, a.get("id"))] = a

    workdir = Path(tempfile.mkdtemp(prefix="vision_verify_"))
    print(f"[vision] working dir: {workdir}")

    decisions: dict[str, dict] = {}
    for i, le in enumerate(fuzzy_groups, start=1):
        lid = le["logical_id"]
        occs_data: list[dict] = []
        crops: list[Image.Image] = []
        skip = False
        for occ in le["occurrences"]:
            ann = ann_lookup.get((occ["page"], occ["id"]))
            if not ann:
                continue
            page_png = pages_dir / f"{occ['page']}.png"
            if not page_png.exists():
                continue
            try:
                crop = crop_bbox(page_png, ann.get("bbox_png") or {})
                # Cap crop size to keep image manageable
                crop.thumbnail((600, 400))
                crops.append(crop)
                occs_data.append({"page": occ["page"], "id": occ["id"], "reasoning": ann.get("reasoning", "")})
            except Exception as e:
                print(f"  [{lid}] crop err {occ}: {e}")
        if len(crops) < 2:
            decisions[lid] = {"verdict": "UNCLEAR", "reason": "fewer than 2 crops available", "_error": "missing_crops"}
            continue

        stacked = stack_horizontal(crops)
        stacked_path = workdir / f"{lid}.png"
        stacked.save(stacked_path)

        prompt = build_prompt(le, occs_data)
        print(f"[vision] [{i}/{len(fuzzy_groups)}] {lid} — {len(crops)} crops — calling Claude…", flush=True)
        ans = call_claude_vision(prompt, stacked_path)
        decisions[lid] = ans
        print(f"  → {ans.get('verdict')}: {ans.get('reason','')[:100]}")

    # ----- Apply decisions -----
    counter = {"SAME": 0, "DIFFERENT": 0, "UNCLEAR": 0}
    new_logical_elements: list[dict] = []
    next_orphan_idx = max(
        (int(le["logical_id"].split("_")[1]) for le in idx["logical_elements"] if le["logical_id"].startswith("le_")),
        default=0,
    )
    new_a2l: dict[str, str] = dict(idx["annotation_to_logical"])

    for le in idx["logical_elements"]:
        lid = le["logical_id"]
        if lid in decisions:
            v = decisions[lid].get("verdict", "UNCLEAR")
            counter[v] = counter.get(v, 0) + 1
            if v == "SAME":
                le2 = dict(le)
                le2["grouping_method"] = "vision_verified_match"
                le2["vision_reason"] = decisions[lid].get("reason", "")
                new_logical_elements.append(le2)
            elif v == "DIFFERENT":
                # Split back into singletons
                for occ in le["occurrences"]:
                    next_orphan_idx += 1
                    new_lid = f"le_{next_orphan_idx:03d}"
                    new_logical_elements.append(
                        {
                            **le,
                            "logical_id": new_lid,
                            "grouping_method": "vision_split_singleton",
                            "vision_reason": decisions[lid].get("reason", ""),
                            "occurrences": [occ],
                            "page_count": 1,
                            "pages": [occ["page"]],
                        }
                    )
                    new_a2l[f"{occ['page']}-{occ['id']}"] = new_lid
            else:
                # UNCLEAR — keep as fuzzy with a note
                le2 = dict(le)
                le2["grouping_method"] = "fuzzy_position_match_unclear"
                le2["vision_reason"] = decisions[lid].get("reason", "")
                new_logical_elements.append(le2)
        else:
            new_logical_elements.append(le)

    # Re-stat
    methods = {}
    for le in new_logical_elements:
        methods[le["grouping_method"]] = methods.get(le["grouping_method"], 0) + 1

    out = {
        "logical_elements": new_logical_elements,
        "annotation_to_logical": new_a2l,
        "stats": {
            **idx.get("stats", {}),
            "total_logical_elements": len(new_logical_elements),
            "groups_by_method": methods,
            "vision_verified_summary": counter,
        },
    }
    verified_path = inputs_dir / "interaction_index.verified.json"
    active_path   = inputs_dir / "interaction_index.json"
    payload = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    verified_path.write_text(payload, encoding="utf-8")
    # Overwrite the active index too, so downstream (agent prompt, eval) reads
    # the best available data without needing to know about .verified.
    active_path.write_text(payload, encoding="utf-8")

    print()
    print(f"[vision] decisions: SAME={counter['SAME']}  DIFFERENT={counter['DIFFERENT']}  UNCLEAR={counter['UNCLEAR']}")
    print(f"[vision] groups_by_method now: {methods}")
    print(f"[vision] wrote {verified_path}")
    print(f"[vision] also overwrote active {active_path}")
    print(f"[vision] crops kept at {workdir} for inspection — rm -rf when done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
