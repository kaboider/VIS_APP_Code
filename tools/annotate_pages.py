#!/usr/bin/env python3
"""
Render annotated page PNGs that overlay every interactive logical_element's
bounding box on top of the original mockup, with the logical_id labeled.

This bridges the agent-visual gap: when `interaction_summary.md` says
`le_030 | navigate | (no description)`, the agent can open
`inputs/pages_annotated/<page>.annotated.png` and instantly see WHICH button
le_030 is on the page.

Color scheme (matches the summary tier system):
  - critical  → red box     (must implement + must work)
  - bonus     → orange box   (must have testid; behavior unconstrained)
  - skip      → muted gray   (decorative click_dead — drawn lightly for context)

Output:
  <inputs_dir>/pages_annotated/<page>.annotated.png

Usage:
    python3 annotate_pages.py <inputs_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow required. pip install Pillow", file=sys.stderr)
    sys.exit(2)


CRITICAL_CLICK_SUBTYPES = {
    "click_popout","click_external","click_upload_file","click_social_oauth",
    "click_play_music","click_vol","click_next_misuc","click_pre_misuc",
}


def tier_for(le: dict) -> str:
    if not le.get("interactable", True):
        return "skip"
    t = le.get("type")
    if t in ("navigate","input","toggle"):
        return "critical"
    if t == "click":
        st = le.get("subtype")
        if st == "click_dead": return "skip"
        if st in CRITICAL_CLICK_SUBTYPES: return "critical"
        if st == "click_unknown_nav": return "bonus"
        if st and st.startswith("click_"): return "critical"
    return "bonus"


# RGBA: alpha so we can see the design underneath
COLORS = {
    "critical": (228, 26, 28, 230),    # red
    "bonus":    (255, 127, 0, 220),    # orange
    "skip":     (160, 160, 160, 140),  # muted gray
}
TEXT_COLORS = {
    "critical": (255, 255, 255, 255),
    "bonus":    (255, 255, 255, 255),
    "skip":     (255, 255, 255, 220),
}


def get_font(size: int) -> ImageFont.FreeTypeFont:
    # Try a few common system fonts; fall back to default.
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",          # macOS
        "/System/Library/Fonts/SFNS.ttf",                        # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",           # Linux
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_box(draw: ImageDraw.ImageDraw, bbox: dict, label: str, tier: str,
             font: ImageFont.FreeTypeFont) -> None:
    x = int(bbox.get("x", 0))
    y = int(bbox.get("y", 0))
    w = int(bbox.get("w", 0))
    h = int(bbox.get("h", 0))
    if w <= 0 or h <= 0:
        return

    line_color = COLORS.get(tier, COLORS["bonus"])
    text_color = TEXT_COLORS.get(tier, TEXT_COLORS["bonus"])
    width = 3 if tier == "critical" else (2 if tier == "bonus" else 1)

    # Box
    draw.rectangle([x, y, x + w, y + h], outline=line_color, width=width)

    # Label background — small chip on the top-left of the box
    try:
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
    except AttributeError:
        # Pillow <10
        tw, th = font.getsize(label)
    pad = 3
    chip_x1 = x
    chip_y1 = max(0, y - th - pad * 2)
    chip_x2 = x + tw + pad * 2
    chip_y2 = chip_y1 + th + pad * 2
    draw.rectangle([chip_x1, chip_y1, chip_x2, chip_y2], fill=line_color)
    draw.text((chip_x1 + pad, chip_y1 + pad), label, fill=text_color, font=font)


def main() -> int:
    # Args: <inputs_dir> [<index_path>]
    if len(sys.argv) not in (2, 3):
        print("usage: annotate_pages.py <inputs_dir> [<index_path>]", file=sys.stderr)
        return 2
    inputs_dir = Path(sys.argv[1]).resolve()
    pages_dir = inputs_dir / "pages"
    out_dir = inputs_dir / "pages_annotated"
    if len(sys.argv) == 3:
        idx_path = Path(sys.argv[2]).resolve()
    else:
        idx_path = inputs_dir / "interaction_index.json"

    if not pages_dir.is_dir():
        print(f"No pages/ at {pages_dir}", file=sys.stderr); return 1
    if not idx_path.exists():
        print(f"No interaction_index.json at {idx_path}", file=sys.stderr); return 1
    out_dir.mkdir(exist_ok=True)

    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    interaction_dir = inputs_dir / "interaction"

    # Build (page, ann_id) → annotation lookup
    ann_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for f in sorted(interaction_dir.glob("*_human_interaction_annotation.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        page = doc.get("page_name") or f.stem.replace("_human_interaction_annotation", "")
        for a in doc.get("annotations", []):
            ann_lookup[(page, a.get("id"))] = a

    # Build: per page, list of (logical_id, ann, tier)
    per_page: dict[str, list[tuple[str, dict, str]]] = {}
    for le in idx["logical_elements"]:
        t = tier_for(le)
        for occ in le["occurrences"]:
            ann = ann_lookup.get((occ["page"], occ["id"]))
            if not ann:
                continue
            per_page.setdefault(occ["page"], []).append((le["logical_id"], ann, t))

    # Render
    rendered = 0
    for page_name, items in sorted(per_page.items()):
        src_png = pages_dir / f"{page_name}.png"
        if not src_png.exists():
            print(f"  [skip] {page_name}: no PNG at {src_png}")
            continue

        base = Image.open(src_png).convert("RGBA")
        # Translucent overlay on its own layer so boxes don't bake into base
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Font size scales with image height; cap at 18px so it stays readable
        font_size = max(11, min(18, base.height // 80))
        font = get_font(font_size)

        # Render in tier order so critical is drawn last (on top)
        order = {"skip": 0, "bonus": 1, "critical": 2}
        items_sorted = sorted(items, key=lambda x: order.get(x[2], 0))
        for lid, ann, t in items_sorted:
            draw_box(draw, ann.get("bbox_png") or {}, lid, t, font)

        composed = Image.alpha_composite(base, overlay).convert("RGB")
        out_path = out_dir / f"{page_name}.annotated.png"
        composed.save(out_path, format="PNG", optimize=True)
        rendered += 1

    print(f"[annotate] wrote {rendered} annotated PNGs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
