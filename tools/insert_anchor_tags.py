#!/usr/bin/env python3
"""
Take each task's <task>_anchors.json (exported from anchor_marker.html),
filter out cross-task localStorage pollution, then insert `<testid>` inline
markers into every variant's description.md.

For each anchor, we pick an insertion point inside the page's "Description"
prose by keyword-match on (testid words + reasoning words). The marker is
placed immediately after the first keyword hit, e.g.:

  before:  "...a centered purple BlogSprout logo with a Subscribe CTA..."
  anchor:  testid="home", reasoning="..."
  after:   "...a centered purple BlogSprout <home> logo with a Subscribe CTA..."

Outputs:
  <task>/<task>_anchors.cleaned.json     filtered to own pages
  <variant>/<task>/description.annotated.md   with inline tags
                                               (won't overwrite description.md)

Run:
  python3 tools/insert_anchor_tags.py             # process all 10 tasks
  python3 tools/insert_anchor_tags.py 1_newsletter # one task only
  python3 tools/insert_anchor_tags.py --apply     # also overwrite description.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

# Stop words that are too generic for keyword-based matching
STOP = {
    "page","frame","rectangle","element","button","clickable","unknown",
    "navigation","link","group","item","input","field","section","layout",
    "click","trigger","triggers","opens","represents","another","application",
    "from","this","that","into","onto","with","default","defaulting","default",
    "unclassified","navigate","submits","navigates","likely",
}

# When a testid word doesn't appear verbatim in the description, also try
# these synonyms. Helps "search" find "magnifier"/"results", "login" find
# "sign in", etc. Each maps to the EXTRA terms beyond the word itself.
SYNONYMS = {
    "home":      ["logo", "wordmark", "brand", "back", "homepage"],
    "logo":      ["wordmark", "brand", "logotype"],
    "search":    ["magnifier", "magnifying", "lookup", "query", "find", "results"],
    "login":     ["sign in", "signin", "log in", "auth"],
    "signin":    ["sign in", "log in", "login", "auth"],
    "signup":    ["sign up", "register", "create account"],
    "register":  ["sign up", "signup", "create account"],
    "logout":    ["sign out", "log out"],
    "subscribe": ["newsletter", "subscription", "stay informed", "join"],
    "submit":    ["send", "create", "save", "apply", "post"],
    "send":      ["submit", "post", "share"],
    "title":     ["heading", "headline", "h1", "h2", "h3"],
    "heading":   ["title", "headline"],
    "headline":  ["title", "heading"],
    "name":      ["first name", "last name", "full name", "your name"],
    "email":     ["e-mail", "address", "mail"],
    "password":  ["passwd", "pass"],
    "phone":     ["mobile", "cell"],
    "next":      ["continue", "proceed", "forward"],
    "previous":  ["prev", "back"],
    "play":      ["pause"],
    "pause":     ["play"],
    "volume":    ["vol", "audio", "speaker"],
    "follow":    ["follow us", "subscribe"],
    "share":     ["sharing", "social"],
    "like":      ["heart", "favorite", "favourite"],
    "comment":   ["reply", "discussion"],
    "post":      ["article", "story"],
    "article":   ["post", "story", "card", "blog"],
    "category":  ["categories", "topic", "section"],
    "tag":       ["tags", "label", "chip"],
    "author":    ["profile", "byline", "writer"],
    "contact":   ["message", "inquiry", "reach"],
    "about":     ["bio", "profile", "info"],
    "cart":      ["basket", "bag", "shopping"],
    "checkout":  ["place order", "pay", "buy"],
    "filter":    ["filters", "filtering", "facet"],
    "sort":      ["sorting", "order"],
    "footer":    ["footer", "bottom"],
    "header":    ["header", "top", "nav"],
    "upload":    ["choose file", "attach", "drop", "select"],
    "download":  ["save as", "export"],
    "menu":      ["sidebar", "drawer", "navigation"],
    "nav":       ["navigation", "menu"],
    "service":   ["services", "offerings"],
    "buy":       ["purchase", "order"],
    "play-music": ["play", "audio"],
    "send-message": ["send", "message", "submit"],
}


def slug_for_match(page_name: str) -> set[str]:
    """01_Sign_in → {sign, in}; 03_Home_page → {home}."""
    s = re.sub(r"^\d+_", "", page_name).lower()
    s = s.replace("-", "_")
    return {w for w in s.split("_") if len(w) >= 2 and w not in {"page"}}


def find_page_section(description_md: str, page_name: str) -> tuple[int, int]:
    """Return (start_line_idx, end_line_idx) of `### N. <Page>` ... up to next `### `.
    -1, -1 if not found."""
    target_words = slug_for_match(page_name)
    lines = description_md.splitlines()
    start = -1
    for i, line in enumerate(lines):
        if not line.startswith("### "):
            continue
        head = line[4:].strip().lower()
        head = re.sub(r"^\d+\.\s*", "", head)
        head_words = set(re.findall(r"[a-z]+", head))
        if target_words & head_words:
            start = i; break
    if start == -1:
        return -1, -1
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("### "):
            end = j; break
    return start, end


def keywords_for_anchor(anchor: dict) -> list[str]:
    """Extract distinct keywords for keyword-search: testid pieces + reasoning."""
    kws: list[str] = []
    seen: set[str] = set()
    # testid kebab parts: 'send-message' → ['send', 'message']
    for w in anchor.get("testid", "").split("-"):
        wl = w.lower()
        if len(wl) >= 3 and wl not in STOP and wl not in seen:
            seen.add(wl); kws.append(wl)
    # reasoning words
    for w in re.findall(r"[A-Za-z]{4,}", anchor.get("reasoning") or ""):
        wl = w.lower()
        if wl not in STOP and wl not in seen:
            seen.add(wl); kws.append(wl)
    return kws


def insert_tags_in_section(section_text: str, anchors_for_page: list[dict]) -> tuple[str, dict]:
    """Insert <testid> markers within section_text. Returns (new_text, stats)."""
    text = section_text
    inserted: list[str] = []
    failed: list[str] = []
    used_offsets: list[tuple[int, int]] = []  # claimed (start, end) ranges to avoid overlap

    # Process top-down: smaller bbox_png.y first
    sorted_anchors = sorted(
        anchors_for_page,
        key=lambda a: (a.get("bbox_png") or {}).get("y", 0),
    )

    for anchor in sorted_anchors:
        testid = anchor["testid"]
        if f"<{testid}>" in text:
            inserted.append(f"{testid} (already present)")
            continue
        kws = keywords_for_anchor(anchor)
        if not kws:
            failed.append(f"{testid} (no keywords)")
            continue
        # Try each keyword in order, find first non-overlapping match
        chosen_pos = -1
        chosen_kw = None
        for kw in kws:
            # Find all matches; pick the FIRST that doesn't fall inside an already-claimed region
            for m in re.finditer(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
                if any(s <= m.start() <= e for s, e in used_offsets):
                    continue
                chosen_pos = m.end()
                chosen_kw = kw
                break
            if chosen_pos >= 0:
                break
        if chosen_pos < 0:
            failed.append(f"{testid} (no keyword match: {kws[:3]})")
            continue
        # Insert ` <testid>` after the matched word
        marker = f" <{testid}>"
        text = text[:chosen_pos] + marker + text[chosen_pos:]
        # Track the inserted region so subsequent anchors won't pile up at the same spot
        used_offsets.append((chosen_pos, chosen_pos + len(marker)))
        # Shift all previously-claimed offsets after this point — actually they're
        # all before this, so no shift needed (we always insert at chosen_pos which
        # is after their ends).
        inserted.append(f"{testid} after `{chosen_kw}`")

    return text, {"inserted": inserted, "failed": failed}


def clean_task_anchors(task_dir: Path) -> Optional[dict]:
    """Strip cross-task pollution from a task's anchors JSON."""
    f = task_dir / f"{task_dir.name}_anchors.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text(encoding="utf-8"))
    own_pages = {p.stem for p in (task_dir / "pages").glob("*.png")}
    cleaned: dict[str, list] = {}
    for page, anchors in (data.get("anchors") or {}).items():
        if page in own_pages and anchors:
            cleaned[page] = anchors
    out = {"task": data.get("task", task_dir.name), "anchors": cleaned}
    (task_dir / f"{task_dir.name}_anchors.cleaned.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    raw_total = sum(len(v) for v in (data.get("anchors") or {}).values())
    raw_pages = len([k for k, v in (data.get("anchors") or {}).items() if v])
    clean_total = sum(len(v) for v in cleaned.values())
    print(f"[clean] {task_dir.name}: pages {raw_pages} → {len(cleaned)},  "
          f"anchors {raw_total} → {clean_total}")
    return out


def annotate_description(desc_path: Path, anchors_by_page: dict[str, list[dict]],
                         apply_inplace: bool) -> dict:
    """Insert tags into desc_path's description.md. Save to .annotated.md (or
    overwrite if apply_inplace)."""
    if not desc_path.exists():
        return {"error": f"missing {desc_path}"}
    text = desc_path.read_text(encoding="utf-8")
    summary: dict[str, list[str]] = {}
    for page_name, anchor_list in anchors_by_page.items():
        if not anchor_list:
            continue
        s, e = find_page_section(text, page_name)
        if s == -1:
            summary.setdefault("missing_page_sections", []).append(page_name)
            continue
        lines = text.splitlines(keepends=False)
        sec = "\n".join(lines[s:e])
        new_sec, stats = insert_tags_in_section(sec, anchor_list)
        text = "\n".join(lines[:s]) + ("\n" if s > 0 else "") + new_sec + "\n" + "\n".join(lines[e:])
        # Re-split so subsequent pages still find their sections by line number
        # (they'd be shifted by our inserts which are intra-section, so still OK)
        summary.setdefault("inserted", []).extend(f"{page_name}: {x}" for x in stats["inserted"])
        summary.setdefault("failed", []).extend(f"{page_name}: {x}" for x in stats["failed"])

    out_path = desc_path if apply_inplace else desc_path.with_name("description.annotated.md")
    out_path.write_text(text, encoding="utf-8")
    summary["written_to"] = str(out_path)
    return summary


def main():
    tasks_root = Path(__file__).parent.parent.resolve()
    only_task = None
    apply_inplace = False
    for arg in sys.argv[1:]:
        if arg == "--apply":
            apply_inplace = True
        elif not arg.startswith("--"):
            only_task = arg

    # Variants that may contain a description.md per task
    variant_dirs = ["c0", "c2", "c3", "c1/pick_A", "c1/pick_B", "c1/pick_C"]

    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name[0:1].isdigit():
            continue
        if only_task and task_dir.name != only_task:
            continue
        cleaned = clean_task_anchors(task_dir)
        if not cleaned or not cleaned["anchors"]:
            print(f"  [skip] {task_dir.name}: no anchors after cleaning")
            continue

        for variant in variant_dirs:
            desc = tasks_root / variant / task_dir.name / "description.md"
            if not desc.exists():
                continue
            res = annotate_description(desc, cleaned["anchors"], apply_inplace)
            n_ok = len(res.get("inserted", []))
            n_fail = len(res.get("failed", []))
            n_miss = len(res.get("missing_page_sections", []))
            print(f"  [annotate] {variant}/{task_dir.name}: "
                  f"inserted={n_ok}, failed={n_fail}, missing_sections={n_miss}  "
                  f"→ {Path(res['written_to']).name}")
            if n_fail:
                for x in res["failed"][:5]:
                    print(f"     ⚠ {x}")
            if n_miss:
                for x in res["missing_page_sections"]:
                    print(f"     ✗ section not found: {x}")


if __name__ == "__main__":
    main()
