#!/usr/bin/env python3
"""
eval_run_vlm.py — VLM-augmented eval (drop-in alternative to eval_run.py).

What's new vs eval_run.py
-------------------------
1. Once per page, ask a VLM to locate every interactive UI component in the
   rendered full-page screenshot. Returns 50–200 (bbox, type, text) tuples
   in rendered-screenshot pixel coordinates.
2. Match those VLM elements against the GT annotations for the page,
   producing a dense set of (mockup ↔ rendered) anchor pairs (typically
   10–50 per page, vs the 2–8 testid/semantic anchors found before).
3. With many anchors, replace the single global affine with a k-NN
   distance-weighted local translation per annotation. Solves the
   "global (sx, sy, tx, ty) cannot fit non-uniform vertical drift on long
   pages" failure mode.
4. Playwright behavior checks unchanged — VLM is only used for localization.

The Playwright + DOM logic (login, route discovery, candidate selection,
behavior checks, screenshot rendering, summary writing) is reused unchanged
by importing eval_run and monkey-patching just two functions:
    eval_run.find_anchors_all   ← prepends VLM-derived anchors
    eval_run.apply_affine       ← swaps to k-NN local when anchors exist

Backward compatibility
----------------------
If EVAL_VLM_ANCHORS is unset or != "1", behaves identically to eval_run.py
(both patches are no-ops in that mode).

Usage
-----
    # Default: identical to eval_run.py
    python3 tools/eval_run_vlm.py <run_dir>

    # With VLM (Anthropic Haiku, the cheap default):
    EVAL_VLM_ANCHORS=1 ANTHROPIC_API_KEY=sk-... \\
        python3 tools/eval_run_vlm.py <run_dir>

Env vars
--------
    EVAL_VLM_ANCHORS    "1" to enable (default unset = behave like eval_run.py)
    EVAL_VLM_PROVIDER   "anthropic" | "openai" | "gemini"  (default anthropic)
    EVAL_VLM_MODEL      override default model for the chosen provider
    EVAL_VLM_K          k-NN neighbours for local transform (default 4)
    EVAL_VLM_MAX_RADIUS px in mockup-CSS space; anchors farther are dropped
                        (default 800)
    EVAL_VLM_CACHE      "0" to disable cache (default on; key = screenshot md5)
    ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY  per provider

Outputs (when VLM enabled)
--------------------------
    logs/vlm_anchors_cache.json   per-page VLM responses (incremental, persists)
    logs/vlm_pairs.json           debug dump of GT↔VLM pairs per page
    summary.vlm_enabled           = true
    summary.vlm_provider, vlm_model
    summary.vlm_pairs_per_page    {page_path: n_pairs}
    summary.vlm_total_pairs       sum of all pairs
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# Make eval_run importable when this file lives next to it under tools/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_run as _er  # noqa: E402


# ============================================================
# VLM client (Anthropic / OpenAI / Gemini)
# ============================================================

VLM_PROMPT = """\
This is a screenshot of a rendered webpage. Identify EVERY interactive UI \
component you can see.

For each component, output ONE JSON object with these fields:
  - "bbox": [x, y, width, height] in pixels of THIS screenshot (top-left origin, 0-indexed)
  - "type": one of "button" | "link" | "input" | "toggle" | "card" | "icon" | "image" | "text"
  - "text": visible label, placeholder, or aria-label (max 100 chars). Use "" if none.
  - "confidence": 0.0–1.0

Include: buttons, links, form inputs (text/email/password/textarea/select), \
toggles/switches/checkboxes/radios, interactive cards in lists or grids, \
clickable icons, tabs, accordion headers, dropdown triggers.

Return ONLY a JSON array. No prose. No markdown fences. Example:
[
  {"bbox":[120,40,80,32],"type":"button","text":"Submit","confidence":0.95},
  {"bbox":[200,40,300,32],"type":"input","text":"Email","confidence":0.9}
]
"""

_PROVIDER_DEFAULTS = {
    "anthropic": "claude-haiku-4-5",
    "openai":    "gpt-4o-mini",
    "gemini":    "gemini-2.5-flash",
}


def _call_vlm_anthropic(image_bytes: bytes, model: str) -> Optional[str]:
    try:
        import anthropic
    except ImportError:
        print("[vlm] anthropic SDK not installed; pip install anthropic", file=sys.stderr)
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[vlm] ANTHROPIC_API_KEY not set", file=sys.stderr)
        return None
    client = anthropic.Anthropic(api_key=api_key)
    img_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=8192,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/png",
                                "data": img_b64}},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }],
        )
    except Exception as e:
        print(f"[vlm] anthropic call failed: {e}", file=sys.stderr)
        return None
    return resp.content[0].text if resp.content else None


def _call_vlm_openai(image_bytes: bytes, model: str) -> Optional[str]:
    try:
        from openai import OpenAI
    except ImportError:
        print("[vlm] openai SDK not installed; pip install openai", file=sys.stderr)
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[vlm] OPENAI_API_KEY not set", file=sys.stderr)
        return None
    client = OpenAI(api_key=api_key)
    img_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=8192,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }],
        )
    except Exception as e:
        print(f"[vlm] openai call failed: {e}", file=sys.stderr)
        return None
    return resp.choices[0].message.content


def _call_vlm_gemini(image_bytes: bytes, model: str) -> Optional[str]:
    try:
        import google.generativeai as genai
    except ImportError:
        print("[vlm] google-generativeai SDK not installed; "
              "pip install google-generativeai", file=sys.stderr)
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[vlm] GEMINI_API_KEY not set", file=sys.stderr)
        return None
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model)
    try:
        resp = m.generate_content(
            [VLM_PROMPT, {"mime_type": "image/png", "data": image_bytes}]
        )
    except Exception as e:
        print(f"[vlm] gemini call failed: {e}", file=sys.stderr)
        return None
    return resp.text


def _parse_vlm_output(text: str) -> Optional[list[dict]]:
    """Strip optional ```json fences, parse JSON array, validate shape."""
    s = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\[.+?\])\s*```", s, re.DOTALL)
    if fence:
        s = fence.group(1)
    start, end = s.find("["), s.rfind("]")
    if start < 0 or end <= start:
        print("[vlm] couldn't locate JSON array in response", file=sys.stderr)
        return None
    try:
        arr = json.loads(s[start:end + 1])
    except Exception as e:
        print(f"[vlm] JSON parse error: {e}", file=sys.stderr)
        return None
    if not isinstance(arr, list):
        return None
    out: list[dict] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        bb = item.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        try:
            x, y, w, h = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        out.append({
            "bbox": [x, y, w, h],
            "type": str(item.get("type", "")).lower(),
            "text": str(item.get("text", ""))[:120],
            "confidence": float(item.get("confidence", 0.5) or 0.5),
        })
    return out


def call_vlm(image_bytes: bytes) -> Optional[list[dict]]:
    """Dispatch to configured provider. Returns parsed elements or None."""
    provider = os.environ.get("EVAL_VLM_PROVIDER", "anthropic").lower()
    model = os.environ.get("EVAL_VLM_MODEL") or _PROVIDER_DEFAULTS.get(provider)
    if not model:
        print(f"[vlm] unknown provider {provider!r}", file=sys.stderr)
        return None
    raw = None
    if provider == "anthropic":
        raw = _call_vlm_anthropic(image_bytes, model)
    elif provider == "openai":
        raw = _call_vlm_openai(image_bytes, model)
    elif provider == "gemini":
        raw = _call_vlm_gemini(image_bytes, model)
    if not raw:
        return None
    return _parse_vlm_output(raw)


# ============================================================
# Cache (persisted to <run_dir>/logs/vlm_anchors_cache.json)
# ============================================================

def _cache_path(run_dir: Path) -> Path:
    return run_dir / "logs" / "vlm_anchors_cache.json"


def cache_load(run_dir: Path) -> dict:
    p = _cache_path(run_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def cache_save(run_dir: Path, cache: dict) -> None:
    p = _cache_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                 encoding="utf-8")


def vlm_locate(page, run_dir: Path, page_path: str, cache: dict) -> list[dict]:
    """Take full-page screenshot, return parsed VLM elements (with caching)."""
    use_cache = os.environ.get("EVAL_VLM_CACHE", "1") != "0"
    try:
        png = page.screenshot(full_page=True)
    except Exception as e:
        print(f"[vlm] screenshot failed: {e}", file=sys.stderr)
        return []
    h = hashlib.md5(png).hexdigest()[:16]
    key = f"{page_path}::{h}"
    if use_cache and key in cache:
        print(f"[vlm] cache hit {page_path} ({h})")
        return cache[key]
    provider = os.environ.get("EVAL_VLM_PROVIDER", "anthropic")
    print(f"[vlm] calling {provider} for {page_path} "
          f"(screenshot {len(png)} bytes, hash {h})")
    elems = call_vlm(png) or []
    print(f"[vlm] {page_path}: VLM returned {len(elems)} elements")
    if use_cache:
        cache[key] = elems
        cache_save(run_dir, cache)
    return elems


# ============================================================
# Match VLM elements ↔ GT annotations
# ============================================================

# GT type → preferred VLM types (1.0 = perfect, 0.3 = poor cross-type)
_TYPE_COMPAT = {
    "navigate": {"link": 1.0, "button": 0.7, "card": 0.6, "icon": 0.4},
    "click":    {"button": 1.0, "link": 0.7, "icon": 0.6, "card": 0.6, "toggle": 0.4},
    "input":    {"input": 1.0},
    "toggle":   {"toggle": 1.0, "button": 0.6, "input": 0.4},
}


def _type_score(gt_type: str, vlm_type: str) -> float:
    return _TYPE_COMPAT.get(gt_type, {}).get(vlm_type, 0.3)


def _word_set(s: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", (s or "").lower()))


def _text_score(gt_reasoning: str, vlm_text: str) -> float:
    g, v = _word_set(gt_reasoning), _word_set(vlm_text)
    if not g or not v:
        return 0.0
    return len(g & v) / len(g)


def _size_score(gt_w: float, gt_h: float, vlm_w: float, vlm_h: float) -> float:
    """1.0 when areas match; decays smoothly with mismatch."""
    a, b = gt_w * gt_h, vlm_w * vlm_h
    if a <= 0 or b <= 0:
        return 0.5
    r = a / b
    if r > 1:
        r = 1 / r
    return r


def match_vlm_to_gt(vlm_elements: list[dict],
                     gt_anns: list[dict],
                     scale: float,
                     min_score: float = 0.45,
                     max_size_ratio: float = 5.0) -> list[dict]:
    """Greedy 1-to-1 matching. Returns anchors in find_anchors_*() format
    (target_cx/cy in mockup-CSS-px, rendered_cx/cy in screenshot-px).

    Hard filters (applied BEFORE scoring):
      * GT bbox area vs VLM bbox area must be within `max_size_ratio` either
        way. Stops VLM container/card bboxes (700x200) from being matched to
        small button GTs (60x30) just because the card text contains the
        button label — that pollutes the k-NN translation pool with garbage
        deltas (off by 500+ px).
    """
    if not vlm_elements or not gt_anns:
        return []

    # Process longer-reasoning GTs first — most distinctive matches go first.
    gt_sorted = sorted(
        gt_anns,
        key=lambda a: -len(a.get("reasoning") or ""),
    )
    used: set[int] = set()
    out: list[dict] = []

    for ann in gt_sorted:
        gt_type = ann.get("type", "click")
        gt_reasoning = " ".join(filter(None, [
            ann.get("reasoning") or "",
            ann.get("note") or "",
            ann.get("subtype") or "",
        ]))
        bb = ann.get("bbox_png") or {}
        gt_cx = (bb.get("x", 0) + bb.get("w", 0) / 2) / max(scale, 1e-6)
        gt_cy = (bb.get("y", 0) + bb.get("h", 0) / 2) / max(scale, 1e-6)
        gt_w = bb.get("w", 0) / max(scale, 1e-6)
        gt_h = bb.get("h", 0) / max(scale, 1e-6)

        gt_area = max(1.0, gt_w * gt_h)

        best_idx, best_score = -1, 0.0
        for i, v in enumerate(vlm_elements):
            if i in used:
                continue
            vx, vy, vw, vh = v["bbox"]
            # Hard size-ratio filter — drop "button matched to entire card"
            # cases where text_score alone would let the match through.
            vlm_area = max(1.0, vw * vh)
            ratio = max(gt_area, vlm_area) / min(gt_area, vlm_area)
            if ratio > max_size_ratio:
                continue
            text_s = _text_score(gt_reasoning, v.get("text", ""))
            type_s = _type_score(gt_type, v.get("type", ""))
            size_s = _size_score(gt_w, gt_h, vw, vh)
            # x-position: viewport_w == figma_w, so x is comparable across
            # the two coordinate spaces. y is what we're trying to FIND drift
            # in, so don't penalize y mismatch here.
            x_diff = abs(gt_cx - (vx + vw / 2))
            x_pos_s = max(0.0, 1.0 - x_diff / 600)
            score = (0.45 * text_s + 0.30 * type_s
                     + 0.15 * size_s + 0.10 * x_pos_s)
            score *= max(0.5, v.get("confidence", 0.8))
            if score > best_score:
                best_idx, best_score = i, score

        if best_idx >= 0 and best_score >= min_score:
            v = vlm_elements[best_idx]
            vx, vy, vw, vh = v["bbox"]
            out.append({
                "ann_id": ann["id"],
                "target_cx": gt_cx, "target_cy": gt_cy,
                "rendered_cx": vx + vw / 2,
                "rendered_cy": vy + vh / 2,
                "rendered_bbox": {"x": vx, "y": vy, "width": vw, "height": vh},
                "method": "vlm",
                "score": round(best_score, 3),
                "match_text": (v.get("text") or v.get("type") or "")[:40],
            })
            used.add(best_idx)

    return out


# ============================================================
# k-NN distance-weighted local translation
# ============================================================

_MIN_ANCHORS_FOR_KNN = int(os.environ.get("EVAL_VLM_MIN_ANCHORS", "4"))


def apply_local_transform(target: dict, anchors: list[dict],
                          k: int = 4, max_radius: float = 800.0,
                          fallback_T: Optional[dict] = None) -> dict:
    """Return target shifted by inverse-distance weighted average of nearby
    anchor deltas. Width/height unchanged (viewport_w == figma_w → no scale).

    Safety nets:
      * If fewer than _MIN_ANCHORS_FOR_KNN total anchors are available, fall
        back to fallback_T (the original global affine) — too few points to
        produce a stable local field.
      * If no anchor is within max_radius of this target, also fall back —
        a single distant anchor would dominate and likely give a wrong delta.
    """
    if len(anchors) < _MIN_ANCHORS_FOR_KNN:
        # Sparse anchor pool — global affine is more reliable than 1–3 noisy
        # local points.
        return _orig_apply_affine(target, fallback_T) if fallback_T else target
    if not anchors:
        return target
    cx = target["x"] + target["width"] / 2
    cy = target["y"] + target["height"] / 2
    pool: list[tuple[float, dict]] = []
    for a in anchors:
        d = ((a["target_cx"] - cx) ** 2 + (a["target_cy"] - cy) ** 2) ** 0.5
        if d <= max_radius:
            pool.append((d, a))
    if not pool:
        # Target is far from every anchor — trust global affine instead of
        # extrapolating from a single distant anchor.
        return _orig_apply_affine(target, fallback_T) if fallback_T else target
    pool.sort(key=lambda x: x[0])
    pool = pool[:max(1, k)]
    eps = 1.0
    w_sum = 0.0
    tx_sum = 0.0
    ty_sum = 0.0
    for d, a in pool:
        # Weight by anchor's own match score so noisy VLM anchors contribute
        # less than high-confidence DOM/testid anchors.
        s = float(a.get("score", 1.0))
        w = s / (d + eps)
        w_sum += w
        tx_sum += w * (a["rendered_cx"] - a["target_cx"])
        ty_sum += w * (a["rendered_cy"] - a["target_cy"])
    tx = tx_sum / w_sum
    ty = ty_sum / w_sum
    return {
        "x": target["x"] + tx,
        "y": target["y"] + ty,
        "width": target["width"],
        "height": target["height"],
    }


# ============================================================
# Monkey-patch eval_run
# ============================================================

# Module state used by the patches:
_anchors_state: list[dict] = []      # anchors for the page CURRENTLY being processed
_run_dir: Optional[Path] = None      # set in main()
_vlm_cache: dict = {}                # screenshot-keyed cache, lazily loaded
_vlm_pairs_log: dict[str, list] = {} # page_path → matched pairs (debug)

_K = int(os.environ.get("EVAL_VLM_K", "4"))
_MAX_R = float(os.environ.get("EVAL_VLM_MAX_RADIUS", "800"))

_orig_find_anchors_all = _er.find_anchors_all
_orig_apply_affine = _er.apply_affine


def _patched_find_anchors_all(page, anns, scale, description_md="",
                               anchors_for_page=None):
    """Run the original anchor pipeline + VLM in parallel.

    IMPORTANT: VLM-derived anchors are intentionally NOT included in the
    return value. Returning them would make main()'s anchor short-circuit
    fire (set tier_n=1 and use a synthetic `chosen` record without DOM
    attrs like href/aria), which crashes behavior scoring because:
      * VLM bboxes are imprecise (±10–30px), so the "exact" T1 box is
        actually drawn off the real element.
      * The synthetic chosen has no href/aria_label/input_type → every
        downstream behavior check (navigate / external / input / popout)
        fails immediately.

    Instead, we keep VLM anchors only in `_anchors_state`, which is read
    by `_patched_apply_affine` for the k-NN local transform. Annotations
    then go through the regular pick_best() → DOM search path with their
    target already shifted to the right neighbourhood, finding the real
    DOM element (with full attrs) for the behavior check.

    Original-system anchors (testid, semantic, subtype, href, anchor_json)
    DO go through the short-circuit as before — their rendered_bbox is
    DOM-precise.
    """
    global _anchors_state

    # 1. Original anchor pipeline — these are returned and trigger the
    #    short-circuit (rendered_bbox is precise, rendered_element is full).
    orig_anchors = _orig_find_anchors_all(
        page, anns, scale,
        description_md=description_md,
        anchors_for_page=anchors_for_page,
    )
    orig_ann_ids = {a["ann_id"] for a in orig_anchors}

    # 2. VLM anchors — kept ONLY for the k-NN transform.
    vlm_anchors: list[dict] = []
    if (os.environ.get("EVAL_VLM_ANCHORS") == "1"
            and _run_dir is not None):
        try:
            page_path = urlparse(page.url).path or "/"
        except Exception:
            page_path = "?"
        vlm_elems = vlm_locate(page, _run_dir, page_path, _vlm_cache)
        if vlm_elems:
            vlm_anchors = match_vlm_to_gt(vlm_elems, anns, scale)
            _vlm_pairs_log[page_path] = vlm_anchors
            print(f"[vlm] {page_path}: matched {len(vlm_anchors)}/{len(anns)} "
                  f"GT annotations from {len(vlm_elems)} VLM elements "
                  f"(orig pipeline produced {len(orig_anchors)})")

    # 3. Combine for k-NN transform. Avoid double-listing the same ann_id —
    #    if a GT was matched by both, prefer the original (more reliable).
    _anchors_state = list(orig_anchors) + [
        a for a in vlm_anchors if a["ann_id"] not in orig_ann_ids
    ]

    # 4. Return only original anchors so main()'s short-circuit fires only
    #    on DOM-precise anchors. VLM anchors silently shape the transform.
    return orig_anchors


def _patched_apply_affine(target, T):
    """Use k-NN local translation when VLM mode is on AND the anchor pool is
    rich enough; otherwise fall back to the global affine T.

    apply_local_transform itself decides per-target whether to use k-NN
    (enough anchors + at least one within max_radius) or fall back to T."""
    if (os.environ.get("EVAL_VLM_ANCHORS") == "1"
            and _anchors_state):
        return apply_local_transform(target, _anchors_state,
                                      k=_K, max_radius=_MAX_R,
                                      fallback_T=T)
    return _orig_apply_affine(target, T)


_er.find_anchors_all = _patched_find_anchors_all
_er.apply_affine = _patched_apply_affine


# ============================================================
# Main (delegates to eval_run.main; pre/post hooks add VLM bookkeeping)
# ============================================================

def main() -> int:
    global _run_dir, _vlm_cache

    if len(sys.argv) != 2:
        print("usage: eval_run_vlm.py <run_dir>", file=sys.stderr)
        return 2
    _run_dir = Path(sys.argv[1]).resolve()

    vlm_on = os.environ.get("EVAL_VLM_ANCHORS") == "1"
    if vlm_on:
        provider = os.environ.get("EVAL_VLM_PROVIDER", "anthropic")
        model = (os.environ.get("EVAL_VLM_MODEL")
                 or _PROVIDER_DEFAULTS.get(provider, "?"))
        print(f"[vlm] enabled — provider={provider} model={model} "
              f"k={_K} max_radius={_MAX_R}")
        _vlm_cache = cache_load(_run_dir)
        if _vlm_cache:
            print(f"[vlm] loaded cache with {len(_vlm_cache)} entries")
    else:
        print("[vlm] disabled (set EVAL_VLM_ANCHORS=1 to enable). "
              "Behaving exactly like eval_run.py.")

    rc = _er.main()

    # ---- Post-run: write VLM debug + augment summary ----
    if vlm_on:
        try:
            pairs_path = _run_dir / "logs" / "vlm_pairs.json"
            pairs_path.parent.mkdir(parents=True, exist_ok=True)
            pairs_path.write_text(
                json.dumps(_vlm_pairs_log, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[vlm] wrote {pairs_path}")
        except Exception as e:
            print(f"[vlm] failed to write vlm_pairs.json: {e}",
                  file=sys.stderr)

        try:
            result_path = _run_dir / "logs" / "eval_result.json"
            doc = json.loads(result_path.read_text(encoding="utf-8"))
            provider = os.environ.get("EVAL_VLM_PROVIDER", "anthropic")
            doc["summary"]["vlm_enabled"] = True
            doc["summary"]["vlm_provider"] = provider
            doc["summary"]["vlm_model"] = (
                os.environ.get("EVAL_VLM_MODEL")
                or _PROVIDER_DEFAULTS.get(provider, "?")
            )
            doc["summary"]["vlm_pairs_per_page"] = {
                p: len(v) for p, v in _vlm_pairs_log.items()
            }
            doc["summary"]["vlm_total_pairs"] = sum(
                len(v) for v in _vlm_pairs_log.values()
            )
            result_path.write_text(
                json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"[vlm] augmented summary in {result_path}  "
                  f"(total pairs={doc['summary']['vlm_total_pairs']})")
        except Exception as e:
            print(f"[vlm] failed to augment summary: {e}", file=sys.stderr)

    return rc


if __name__ == "__main__":
    sys.exit(main())
