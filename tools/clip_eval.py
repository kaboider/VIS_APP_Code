#!/usr/bin/env python3
"""
clip_eval.py — visual similarity scoring (mockup vs agent render) using CLIP.

Standalone batch script — does NOT depend on eval_run.py output (but will
reuse it when present to avoid re-resolving URLs).

For each run dir under <runs_root>:
  1. docker compose up the agent's app (skippable)
  2. wait for the frontend port to actually serve HTTP
  3. for each mockup PNG in inputs/pages/:
     a. resolve its URL (README ## Pages → eval_result.json fallback → /<slug>)
     b. capture a CLEAN full-page screenshot (no annotations)
     c. compute CLIP cosine similarity vs the mockup PNG
  4. save per-page + average to logs/clip_similarity.json
  5. docker compose down

Outputs:
  <run_dir>/logs/clip_similarity.json   per-run details
  <run_dir>/logs/clip_screenshots/*.png clean screenshots (for inspection)
  <runs_root>/clip_summary.csv          per-run aggregate leaderboard

CPU is fine for ViT-B-32 (~100ms/image, ~20s per typical run).

Usage:
    python3 clip_eval.py                          # default _runs dir
    python3 clip_eval.py /path/to/_runs
    python3 clip_eval.py --filter '*c3*'          # subset of runs
    python3 clip_eval.py --skip-docker            # services already up
    python3 clip_eval.py --force                  # re-run even if json exists
    python3 clip_eval.py --model ViT-L-14         # better but slower on CPU
    python3 clip_eval.py --http-timeout 180       # slow startups

First run downloads ~150MB of model weights to ~/.cache/clip/.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ----------------------------------------------------------------------
# Lazy imports — keep --help / --filter snappy without paying torch import cost
# ----------------------------------------------------------------------

_clip_cache: dict = {}


def load_clip(model_name: str = "ViT-B-32",
              pretrained: str = "openai") -> tuple:
    """Return (model, preprocess, torch, F, Image). Cached per-process. LOCAL backend only."""
    if "model" in _clip_cache:
        return _clip_cache["bundle"]
    try:
        import open_clip
        import torch
        import torch.nn.functional as F
        from PIL import Image
    except ImportError:
        print("ERROR: missing deps for local backend. Install:\n"
              "  pip install open_clip_torch torch pillow playwright\n"
              "  playwright install chromium\n"
              "Or use --backend replicate / --backend huggingface / --backend vertex",
              file=sys.stderr)
        sys.exit(1)
    print(f"[clip] loading {model_name} ({pretrained})...", file=sys.stderr)
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()
    print(f"[clip] loaded in {time.time()-t0:.1f}s", file=sys.stderr)
    _clip_cache["model"] = True
    _clip_cache["bundle"] = (model, preprocess, torch, F, Image)
    return _clip_cache["bundle"]


def _cos(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length float lists. No torch dependency."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return float("nan")
    return dot / (na * nb)


def _embed_local(path: Path, model_name: str) -> list[float]:
    try:
        model, preprocess, torch, F, Image = load_clip(model_name)
        im = Image.open(path).convert("RGB")
        with torch.no_grad():
            x = preprocess(im).unsqueeze(0)
            feat = model.encode_image(x)
            feat = F.normalize(feat, dim=-1)
        return feat[0].tolist()
    except Exception as e:
        raise EmbedError(f"local: {e}") from e


class EmbedError(Exception):
    """Raised when an embedding backend fails (rate limit, network, auth, etc.).
    Caller catches and may rotate to a different backend OR abort writing."""


# --- Model name normalization ---
# Users pass --model in open_clip style (e.g. "ViT-B-32" / "ViT-L-14"). Each
# backend speaks a different naming convention; normalize per-backend so the
# user can use one --model flag across all of them.

_HF_MODEL_MAP = {
    "ViT-B-32": "openai/clip-vit-base-patch32",
    "ViT-B-16": "openai/clip-vit-base-patch16",
    "ViT-L-14": "openai/clip-vit-large-patch14",
    "ViT-L-14-336": "openai/clip-vit-large-patch14-336",
}


def _hf_model_name(name: str) -> str:
    """Map open_clip-style names to HF repo paths. If already an HF repo path
    (contains '/'), pass through unchanged."""
    if "/" in name:
        return name
    return _HF_MODEL_MAP.get(name, "openai/clip-vit-base-patch32")


def _embed_replicate(path: Path, model_name: str = "ViT-B-32") -> list[float]:
    """Use Replicate's official `openai/clip` model for image features.
    Needs REPLICATE_API_TOKEN. Returns a ViT-B/32 512-d embedding (model
    is pinned — model_name argument ignored).
    Raises EmbedError on failure (incl. 429 rate limit).

    History:
    - `andreasjansson/clip-features`: schema changed to text-only string input.
    - `krthr/clip-embeddings`: works but ViT-L/14 768-d (mismatched with HF).
    - `openai/clip` (current): ViT-B/32 512-d, matches HF default."""
    try:
        import replicate
    except ImportError:
        raise EmbedError("pip install replicate")
    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise EmbedError("REPLICATE_API_TOKEN env var not set")
    try:
        with path.open("rb") as f:
            out = replicate.run(
                "openai/clip",
                input={"image": f},
            )
    except Exception as e:
        # Most common: ReplicateError 429 "Request was throttled"
        msg = str(e)[:300]
        raise EmbedError(f"replicate: {msg}") from e
    if not out:
        raise EmbedError("replicate: empty response")
    # openai/clip returns {"embedding": [...]} (single object, not array)
    rec = out[0] if isinstance(out, list) else out
    emb = rec.get("embedding") if isinstance(rec, dict) else None
    if not emb:
        raise EmbedError(f"replicate: unexpected response shape {str(rec)[:100]}")
    return emb


def _embed_huggingface(path: Path, model_name: str = "openai/clip-vit-base-patch32") -> list[float]:
    """Use HF Inference API. Needs HF_TOKEN.
    Auto-normalizes open_clip-style names (ViT-B-32) to HF repo paths.
    Raises EmbedError on failure (incl. 429 / 503 cold-load).

    HF deprecated `/pipeline/<task>/<repo>` in favor of `/models/<repo>`
    plus an explicit `task` form-field for the new InferenceProviders router.
    For CLIP image features we call `image-feature-extraction`."""
    import urllib.request, urllib.error
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise EmbedError("HF_TOKEN env var not set")
    repo = _hf_model_name(model_name)
    # Modern HF Serverless Inference URL — no `/pipeline/<task>/` prefix
    url = f"https://api-inference.huggingface.co/models/{repo}"
    data = path.read_bytes()
    req = urllib.request.Request(url, data=data,
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/octet-stream",
                                          "x-wait-for-model": "true"})
    try:
        resp = urllib.request.urlopen(req, timeout=60).read()
    except urllib.error.HTTPError as e:
        body = ""
        try: body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception: pass
        raise EmbedError(f"huggingface: {e.code} {body}") from e
    except Exception as e:
        raise EmbedError(f"huggingface: {e}") from e
    try:
        out = json.loads(resp)
    except Exception as e:
        raise EmbedError(f"huggingface: bad json ({e})") from e
    # HF returns nested list; flatten to 1D
    while isinstance(out, list) and out and isinstance(out[0], list):
        out = out[0]
    if not out:
        raise EmbedError("huggingface: empty embedding")
    return out


# Rotation state: each call cycles through the configured backend list.
_rotation_state: dict = {"index": 0}


def _embed_rotate(path: Path, model_name: str,
                  backends: list[str]) -> list[float]:
    """Round-robin between backends. Tries each one once on failure before raising."""
    embed_fns = {
        "local": _embed_local,
        "replicate": _embed_replicate,
        "huggingface": _embed_huggingface,
        "vertex": _embed_vertex,
    }
    n = len(backends)
    last_err = None
    for offset in range(n):
        idx = (_rotation_state["index"] + offset) % n
        be = backends[idx]
        fn = embed_fns.get(be)
        if fn is None:
            last_err = EmbedError(f"unknown backend: {be}")
            continue
        try:
            emb = fn(path, model_name)
            # Advance rotation so next call starts at the NEXT backend
            _rotation_state["index"] = (idx + 1) % n
            return emb
        except EmbedError as e:
            print(f"  [{be}] {e} — trying next backend", file=sys.stderr)
            last_err = e
    raise last_err if last_err else EmbedError("all backends failed")


def _embed_vertex(path: Path, model_name: str = "multimodalembedding@001") -> list[float]:
    """Use Vertex AI Multimodal Embeddings ($0.0002/image). Needs:
       GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json AND GOOGLE_CLOUD_PROJECT."""
    try:
        from vertexai.vision_models import MultiModalEmbeddingModel, Image as VImage
        import vertexai
    except ImportError:
        raise EmbedError("pip install google-cloud-aiplatform")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise EmbedError("GOOGLE_CLOUD_PROJECT env var not set")
    try:
        vertexai.init(project=project, location=location)
        model = MultiModalEmbeddingModel.from_pretrained(model_name)
        img = VImage.load_from_file(str(path))
        emb = model.get_embeddings(image=img, dimension=1408)
        return list(emb.image_embedding)
    except Exception as e:
        raise EmbedError(f"vertex: {e}") from e


def clip_similarity(img_a: Path, img_b: Path,
                    model_name: str = "ViT-B-32",
                    backend: str = "local",
                    rotate_backends: Optional[list[str]] = None) -> float:
    """Cosine similarity of CLIP image embeddings, range [-1, 1].

    Raises EmbedError on failure (so caller can choose to skip writing
    rather than write a NaN result).
    """
    if backend == "rotate":
        if not rotate_backends:
            raise EmbedError("rotate backend requires --rotate-backends")
        ea = _embed_rotate(img_a, model_name, rotate_backends)
        eb = _embed_rotate(img_b, model_name, rotate_backends)
    else:
        embed_fn = {
            "local": _embed_local,
            "replicate": _embed_replicate,
            "huggingface": _embed_huggingface,
            "vertex": _embed_vertex,
        }.get(backend)
        if embed_fn is None:
            raise EmbedError(f"Unknown backend: {backend}")
        ea = embed_fn(img_a, model_name)
        eb = embed_fn(img_b, model_name)
    if not ea or not eb:
        raise EmbedError(f"empty embedding (a={len(ea) if ea else 0}, b={len(eb) if eb else 0})")
    return _cos(ea, eb)


# ----------------------------------------------------------------------
# Run dir helpers
# ----------------------------------------------------------------------

def read_meta(run_dir: Path) -> dict:
    p = run_dir / "meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_mockups(run_dir: Path) -> list[Path]:
    """List inputs/pages/<page>.png (excluding the *_structure-only.json siblings)."""
    pages_dir = run_dir / "inputs" / "pages"
    if not pages_dir.is_dir():
        return []
    return sorted(p for p in pages_dir.glob("*.png") if p.is_file())


def parse_readme_pages(workspace: Path) -> dict[str, str]:
    """README's `## Pages` table → {page_name_or_index: url}.

    Tolerant: matches both '01_Foo' style and bare numeric indices, strips
    backticks/markdown links from URLs.
    """
    readme = workspace / "README.md"
    if not readme.is_file():
        return {}
    text = readme.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^##\s+Pages\b", text, re.M | re.I)
    if not m:
        return {}
    section = text[m.end():]
    end = re.search(r"\n##\s+\S", section)
    if end:
        section = section[: end.start()]

    out: dict[str, str] = {}
    for line in section.splitlines():
        cols = [c.strip() for c in line.split("|") if c.strip()]
        if len(cols) < 2:
            continue
        # First col: page index ("1", "2"...) or page name; second: URL
        idx = cols[0]
        url = cols[1]
        # Strip backticks, [text](url), surrounding quotes
        url = url.strip("`'\"")
        m2 = re.match(r"\[([^\]]*)\]\(([^)]+)\)", url)
        if m2:
            url = m2.group(2)
        url = url.strip("`'\"").split("?")[0].split("#")[0]
        if not url.startswith("/"):
            continue
        out[idx] = url
    return out


def resolve_page_url(mockup_path: Path,
                     readme_map: dict[str, str],
                     existing_eval: Optional[dict]) -> Optional[str]:
    """Best-effort URL resolution for a mockup, from cheapest source first."""
    name = mockup_path.stem  # e.g. "03_Main"

    # 1. Existing eval_result.json may already have URL → page mapping
    if existing_eval:
        for r in existing_eval.get("results", []):
            if r.get("page") == name and r.get("url"):
                return r["url"]

    # 2. README map by page index (strip leading zeros)
    m = re.match(r"^(\d+)", name)
    if m:
        idx = str(int(m.group(1)))
        if idx in readme_map:
            return readme_map[idx]
        if m.group(1) in readme_map:
            return readme_map[m.group(1)]

    # 3. README map by name
    if name in readme_map:
        return readme_map[name]

    # 4. Bare-name slug fallback (e.g. "03_Main" → "/main")
    slug = re.sub(r"^\d+_+", "", name).lower().replace("_", "-")
    return f"/{slug}" if slug else "/"


# ----------------------------------------------------------------------
# Docker helpers (mirror eval_all_runs.sh logic)
# ----------------------------------------------------------------------

def workspace_has_compose(ws: Path) -> bool:
    return any((ws / n).is_file() for n in
               ("docker-compose.yml", "compose.yaml", "compose.yml"))


def docker_up(ws: Path, project: str, timeout: int = 300) -> bool:
    if not workspace_has_compose(ws):
        return False
    try:
        subprocess.run(
            ["docker", "compose", "-p", project, "up", "-d", "--build", "--wait"],
            cwd=ws, check=False, timeout=timeout,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.TimeoutExpired:
        print(f"  [docker] up timed out after {timeout}s", file=sys.stderr)
        return False


def docker_down(ws: Path, project: str) -> None:
    try:
        subprocess.run(
            ["docker", "compose", "-p", project, "down", "--remove-orphans", "-v"],
            cwd=ws, check=False, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def docker_nuke_others(skip_project: str) -> None:
    """Tear down anything still running on this host (single-docker-host assumption)."""
    try:
        proc = subprocess.run(
            ["docker", "compose", "ls", "-q"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        for p in (proc.stdout or "").splitlines():
            p = p.strip()
            if not p or p == skip_project:
                continue
            subprocess.run(
                ["docker", "compose", "-p", p, "down", "--remove-orphans"],
                check=False, timeout=30,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass


def wait_http(url: str, timeout: int = 90, step: int = 2) -> bool:
    """Poll URL until any HTTP response (2xx/3xx/4xx/5xx) or timeout."""
    import urllib.request, urllib.error
    elapsed = 0
    while elapsed < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            urllib.request.urlopen(req, timeout=5)
            return True
        except urllib.error.HTTPError:
            return True  # got a real HTTP response, even if 4xx/5xx
        except Exception:
            time.sleep(step)
            elapsed += step
    return False


# ----------------------------------------------------------------------
# Screenshot capture
# ----------------------------------------------------------------------

def capture_screenshots(base_url: str, page_urls: dict[str, str],
                        out_dir: Path,
                        viewport: tuple[int, int] = (1440, 900)) -> dict[str, Path]:
    """Visit each URL, save full-page PNG. Returns {page_name: png_path}."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium",
              file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    captured: dict[str, Path] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]},
            ignore_https_errors=True,
        )
        page = ctx.new_page()
        for name, url_path in page_urls.items():
            full_url = f"{base_url.rstrip('/')}{url_path}"
            png = out_dir / f"{name}.png"
            try:
                page.goto(full_url, wait_until="networkidle", timeout=15000)
            except Exception:
                try:
                    page.goto(full_url, wait_until="domcontentloaded", timeout=10000)
                except Exception as e:
                    print(f"  [shot] {name} → {full_url} : navigate failed ({e})", file=sys.stderr)
                    continue
            time.sleep(1.0)  # let SPA hydration settle
            try:
                page.screenshot(path=str(png), full_page=True)
                captured[name] = png
            except Exception as e:
                print(f"  [shot] {name} screenshot failed: {e}", file=sys.stderr)
        ctx.close()
        browser.close()
    return captured


# ----------------------------------------------------------------------
# Per-run pipeline
# ----------------------------------------------------------------------

def eval_one_run(run_dir: Path,
                 *,
                 model_name: str,
                 backend: str,
                 rotate_backends: Optional[list[str]],
                 skip_docker: bool,
                 force: bool,
                 rescore_only: bool,
                 http_timeout: int) -> dict:
    """Process one run dir; return its summary record (for the CSV)."""
    run_id = run_dir.name
    meta = read_meta(run_dir)
    workspace = run_dir / "workspace"
    out_json = run_dir / "logs" / "clip_similarity.json"
    out_shots = run_dir / "logs" / "clip_screenshots"

    record: dict = {
        "run_id": run_id,
        "task": meta.get("task", ""),
        "variant": meta.get("variant", ""),
        "cli": meta.get("cli", ""),
        "model": meta.get("model", ""),
        "n_pages": 0,
        "avg_clip": None,
        "min_clip": None,
        "max_clip": None,
        "status": "OK",
    }

    if not meta:
        record["status"] = "NO_META"
        return record

    # c0 is text-only by design (no mockup PNGs in the task description).
    # CLIP visual similarity has no reference image to compare against, so
    # skip it explicitly with a clear status instead of a generic NO_MOCKUPS.
    variant = meta.get("variant", "")
    if variant == "c0":
        record["status"] = "SKIP_TEXT_ONLY_VARIANT"
        return record

    project = meta.get("compose_project", "")
    fport = meta.get("frontend_port", 38000)

    # Cached? (only honored when neither --force nor --rescore-only)
    if out_json.exists() and not force and not rescore_only:
        try:
            cached = json.loads(out_json.read_text())
            scores = [s["clip_similarity"] for s in cached.get("pages", [])
                      if isinstance(s.get("clip_similarity"), (int, float))]
            if scores:
                record.update({
                    "n_pages": len(scores),
                    "avg_clip": round(sum(scores) / len(scores), 4),
                    "min_clip": round(min(scores), 4),
                    "max_clip": round(max(scores), 4),
                    "status": "CACHED",
                })
                return record
        except Exception:
            pass  # fall through and recompute

    mockups = list_mockups(run_dir)
    if not mockups:
        record["status"] = "NO_MOCKUPS"
        return record

    docker_was_up = False
    captured: dict[str, Path] = {}
    page_urls: dict[str, str] = {}
    base_url = f"http://localhost:{fport}"

    # Auto-detect: if clip_screenshots/ already has every mockup's PNG, treat
    # this as an implicit --rescore-only (skip docker + skip capture). User
    # can force a fresh capture with --force.
    auto_rescore = False
    if not rescore_only and not force and out_shots.is_dir():
        existing = {mp.stem for mp in mockups
                    if (out_shots / f"{mp.stem}.png").is_file()}
        if existing and len(existing) == len(mockups):
            auto_rescore = True
            print(f"  [auto-rescore] {out_shots.name}/ already has all "
                  f"{len(existing)} screenshots — skipping docker", file=sys.stderr)

    if rescore_only or auto_rescore:
        # Re-score using existing screenshots — skip docker + capture.
        if not out_shots.is_dir():
            record["status"] = "NO_EXISTING_SCREENSHOTS"
            print(f"  [rescore] {out_shots} doesn't exist; need a fresh run first",
                  file=sys.stderr)
            return record
        for mp in mockups:
            png = out_shots / f"{mp.stem}.png"
            if png.is_file():
                captured[mp.stem] = png
        if not captured:
            record["status"] = "NO_EXISTING_SCREENSHOTS"
            return record
        if not auto_rescore:
            print(f"  [rescore] reusing {len(captured)} existing screenshots", file=sys.stderr)
    else:
        # ---------- bring up agent's app ----------
        if not skip_docker:
            if not workspace_has_compose(workspace):
                record["status"] = "NO_COMPOSE"
                return record
            docker_nuke_others(skip_project=project)
            if not docker_up(workspace, project):
                record["status"] = "DOCKER_UP_FAILED"
                return record
            docker_was_up = True

            if not wait_http(f"http://localhost:{fport}/", timeout=http_timeout):
                print(f"  [http] frontend not responding within {http_timeout}s", file=sys.stderr)
                # Don't bail — try to capture whatever's there

        # ---------- resolve URLs ----------
        readme_map = parse_readme_pages(workspace)
        existing_eval = None
        eval_path = run_dir / "logs" / "eval_result.json"
        if eval_path.exists():
            try:
                existing_eval = json.loads(eval_path.read_text())
            except Exception:
                pass

        page_urls = {}
        for mp in mockups:
            url = resolve_page_url(mp, readme_map, existing_eval)
            if url:
                page_urls[mp.stem] = url

        # ---------- capture clean screenshots ----------
        captured = capture_screenshots(base_url, page_urls, out_shots)

    # ---------- CLIP per page ----------
    page_results = []
    for mp in mockups:
        name = mp.stem
        shot = captured.get(name)
        rec = {
            "page": name,
            "mockup": str(mp.relative_to(run_dir)),
            "url": page_urls.get(name),
            "screenshot": str(shot.relative_to(run_dir)) if shot else None,
            "clip_similarity": None,
        }
        if shot:
            try:
                sim = clip_similarity(mp, shot,
                                      model_name=model_name, backend=backend,
                                      rotate_backends=rotate_backends)
                rec["clip_similarity"] = round(sim, 4)
            except EmbedError as e:
                # ANY embedding failure → abort writing this run's JSON entirely.
                # Screenshots are kept (caller may reuse them via --rescore-only).
                print(f"  [clip] {name}: {e} — abort run (no JSON written)", file=sys.stderr)
                record["status"] = f"EMBED_FAILED: {str(e)[:80]}"
                if docker_was_up:
                    docker_down(workspace, project)
                return record
        page_results.append(rec)

    # ---------- save per-run JSON ----------
    summary = {
        "run_id": run_id,
        "model": model_name,
        "backend": backend,
        "base_url": base_url,
        "pages": page_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    # ---------- compute aggregates ----------
    scores = [r["clip_similarity"] for r in page_results
              if isinstance(r["clip_similarity"], (int, float))]
    if scores:
        record.update({
            "n_pages": len(scores),
            "avg_clip": round(sum(scores) / len(scores), 4),
            "min_clip": round(min(scores), 4),
            "max_clip": round(max(scores), 4),
        })

    # ---------- tear down ----------
    if docker_was_up:
        docker_down(workspace, project)

    return record


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_root", nargs="?", default=None,
                    help="path to _runs directory (default: ./_runs)")
    ap.add_argument("--filter", default="*",
                    help="glob filter on run dir names (default: *)")
    ap.add_argument("--skip-docker", action="store_true",
                    help="don't bring docker up/down — assume services running")
    ap.add_argument("--force", action="store_true",
                    help="re-run from scratch (re-bring-up docker, re-screenshot, re-embed) even if clip_similarity.json exists")
    ap.add_argument("--rescore-only", action="store_true",
                    help="reuse existing logs/clip_screenshots/*.png — skip docker + skip capture, "
                         "only re-compute CLIP similarity. Overwrites clip_similarity.json. "
                         "Use this to switch backends or models without re-running docker / screenshots.")
    ap.add_argument("--model", default="ViT-B-32",
                    help="model name. local: open_clip name (ViT-B-32 / ViT-L-14). "
                         "huggingface: HF repo (openai/clip-vit-base-patch32). "
                         "replicate / vertex: backend default")
    ap.add_argument("--backend", default="local",
                    choices=("local", "replicate", "huggingface", "vertex", "rotate"),
                    help="where to compute embeddings. "
                         "local = open_clip on this CPU (free, ~100ms/img). "
                         "replicate = Replicate API ($0.001/img, needs REPLICATE_API_TOKEN; "
                         "  rate-limited to 6 RPM unless account credit ≥ $5). "
                         "huggingface = HF Inference (free tier ~50 RPM, needs HF_TOKEN). "
                         "vertex = Vertex AI ($0.0002/img, needs GOOGLE_CLOUD_PROJECT + creds). "
                         "rotate = round-robin over --rotate-backends (default: replicate,huggingface) "
                         "  to double effective throughput when one is rate-limited.")
    ap.add_argument("--rotate-backends", default="replicate,huggingface",
                    help="comma-separated backends to cycle through for --backend rotate "
                         "(default: replicate,huggingface)")
    ap.add_argument("--http-timeout", type=int, default=90,
                    help="seconds to wait for frontend HTTP after docker up (default: 90)")
    args = ap.parse_args()

    # Resolve runs_root
    here = Path(__file__).parent.parent  # tools/.. = tasks/
    runs_root = Path(args.runs_root) if args.runs_root else here / "_runs"
    runs_root = runs_root.resolve()
    if not runs_root.is_dir():
        print(f"ERROR: runs_root not found: {runs_root}", file=sys.stderr)
        return 1

    # A real run dir always has a meta.json. Folders like `_batch_logs`
    # (created by run_all.sh) and other ad-hoc directories under _runs/
    # don't, so filter them out instead of marking them NO_META later.
    # Also skip names starting with `_` and `.` as a sanity guard.
    run_dirs = sorted(
        d for d in runs_root.iterdir()
        if d.is_dir()
        and not d.name.startswith(("_", "."))
        and (d / "meta.json").is_file()
        and fnmatch.fnmatch(d.name, args.filter)
    )
    if not run_dirs:
        print(f"No run dirs match filter '{args.filter}' under {runs_root}", file=sys.stderr)
        return 0

    print("=" * 64)
    print(f"CLIP eval")
    print(f"  runs_root:    {runs_root}")
    print(f"  filter:       {args.filter}")
    print(f"  backend:      {args.backend}")
    print(f"  model:        {args.model}")
    print(f"  skip-docker:  {args.skip_docker}")
    print(f"  force:        {args.force}")
    print(f"  n runs:       {len(run_dirs)}")
    print("=" * 64)

    records: list[dict] = []
    t_start = time.time()
    for i, rd in enumerate(run_dirs, 1):
        print(f"\n[{i}/{len(run_dirs)}] {rd.name}")
        try:
            rotate_backends = [b.strip() for b in args.rotate_backends.split(",") if b.strip()]
            rec = eval_one_run(
                rd,
                model_name=args.model,
                backend=args.backend,
                rotate_backends=rotate_backends,
                skip_docker=args.skip_docker,
                force=args.force,
                rescore_only=args.rescore_only,
                http_timeout=args.http_timeout,
            )
        except KeyboardInterrupt:
            print("\n[clip] interrupted", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            rec = {"run_id": rd.name, "status": f"EXCEPTION: {e}"}
        records.append(rec)
        if rec.get("avg_clip") is not None:
            print(f"  → avg_clip={rec['avg_clip']:.3f}  "
                  f"(min={rec['min_clip']:.3f}, max={rec['max_clip']:.3f}, "
                  f"n={rec['n_pages']}, status={rec['status']})")
        else:
            print(f"  → (no scores; status={rec.get('status','?')})")

    # ---------- aggregate CSV ----------
    csv_path = runs_root / "clip_summary.csv"
    fields = ["run_id", "task", "variant", "cli", "model",
              "n_pages", "avg_clip", "min_clip", "max_clip", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})

    print()
    print("=" * 64)
    print(f" Wrote {csv_path}  ({time.time()-t_start:.0f}s total)")
    print("=" * 64)

    # Pretty leaderboard (top by avg_clip)
    ranked = sorted(
        [r for r in records if isinstance(r.get("avg_clip"), (int, float))],
        key=lambda r: -r["avg_clip"],
    )
    if ranked:
        print()
        print("Leaderboard (avg_clip desc):")
        print(f"  {'avg':>6}  {'min':>6}  {'max':>6}  {'n':>3}  run_id")
        for r in ranked[:30]:
            print(f"  {r['avg_clip']:>6.3f}  {r['min_clip']:>6.3f}  "
                  f"{r['max_clip']:>6.3f}  {r['n_pages']:>3}  {r['run_id']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
