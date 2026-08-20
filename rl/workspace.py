"""Rollout workspace helpers and structural (no-Docker) scoring."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

COMPOSE_NAMES = ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml")
FRONTEND_NAMES = (
    "index.html", "home.html", "base.html", "app.html",
    "app.tsx", "app.jsx", "app.vue", "page.tsx", "page.jsx",
    "+page.svelte", "main.tsx", "main.jsx",
)
_FRONTEND_SUFFIXES = (".html", ".tsx", ".jsx", ".vue", ".svelte")
BACKEND_NAMES = (
    "main.py", "app.py", "server.js", "server.ts", "index.js", "index.ts",
    "route.ts", "urls.py", "manage.py",
)
SKIP_DIR_NAMES = frozenset({
    "node_modules", ".git", ".next", "dist", "build", ".initial_env",
    "terminal_logs", "__pycache__",
})

_COMPOSE_SET = {n.lower() for n in COMPOSE_NAMES}
_FRONTEND_SET = {n.lower() for n in FRONTEND_NAMES}
_BACKEND_SET = {n.lower() for n in BACKEND_NAMES}

_WEIGHT_COMPOSE_NAME = 0.40
_WEIGHT_COMPOSE_OK = 0.20
_WEIGHT_FRONTEND = 0.20
_WEIGHT_BACKEND = 0.20


def _basename(path: str) -> str:
    return path.rstrip("/").split("/")[-1].lower()


def is_junk_relpath(rel: str) -> bool:
    parts = Path(rel.replace("\\", "/")).parts
    return any(p in SKIP_DIR_NAMES for p in parts)


def _has_named(files: dict[str, str], names: set[str]) -> bool:
    return any(_basename(p) in names for p in files)


def _compose_ok(files: dict[str, str]) -> bool:
    body = ""
    for path, content in files.items():
        if _basename(path) in _COMPOSE_SET:
            body = content or ""
            break
    if not body:
        return False
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(body)
        return isinstance(data, dict) and bool(data.get("services"))
    except Exception:
        return bool(re.search(r"(?m)^services:\s*$", body)) or ("services:" in body)


def score_files(files: dict[str, str]) -> dict[str, Any]:
    """Score a path→content map (disk scan or markdown-fence fallback)."""
    has_compose = _has_named(files, _COMPOSE_SET)
    compose_ok = _compose_ok(files)
    has_frontend = _has_named(files, _FRONTEND_SET) or any(
        p.lower().endswith(_FRONTEND_SUFFIXES) for p in files
    )
    has_backend = _has_named(files, _BACKEND_SET)
    score = 0.0
    if has_compose:
        score += _WEIGHT_COMPOSE_NAME
    if compose_ok:
        score += _WEIGHT_COMPOSE_OK
    if has_frontend:
        score += _WEIGHT_FRONTEND
    if has_backend:
        score += _WEIGHT_BACKEND
    # Small density term so GRPO groups are not all-zero before compose exists.
    score += min(0.15, 0.015 * len(files))
    score = float(min(1.0, round(score, 4)))
    return {
        "score": score,
        "has_compose": bool(has_compose),
        "compose_ok": bool(compose_ok),
        "has_frontend": bool(has_frontend),
        "has_backend": bool(has_backend),
        "n_files": int(len(files)),
        "combined_score_critical": score,
    }


def _eval_override(workspace_dir: Path) -> float | None:
    """Phase 2 hook: honor Playwright/harness combined_score_critical if present."""
    parent = workspace_dir.parent
    candidates = (
        workspace_dir / "eval_result.json",
        workspace_dir / "logs" / "eval_result.json",
        parent / "logs" / "eval_result.json",
        parent / "eval_result.json",
    )
    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            data = json.loads(cand.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        blobs: list[Any] = [data]
        if isinstance(data, dict):
            blobs.append(data.get("summary"))
            blobs.append(data.get("eval"))
        for obj in blobs:
            if not isinstance(obj, dict):
                continue
            if "combined_score_critical" in obj and obj["combined_score_critical"] is not None:
                try:
                    return float(obj["combined_score_critical"])
                except (TypeError, ValueError):
                    continue
    return None


def workspace_is_complete(workspace_dir: Path | str | None) -> bool:
    """True when compose + frontend + backend are all present on disk."""
    if not workspace_dir:
        return False
    result = score_workspace(workspace_dir)
    return bool(result.get("has_compose") and result.get("has_frontend") and result.get("has_backend"))


def score_workspace(workspace_dir: Path | str) -> dict[str, Any]:
    """Scan files on disk and return a structural score in [0, 1].

    Does not call docker or Playwright. If eval_result.json already contains
    ``combined_score_critical``, that value overrides the structural proxy.
    """
    root = Path(workspace_dir)
    empty = {
        "score": 0.0,
        "has_compose": False,
        "compose_ok": False,
        "has_frontend": False,
        "has_backend": False,
        "n_files": 0,
        "combined_score_critical": 0.0,
    }
    if not root.is_dir():
        return empty

    files: dict[str, str] = {}
    n_files = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if is_junk_relpath(rel):
            continue
        n_files += 1
        if _basename(rel) in _COMPOSE_SET:
            try:
                files[rel] = path.read_text(encoding="utf-8", errors="replace")[:100_000]
            except OSError:
                files[rel] = ""
        else:
            files[rel] = ""

    result = score_files(files)
    result["n_files"] = n_files
    override = _eval_override(root)
    if override is not None:
        clipped = float(min(1.0, max(0.0, override)))
        result["score"] = clipped
        result["combined_score_critical"] = clipped
    return result


def make_rollout_workspace(task_dir: str | Path | None, extra_info: dict[str, Any] | None = None) -> Path:
    """Create a temp rollout tree: ``inputs/`` (copied spec) + empty ``workspace/``.

    Returns the workspace *root* (parent of ``workspace/`` and ``inputs/``).
    """
    _ = extra_info
    base = os.environ.get("VISAPP_RL_WORKSPACE_ROOT") or None
    if base:
        Path(base).mkdir(parents=True, exist_ok=True)
    workspace_root = Path(tempfile.mkdtemp(prefix="visapp_rl_", dir=base))
    inputs = workspace_root / "inputs"
    workspace = workspace_root / "workspace"
    inputs.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    src = Path(task_dir) if task_dir else None
    if src is not None and src.is_dir():
        desc = src / "description.md"
        if desc.is_file():
            shutil.copy2(desc, inputs / "description.md")
        pages = src / "pages"
        if pages.is_dir():
            shutil.copytree(pages, inputs / "pages", dirs_exist_ok=True)
    return workspace_root


def cleanup_workspace(path: Path | str | None) -> None:
    if not path:
        return
    shutil.rmtree(path, ignore_errors=True)


def keep_workspace_enabled() -> bool:
    return os.environ.get("VISAPP_RL_KEEP_WORKSPACE", "").strip().lower() in {"1", "true", "yes", "on"}
