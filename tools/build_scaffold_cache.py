#!/usr/bin/env python3
"""
build_scaffold_cache.py — pre-generate scaffold projects and cache their file lists.

Reads tools/scaffold_cache/registry.json, runs each scaffold command in a
temp dir, and writes <key>.json under tools/scaffold_cache/ containing every
scaffold-generated file path + SHA256 hash.

analyze_edits.py then loads these caches to classify Writes as
"new" / "overwrite_scaffold" / "overwrite_self".

Usage:
    python3 build_scaffold_cache.py             # build all entries
    python3 build_scaffold_cache.py KEY [KEY..] # build only the named entries
    python3 build_scaffold_cache.py --list      # list registry entries

node_modules/, .git/, package-lock.json are excluded from the cache by
default — they are noisy and agents do not Write to them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_NAME = "scaffold_probe"

EXCLUDE_DIR_NAMES = {
    "node_modules",
    ".git",
    ".turbo",
    ".next",
    ".nuxt",
    ".output",
    ".vercel",
    "dist",
    "build",
}
EXCLUDE_FILE_NAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lockb",
    ".DS_Store",
}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in root.rglob("*"):
        # skip directories themselves (we want files)
        if not p.is_file():
            continue
        # skip anything under an excluded directory
        rel = p.relative_to(root)
        parts = rel.parts
        if any(seg in EXCLUDE_DIR_NAMES for seg in parts):
            continue
        if p.name in EXCLUDE_FILE_NAMES:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        try:
            digest = sha256_of(p)
        except OSError:
            digest = ""
        out[str(rel)] = {"size": size, "sha256": digest}
    return out


def run_scaffold(cmd_template: str, work_dir: Path, timeout_s: int = 90) -> tuple[bool, str]:
    cmd = cmd_template.format(project=PROJECT_NAME)
    print(f"  $ {cmd}")
    # Open /dev/null for stdin so any unexpected interactive prompt fails
    # immediately (CLIs that read EOF and crash) instead of hanging the full
    # timeout. Force CI=1 / non-TTY env to encourage non-interactive behavior.
    env = os.environ.copy()
    env.setdefault("CI", "1")
    env.setdefault("npm_config_yes", "true")
    try:
        with open(os.devnull, "rb") as devnull:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=work_dir,
                check=False,
                stdin=devnull,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout_s}s (probably hit an interactive prompt; try simplifying the command)"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return False, f"exit={proc.returncode}\n{tail}"
    return True, ""


def build_one(key: str, entry: dict[str, Any], cache_dir: Path, timeout_s: int = 90) -> bool:
    print(f"[{key}] {entry.get('description', '')}")
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"scaffold_{key}_") as td:
        td_path = Path(td)
        ok, err = run_scaffold(entry["command"], td_path, timeout_s=timeout_s)
        project_root = td_path / PROJECT_NAME
        if not project_root.exists():
            # Some scaffolds drop files directly in cwd (no nested dir). Try td itself.
            project_root = td_path
        files = collect_files(project_root)
        # Tolerate non-zero exit codes IF the scaffold actually produced files
        # (e.g. create-react-router exits 1 on a benign "git init failed" warning).
        if not ok and not files:
            print(f"  FAILED ({key}): {err}", file=sys.stderr)
            return False
        if not ok and files:
            print(f"  WARNING ({key}): non-zero exit but {len(files)} files produced — proceeding.")
        elapsed = time.time() - t0
        out = {
            "key": key,
            "command": entry["command"],
            "framework": entry.get("framework"),
            "description": entry.get("description"),
            "match_patterns": entry.get("match_patterns", []),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed, 1),
            "file_count": len(files),
            "files": files,
        }
        cache_path = cache_dir / f"{key}.json"
        cache_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  ok: {len(files)} files cached → {cache_path.name} ({elapsed:.1f}s)")
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", nargs="*", help="specific registry keys to build (default: all)")
    parser.add_argument("--list", action="store_true", help="list registry entries and exit")
    parser.add_argument("--force", action="store_true", help="rebuild even if cache exists")
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="per-command timeout in seconds (default: 90; raise for slow networks)",
    )
    args = parser.parse_args()

    cache_dir = Path(__file__).parent / "scaffold_cache"
    registry_path = cache_dir / "registry.json"
    if not registry_path.exists():
        print(f"registry not found: {registry_path}", file=sys.stderr)
        return 1
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    scaffolds: dict[str, dict[str, Any]] = registry.get("scaffolds", {})

    if args.list:
        for key, entry in scaffolds.items():
            cached = (cache_dir / f"{key}.json").exists()
            mark = "[cached]" if cached else "[       ]"
            print(f"  {mark} {key:<22} {entry.get('description', '')}")
        return 0

    if args.keys:
        target = [(k, scaffolds[k]) for k in args.keys if k in scaffolds]
        unknown = [k for k in args.keys if k not in scaffolds]
        for k in unknown:
            print(f"  unknown key: {k}", file=sys.stderr)
    else:
        target = list(scaffolds.items())

    if not shutil.which("npx"):
        print("npx not found in PATH — install Node.js", file=sys.stderr)
        return 1

    failures = 0
    for key, entry in target:
        cache_path = cache_dir / f"{key}.json"
        if cache_path.exists() and not args.force:
            print(f"[{key}] cache exists ({cache_path.name}) — skip (use --force to rebuild)")
            continue
        if not build_one(key, entry, cache_dir, timeout_s=args.timeout):
            failures += 1

    print(f"\nDone. {len(target) - failures}/{len(target)} succeeded.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
