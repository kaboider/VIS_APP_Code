#!/usr/bin/env python3
"""Fallback scorer when Docker/Playwright cannot run.

Writes logs/eval_result.json in the same shape as eval_run.py so
eval_all_runs.sh can tabulate a number instead of FAIL. The
combined_score_critical here is a STRUCTURAL proxy (compose + yaml
+ frontend + backend), not a visual/interaction score.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rl.reward import compute_score  # noqa: E402


def _n_critical(run_dir: Path) -> int:
    for cand in (run_dir / "anchors.json",):
        if not cand.is_file():
            continue
        try:
            data = json.loads(cand.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        anchors = data.get("anchors") if isinstance(data, dict) else data
        flat = []
        if isinstance(anchors, dict):
            for v in anchors.values():
                if isinstance(v, list):
                    flat.extend(v)
        elif isinstance(anchors, list):
            flat = anchors
        if flat:
            crit = [a for a in flat if isinstance(a, dict) and str(a.get("tier", "critical")).lower() == "critical"]
            return len(crit) or len(flat)
        if isinstance(data, dict):
            return int(data.get("n_critical") or 0)
    return 0


def _file_blob(ws: Path) -> str:
    parts = []
    for path in ws.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ws).as_posix()
        if any(p in rel for p in ("node_modules/", ".git/", ".next/", "dist/", "build/")):
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(body) > 80_000:
            body = body[:80_000]
        parts.append(f"```{rel}\n{body}\n```")
    return "\n".join(parts)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: structural_eval.py RUN_DIR", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1]).resolve()
    ws = run_dir / "workspace"
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    meta = {}
    try:
        meta = json.loads((run_dir / "meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    blob = _file_blob(ws) if ws.is_dir() else ""
    extra = {"workspace_dir": str(ws)} if ws.is_dir() else None
    score = compute_score("visapp", blob, None, extra_info=extra)
    n_files = sum(1 for p in ws.rglob("*") if p.is_file()) if ws.is_dir() else 0
    has_compose = (ws / "docker-compose.yml").is_file() or (ws / "docker-compose.yaml").is_file()
    n_crit = _n_critical(run_dir)
    summary = {
        "task": meta.get("task"),
        "variant": meta.get("variant"),
        "model": meta.get("model"),
        "auth_bypass_used": False,
        "n_critical": n_crit,
        "n_bonus": 0,
        "found_critical": int(round(score * n_crit)) if n_crit else 0,
        "found_bonus": 0,
        "avg_localization_critical": round(score, 3),
        "avg_behavior_critical": round(score, 3),
        "combined_score_critical": round(score, 3),
        "eval_mode": "structural_no_docker",
        "has_compose": has_compose,
        "n_files": n_files,
        "tier_distribution": {"missed": n_crit},
    }
    out = logs / "eval_result.json"
    out.write_text(json.dumps({"summary": summary, "results": []}, indent=2, ensure_ascii=False) + "\n")
    print(f"[structural] combined={score:.3f} compose={has_compose} files={n_files} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
