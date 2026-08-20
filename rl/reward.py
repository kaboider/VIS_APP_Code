"""verl custom reward for VIS_APP camel-style coding rollouts.

Primary signal is a workspace-directory structural score (compose + yaml +
frontend + backend). Token-level PRM and LLM-as-judge are out of scope.
Playwright ``combined_score_critical`` is a Phase 2 hook only: if the value is
already present on extra_info or in eval_result.json, it overrides the proxy.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rl.workspace import score_files, score_workspace


_XML_WRITE = re.compile(
    r"<function=write_file>\s*"
    r"<parameter=path>(.*?)</parameter>\s*"
    r"<parameter=content>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def extract_xml_write_files(solution_str: str) -> dict[str, str]:
    """Parse Qwen XML write_file calls out of a decoded trajectory."""
    files: dict[str, str] = {}
    for match in _XML_WRITE.finditer(solution_str or ""):
        path = (match.group(1) or "").strip().lstrip("./")
        if path:
            files[path.replace("\\", "/")] = match.group(2)
    return files


def _extract_files(solution_str: str) -> dict[str, str]:
    files: dict[str, str] = {}
    text = solution_str or ""
    # Language tag, if present, must sit on the same line as the path.
    # Do not let `\s` eat the newline or ` ```docker-compose.yml` is parsed as a
    # language with the first content line as the path.
    fence = re.compile(
        r"```(?:(?:[\w.+-]+)[^\S\n]+)?([^\n`]+)\n(.*?)```",
        re.DOTALL,
    )
    for match in fence.finditer(text):
        path = match.group(1).strip().strip("`").strip()
        path = path.split()[0] if path else ""
        path = path.lstrip("./")
        if not path or path.lower() in {"python", "bash", "json", "yaml", "yml", "html", "js", "ts"}:
            body = match.group(2)
            if "services:" in body and path.lower() in {"yaml", "yml"}:
                files.setdefault("docker-compose.yml", body)
            continue
        files[path.replace("\\", "/")] = match.group(2)
    if not files and "services:" in text:
        files["docker-compose.yml"] = text
    return files


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _phase2_score(extra_info: dict[str, Any] | None) -> float | None:
    if not extra_info:
        return None
    direct = _as_float(extra_info.get("combined_score_critical"))
    if direct is not None:
        return direct
    for nested_key in ("eval", "eval_result", "summary", "rollout_reward_scores"):
        nested = extra_info.get(nested_key)
        if isinstance(nested, dict):
            found = _as_float(nested.get("combined_score_critical"))
            if found is not None:
                return found
    return None


def _workspace_path_from_extra(extra_info: dict[str, Any] | None) -> Path | None:
    if not extra_info:
        return None
    for key in ("workspace_dir", "workspace"):
        raw = extra_info.get(key)
        if raw:
            path = Path(str(raw))
            if path.exists():
                return path
    tef = extra_info.get("tool_extra_fields")
    if isinstance(tef, dict):
        for key in ("workspace_dir", "workspace"):
            raw = tef.get(key)
            if raw:
                path = Path(str(raw))
                if path.exists():
                    return path
    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    _ = data_source, ground_truth
    phase2 = _phase2_score(extra_info)
    if phase2 is not None:
        return float(min(1.0, max(0.0, phase2)))

    if extra_info:
        prior = _as_float(extra_info.get("score"))
        if prior is not None and prior > 0:
            return float(min(1.0, max(0.0, prior)))

    ws = _workspace_path_from_extra(extra_info)
    if ws is not None:
        return float(score_workspace(ws)["score"])

    files = _extract_files(solution_str)
    files.update(extract_xml_write_files(solution_str))
    return float(score_files(files)["score"])
