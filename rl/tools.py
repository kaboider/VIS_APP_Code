"""Native verl tools that operate inside a per-rollout workspace directory.

Workspace path lives on ``agent_data.extra_fields["workspace_dir"]`` because
verl create/release is per tool *call*, not per trajectory.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from rl.workspace import SKIP_DIR_NAMES, is_junk_relpath

_WRITE_MAX_CHARS = 200_000
_READ_MAX_CHARS = 8_000
_LIST_MAX_ENTRIES = 400
_SHELL_DEFAULT_TIMEOUT = 30
_SHELL_MAX_TIMEOUT = 60
_SHELL_OUTPUT_CHARS = 8_000

_DENY_DOCKER = re.compile(r"\b(docker|docker-compose|podman|nerdctl)\b", re.I)
_DENY_SUDO = re.compile(r"\b(sudo|pkexec)\b", re.I)
_DENY_RM_ROOT = re.compile(
    r"\brm\s+(?:-[a-zA-Z]*r[a-zA-Z]*f|[a-zA-Z]*-fr|--recursive\s+-f|-f\s+--recursive)\s+(?:/\s*(?:\*|home|usr|etc|bin)?|~(?:/|\s|$))",
    re.I,
)
_DENY_RUNTIME_FETCH = re.compile(
    r"(wget|curl).*(nodejs\.org|python\.org|nvm\.sh|fnm\.sh|miniconda|anaconda\.com|pyenv|"
    r"node-v\d|python-\d)",
    re.I,
)
_DENY_PIPE_SHELL = re.compile(r"(curl|wget).*\|\s*(?:sudo\s+)?(?:ba)?sh\b", re.I)
_DENY_TARBALL_PM = re.compile(
    r"\b(npm|npx|yarn|pnpm|pip3?|easy_install)\b.*https?://[^\s;|&]+\.(?:tgz|tar\.gz|zip)\b",
    re.I,
)


class WorkspaceMissing(RuntimeError):
    pass


def get_workspace(agent_data: Any) -> Path:
    extra = getattr(agent_data, "extra_fields", None) if agent_data is not None else None
    if not isinstance(extra, dict):
        extra = {}
    raw = extra.get("workspace_dir") or extra.get("workspace")
    if not raw:
        raise WorkspaceMissing("workspace_dir missing on agent_data.extra_fields")
    path = Path(str(raw))
    if not path.is_dir():
        raise WorkspaceMissing(f"workspace_dir does not exist: {path}")
    return path.resolve()


def _workspace_from_kwargs(kwargs: dict[str, Any]) -> Path:
    return get_workspace(kwargs.get("agent_data"))


def _ok(text: str, reward: float = 0.0) -> tuple[ToolResponse, float, dict]:
    return ToolResponse(text=text), reward, {}


def _err(text: str) -> tuple[ToolResponse, float, dict]:
    return ToolResponse(text=text), 0.0, {}


def resolve_tool_path(
    workspace: Path,
    path: str,
    *,
    allow_inputs: bool = False,
    allow_dotdot: bool = False,
) -> Path:
    """Resolve ``path`` inside the rollout workspace (and optionally ``../inputs``)."""
    if path is None or str(path).strip() == "":
        raise ValueError("path is required")
    raw = str(path).strip()
    if os.path.isabs(raw):
        raise ValueError("absolute paths are not allowed")
    parts = Path(raw.replace("\\", "/")).parts
    if not allow_dotdot and ".." in parts:
        raise ValueError("path traversal (..) is not allowed")
    candidate = (workspace / raw).resolve()
    ws = workspace.resolve()
    if candidate == ws or ws in candidate.parents:
        return candidate
    if allow_inputs:
        inputs = (ws.parent / "inputs").resolve()
        if candidate == inputs or inputs in candidate.parents:
            return candidate
    raise ValueError("path is outside the workspace")


class WriteFileTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        _ = instance_id
        try:
            workspace = _workspace_from_kwargs(kwargs)
            dest = resolve_tool_path(workspace, parameters.get("path", ""), allow_dotdot=False)
            content = parameters.get("content")
            if content is None:
                print(
                    f"[visapp] write_file error: content missing path={parameters.get('path')}",
                    flush=True,
                )
                return _err("Error: content is required")
            text = content if isinstance(content, str) else str(content)
            if len(text) > _WRITE_MAX_CHARS:
                return _err(f"Error: content too large ({len(text)} chars, max {_WRITE_MAX_CHARS})")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            rel = dest.relative_to(workspace).as_posix()
            print(f"[visapp] write_file {rel} ({len(text)} chars)", flush=True)
            return _ok(f"wrote {rel} ({len(text)} chars)")
        except (WorkspaceMissing, ValueError, OSError) as exc:
            print(f"[visapp] write_file error: {exc}", flush=True)
            return _err(f"Error: {exc}")


class ReadFileTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        _ = instance_id
        try:
            workspace = _workspace_from_kwargs(kwargs)
            src = resolve_tool_path(
                workspace,
                parameters.get("path", ""),
                allow_inputs=True,
                allow_dotdot=True,
            )
            if not src.is_file():
                hint = ""
                if str(parameters.get("path") or "").rstrip("/").endswith(
                    ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "index.html", "app.py")
                ) or str(parameters.get("path") or "") in {".", ""}:
                    hint = " Call write_file to create it."
                return _err(f"Error: not a file: {parameters.get('path')}.{hint}")
            text = src.read_text(encoding="utf-8", errors="replace")
            truncated = False
            if len(text) > _READ_MAX_CHARS:
                text = text[:_READ_MAX_CHARS]
                truncated = True
            suffix = "\n...[truncated]" if truncated else ""
            return _ok(text + suffix)
        except (WorkspaceMissing, ValueError, OSError) as exc:
            return _err(f"Error: {exc}")


class ListDirTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        _ = instance_id
        try:
            workspace = _workspace_from_kwargs(kwargs)
            raw = parameters.get("path") or "."
            target = resolve_tool_path(
                workspace, str(raw), allow_inputs=True, allow_dotdot=True
            )
            if not target.is_dir():
                return _err(f"Error: not a directory: {raw}")
            entries: list[str] = []
            try:
                children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError as exc:
                return _err(f"Error: {exc}")
            for child in children:
                if child.name in SKIP_DIR_NAMES or child.name.startswith(".git"):
                    continue
                try:
                    rel = child.relative_to(workspace).as_posix()
                except ValueError:
                    try:
                        rel = child.relative_to(workspace.parent).as_posix()
                    except ValueError:
                        rel = child.name
                if is_junk_relpath(rel):
                    continue
                marker = "/" if child.is_dir() else ""
                entries.append(f"{rel}{marker}")
                if len(entries) >= _LIST_MAX_ENTRIES:
                    entries.append("...[truncated]")
                    break
            if not entries:
                return _ok(
                    "(empty) workspace has no files yet. "
                    "Call write_file with path=docker-compose.yml next."
                )
            return _ok("\n".join(entries))
        except (WorkspaceMissing, ValueError, OSError) as exc:
            return _err(f"Error: {exc}")


def _shell_denied(command: str) -> str | None:
    if _DENY_DOCKER.search(command):
        return "docker/podman is not allowed during GRPO rollouts"
    if _DENY_SUDO.search(command):
        return "sudo is not allowed"
    if _DENY_RM_ROOT.search(command) or re.search(r"\brm\s+-rf\s+/\s*$", command):
        return "refusing destructive rm of /"
    if _DENY_PIPE_SHELL.search(command):
        return "piping curl/wget into a shell is not allowed"
    if _DENY_RUNTIME_FETCH.search(command):
        return "downloading Node/Python runtimes is not allowed"
    if _DENY_TARBALL_PM.search(command):
        return "installing node/python tarballs over the network is not allowed"
    tokens = re.split(r"\s+", command.strip(), maxsplit=1)
    head = tokens[0] if tokens else ""
    if head in {"npm", "npx", "yarn", "pnpm"} and shutil.which(head) is None:
        return f"{head} is not installed on PATH"
    return None


class ShellExecTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        _ = instance_id
        try:
            workspace = _workspace_from_kwargs(kwargs)
        except WorkspaceMissing as exc:
            return _err(f"Error: {exc}")
        command = parameters.get("command")
        if not command or not str(command).strip():
            return _err("Error: command is required")
        command = str(command)
        denied = _shell_denied(command)
        if denied:
            return _err(f"Error: {denied}")
        timeout = parameters.get("timeout", _SHELL_DEFAULT_TIMEOUT)
        try:
            timeout_s = int(timeout)
        except (TypeError, ValueError):
            timeout_s = _SHELL_DEFAULT_TIMEOUT
        timeout_s = max(1, min(timeout_s, _SHELL_MAX_TIMEOUT))

        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                command,
                shell=True,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={**os.environ, "PWD": str(workspace)},
            )

        try:
            proc = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired:
            return _err(f"Error: command timed out after {timeout_s}s")
        except OSError as exc:
            return _err(f"Error: {exc}")
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        if len(out) > _SHELL_OUTPUT_CHARS:
            keep = _SHELL_OUTPUT_CHARS // 2
            out = out[:keep] + "\n...(truncated)...\n" + out[-keep:]
        return _ok(f"exit={proc.returncode}\n{out}".rstrip())
