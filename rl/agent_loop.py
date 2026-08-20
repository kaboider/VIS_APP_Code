"""VIS_APP multi-turn tool agent loop for verl GRPO."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop

from rl.reward import extract_xml_write_files
from rl.tools import resolve_tool_path
from rl.workspace import (
    cleanup_workspace,
    keep_workspace_enabled,
    make_rollout_workspace,
    score_files,
    score_workspace,
    workspace_is_complete,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)


def _bounded_sampling_params(
    sampling_params: dict,
    *,
    prompt_tokens: int,
    response_tokens: int,
    max_model_length: int,
    max_response_length: int,
) -> dict | None:
    """Limit one generation to the context and rollout space still available."""
    available = min(
        max_model_length - prompt_tokens,
        max_response_length - response_tokens,
    )
    if available <= 0:
        return None
    bounded = dict(sampling_params)
    configured = bounded.get("max_tokens")
    if configured is None:
        configured = bounded.get("max_new_tokens")
    if configured is None or int(configured) > available:
        bounded["max_tokens"] = int(available)
    return bounded


def _task_dir_from_kwargs(extra_info: dict[str, Any], kwargs: dict[str, Any]) -> str | None:
    task_dir = extra_info.get("task_dir")
    if task_dir:
        return str(task_dir)
    gt: Any = extra_info.get("ground_truth")
    if gt is None:
        rm = kwargs.get("reward_model") or {}
        if isinstance(rm, dict):
            gt = rm.get("ground_truth")
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except json.JSONDecodeError:
            gt = None
    if isinstance(gt, dict):
        td = gt.get("task_dir")
        if td:
            return str(td)
    return None


@register("visapp_agent")
class VisappAgentLoop(ToolAgentLoop):
    """Camel-style write-files-via-tools loop with a workspace-directory reward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        configured_model_length = getattr(self.rollout_config, "max_model_len", None)
        self.max_model_length = int(
            configured_model_length or (self.prompt_length + self.response_length)
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info") or {}
        if not isinstance(extra_info, dict):
            extra_info = dict(extra_info) if extra_info else {}
        task_dir = _task_dir_from_kwargs(extra_info, kwargs)
        task_id = extra_info.get("task_id")
        if not task_id and task_dir:
            task_id = Path(str(task_dir)).name

        workspace_root = make_rollout_workspace(task_dir, extra_info=extra_info)
        workspace_dir = workspace_root / "workspace"

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = {}
        getter = getattr(self, "_get_mm_processor_kwargs", None)
        if callable(getter):
            mm_processor_kwargs = getter(audios) or {}
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {}) or {}

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )
        agent_data.extra_fields.update(
            {
                "workspace_dir": str(workspace_dir),
                "workspace_root": str(workspace_root),
                "task_id": task_id,
                "task_dir": str(task_dir) if task_dir else None,
                "tool_names": [],
                "write_paths": [],
            }
        )
        agent_data._visapp_stop_after_tools = False

        result: dict[str, Any] = {"score": 0.0}
        try:
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                    if state == AgentState.TERMINATED:
                        break
                    if state != AgentState.PROCESSING_TOOLS and self._can_early_stop(
                        agent_data, had_tool_calls=False
                    ):
                        state = AgentState.TERMINATED
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                    if getattr(agent_data, "_visapp_stop_after_tools", False):
                        state = AgentState.TERMINATED
                    elif state != AgentState.TERMINATED and self._can_early_stop(
                        agent_data, had_tool_calls=True
                    ):
                        state = AgentState.TERMINATED
                elif state == AgentState.INTERACTING:
                    state = await self._handle_interacting_state(agent_data)
                else:
                    logger.error("Invalid state: %s", state)
                    state = AgentState.TERMINATED
        finally:
            try:
                result = score_workspace(workspace_dir)
            except Exception:
                logger.exception("score_workspace failed")
                result = {"score": 0.0, "n_files": 0}
            if not workspace_is_complete(workspace_dir):
                try:
                    decoded = self.tokenizer.decode(
                        agent_data.response_ids or [], skip_special_tokens=True
                    )
                except Exception:
                    decoded = ""
                xml_files = extract_xml_write_files(decoded)
                wrote = 0
                for rel, content in xml_files.items():
                    if not (content or "").strip():
                        continue
                    try:
                        dest = resolve_tool_path(workspace_dir, rel, allow_dotdot=False)
                    except ValueError:
                        continue
                    if dest.exists():
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                    wrote += 1
                if wrote:
                    print(f"[visapp] xml-materialize n={wrote}", flush=True)
                    try:
                        result = score_workspace(workspace_dir)
                    except Exception:
                        logger.exception("score_workspace failed after xml-materialize")
                elif xml_files and not result.get("n_files"):
                    result = score_files(xml_files)
            agent_data.extra_fields["score"] = result.get("score", 0.0)
            agent_data.extra_fields["score_breakdown"] = {
                k: result.get(k)
                for k in (
                    "score",
                    "has_compose",
                    "compose_ok",
                    "has_frontend",
                    "has_backend",
                    "n_files",
                    "combined_score_critical",
                )
            }
            print(
                f"[visapp] score={result.get('score')} n_files={result.get('n_files')} "
                f"compose={result.get('has_compose')} tools={agent_data.extra_fields.get('tool_names')} "
                f"writes={agent_data.extra_fields.get('write_paths')}",
                flush=True,
            )
            if not keep_workspace_enabled():
                cleanup_workspace(workspace_root)

        mask_len = len(agent_data.response_mask)
        if mask_len:
            response_ids = agent_data.prompt_ids[-mask_len:]
            prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - mask_len]
        else:
            response_ids = []
            prompt_ids = list(agent_data.prompt_ids)
        output_multi_modal_data: dict[str, Any] = {}
        if agent_data.image_data is not None:
            output_multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            output_multi_modal_data["videos"] = agent_data.video_data
        extra_fields = dict(agent_data.extra_fields)
        extra_fields.pop("_visapp_stop_after_tools", None)
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=output_multi_modal_data,
            response_logprobs=(
                agent_data.response_logprobs[: self.response_length]
                if agent_data.response_logprobs
                else None
            ),
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=None,
            extra_fields=extra_fields,
            reward_score=float(result.get("score") or 0.0),
        )
        output.extra_fields.update(
            {"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards}
        )
        output.reward_score = float(result.get("score") or 0.0)
        return output

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        bounded = _bounded_sampling_params(
            sampling_params,
            prompt_tokens=len(agent_data.prompt_ids),
            response_tokens=len(agent_data.response_mask),
            max_model_length=self.max_model_length,
            max_response_length=self.response_length,
        )
        if bounded is None:
            agent_data.metrics["context_limit_terminated"] = 1
            return AgentState.TERMINATED
        state = await super()._handle_generating_state(
            agent_data, bounded, ignore_termination=ignore_termination
        )
        # verl terminates on max_assistant_turns / response cap *before* extracting
        # tool calls, so the last write_file never lands on disk. Recover it once.
        if state == AgentState.TERMINATED and await self._recover_pending_tool_calls(agent_data):
            agent_data._visapp_stop_after_tools = True
            return AgentState.PROCESSING_TOOLS
        return state

    async def _recover_pending_tool_calls(self, agent_data: AgentData) -> bool:
        if not agent_data.response_ids:
            return False
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        try:
            _, tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
        except Exception:
            logger.exception("recover_pending_tool_calls failed")
            return False
        if not tool_calls:
            return False
        agent_data.tool_calls = tool_calls
        return True

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        names = [tc.name for tc in (agent_data.tool_calls or [])]
        agent_data.extra_fields.setdefault("tool_names", []).extend(names)
        for tc in agent_data.tool_calls or []:
            if tc.name != "write_file":
                continue
            try:
                args = json.loads(tc.arguments or "{}")
            except (TypeError, json.JSONDecodeError):
                args = {}
            agent_data.extra_fields.setdefault("write_paths", []).append(str(args.get("path") or ""))
        return await super()._handle_processing_tools_state(agent_data)

    def _workspace_complete(self, agent_data: AgentData) -> bool:
        raw = (agent_data.extra_fields or {}).get("workspace_dir")
        if not raw:
            return False
        return workspace_is_complete(Path(str(raw)))

    def _assistant_said_done(self, agent_data: AgentData) -> bool:
        if not agent_data.response_ids:
            return False
        try:
            text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        except Exception:
            return False
        stripped = _TOOL_CALL_BLOCK.sub(" ", text)
        return bool(re.search(r"\bDONE\b", stripped, re.I))

    def _can_early_stop(self, agent_data: AgentData, *, had_tool_calls: bool) -> bool:
        # Do not stop on compose-only: that capped most 32-step rollouts at 0.615.
        if not self._workspace_complete(agent_data):
            return False
        if not had_tool_calls:
            return True
        return self._assistant_said_done(agent_data)
