#!/usr/bin/env python3
"""camel_runner_single.py — SINGLE-agent CAMEL runner (no Workforce). One ChatAgent
builds the web app in the run's workspace/, using CAMEL's OFFICIAL toolkits the way
examples/workforce/eigent.py wires them (TerminalToolkit + FileToolkit +
NoteTakingToolkit via get_tools()), instead of the hand-rolled FunctionTools used by
the multi-agent camel_runner.py. This is the single-agent A/B arm vs the Workforce.

Backend: OpenAI (default gpt-5.6-luna), api_mode="responses" so a reasoning model can
use function tools. The ChatAgent auto-runs its tool-call loop each step(). Inner tool-calling
is hard-capped with max_iteration (default 12). Outer continue nudges run
until the model replies DONE or --max-steps is hit; compose on disk is not
an automatic stop — the model can inspect and keep building.

Post-run we STRIP the TerminalToolkit scaffolding (.initial_env venv + terminal_logs)
from the workspace so the docker build context / eval only sees the agent's real app.

Emits logs/summary.json {is_error, has_compose, n_files, steps, ...} — same keep-logic
as the other CLIs.

Usage:
  camel_runner_single.py --run-dir <RUN_DIR> --api-key-file <path> \
     [--model gpt-5.6-luna] [--system-prompt <md>] [--max-steps 12] [--timeout 3600]
"""
import os, sys, json, argparse, time, traceback, threading, shutil, hashlib, functools, re, urllib.request

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--system-prompt", default="")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE_URL",
                                                         os.environ.get("CAMEL_API_URL", "")))
    ap.add_argument("--api-mode", default=os.environ.get("CAMEL_API_MODE", "responses"),
                    choices=("responses", "chat"),
                    help="responses = OpenAI Responses API (gpt-5.x + tools). "
                         "chat = Chat Completions (local vLLM / OpenAI-compatible).")
    ap.add_argument("--reasoning-effort", default=os.environ.get("CAMEL_REASONING_EFFORT", "medium"),
                    choices=("none", "minimal", "low", "medium", "high", "xhigh"))
    ap.add_argument("--prompt-cache-key", default=os.environ.get("CAMEL_PROMPT_CACHE_KEY", ""))
    ap.add_argument("--prompt-cache-retention",
                    default=os.environ.get("CAMEL_PROMPT_CACHE_RETENTION", "24h"),
                    choices=("in_memory", "24h"))
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-iteration", type=int,
                    default=int(os.environ.get("CAMEL_MAX_ITERATION", "6")),
                    help="Hard cap on model calls inside one ChatAgent.step()")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    run_dir = os.path.abspath(a.run_dir)
    ws  = os.path.join(run_dir, "workspace")
    inp = os.path.join(run_dir, "inputs")
    logs = os.path.join(run_dir, "logs"); os.makedirs(logs, exist_ok=True)
    os.makedirs(ws, exist_ok=True)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key and a.api_key_file and os.path.isfile(a.api_key_file):
        api_key = open(a.api_key_file).read().strip()
    if not api_key:
        ap.error("set OPENAI_API_KEY/CAMEL_API_KEY or provide --api-key-file")
    os.environ["OPENAI_API_KEY"] = api_key
    if a.base_url:
        os.environ["OPENAI_API_BASE_URL"] = a.base_url

    meta = {}
    try:
        meta = json.load(open(os.path.join(run_dir, "meta.json")))
    except (OSError, json.JSONDecodeError):
        pass
    cache_identity = "|".join(("camel-single", a.model, a.reasoning_effort,
                               str(meta.get("task", "unknown")),
                               str(meta.get("variant", "unknown"))))
    cache_key = a.prompt_cache_key or (
        "camel-single-" + hashlib.sha256(cache_identity.encode()).hexdigest()[:32]
    )

    trace_path = os.path.join(logs, "camel_single_trace.jsonl")
    trace_lock = threading.Lock()

    def trace(event):
        with trace_lock, open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    from camel.types.enums import OpenAIBackendRole
    from camel.toolkits import TerminalToolkit, FileToolkit, NoteTakingToolkit, FunctionTool
    from camel.memories.blocks.chat_history_block import ChatHistoryBlock
    from camel.memories.records import ContextRecord, MemoryRecord
    import camel.agents.chat_agent as camel_chat_agent

    # ChatHistoryMemory windowing keeps SYSTEM but slides off the first USER
    # task prompt; vLLM then raises "No user query found in messages".
    _orig_chb_retrieve = ChatHistoryBlock.retrieve
    def _retrieve_keep_task_user(self, window_size=None):
        if window_size is None or window_size < 0:
            return _orig_chb_retrieve(self, window_size)
        record_dicts = self.storage.load()
        if not record_dicts:
            return []
        start_index = 0
        if record_dicts[0].get("role_at_backend") in {
            OpenAIBackendRole.SYSTEM.value, OpenAIBackendRole.DEVELOPER.value,
        }:
            start_index = 1
        if (start_index < len(record_dicts)
                and record_dicts[start_index].get("role_at_backend")
                == OpenAIBackendRole.USER.value):
            start_index += 1
        preserved = record_dicts[:start_index]
        sliding = record_dicts[start_index:]
        truncated = [] if window_size == 0 else sliding[-window_size:]
        chat_records = [MemoryRecord.from_dict(r) for r in preserved + truncated]
        output_records = []
        score = 1.0
        for record in reversed(chat_records):
            if record.role_at_backend == OpenAIBackendRole.SYSTEM:
                output_records.append(ContextRecord(
                    memory_record=record, score=1.0, timestamp=record.timestamp))
            else:
                score *= self.keep_rate
                output_records.append(ContextRecord(
                    memory_record=record, score=score, timestamp=record.timestamp))
        output_records.reverse()
        return output_records
    ChatHistoryBlock.retrieve = _retrieve_keep_task_user
    try:
        from openai import BadRequestError
    except ImportError:
        BadRequestError = type("BadRequestError", (Exception,), {})

    def _as_bool(value, default=True):
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        s = str(value).strip().lower()
        if s in {"", "none", "null"}:
            return default
        return s in {"1", "true", "yes", "y"}

    def _as_float(value, default=20.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _repair_json(raw: str) -> str:
        s = (raw or "").strip()
        if not s:
            return "{}"
        in_str = False
        escape = False
        quote = None
        stack = []
        for ch in s:
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_str = False
                continue
            if ch in ('"', "'"):
                in_str = True
                quote = ch
                continue
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]" and stack and stack[-1] == ch:
                stack.pop()
        if in_str and quote:
            s += quote
        s = re.sub(r",\s*$", "", s)
        while stack:
            s += stack.pop()
        return s or "{}"

    _orig_json_loads = camel_chat_agent.json.loads
    def _loads_tool_args(s, *args, **kwargs):
        if isinstance(s, (dict, list)):
            return s
        if s is None:
            return {}
        try:
            return _orig_json_loads(s, *args, **kwargs)
        except json.JSONDecodeError:
            repaired = _repair_json(s if isinstance(s, str) else str(s))
            try:
                return _orig_json_loads(repaired, *args, **kwargs)
            except json.JSONDecodeError:
                return {}
    camel_chat_agent.json.loads = _loads_tool_args

    def _probe_max_model_len(url: str, default: int = 32768) -> int:
        env = os.environ.get("CAMEL_MAX_MODEL_LEN", "").strip()
        if env:
            try:
                return int(env)
            except ValueError:
                pass
        if not url:
            return default
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as resp:
                data = json.loads(resp.read().decode())
            for item in data.get("data") or []:
                n = item.get("max_model_len")
                if n:
                    return int(n)
        except Exception:
            pass
        return default

    def _msg_content(msg):
        return msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")

    def _content_chars(content) -> int:
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            n = 0
            for part in content:
                if isinstance(part, dict):
                    n += len(str(part.get("text") or part.get("content") or ""))
                else:
                    n += len(str(part))
            return n
        return len(str(content or ""))

    def _set_msg_content(msg, text):
        if isinstance(msg, dict):
            msg["content"] = text
        else:
            try:
                msg.content = text
            except Exception:
                pass

    def _clip_text(text: str, cap: int = 4000) -> str:
        if not isinstance(text, str) or len(text) <= cap:
            return text
        keep_head, keep_tail = cap * 2 // 3, cap // 3
        return text[:keep_head] + "\n...[truncated]...\n" + text[-keep_tail:]

    def _msg_role(msg) -> str:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        return str(role or "").lower()

    def _trim_messages(messages, max_len: int, max_out: int):
        """Drop/clip history so prompt+output fit in the served context window.

        Never drop the system message or the first user message (vLLM chat
        templates error with "No user query found" if the task prompt is gone).
        """
        if not messages:
            return messages
        msgs = list(messages)
        for msg in msgs:
            content = _msg_content(msg)
            if isinstance(content, str) and len(content) > 4000:
                _set_msg_content(msg, _clip_text(content, 4000))
        pinned, rest = [], msgs
        if rest and _msg_role(rest[0]) in {"system", "developer"}:
            pinned.append(rest[0])
            rest = rest[1:]
        if rest and _msg_role(rest[0]) == "user":
            pinned.append(rest[0])
            rest = rest[1:]
        budget_chars = max(8000, (max_len - max(max_out, 256) - 512) * 3)
        def total():
            return sum(_content_chars(_msg_content(m)) for m in pinned + rest) + 2048 * 3
        while len(rest) > 1 and total() > budget_chars:
            rest.pop(0)
        return pinned + rest

    def _estimate_prompt_tokens(messages) -> int:
        n = 0
        for msg in messages or []:
            n += max(1, _content_chars(_msg_content(msg)) // 3)
        return n + 2048

    # CAMEL TerminalToolkit.shell_exec does `monotonic() + timeout`. vLLM
    # qwen3_xml tool-calls often pass timeout/block as strings. Also bump
    # timeouts for slow scaffold commands (npm/pip/docker).
    _orig_shell_exec = TerminalToolkit.shell_exec
    @functools.wraps(_orig_shell_exec)
    def _shell_exec_float_timeout(self, id, command, block=True, timeout=20.0):
        timeout = _as_float(timeout, 20.0)
        block = _as_bool(block, True)
        cmd = command if isinstance(command, str) else str(command)
        low = cmd.lower()
        if any(tok in low for tok in ("nodejs.org/dist", "node-v", "node.tar")) and \
                any(tok in low for tok in ("wget", "curl", "tar ")):
            return ("DENIED: Node.js and python3 are already on PATH. "
                    "Do not download runtimes into the workspace. Use `node`, `npm`, `npx`, `python3`.")
        slow = any(tok in cmd for tok in (
            "npm ", "npx ", "yarn ", "pnpm ", "pip ", "pip3 ",
            "docker ", "apt", "composer ", "bundle ", "cargo ",
        ))
        min_t = _as_float(os.environ.get("CAMEL_SHELL_TIMEOUT", "300" if slow else "60"), 60.0)
        timeout = max(timeout, min_t)
        # Qwen burns the whole budget on ls/find/cat compose. Force write_file.
        inspect_only = bool(re.match(
            r"^\s*(ls|ll|find|tree|pwd|du|stat|head|tail|cat|wc|file)\b", cmd))
        if inspect_only or (low.strip().startswith("ls") or "find " in low[:40]):
            return ("DENIED: do not inspect with shell_exec. Use list_dir / read_file. "
                    "Next: write_file path=backend/app.py (Flask/FastAPI), then "
                    "write_file path=frontend/index.html. Do not rewrite docker-compose.yml.")
        if "inputs" in low and any(tok in low for tok in ("cp ", "rsync ", "mv ", "tar ")):
            return ("DENIED: do not copy ../inputs into the workspace. "
                    "Read inputs with read_file/list_dir and write app source with write_file.")
        return _orig_shell_exec(self, id=id, command=cmd, block=block,
                                timeout=timeout)
    TerminalToolkit.shell_exec = _shell_exec_float_timeout

    _orig_write = FileToolkit.write_to_file
    @functools.wraps(_orig_write)
    def _write_to_file_sane_encoding(self, title, content, filename,
                                     encoding=None, use_latex=False):
        if encoding is None or str(encoding).strip().lower() in {"", "none", "null"}:
            encoding = None
        use_latex = _as_bool(use_latex, False)
        return _orig_write(self, title, content, filename,
                           encoding=encoding, use_latex=use_latex)
    FileToolkit.write_to_file = _write_to_file_sane_encoding

    _orig_read = FileToolkit.read_file
    @functools.wraps(_orig_read)
    def _read_file_sane_paths(self, file_paths):
        if isinstance(file_paths, str) and file_paths.strip().startswith("["):
            try:
                parsed = json.loads(file_paths)
                if isinstance(parsed, list):
                    file_paths = parsed
            except json.JSONDecodeError:
                pass
        out = _orig_read(self, file_paths)
        if isinstance(out, str):
            return _clip_text(out, 6000)
        if isinstance(out, dict):
            return {k: _clip_text(v, 4000) if isinstance(v, str) else v
                    for k, v in out.items()}
        return out
    FileToolkit.read_file = _read_file_sane_paths

    model_call_seq = 0
    cache_stats = {"input_tokens": 0, "cached_tokens": 0,
                   "hit_calls": 0, "model_calls": 0}

    max_model_len = _probe_max_model_len(a.base_url)
    want_max_tokens = int(os.environ.get("CAMEL_MAX_TOKENS", "16384"))
    print(f"[camel1] max_model_len={max_model_len} want_max_tokens={want_max_tokens}",
          file=sys.stderr)

    def mk_model():
        nonlocal model_call_seq
        kw = {"url": a.base_url} if a.base_url else {}
        raw_usage_holder = {}
        if a.api_mode == "chat":
            # Local vLLM / OpenAI-compatible: Chat Completions + tools.
            # OpenAI backend accepts arbitrary served-model names as model_type.
            cfg = {}
            if want_max_tokens > 0:
                cfg["max_tokens"] = want_max_tokens
            temp = os.environ.get("CAMEL_TEMPERATURE", "")
            if temp:
                cfg["temperature"] = float(temp)
            backend = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI, model_type=a.model,
                api_key=api_key, api_mode="chat_completions",
                model_config_dict=cfg, **kw)
        else:
            # reasoning model + function tools -> must go through the Responses API.
            # store=False keeps tool-call continuation stateless and works with
            # Zero Data Retention gateways that reject previous_response_id.
            backend = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=a.model, api_mode="responses",
                model_config_dict={
                    "store": False,
                    "reasoning": {"effort": a.reasoning_effort},
                    "prompt_cache_key": cache_key,
                    "prompt_cache_retention": a.prompt_cache_retention,
                }, **kw)

        def _attach_usage_hook(create_fn):
            def traced_create(*args, **kwargs):
                response = create_fn(*args, **kwargs)
                raw_usage = getattr(response, "usage", None)
                if hasattr(raw_usage, "model_dump"):
                    raw_usage = raw_usage.model_dump()
                raw_usage_holder["usage"] = raw_usage or {}
                return response
            return traced_create

        client = getattr(backend, "_client", None)
        if client is not None:
            if a.api_mode == "chat" and hasattr(client, "chat"):
                client.chat.completions.create = _attach_usage_hook(
                    client.chat.completions.create)
            elif hasattr(client, "responses"):
                client.responses.create = _attach_usage_hook(client.responses.create)
        raw_run = backend.run

        def traced_run(messages, response_format=None, tools=None):
            nonlocal model_call_seq
            model_call_seq += 1
            call_number = model_call_seq
            started = time.time()
            base = {"event": "model_call", "call": call_number,
                    "started_at": started, "message_count": len(messages),
                    "tool_schema_count": len(tools or []),
                    "prompt_cache_key": cache_key,
                    "prompt_cache_retention": a.prompt_cache_retention}
            raw_usage_holder.clear()
            if a.api_mode == "chat":
                messages = _trim_messages(messages, max_model_len, want_max_tokens)
            if a.api_mode == "chat" and hasattr(backend, "model_config_dict"):
                est = _estimate_prompt_tokens(messages)
                remain = max(256, max_model_len - est - 64)
                cap = want_max_tokens if want_max_tokens > 0 else remain
                backend.model_config_dict["max_tokens"] = int(max(256, min(cap, remain)))
                base["max_tokens"] = backend.model_config_dict["max_tokens"]
                base["prompt_est"] = est
                base["message_count"] = len(messages)
            try:
                response = raw_run(messages, response_format, tools)
            except Exception as exc:
                msg = str(exc)
                # Last-ditch: shrink output budget and retry once on context overflow.
                if a.api_mode == "chat" and hasattr(backend, "model_config_dict") and (
                        "maximum context length" in msg or "input_tokens" in msg):
                    backend.model_config_dict["max_tokens"] = 256
                    try:
                        response = raw_run(messages, response_format, tools)
                    except Exception as exc2:
                        trace({**base, "ended_at": time.time(),
                               "elapsed_s": round(time.time() - started, 3),
                               "status": "error",
                               "error": f"{type(exc2).__name__}: {exc2}"})
                        raise
                else:
                    trace({**base, "ended_at": time.time(),
                           "elapsed_s": round(time.time() - started, 3),
                           "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                    raise
            raw_usage = raw_usage_holder.get("usage") or {}
            details = (raw_usage.get("input_tokens_details")
                       or raw_usage.get("prompt_tokens_details") or {})
            if isinstance(details, dict):
                cached = int(details.get("cached_tokens") or 0)
            else:
                cached = int(getattr(details, "cached_tokens", 0) or 0)
            usage = {
                "prompt_tokens": int(raw_usage.get("input_tokens")
                                     or raw_usage.get("prompt_tokens") or 0),
                "completion_tokens": int(raw_usage.get("output_tokens")
                                         or raw_usage.get("completion_tokens") or 0),
                "total_tokens": int(raw_usage.get("total_tokens") or 0),
                "cached_tokens": cached,
            }
            if not usage["total_tokens"]:
                adapted = getattr(response, "usage", None)
                if hasattr(adapted, "model_dump"):
                    adapted = adapted.model_dump()
                adapted = adapted or {}
                usage.update({k: int(adapted.get(k) or 0) for k in
                              ("prompt_tokens", "completion_tokens", "total_tokens")})
            cache_stats["model_calls"] += 1
            cache_stats["input_tokens"] += usage["prompt_tokens"]
            cache_stats["cached_tokens"] += usage["cached_tokens"]
            cache_stats["hit_calls"] += int(usage["cached_tokens"] > 0)
            trace({**base, "ended_at": time.time(),
                   "elapsed_s": round(time.time() - started, 3),
                   "status": "ok", "response_id": getattr(response, "id", None),
                   "usage": usage})
            return response

        backend.run = traced_run
        return backend

    # eigent-style official toolkits, rooted at the app workspace.
    # Only shell_exec: the other TerminalToolkit tools (shell_write_*, ask_user)
    # steal turns from write_file on local Qwen.
    tools = []
    try:
        _term = TerminalToolkit(working_directory=ws, safe_mode=True,
                                clone_current_env=False)
        tools.append(FunctionTool(_term.shell_exec))
    except Exception as e:
        print(f"[camel1] TerminalToolkit init failed: {e}", file=sys.stderr)

    def _safe_ws_path(rel: str, *, write: bool = False) -> str:
        rel = (rel or ".").lstrip("/")
        dest = os.path.realpath(os.path.normpath(os.path.join(ws, rel)))
        root = os.path.realpath(ws)
        inp_root = os.path.realpath(inp)
        if dest == root or dest.startswith(root + os.sep):
            return dest
        if (not write) and (dest == inp_root or dest.startswith(inp_root + os.sep)):
            return dest
        raise ValueError(f"path escapes workspace: {rel}")

    def write_file(path: str, content: str) -> str:
        """Write a UTF-8 text file relative to the workspace root (e.g. docker-compose.yml)."""
        dest = _safe_ws_path(path, write=True)
        base = os.path.basename(dest).lower()
        # Local Qwen rewrites compose every step and never creates app source.
        if base in {"docker-compose.yml", "docker-compose.yaml"} and os.path.isfile(dest) \
                and os.path.getsize(dest) > 40:
            return ("DENIED: docker-compose.yml already exists. Do not rewrite it. "
                    "Next required: write_file path=backend/app.py then "
                    "write_file path=frontend/index.html.")
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        text = content if isinstance(content, str) else str(content)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(text)
        return f"wrote {path} ({len(text)} chars)"

    def read_file(path: str) -> str:
        """Read a UTF-8 text file relative to the workspace or ../inputs."""
        dest = _safe_ws_path(path, write=False)
        with open(dest, encoding="utf-8", errors="replace") as f:
            return _clip_text(f.read(), 6000)

    def list_dir(path: str = ".") -> str:
        """List files under a workspace-relative directory (../inputs is allowed)."""
        dest = _safe_ws_path(path or ".", write=False)
        if not os.path.isdir(dest):
            return f"not a directory: {path}"
        names = sorted(os.listdir(dest))[:80]
        return "\n".join(names) if names else "(empty)"

    tools += [FunctionTool(write_file), FunctionTool(read_file), FunctionTool(list_dir)]
    # Skip FileToolkit.write_to_file (title/filename/encoding) — local Qwen
    # routinely mis-fills that schema. Keep simple path+content write_file.
    # NoteTakingToolkit is skipped: Qwen spent the whole budget on notes.

    sysmsg = ""
    if a.system_prompt and os.path.exists(a.system_prompt):
        sysmsg = open(a.system_prompt).read()
    desc = os.path.join(inp, "description.md")
    task_spec = open(desc).read() if os.path.exists(desc) else "(no description.md)"

    system_message = (sysmsg + "\n\n---\nYou are a full-stack coding agent. Your cwd is "
        "the app workspace. Tools: write_file(path, content), read_file(path), "
        "list_dir(path), and shell_exec. CREATE REAL, RUNNABLE source files. Read task "
        "inputs from ../inputs/. The finished app MUST launch with a single "
        "`docker compose up` from the workspace root — there MUST be a working "
        "docker-compose.yml there. Build actual working code, not placeholders.\n"
        "FIRST write_file MUST be docker-compose.yml (frontend + backend + db as the "
        "spec needs). Then write backend/app.py and frontend/index.html. "
        "Write files in chunks (under ~200 lines per tool call). Prefer write_file for "
        "source; use shell_exec heredoc only if write_file fails. Do NOT rewrite "
        "docker-compose.yml after the first write. Do NOT run "
        "`docker compose up` yourself — evaluation starts the stack. Do NOT take notes "
        "or write session summaries. node/npm/python3 are already on PATH; NEVER wget/"
        "curl Node or Python runtimes into the workspace.")
    token_limit = max(4096, max_model_len - max(want_max_tokens, 1024) - 512)
    agent = ChatAgent(system_message=system_message, model=mk_model(), tools=tools,
                      token_limit=token_limit, prune_tool_calls_from_memory=True,
                      message_window_size=48, summarize_threshold=None,
                      max_iteration=a.max_iteration, retry_attempts=2)

    prompt = (
        "Build the web application specified below. Read ALL task inputs in ../inputs/ "
        "(description.md and pages/*.png visual mockups). "
        "Your FIRST tool call must be write_file path=docker-compose.yml. "
        "Your SECOND must be write_file path=backend/app.py. "
        "Your THIRD must be write_file path=frontend/index.html. "
        "Then keep writing source so `docker compose up` from the workspace root "
        "launches the app. Match the visual mockups. Do not reply DONE until "
        "backend and frontend source both exist.\n\n"
        f"--- TASK SPEC ---\n{task_spec}")

    def has_compose():
        return (os.path.exists(os.path.join(ws, "docker-compose.yml"))
                or os.path.exists(os.path.join(ws, "docker-compose.yaml")))

    def _is_source(rel: str) -> bool:
        rel_l = rel.replace("\\", "/").lower()
        if rel_l.startswith(("terminal_logs/", ".initial_env/", "context_files/",
                             "scaffold_ref/")):
            return False
        return os.path.splitext(rel_l)[1] in {
            ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".vue",
            ".svelte", ".go", ".rs", ".java", ".json",
        }

    def _has_backend(files):
        for rel in files:
            r = rel.replace("\\", "/").lower()
            if r.startswith("backend/") and _is_source(rel):
                return True
            if os.path.basename(r) in {"app.py", "server.py", "main.py", "main.go"}:
                return True
        return False

    def _has_frontend(files):
        for rel in files:
            r = rel.replace("\\", "/").lower()
            if r.startswith("frontend/") and _is_source(rel):
                return True
            if os.path.basename(r) in {"index.html", "app.jsx", "app.tsx", "app.vue",
                                       "+page.svelte"}:
                return True
        return False

    trace({"event": "start", "system_message": system_message,
           "prompt": prompt, "model": a.model,
           "reasoning_effort": a.reasoning_effort,
           "prompt_cache_key": cache_key,
           "prompt_cache_retention": a.prompt_cache_retention,
           "max_iteration": a.max_iteration, "max_steps": a.max_steps,
           "time": time.time()})

    err = None; steps = 0; last = ""

    def _workspace_listing(limit=80):
        out = []
        for root, dirs, files in os.walk(ws):
            dirs[:] = [d for d in dirs if d not in
                       {".initial_env", "terminal_logs", "node_modules",
                        "context_files", ".git"}]
            for fn in files:
                out.append(os.path.relpath(os.path.join(root, fn), ws))
                if len(out) >= limit:
                    return out
        return out

    def _compact_agent():
        try:
            agent.reset()
        except Exception:
            try:
                agent.init_messages()
            except Exception:
                pass
        listing = _workspace_listing()
        return ("Context was compacted after overflow. Files already on disk:\n- "
                + "\n- ".join(listing or ["(empty)"])
                + "\nWrite remaining REAL source files and docker-compose.yml. "
                  "Do not take notes. Do not re-read large images.")

    def run():
        nonlocal steps, last
        msg = prompt
        for i in range(a.max_steps):
            steps = i + 1
            try:
                r = agent.step(msg)
            except json.JSONDecodeError as exc:
                trace({"event": "step_json_error", "step": steps,
                       "error": str(exc), "time": time.time()})
                msg = ("Your last tool-call JSON was truncated or invalid. "
                       "Rewrite the same file in smaller chunks via shell_exec "
                       "heredoc. Continue building.")
                continue
            except BadRequestError as exc:
                trace({"event": "step_context_error", "step": steps,
                       "error": str(exc), "time": time.time()})
                msg = _compact_agent()
                continue
            except Exception as exc:
                err_s = f"{type(exc).__name__}: {exc}"
                if "maximum context length" in str(exc) or "JSON" in err_s:
                    trace({"event": "step_recoverable", "step": steps,
                           "error": err_s, "time": time.time()})
                    msg = _compact_agent()
                    continue
                raise
            trace({"event": "agent_step", "step": steps, "input": msg,
                   "response": r.model_dump(mode="json", fallback=str),
                   "time": time.time()})
            last = (r.msgs[0].content if r.msgs else "") or ""
            listing = _workspace_listing()
            need_be = not _has_backend(listing)
            need_fe = not _has_frontend(listing)
            # Compose on disk is not a stop. DONE is only honored once app source exists.
            if "DONE" in last.upper() and has_compose() and not need_be and not need_fe:
                break
            if not has_compose():
                msg = (
                    "STOP. Call write_file NOW: path=docker-compose.yml with frontend + "
                    "backend services. Do not shell_exec. Files on disk:\n- "
                    + "\n- ".join(listing or ["(empty)"])
                )
            elif need_be:
                msg = (
                    "docker-compose.yml exists — do NOT rewrite it and do NOT reply DONE. "
                    "Call write_file NOW: path=backend/app.py (Flask or FastAPI, CORS, "
                    "/health). Then frontend/index.html. Files:\n- "
                    + "\n- ".join(listing or ["(empty)"])
                )
            elif need_fe:
                msg = (
                    "Backend exists. Do NOT rewrite compose. Call write_file NOW: "
                    "path=frontend/index.html (real UI matching the mockups). Files:\n- "
                    + "\n- ".join(listing or ["(empty)"])
                )
            else:
                msg = (
                    "Compose + frontend + backend exist. Files:\n- "
                    + "\n- ".join(listing or ["(empty)"])
                    + "\nKeep writing remaining pages/API with write_file. "
                      "If the app is runnable, reply DONE."
                )

    holder = {}
    def wrap():
        try: run()
        except Exception as e:
            holder["err"] = f"{type(e).__name__}: {e}"; holder["tb"] = traceback.format_exc()
            trace({"event": "error", "error": holder["err"],
                   "traceback": holder["tb"], "time": time.time()})
    t0 = time.time()
    th = threading.Thread(target=wrap, daemon=True); th.start(); th.join(a.timeout)
    dt = time.time() - t0
    if th.is_alive(): err = f"TIMEOUT after {a.timeout}s"
    elif "err" in holder: err = holder["err"]; sys.stderr.write(holder.get("tb", ""))

    # Normally strip TerminalToolkit scaffolding so eval sees only the app.
    # Set CAMEL_KEEP_TOOLKIT_ARTIFACTS=1 for forensic/debug runs.
    if os.environ.get("CAMEL_KEEP_TOOLKIT_ARTIFACTS") != "1":
        for junk in (".initial_env", "terminal_logs", "context_files"):
            p = os.path.join(ws, junk)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        for p in os.listdir(ws):
            if p.startswith("node-v") or p.endswith(".tar.gz"):
                fp = os.path.join(ws, p)
                if os.path.isdir(fp):
                    shutil.rmtree(fp, ignore_errors=True)
                else:
                    try:
                        os.remove(fp)
                    except OSError:
                        pass

    open(os.path.join(logs, "camel_single_result.txt"), "w").write(last or "")
    has = has_compose()
    n_files = sum(len(f) for _, _, f in os.walk(ws))
    is_error = bool(err) or not has
    if cache_stats["model_calls"] > 1 and cache_stats["cached_tokens"] == 0:
        print("[camel1] WARNING: prompt cache configured but no cached tokens "
              "were reported", file=sys.stderr)
    summary = {"summary": {
        "task": os.path.basename(run_dir), "model": a.model, "cli": "camel",
        "topology": "single", "is_error": is_error, "num_turns": steps,
        "reasoning_effort": a.reasoning_effort,
        "prompt_cache": {**cache_stats,
                         "key": cache_key,
                         "retention": a.prompt_cache_retention,
                         "verified_hit": cache_stats["cached_tokens"] > 0,
                         "hit_rate": round(cache_stats["cached_tokens"] /
                                           max(cache_stats["input_tokens"], 1), 4)},
        "elapsed_s": round(dt, 1), "error": err, "has_compose": has, "n_files": n_files,
        "result": (last or "")[:2000]}}
    json.dump(summary, open(os.path.join(logs, "summary.json"), "w"),
              ensure_ascii=False, indent=1)
    trace({"event": "finish", "summary": summary, "time": time.time()})
    try:
        from camel_trace_viewer import build_viewer
        build_viewer(run_dir)
    except Exception as e:
        print(f"[camel1] trace viewer generation failed: {e}", file=sys.stderr)
    print(f"[camel1] done {dt:.0f}s steps={steps} is_error={is_error} compose={has} "
          f"files={n_files} err={err}")
    return 0 if not is_error else 1

if __name__ == "__main__":
    sys.exit(main())
