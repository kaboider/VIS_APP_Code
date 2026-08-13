#!/usr/bin/env python3
"""camel_runner_single.py — SINGLE-agent CAMEL runner (no Workforce). One ChatAgent
builds the web app in the run's workspace/, using CAMEL's OFFICIAL toolkits the way
examples/workforce/eigent.py wires them (TerminalToolkit + FileToolkit +
NoteTakingToolkit via get_tools()), instead of the hand-rolled FunctionTools used by
the multi-agent camel_runner.py. This is the single-agent A/B arm vs the Workforce.

Backend: OpenAI (default gpt-5.6-luna), api_mode="responses" so a reasoning model can
use function tools. The ChatAgent auto-runs its tool-call loop each step(); we drive a
few "continue" nudges until docker-compose.yml exists or a step budget is hit.

Post-run we STRIP the TerminalToolkit scaffolding (.initial_env venv + terminal_logs)
from the workspace so the docker build context / eval only sees the agent's real app.

Emits logs/summary.json {is_error, has_compose, n_files, steps, ...} — same keep-logic
as the other CLIs.

Usage:
  camel_runner_single.py --run-dir <RUN_DIR> --api-key-file <path> \
     [--model gpt-5.6-luna] [--system-prompt <md>] [--max-steps 12] [--timeout 3600]
"""
import os, sys, json, argparse, time, traceback, threading, shutil, hashlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--system-prompt", default="")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE_URL", ""))
    ap.add_argument("--reasoning-effort", default=os.environ.get("CAMEL_REASONING_EFFORT", "medium"),
                    choices=("none", "minimal", "low", "medium", "high", "xhigh"))
    ap.add_argument("--prompt-cache-key", default=os.environ.get("CAMEL_PROMPT_CACHE_KEY", ""))
    ap.add_argument("--prompt-cache-retention",
                    default=os.environ.get("CAMEL_PROMPT_CACHE_RETENTION", "24h"),
                    choices=("in_memory", "24h"))
    ap.add_argument("--max-steps", type=int, default=12)
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
    from camel.toolkits import TerminalToolkit, FileToolkit, NoteTakingToolkit

    model_call_seq = 0
    cache_stats = {"input_tokens": 0, "cached_tokens": 0,
                   "hit_calls": 0, "model_calls": 0}

    def mk_model():
        nonlocal model_call_seq
        kw = {"url": a.base_url} if a.base_url else {}
        # reasoning model + function tools -> must go through the Responses API.
        # store=False keeps tool-call continuation stateless and works with
        # Zero Data Retention gateways that reject previous_response_id.
        backend = ModelFactory.create(model_platform=ModelPlatformType.OPENAI,
                                      model_type=a.model, api_mode="responses",
                                      model_config_dict={
                                          "store": False,
                                          "reasoning": {"effort": a.reasoning_effort},
                                          "prompt_cache_key": cache_key,
                                          "prompt_cache_retention": a.prompt_cache_retention,
                                      }, **kw)
        raw_create = backend._client.responses.create
        raw_usage_holder = {}

        def traced_create(*args, **kwargs):
            response = raw_create(*args, **kwargs)
            raw_usage = getattr(response, "usage", None)
            if hasattr(raw_usage, "model_dump"):
                raw_usage = raw_usage.model_dump()
            raw_usage_holder["usage"] = raw_usage or {}
            return response

        backend._client.responses.create = traced_create
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
            try:
                response = raw_run(messages, response_format, tools)
            except Exception as exc:
                trace({**base, "ended_at": time.time(),
                       "elapsed_s": round(time.time() - started, 3),
                       "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                raise
            raw_usage = raw_usage_holder.get("usage") or {}
            details = raw_usage.get("input_tokens_details") or {}
            usage = {
                "prompt_tokens": int(raw_usage.get("input_tokens") or 0),
                "completion_tokens": int(raw_usage.get("output_tokens") or 0),
                "total_tokens": int(raw_usage.get("total_tokens") or 0),
                "cached_tokens": int(details.get("cached_tokens") or 0),
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
    tools = []
    try:
        tools += TerminalToolkit(working_directory=ws, safe_mode=True,
                                 clone_current_env=False).get_tools()
    except Exception as e:
        print(f"[camel1] TerminalToolkit init failed: {e}", file=sys.stderr)
    try:
        tools += FileToolkit(working_directory=ws).get_tools()
    except Exception as e:
        print(f"[camel1] FileToolkit init failed: {e}", file=sys.stderr)
    try:
        tools += NoteTakingToolkit(working_directory=ws).get_tools()
    except Exception as e:
        print(f"[camel1] NoteTakingToolkit init failed: {e}", file=sys.stderr)

    sysmsg = ""
    if a.system_prompt and os.path.exists(a.system_prompt):
        sysmsg = open(a.system_prompt).read()
    desc = os.path.join(inp, "description.md")
    task_spec = open(desc).read() if os.path.exists(desc) else "(no description.md)"

    system_message = (sysmsg + "\n\n---\nYou are a full-stack coding agent. Your cwd is "
        "the app workspace. Use the terminal tools (shell_exec, shell_write_content_to_file) "
        "and file tools to CREATE REAL, RUNNABLE source files. Read task inputs from "
        "../inputs/. The finished app MUST launch with a single `docker compose up` from "
        "the workspace root — there MUST be a working docker-compose.yml there. Build "
        "actual working code, not placeholders.")
    agent = ChatAgent(system_message=system_message, model=mk_model(), tools=tools)

    prompt = (
        "Build the web application specified below. Read ALL task inputs in ../inputs/ "
        "(description.md and pages/*.png visual mockups). Write every source file into the "
        "current workspace. The app MUST be launchable with a single `docker compose up` "
        "from the workspace root: include a working docker-compose.yml plus frontend and "
        "backend as the spec requires. Match the visual mockups.\n\n"
        f"--- TASK SPEC ---\n{task_spec}")

    def has_compose():
        return (os.path.exists(os.path.join(ws, "docker-compose.yml"))
                or os.path.exists(os.path.join(ws, "docker-compose.yaml")))

    trace({"event": "start", "system_message": system_message,
           "prompt": prompt, "model": a.model,
           "reasoning_effort": a.reasoning_effort,
           "prompt_cache_key": cache_key,
           "prompt_cache_retention": a.prompt_cache_retention,
           "time": time.time()})

    err = None; steps = 0; last = ""
    def run():
        nonlocal steps, last
        msg = prompt
        for i in range(a.max_steps):
            steps = i + 1
            r = agent.step(msg)
            trace({"event": "agent_step", "step": steps, "input": msg,
                   "response": r.model_dump(mode="json", fallback=str),
                   "time": time.time()})
            last = (r.msgs[0].content if r.msgs else "") or ""
            # stop once a compose file exists and the agent isn't asking to continue
            if has_compose() and ("continue" not in last.lower()):
                break
            msg = ("Continue until the app is COMPLETE and runnable: ensure "
                   "docker-compose.yml exists at the workspace root and every service the "
                   "spec needs is wired. If it already runs, reply DONE.")
            if "DONE" in last and has_compose():
                break

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
        for junk in (".initial_env", "terminal_logs"):
            p = os.path.join(ws, junk)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)

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
