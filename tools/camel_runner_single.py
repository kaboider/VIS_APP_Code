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
import os, sys, json, argparse, time, traceback, threading, shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--system-prompt", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    run_dir = os.path.abspath(a.run_dir)
    ws  = os.path.join(run_dir, "workspace")
    inp = os.path.join(run_dir, "inputs")
    logs = os.path.join(run_dir, "logs"); os.makedirs(logs, exist_ok=True)
    os.makedirs(ws, exist_ok=True)
    os.environ["OPENAI_API_KEY"] = open(a.api_key_file).read().strip()
    if a.base_url:
        os.environ["OPENAI_API_BASE_URL"] = a.base_url

    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    from camel.toolkits import TerminalToolkit, FileToolkit, NoteTakingToolkit

    def mk_model():
        kw = {"url": a.base_url} if a.base_url else {}
        # reasoning model + function tools -> must go through the Responses API.
        return ModelFactory.create(model_platform=ModelPlatformType.OPENAI,
                                   model_type=a.model, api_mode="responses", **kw)

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

    err = None; steps = 0; last = ""
    def run():
        nonlocal steps, last
        msg = prompt
        for i in range(a.max_steps):
            steps = i + 1
            r = agent.step(msg)
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
    t0 = time.time()
    th = threading.Thread(target=wrap, daemon=True); th.start(); th.join(a.timeout)
    dt = time.time() - t0
    if th.is_alive(): err = f"TIMEOUT after {a.timeout}s"
    elif "err" in holder: err = holder["err"]; sys.stderr.write(holder.get("tb", ""))

    # strip TerminalToolkit scaffolding so the eval only sees the real app
    for junk in (".initial_env", "terminal_logs"):
        p = os.path.join(ws, junk)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)

    open(os.path.join(logs, "camel_single_result.txt"), "w").write(last or "")
    has = has_compose()
    n_files = sum(len(f) for _, _, f in os.walk(ws))
    is_error = bool(err) or not has
    summary = {"summary": {
        "task": os.path.basename(run_dir), "model": a.model, "cli": "camel",
        "topology": "single", "is_error": is_error, "num_turns": steps,
        "elapsed_s": round(dt, 1), "error": err, "has_compose": has, "n_files": n_files,
        "result": (last or "")[:2000]}}
    json.dump(summary, open(os.path.join(logs, "summary.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"[camel1] done {dt:.0f}s steps={steps} is_error={is_error} compose={has} "
          f"files={n_files} err={err}")
    return 0 if not is_error else 1

if __name__ == "__main__":
    sys.exit(main())
