#!/usr/bin/env python3
"""camel_runner.py — drive a CAMEL multi-agent Workforce to BUILD the web app in a
run's workspace/, from the task spec in inputs/. This is the agent-launch step for
`run_eval.sh --cli camel`: it replaces a single coding CLI with a coordinated team
(Coordinator + Task planner + N worker agents), so we can measure the multi-agent
effect on the same visual-spec-to-web-app benchmark.

Backend: OpenAI (default gpt-5.6-luna). All agents share one model backend.
Workers get TerminalToolkit (shell_exec + file writes, cwd=workspace) as the main
coding tool, plus FileToolkit's read/grep/glob for navigation.

Emits into logs/:
  summary.json          {summary:{is_error,num_turns,model,cli,elapsed_s,...}}  (keep-logic + eval read this)
  camel_workforce.json  raw workforce log tree (transcript / who-did-what)
  camel_result.txt      the final task result text

is_error = (an exception was raised) OR (no docker-compose.yml at workspace root),
matching the benchmark's "clean run" criterion.

Usage:
  camel_runner.py --run-dir <RUN_DIR> --api-key-file <path> \
                  [--model gpt-5.6-luna] [--workers 3] [--system-prompt <md>] \
                  [--timeout 3600]
"""
import os, sys, json, argparse, time, traceback, threading

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--system-prompt", default="")
    ap.add_argument("--base-url", default="")          # for OpenAI-compatible endpoints
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

    import subprocess
    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    from camel.societies.workforce import Workforce
    from camel.toolkits import FunctionTool
    from camel.tasks import Task

    def mk_model():
        kw = {}
        if a.base_url:
            kw["url"] = a.base_url
        # gpt-5.x reasoning models reject function tools on /v1/chat/completions; route
        # through the OpenAI Responses API (camel 0.2.90 api_mode) so tools work WITH
        # full reasoning. Harmless for non-reasoning models.
        return ModelFactory.create(model_platform=ModelPlatformType.OPENAI,
                                   model_type=a.model, api_mode="responses", **kw)

    # Custom, workspace-rooted coding tools (predictable + no sandbox-venv pollution,
    # unlike TerminalToolkit which drops a .initial_env virtualenv into the workspace).
    def _safe(path):
        fp = os.path.realpath(os.path.join(ws, path))
        if not fp.startswith(os.path.realpath(ws)):
            raise ValueError("path escapes workspace")
        return fp

    def _safe_read(path):
        # reads may target the workspace OR the read-only task inputs/ (spec, mockups,
        # structure JSON). 'inputs/x' and '../inputs/x' both resolve to the inputs dir.
        cand = os.path.realpath(os.path.join(ws, path))
        rws, rinp = os.path.realpath(ws), os.path.realpath(inp)
        if cand.startswith(rws) or cand.startswith(rinp):
            return cand
        alt = os.path.realpath(os.path.join(inp, path.lstrip("./").replace("inputs/", "", 1)))
        if alt.startswith(rinp):
            return alt
        raise ValueError("path outside workspace and inputs")

    def write_file(path: str, content: str) -> str:
        r"""Create or overwrite a file at a RELATIVE path inside the app workspace.
        Args:
            path (str): relative file path, e.g. 'frontend/index.html' or 'docker-compose.yml'.
            content (str): the FULL text content to write.
        Returns:
            str: a short confirmation.
        """
        fp = _safe(path); os.makedirs(os.path.dirname(fp) or ws, exist_ok=True)
        open(fp, "w").write(content)
        return f"wrote {len(content)} bytes to {path}"

    def read_file(path: str) -> str:
        r"""Read a text file at a RELATIVE path inside the workspace.
        Args:
            path (str): relative file path to read.
        Returns:
            str: file content (truncated to 20000 chars) or an error note.
        """
        try:
            return open(_safe(path)).read()[:20000]
        except Exception as e:
            return f"ERROR reading {path}: {e}"

    def list_dir(path: str = ".") -> str:
        r"""List files and directories at a RELATIVE path inside the workspace.
        Args:
            path (str): relative directory path (default workspace root).
        Returns:
            str: newline-separated entries, or an error note.
        """
        try:
            base = _safe(path)
            return "\n".join(sorted(os.listdir(base))) or "(empty)"
        except Exception as e:
            return f"ERROR listing {path}: {e}"

    def run_shell(command: str) -> str:
        r"""Run a shell command with the workspace as the working directory. Use for
        mkdir, ls, moving files, quick checks. Do NOT start long-running servers.
        Args:
            command (str): the shell command to execute.
        Returns:
            str: combined stdout+stderr (truncated to 8000 chars).
        """
        try:
            p = subprocess.run(command, shell=True, cwd=ws, capture_output=True,
                               text=True, timeout=120)
            return (p.stdout + p.stderr)[:8000] or f"(exit {p.returncode}, no output)"
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out (120s)"
        except Exception as e:
            return f"ERROR: {e}"

    def coding_tools():
        return [FunctionTool(write_file), FunctionTool(read_file),
                FunctionTool(list_dir), FunctionTool(run_shell)]

    sysmsg = ""
    if a.system_prompt and os.path.exists(a.system_prompt):
        sysmsg = open(a.system_prompt).read()
    desc_path = os.path.join(inp, "description.md")
    task_spec = open(desc_path).read() if os.path.exists(desc_path) else "(no description.md found)"

    coordinator = ChatAgent(
        system_message="You are the COORDINATOR of a team building one cohesive web app. "
        "Assign each subtask to the worker best suited, keep the architecture consistent "
        "(shared stack, ports, docker-compose), and ensure the final app runs via one "
        "`docker compose up`.", model=mk_model())
    planner = ChatAgent(
        system_message="You DECOMPOSE building a full web app into a small number of "
        "coherent subtasks (e.g. scaffold+compose, backend+API, frontend pages). Avoid "
        "fragmenting into inconsistent pieces.", model=mk_model())

    wf = Workforce("visual-spec web-app build team",
                   coordinator_agent=coordinator, task_agent=planner)

    worker_sys = (sysmsg + "\n\n---\nYou are a full-stack CODING worker. Your cwd is the app "
                  "workspace. Use shell_exec / shell_write_content_to_file to CREATE REAL, "
                  "RUNNABLE source files in the workspace. Read task inputs (mockups, spec) from "
                  "../inputs/. The finished app MUST launch with a single `docker compose up` from "
                  "the workspace root — there MUST be a working docker-compose.yml there. Build "
                  "actual working code, not placeholders.")
    for i in range(a.workers):
        w = ChatAgent(system_message=worker_sys, model=mk_model(), tools=coding_tools())
        wf.add_single_agent_worker(
            f"Full-stack coding worker {i+1}: writes source files, wires frontend/backend, "
            f"docker-compose", worker=w)

    prompt = (
        "Build the web application specified below as a team. Read ALL task inputs in "
        "../inputs/ (description.md and pages/*.png visual mockups). Write every source file "
        "into the current workspace. The app MUST be launchable with a single "
        "`docker compose up` from the workspace root: include a working docker-compose.yml plus "
        "frontend and backend as the spec requires. Match the visual mockups.\n\n"
        f"--- TASK SPEC ---\n{task_spec}")
    task = Task(content=prompt, id="0")

    result_text, err = "", None
    holder = {}
    def run():
        try:
            holder["res"] = wf.process_task(task)
        except Exception as e:
            holder["err"] = f"{type(e).__name__}: {e}"
            holder["tb"] = traceback.format_exc()
    t0 = time.time()
    th = threading.Thread(target=run, daemon=True); th.start()
    th.join(a.timeout)
    dt = time.time() - t0
    if th.is_alive():
        err = f"TIMEOUT after {a.timeout}s"
    elif "err" in holder:
        err = holder["err"]; sys.stderr.write(holder.get("tb", ""))
    else:
        res = holder.get("res")
        result_text = getattr(res, "result", None) or (str(res) if res is not None else "")

    # raw workforce log tree (who did what)
    n_turns = None
    try:
        tree = wf.get_workforce_log_tree() if hasattr(wf, "get_workforce_log_tree") else None
        if tree is not None:
            json.dump(tree, open(os.path.join(logs, "camel_workforce.json"), "w"),
                      default=str, ensure_ascii=False, indent=1)
            s = json.dumps(tree, default=str)
            n_turns = s.count('"worker_id"') or s.count('"task_id"') or None
    except Exception as e:
        print(f"[camel] log dump failed: {e}", file=sys.stderr)
    open(os.path.join(logs, "camel_result.txt"), "w").write(result_text or "")

    has_compose = os.path.exists(os.path.join(ws, "docker-compose.yml")) \
        or os.path.exists(os.path.join(ws, "docker-compose.yaml"))
    n_files = sum(len(f) for _, _, f in os.walk(ws))
    is_error = bool(err) or not has_compose
    summary = {"summary": {
        "task": os.path.basename(run_dir), "model": a.model, "cli": "camel",
        "topology": "workforce", "workers": a.workers,
        "is_error": is_error, "num_turns": n_turns, "elapsed_s": round(dt, 1),
        "error": err, "has_compose": has_compose, "n_files": n_files,
        "result": (result_text or "")[:2000]}}
    json.dump(summary, open(os.path.join(logs, "summary.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"[camel] done {dt:.0f}s  is_error={is_error} compose={has_compose} "
          f"files={n_files} turns={n_turns} err={err}")
    return 0 if not is_error else 1

if __name__ == "__main__":
    sys.exit(main())
