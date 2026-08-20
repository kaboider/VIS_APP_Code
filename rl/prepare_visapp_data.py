#!/usr/bin/env python3
"""Convert VIS_APP c4 tasks into a verl GRPO parquet (camel-style tool prompts).

Each row is one visual-spec-to-web-app task. The policy must write files via
tools into a rollout workspace (docker-compose.yml required). Reward is the
structural workspace score from rl/reward.py / rl/workspace.py — not Playwright.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM = (
    "You are a full-stack coding agent. The workspace starts EMPTY. "
    "Your FIRST tool call must be write_file of docker-compose.yml "
    "(do not list_dir, read_file, or shell_exec before that file exists). "
    "Use Qwen XML tools only, for example:\n"
    "<tool_call>\n"
    "<function=write_file>\n"
    "<parameter=path>docker-compose.yml</parameter>\n"
    "<parameter=content>services:\n  web:\n    image: python:3.12\n    working_dir: /app\n"
    "    volumes:\n      - .:/app\n    command: python app.py\n</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
    "Then your SECOND write_file must be templates/index.html "
    "(a real HTML page, not empty). Third write_file app.py "
    "(or main.py / server.js). Prefer a tiny Flask stack that renders that template. "
    "The task spec is already in the user message; skip exploration. "
    "Do not dump the app as markdown fences. Do not run docker. "
    "Do not download runtimes. Do not ask questions. "
    "When compose + frontend + backend exist, stop and reply DONE."
)


def _tasks(root: Path) -> list[str]:
    names = [
        "1_newsletter", "2_real-estate", "3_job-board", "4_forum",
        "5_travel-booking", "6_chat", "7_cloud-storage", "8_ecommerce",
        "9_project-management", "10_streaming_music-streaming",
    ]
    return [n for n in names if (root / n / "description.md").is_file()]


def _prompt_for(task_dir: Path, max_chars: int) -> str:
    spec = (task_dir / "description.md").read_text(encoding="utf-8", errors="replace")
    pages = sorted((task_dir / "pages").glob("*_structure-only.json")) if (task_dir / "pages").is_dir() else []
    page_names = [p.name.replace("_structure-only.json", "") for p in pages]
    body = (
        "Build the web application specified below.\n"
        "Workspace is empty. First tool call: write_file docker-compose.yml. "
        "Second: write_file templates/index.html. Third: write_file app.py. "
        "Do not list_dir or explore first. Spec is already below.\n\n"
        f"Pages to implement ({len(page_names)}): {', '.join(page_names) or '(see spec)'}\n\n"
        f"--- TASK SPEC ---\n{spec}"
    )
    if len(body) > max_chars:
        body = body[: max_chars - 20] + "\n...[truncated]"
    return body


def convert(root: Path, split: str, max_chars: int, repeat: int = 1) -> list[dict]:
    templates = []
    for idx, name in enumerate(_tasks(root)):
        task_dir = root / name
        prompt = _prompt_for(task_dir, max_chars)
        gt = json.dumps(
            {"task_id": name, "task_dir": str(task_dir), "variant": "c4"},
            ensure_ascii=False,
        )
        templates.append({
            "data_source": "visapp/c4",
            "agent_name": "visapp_agent",
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": split,
                "index": idx,
                "task_id": name,
                "dataset": "visapp",
                "task_dir": str(task_dir),
                "need_tools_kwargs": False,
            },
        })
    repeat = max(1, int(repeat))
    if repeat == 1:
        return templates
    rows = []
    for _ in range(repeat):
        for tmpl in templates:
            row = {
                **tmpl,
                "extra_info": dict(tmpl["extra_info"]),
            }
            row["extra_info"]["index"] = len(rows)
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-root", type=Path,
                    default=Path(__file__).resolve().parents[1] / "c4")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "_rl_runs" / "data")
    ap.add_argument("--max-chars", type=int, default=6500,
                    help="Truncate the user spec so chat prompts fit MAX_PROMPT_LENGTH=2048.")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Oversample the c4 tasks by repeating each row N times.")
    args = ap.parse_args()
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas+pyarrow required: pip install pandas pyarrow")
    rows = convert(args.tasks_root, "train", args.max_chars, repeat=args.repeat)
    if not rows:
        raise SystemExit(f"no c4 tasks under {args.tasks_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.parquet"
    val_path = args.out_dir / "val.parquet"
    pd.DataFrame(rows).to_parquet(train_path, index=False)
    pd.DataFrame(rows).to_parquet(val_path, index=False)
    print(f"wrote {len(rows)} rows -> {train_path}")
    print(f"wrote {len(rows)} rows -> {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
