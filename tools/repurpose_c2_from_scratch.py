#!/usr/bin/env python3
"""
Re-purpose every c2/<task>/description.md so the variant means
"build from scratch, no scaffold" — the deliberate opposite of c3, which
ships a recommended scaffold + stack.

Replaces the trailing `## Stack: ...` section (and any following code block)
with an explicit "no scaffold" instruction that overrides the [SCAFFOLDING]
section of the base system prompt for c2 runs.

Idempotent — re-running over an already-converted file produces no change.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()  # tasks/

NEW_BLOCK = """## Bootstrap: build from scratch — no scaffold, no starter template

This variant **explicitly forbids using any scaffold or starter template**.
This instruction overrides the `[SCAFFOLDING]` guidance in the base system
prompt for this run only.

- Do NOT run `npx create-next-app`, `npm create astro`, `npm create vite`,
  `npm create svelte`, `npm create remix`, `git clone <starter>`, or any
  other scaffolding / template command.
- Hand-write `package.json`, `tsconfig.json`, framework config files, and
  every line of application code yourself.
- You may pick any modern web framework (React, Vue, Svelte, Astro, etc.)
  and any bundler — but bootstrap the project manually.

This variant measures your ability to produce a coherent, runnable web app
**without scaffold assistance**. Output must still satisfy every other
constraint in the base system prompt (Docker, .env, README, visual fidelity,
test contract).
"""


# Match `## Stack: ...` (or `## Stack` alone) and everything after it to EOF.
_STACK_RE = re.compile(r"\n\s*##\s+Stack[ :].*\Z", re.DOTALL)
# Idempotency check: if file already ends with the new bootstrap block.
_NEW_RE = re.compile(r"\n\s*##\s+Bootstrap:\s+build from scratch.*\Z", re.DOTALL)


def repurpose_one(desc: Path) -> str:
    text = desc.read_text(encoding="utf-8")
    if _NEW_RE.search(text):
        return "skipped (already converted)"
    if not _STACK_RE.search(text):
        # No Stack section — append the new block at end.
        new_text = text.rstrip() + "\n\n" + NEW_BLOCK
        action = "appended (no Stack section was present)"
    else:
        new_text = _STACK_RE.sub("\n\n" + NEW_BLOCK, text)
        action = "replaced Stack section"
    if not new_text.endswith("\n"):
        new_text += "\n"
    desc.write_text(new_text, encoding="utf-8")
    return action


def main() -> int:
    only_task = sys.argv[1] if len(sys.argv) > 1 else None
    c2_root = ROOT / "c2"
    if not c2_root.is_dir():
        print(f"ERROR: {c2_root} does not exist", file=sys.stderr)
        return 1

    targets = []
    for task_dir in sorted(c2_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name[:1].isdigit():
            continue
        if only_task and task_dir.name != only_task:
            continue
        desc = task_dir / "description.md"
        if desc.exists():
            targets.append(desc)

    if not targets:
        print("No c2/<task>/description.md files found.")
        return 1

    for desc in targets:
        result = repurpose_one(desc)
        rel = desc.relative_to(ROOT)
        print(f"  {rel}: {result}")

    print(f"\n{len(targets)} files processed.")
    print(f"Net effect: c2 now means 'no scaffold, build from scratch' (vs c3 which keeps the recommended stack).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
