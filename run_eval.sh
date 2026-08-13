#!/usr/bin/env bash
#
# run_eval.sh — Base eval runner. Launch one CLI agent against one task / variant.
#
# Dispatches on --cli:
#   --cli claude      (default) — Anthropic Claude Code CLI; `claude /login` for Max-plan token
#   --cli codex                 — OpenAI Codex CLI; `codex /login` (OAuth) or OPENAI_API_KEY
#   --cli gemini                — Google Gemini CLI; `gemini` (Google login) or GEMINI_API_KEY
#   --cli antigravity           — Google Antigravity CLI (`agy`); `agy` once to sign in
#
# The thin wrappers run_eval_claude.sh / run_eval_codex.sh / run_eval_gemini.sh /
# run_eval_antigravity.sh just `exec` this script with --cli set.
#
# Usage:
#   ./run_eval.sh --task 1_newsletter --variant c0
#   ./run_eval.sh --task 4_forum --variant c1/pick_A
#   ./run_eval.sh --task 8_ecommerce --variant c2 --cli codex
#   ./run_eval.sh --task 1_newsletter --variant c3 --cli gemini --model gemini-2.5-pro
#   ./run_eval.sh --task 4_forum --variant c4 --cli codex   # c4 = c3 minus scaffold hint
#   ./run_eval.sh --task 4_forum --variant c3 --cli claude --skill   # tell agent skills are available
#
# Single-docker host: one eval at a time. Fixed ports 30000 (frontend) /
# 30001 (backend). Pre-flight tears down any compose project still holding
# those ports.
#
# Layout produced under tasks/_runs/<run_id>/ :
#   inputs/                 read-only copy of the variant folder (description, pages, ...)
#   workspace/              agent's cwd, full read/write
#   agent_system_prompt.md  per-variant base prompt + per-run [RUNTIME ASSIGNMENTS]
#   meta.json               run metadata (cli, model, project name, timestamps)
#   logs/...                CLI-specific log artifacts
#                           - claude: events.jsonl + analysis outputs
#                           - codex:  codex_events.jsonl + stderr log + analysis outputs
#                           - gemini: gemini_events.jsonl + stderr log + analysis outputs

set -euo pipefail

# Preserve the exact caller arguments for the optional isolation bootstrap below.
ORIGINAL_ARGS=("$@")

# ---------- args ----------
CLI="claude"           # claude | codex | gemini
TASK=""
VARIANT=""
RUN_ID=""
MODEL=""               # CLI-specific default chosen below
REFINE=""              # if set to a prior run dir: seed workspace with that run's
                       # built app + feed its eval_report.md back so the agent
                       # REFINES its previous attempt (self-improvement experiment)
PRELOAD=false          # if true: copy <variant>/<task>/workspace/ into the run's
                       # workspace before launching agent + use _preload prompt
SKILL=false            # if true: append a hint to the runtime prompt telling
                       # the agent it may freely use any available Skills
                       # (Claude Code skills, Gemini extensions, etc.) without
                       # asking for permission. Off by default — some agents
                       # over-invoke skills and waste model calls.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cli)     CLI="$2";     shift 2 ;;
    --task)    TASK="$2";    shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --run-id)  RUN_ID="$2";  shift 2 ;;
    --model)   MODEL="$2";   shift 2 ;;
    --preload) PRELOAD=true; shift ;;
    --skill)   SKILL=true;   shift ;;
    --refine)  REFINE="$2";  shift 2 ;;
    -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$TASK"    ]] && { echo "Missing --task"    >&2; exit 1; }
[[ -z "$VARIANT" ]] && { echo "Missing --variant (e.g. c0, c1/pick_A, c2, c3, c4)" >&2; exit 1; }

# ---------- per-CLI defaults + auth + binary locator config ----------
EXTRA_PROBE_PATHS=()
case "$CLI" in
  claude)
    : "${MODEL:=sonnet}"
    BIN_NAME="claude"
    BIN_VAR="CLAUDE_BIN"
    EXTRA_PROBE_PATHS=("$HOME/.volta/bin/claude" "$HOME/.bun/bin/claude")
    INSTALL_HINT="Install Claude Code CLI; then run 'claude /login' once for Max-plan token."
    ;;
  codex)
    : "${MODEL:=gpt-5-codex}"
    # Codex auth is either OAuth (codex /login, subscription) OR OPENAI_API_KEY.
    # We don't gate here — let codex itself surface the auth error if neither
    # is set up.
    BIN_NAME="codex"
    BIN_VAR="CODEX_BIN"
    EXTRA_PROBE_PATHS=("$HOME/.codex/bin/codex")
    INSTALL_HINT="npm install -g @openai/codex   OR   curl -fsSL https://github.com/openai/codex/releases/latest/download/install.sh | bash"
    ;;
  gemini)
    : "${MODEL:=gemini-2.5-pro}"
    BIN_NAME="gemini"
    BIN_VAR="GEMINI_BIN"
    INSTALL_HINT="npm install -g @google/gemini-cli   (then run 'gemini' once to do Google OAuth, OR set GEMINI_API_KEY from https://aistudio.google.com/apikey)"
    ;;
  antigravity)
    # Google Antigravity CLI, launched as `agy`. Model is a display-name string
    # (run `agy models` to list) — an unknown value silently falls back to the
    # CLI default, so keep this aligned with `agy models` output.
    : "${MODEL:=Gemini 3.1 Pro (High)}"
    BIN_NAME="agy"
    BIN_VAR="AGY_BIN"
    EXTRA_PROBE_PATHS=("$HOME/.gemini/antigravity-cli/bin/agy")
    INSTALL_HINT="Install the Antigravity CLI so 'agy' is on PATH (default ~/.local/bin/agy); run 'agy' once to sign in. List models with 'agy models'."
    ;;
  cursor)
    # Cursor Agent CLI (`cursor-agent`). Composer is Cursor's own model family;
    # `composer-2.5` is the slug (composer-2.5-fast is Cursor's default).
    # Auth is subscription login (`cursor-agent login`) OR CURSOR_API_KEY.
    : "${MODEL:=composer-2.5}"
    BIN_NAME="cursor-agent"
    BIN_VAR="CURSOR_BIN"
    EXTRA_PROBE_PATHS=("$HOME/.local/bin/cursor-agent" "$HOME/.cursor/bin/cursor-agent")
    INSTALL_HINT="curl https://cursor.com/install -fsS | bash   (then run 'cursor-agent login' once)"
    ;;
  copilot)
    # GitHub Copilot CLI (`copilot`). MAI-Code-1-Flash's model slug is
    # 'mai-code-1-flash-picker' (the plain 'mai-code-1-flash'/'MAI-Code-1-Flash'
    # are rejected as "not available"). Auth is the GitHub Copilot subscription
    # (`copilot` then /login); no API key needed.
    : "${MODEL:=mai-code-1-flash-picker}"
    BIN_NAME="copilot"
    BIN_VAR="COPILOT_BIN"
    EXTRA_PROBE_PATHS=("/opt/homebrew/bin/copilot" "$HOME/.local/bin/copilot")
    INSTALL_HINT="npm install -g @github/copilot   (then run 'copilot' once and /login to authenticate)"
    ;;
  kimi)
    # Kimi Code CLI (`kimi`). Kimi K3's model alias is 'kimi-code/k3' (1M context,
    # effort=max, thinking). Auth = oauth device-code (stored under ~/.kimi-code).
    # The stream-json is OpenAI-chat-shaped (role/tool_calls), NOT Claude-shaped,
    # and carries NO token counts — per-call OUTPUT tokens live only in kimi's
    # per-session log (snapshotted post-run). No input tokens are exposed anywhere.
    : "${MODEL:=kimi-code/k3}"
    BIN_NAME="kimi"
    BIN_VAR="KIMI_BIN"
    EXTRA_PROBE_PATHS=("$HOME/.kimi-code/bin/kimi" "$HOME/.local/bin/kimi")
    INSTALL_HINT="install kimi-code CLI, then 'kimi login' (device-code oauth)"
    ;;
  camel)
    # CAMEL multi-agent Workforce (tools/camel_runner.py) — NOT a CLI binary; runs
    # in the .camel-venv python with a coordinated team of agents. --model is an
    # OpenAI model id (all agents share it). No PATH binary to locate.
    : "${MODEL:=gpt-5.6-luna}"
    BIN_NAME="camel"; BIN_VAR="CAMEL_BIN"
    INSTALL_HINT="python3.13 -m venv tasks/.camel-venv && tasks/.camel-venv/bin/pip install camel-ai 'mcp==1.12.4'"
    ;;
  *)
    echo "Unknown --cli '$CLI' (expected: claude, codex, gemini, antigravity, cursor, copilot, kimi, camel)" >&2
    exit 1
    ;;
esac

# ---------- clean-room bootstrap ----------
# Agent CLIs commonly have broad shell/filesystem access. Running their workspace
# under this repository lets them discover prior runs, evaluator code, anchors,
# and other non-public benchmark files by walking parent directories. By default,
# re-exec the harness from one temporary root while materializing the agent's
# run under a second, separate root. The agent-side tree contains only its
# workspace plus the selected variant/task's public inputs; harness code is not
# placed anywhere above its cwd.
# The generated run is moved back to the requested RUNS_ROOT only after the agent
# exits. Ground-truth anchors are staged after that move, never during generation.
if [[ "${VISTA_ISOLATED_CHILD:-0}" != "1" && "${VISTA_EVAL_ISOLATION:-1}" == "1" ]]; then
  SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PUBLIC_RUNS_ROOT="${RUNS_ROOT:-$SOURCE_ROOT/_runs}"
  [[ "$PUBLIC_RUNS_ROOT" = /* ]] || PUBLIC_RUNS_ROOT="$SOURCE_ROOT/$PUBLIC_RUNS_ROOT"

  ISO_TS="$(date +%Y%m%d_%H%M%S)"
  ISO_SAFE_VARIANT="${VARIANT//\//_}"
  if [[ -z "$RUN_ID" ]]; then
    if [[ "$CLI" == "claude" ]]; then
      RUN_ID="${ISO_TS}_${TASK}_${ISO_SAFE_VARIANT}"
    else
      RUN_ID="${ISO_TS}_${TASK}_${ISO_SAFE_VARIANT}_${CLI}"
    fi
  fi
  PUBLIC_RUN_DIR="$PUBLIC_RUNS_ROOT/$RUN_ID"
  [[ ! -e "$PUBLIC_RUN_DIR" ]] || {
    echo "ERROR: run already exists: $PUBLIC_RUN_DIR" >&2
    exit 1
  }

  ISO_HARNESS_ROOT="$(mktemp -d /tmp/vista_eval_harness.XXXXXX)"
  ISO_AGENT_ROOT="$(mktemp -d /tmp/vista_eval_agent.XXXXXX)"
  mkdir -p "$ISO_HARNESS_ROOT/$VARIANT"
  cp "$SOURCE_ROOT/run_eval.sh" "$ISO_HARNESS_ROOT/run_eval.sh"
  cp "$SOURCE_ROOT"/agent_system_promt_*.md "$ISO_HARNESS_ROOT/"
  rsync -a --exclude='__pycache__' "$SOURCE_ROOT/tools" "$ISO_HARNESS_ROOT/"
  rsync -a "$SOURCE_ROOT/$VARIANT/$TASK" "$ISO_HARNESS_ROOT/$VARIANT/"

  # Native macOS deny-list around the entire child process tree. This is the
  # enforcement boundary: every agent shell/tool inherits it, so an absolute
  # path or `cd` cannot recover private benchmark files from SOURCE_ROOT.
  SANDBOX_PROFILE="$ISO_HARNESS_ROOT/benchmark.sb"
  cat > "$SANDBOX_PROFILE" <<EOF
(version 1)
(allow default)
(deny file-read* (subpath "$SOURCE_ROOT"))
(deny file-write* (subpath "$SOURCE_ROOT"))
; CAMEL imports its installed packages from this venv, but cannot write there.
(allow file-read* (subpath "$SOURCE_ROOT/.camel-venv"))
EOF
  CHILD_COMMAND=(bash "$ISO_HARNESS_ROOT/run_eval.sh" "${ORIGINAL_ARGS[@]}" --run-id "$RUN_ID")
  if command -v sandbox-exec >/dev/null 2>&1; then
    if sandbox-exec -f "$SANDBOX_PROFILE" head -c 1 "$SOURCE_ROOT/run_eval.sh" >/dev/null 2>&1; then
      echo "ERROR: benchmark sandbox self-test failed; private source root remained readable" >&2
      exit 1
    fi
    CHILD_COMMAND=(sandbox-exec -f "$SANDBOX_PROFILE" "${CHILD_COMMAND[@]}")
    OS_SANDBOX_ENFORCED=1
  else
    [[ "${ALLOW_UNSANDBOXED_EVAL:-0}" == "1" ]] || {
      echo "ERROR: sandbox-exec unavailable; set ALLOW_UNSANDBOXED_EVAL=1 to explicitly accept directory-only isolation" >&2
      exit 1
    }
    OS_SANDBOX_ENFORCED=0
  fi

  echo "[run_eval] isolated agent root: $ISO_AGENT_ROOT" >&2
  echo "[run_eval] public result target: $PUBLIC_RUN_DIR" >&2
  set +e
  ( cd "$ISO_HARNESS_ROOT" && \
    VISTA_ISOLATED_CHILD=1 \
    RUNS_ROOT="$ISO_AGENT_ROOT/_runs" \
    SKIP_ANCHOR_STAGE=1 \
    CAMEL_PY="${CAMEL_PY:-$SOURCE_ROOT/.camel-venv/bin/python}" \
    CAMEL_KEY_FILE="${CAMEL_KEY_FILE:-$SOURCE_ROOT/../api_keys/chatgpt.txt}" \
      "${CHILD_COMMAND[@]}" )
  CHILD_EXIT=$?
  set -e

  CHILD_RUN_DIR="$ISO_AGENT_ROOT/_runs/$RUN_ID"
  if [[ -d "$CHILD_RUN_DIR" ]]; then
    mkdir -p "$PUBLIC_RUNS_ROOT"
    mv "$CHILD_RUN_DIR" "$PUBLIC_RUN_DIR"
    ANCHORS_SRC="$SOURCE_ROOT/$TASK/${TASK}_anchors.json"
    if [[ "${POST_STAGE_ANCHORS:-1}" == "1" && -f "$ANCHORS_SRC" ]]; then
      cp "$ANCHORS_SRC" "$PUBLIC_RUN_DIR/anchors.json"
      echo "[run_eval] post-staged anchors after agent exit → $PUBLIC_RUN_DIR/anchors.json" >&2
    fi
    python3 - "$PUBLIC_RUN_DIR/meta.json" "$SOURCE_ROOT" "$PUBLIC_RUN_DIR" "$ISO_AGENT_ROOT" "$OS_SANDBOX_ENFORCED" <<'PY'
import json, os, sys
p, source_root, public_run, isolated_root, sandbox_enforced = sys.argv[1:]
if os.path.isfile(p):
    data = json.load(open(p))
    data.update({
        "tasks_root": source_root,
        "run_dir": public_run,
        "agent_isolated": True,
        "isolated_root_during_agent_run": isolated_root,
        "os_sandbox_enforced": sandbox_enforced == "1",
        "anchors_staged_after_agent_exit": os.path.isfile(os.path.join(public_run, "anchors.json")),
    })
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
PY
    echo "[run_eval] isolated artifacts preserved: $PUBLIC_RUN_DIR" >&2
  else
    echo "ERROR: isolated child produced no run directory: $CHILD_RUN_DIR" >&2
    CHILD_EXIT=1
  fi

  case "$ISO_HARNESS_ROOT" in
    /tmp/vista_eval_harness.*) rm -rf -- "$ISO_HARNESS_ROOT" ;;
    *) echo "[run_eval] refusing unexpected cleanup target: $ISO_HARNESS_ROOT" >&2 ;;
  esac
  case "$ISO_AGENT_ROOT" in
    /tmp/vista_eval_agent.*) rm -rf -- "$ISO_AGENT_ROOT" ;;
    *) echo "[run_eval] refusing unexpected cleanup target: $ISO_AGENT_ROOT" >&2 ;;
  esac
  exit "$CHILD_EXIT"
fi

# ---------- locate the chosen CLI binary ----------
# When invoked via `bash run_eval.sh` from zsh, the child bash often misses
# PATH entries set by nvm / npm-global / Homebrew in zshrc. Try hard.
locate_bin() {
  local override="${!BIN_VAR:-}"
  if [[ -n "$override" && -x "$override" ]]; then
    printf '%s' "$override"; return 0
  fi
  if command -v "$BIN_NAME" >/dev/null 2>&1; then
    command -v "$BIN_NAME"; return 0
  fi
  if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
    # shellcheck source=/dev/null
    \. "$HOME/.nvm/nvm.sh" >/dev/null 2>&1 || true
    if command -v "$BIN_NAME" >/dev/null 2>&1; then
      command -v "$BIN_NAME"; return 0
    fi
  fi
  local candidates=(
    "$HOME/.npm-global/bin/$BIN_NAME"
    "$HOME/.local/bin/$BIN_NAME"
    "/opt/homebrew/bin/$BIN_NAME"
    "/usr/local/bin/$BIN_NAME"
    "${EXTRA_PROBE_PATHS[@]}"
  )
  if [[ -d "$HOME/.nvm/versions/node" ]]; then
    while IFS= read -r p; do candidates+=("$p"); done < <(
      ls "$HOME/.nvm/versions/node"/*/bin/"$BIN_NAME" 2>/dev/null
    )
  fi
  for p in "${candidates[@]}"; do
    [[ -x "$p" ]] && { printf '%s' "$p"; return 0; }
  done
  return 1
}

if [[ "$CLI" == "camel" ]]; then
  # No CLI binary — the agent-launch step is tools/camel_runner.py in the venv.
  CAMEL_PY="${CAMEL_PY:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.camel-venv/bin/python}"
  [[ -x "$CAMEL_PY" ]] || { echo "ERROR: camel venv python not found at $CAMEL_PY. $INSTALL_HINT" >&2; exit 127; }
  BIN="$CAMEL_PY"
else
  BIN="$(locate_bin || true)"
  if [[ -z "$BIN" ]]; then
    cat >&2 <<ERR
ERROR: '$BIN_NAME' CLI not found on this machine.
  Install: $INSTALL_HINT
  Or pass it explicitly: $BIN_VAR=/absolute/path/to/$BIN_NAME ./run_eval.sh ...
  Or run from your zsh shell directly (so PATH is inherited): ./run_eval.sh ...
ERR
    exit 127
  fi
fi
echo "Using $BIN_NAME at: $BIN" >&2

# Capture the harness/CLI version so each run records exactly which binary
# produced it (benchmark reproducibility — the harness matters as much as the
# model). First line of `--version`, e.g. "2.1.126 (Claude Code)".
# copilot self-updates on a bare `--version`, so it would report the newest
# version even when the run is pinned; probe it WITH --no-auto-update so the
# recorded version matches what actually executes (the run invocation also
# passes --no-auto-update).
if [[ "$CLI" == "copilot" ]]; then
  CLI_VERSION="$("$BIN" --no-auto-update --version 2>/dev/null | head -1 | tr -d '\r')"
else
  CLI_VERSION="$("$BIN" --version 2>/dev/null | head -1 | tr -d '\r')"
fi
echo "CLI version:      ${CLI_VERSION:-unknown}" >&2

# ---------- paths ----------
TASKS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # anchor_tasks/web/tasks
# Allow override via env var for custom runs directories
RUNS_ROOT="${RUNS_ROOT:-$TASKS_ROOT/_runs}"

# Each c-variant folder is now self-contained: it owns its own description.md,
# pages/, etc. The shared anchor folder ($TASKS_ROOT/$TASK) is NOT used for
# agent inputs anymore — only for eval-side ground truth (anchors.json) below.
VARIANT_DIR="$TASKS_ROOT/$VARIANT/$TASK"        # full agent inputs live here
ASSETS_DIR="$TASKS_ROOT/$TASK"                  # eval-only: source for *_anchors.json

# Per-variant system prompt (each file is already correct; no runtime stripping).
# When --preload is set on c1/c3, swap to the *_preload.md prompt which tells
# the agent the scaffold is already in its workspace and not to re-run it.
# IMPORTANT: precompute the suffix in an `if/else` block — `set -e` exits silently
# if a command substitution like $([[ ]] && echo ...) returns non-zero (false test
# from the [[ ]] propagates up through the assignment), with no error message.
if [[ "$PRELOAD" == "true" ]]; then
  PRELOAD_SUFFIX="_preload"
  PRELOAD_HINT='- **The framework scaffold is pre-loaded into your workspace** — `ls -la` will show `package.json`, configs, etc. Do NOT re-run the scaffold command; build on top.'
else
  PRELOAD_SUFFIX=""
  PRELOAD_HINT=""
fi

if [[ "$SKILL" == "true" ]]; then
  read -r -d '' SKILL_HINT <<'SKILL_EOF' || true
- **Skills are enabled — invoke when helpful, no permission needed.** Installed skills:
  - `frontend-design` — mockup → UI translation; use when planning a page's visual structure.
  - `next-best-practices` + `vercel-react-best-practices` — Next.js / React patterns (RSC, Server Actions, re-renders, bundling).
  - `svelte5-best-practices` — Svelte 5 / SvelteKit patterns.
  - `vue-best-practices` — Vue 3 Composition API (also useful for Nuxt).
  - `tailwind-design-system` — Tailwind v4 patterns and design tokens.
  - `prisma-database-setup` — Prisma schema, migrations, client setup. **For Docker deployment add `binaryTargets = ["native", "linux-arm64-openssl-3.0.x", "debian-openssl-3.0.x"]` to schema.prisma.**
  - `docker-expert` — multi-stage Dockerfiles, docker-compose patterns, healthchecks.
  - `neon-postgres` — Postgres SQL reference **only**. Do NOT switch to Neon serverless or `@neondatabase/serverless` — task requires LOCAL Docker Postgres.
  Pick skills matching your stack choice; ignore irrelevant ones. No skill exists for Astro / Remix / Refine / Eleventy / Django / Phoenix — for those, use general framework knowledge.
SKILL_EOF
else
  SKILL_HINT=""
fi
case "$VARIANT" in
  c0)         BASE_PROMPT="$TASKS_ROOT/agent_system_promt_c0.md" ;;
  c1|c1/*)    BASE_PROMPT="$TASKS_ROOT/agent_system_promt_c1${PRELOAD_SUFFIX}.md" ;;
  c2)         BASE_PROMPT="$TASKS_ROOT/agent_system_promt_c2.md" ;;
  c3)         BASE_PROMPT="$TASKS_ROOT/agent_system_promt_c3${PRELOAD_SUFFIX}.md" ;;
  c4)         BASE_PROMPT="$TASKS_ROOT/agent_system_promt_c4.md" ;;
  c5)         BASE_PROMPT="$TASKS_ROOT/agent_system_promt_c5.md" ;;
  *)          echo "Unknown --variant: '$VARIANT' (expected c0, c1, c1/pick_*, c2, c3, c4, c5)" >&2; exit 1 ;;
esac

if [[ "$PRELOAD" == "true" && "$VARIANT" != c1* && "$VARIANT" != "c3" ]]; then
  echo "ERROR: --preload only works with c1 or c3 variants (got '$VARIANT'; c4 is no-scaffold by design)" >&2
  exit 1
fi

[[ -d "$VARIANT_DIR" ]] || { echo "Variant dir not found: $VARIANT_DIR" >&2; exit 1; }
[[ -f "$BASE_PROMPT" ]] || { echo "Base prompt not found: $BASE_PROMPT" >&2; exit 1; }

# ---------- run id + ports + compose project ----------
TS="$(date +%Y%m%d_%H%M%S)"
SAFE_VARIANT="${VARIANT//\//_}"

# Single-docker host: fixed ports, deterministic compose project name.
# claude runs use the bare ID for backward compat; codex/gemini append _${CLI}.
if [[ "$CLI" == "claude" ]]; then
  RUN_ID="${RUN_ID:-${TS}_${TASK}_${SAFE_VARIANT}}"
  CLI_TAG=""
else
  RUN_ID="${RUN_ID:-${TS}_${TASK}_${SAFE_VARIANT}_${CLI}}"
  CLI_TAG="_${CLI}"
fi
RUN_DIR="$RUNS_ROOT/$RUN_ID"

# Default ports moved out of the 30000 range — VS Code / Cursor helpers
# sometimes grab ports there (`Code Helper` listening on 30000 is a common
# culprit). Override via env: FRONTEND_PORT=12345 ./run_eval.sh ...
FRONTEND_PORT="${FRONTEND_PORT:-38000}"
BACKEND_PORT="${BACKEND_PORT:-38001}"

COMPOSE_PROJECT="$(printf '%s' "eval_${SAFE_VARIANT}_${TASK}${CLI_TAG}" \
                  | tr '[:upper:]' '[:lower:]' \
                  | tr -c 'a-z0-9_' '_' \
                  | sed 's/_\{2,\}/_/g; s/_$//')"

# ---------- pre-flight: shut down everything currently running on docker ----------
# Single-docker host: any container left over from a prior run could clash on
# memory, ports, or compose network/volume names. Tear them all down before
# starting this run, regardless of which ports they hold.
port_in_use() { (echo > "/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

if command -v docker >/dev/null 2>&1 && [[ "${PRESERVE_OTHER_DOCKER:-0}" != "1" ]]; then
  # 1) Bring down every running compose project (graceful: removes networks + orphans).
  while IFS= read -r p; do
    [[ -z "$p" ]] && continue
    echo "[run_eval] bringing down compose project '$p'"
    docker compose -p "$p" down --remove-orphans >/dev/null 2>&1 || true
  done < <(docker compose ls -q 2>/dev/null)

  # 2) Stop any remaining standalone containers (not managed by compose).
  leftover=$(docker ps -q 2>/dev/null || true)
  if [[ -n "$leftover" ]]; then
    echo "[run_eval] stopping leftover standalone container(s): $leftover"
    docker stop $leftover >/dev/null 2>&1 || true
    docker rm   $leftover >/dev/null 2>&1 || true
  fi

  sleep 1   # let the OS release sockets
fi

if [[ "${PRESERVE_OTHER_DOCKER:-0}" == "1" ]]; then
  echo "[run_eval] PRESERVE_OTHER_DOCKER=1 — skipping global Docker teardown for parallel build"
fi

# Final sanity: our fixed ports must be free now. Docker is already torn down
# above, so any remaining listener is a stray non-docker process — almost always
# an editor's port-forward proxy, Docker/Colima forwarding process, or a leftover
# dev server. Never kill an unknown host process by default: a forwarding helper
# may own the Docker socket/VM lifecycle. Set KILL_STRAY_PORTS=1 only when the
# caller has intentionally opted into terminating non-Docker listeners.
for port in "$FRONTEND_PORT" "$BACKEND_PORT"; do
  if port_in_use "$port" && command -v lsof >/dev/null 2>&1; then
    holders="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | sort -u | tr '\n' ' ')"
    if [[ -n "${holders// /}" ]]; then
      safe_holders=""
      for holder in $holders; do
        holder_cmd="$(ps -p "$holder" -o command= 2>/dev/null || true)"
        case "$holder_cmd" in
          *colima*|*lima*|*hostagent*|*ssh*)
            echo "[run_eval] port $port held by Colima forwarding PID $holder — not killing it" >&2
            ;;
          *) safe_holders="${safe_holders} ${holder}" ;;
        esac
      done
      if [[ -n "${safe_holders// /}" && "${KILL_STRAY_PORTS:-0}" == "1" ]]; then
        echo "[run_eval] port $port held by PID(s)${safe_holders} after teardown — killing stray listener" >&2
        kill $safe_holders 2>/dev/null || true
      fi
      sleep 1
      if port_in_use "$port" && [[ -n "${safe_holders// /}" && "${KILL_STRAY_PORTS:-0}" == "1" ]]; then
        kill -9 $safe_holders 2>/dev/null || true
        sleep 1
      fi
    fi
  fi
  if port_in_use "$port"; then
    echo "ERROR: port $port is still in use after Docker teardown." >&2
    if command -v lsof >/dev/null 2>&1; then
      lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | sed 's/^/    /' >&2
      echo "  Use other ports for this run:" >&2
      echo "    FRONTEND_PORT=39000 BACKEND_PORT=39001 $0 $*" >&2
      echo "  Or, only if the listener is known safe to terminate, set KILL_STRAY_PORTS=1." >&2
    fi
    exit 1
  fi
done

# ---------- materialize run dir ----------
mkdir -p "$RUN_DIR/inputs" "$RUN_DIR/workspace" "$RUN_DIR/logs"

# Single source of agent inputs: the variant folder is self-contained.
# Exclude workspace/ — that's a workspace seed, not part of the task brief.
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude='.DS_Store' --exclude='workspace/' "$VARIANT_DIR/" "$RUN_DIR/inputs/"
else
  ( cd "$VARIANT_DIR" && find . -type f ! -name '.DS_Store' ! -path './workspace/*' ) | while read -r rel; do
    mkdir -p "$RUN_DIR/inputs/$(dirname "$rel")"
    cp "$VARIANT_DIR/$rel" "$RUN_DIR/inputs/$rel"
  done
fi

# Optional: pre-seed workspace with a scaffold so the agent skips that step.
# Triggered by --preload + the variant folder having a workspace/ subdir
# (generated offline by tools/preload_scaffolds.py — that script cd's into
# <task>/workspace/ and runs the scaffold command in place).
if [[ "$PRELOAD" == "true" ]]; then
  if [[ -d "$VARIANT_DIR/workspace" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --exclude='.DS_Store' "$VARIANT_DIR/workspace/" "$RUN_DIR/workspace/"
    else
      ( cd "$VARIANT_DIR/workspace" && find . -type f ! -name '.DS_Store' ) | while read -r rel; do
        mkdir -p "$RUN_DIR/workspace/$(dirname "$rel")"
        cp "$VARIANT_DIR/workspace/$rel" "$RUN_DIR/workspace/$rel"
      done
    fi
    n_files=$(find "$RUN_DIR/workspace" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "[run_eval] preloaded scaffold ($n_files files) → workspace/"
  else
    echo "[run_eval] WARNING: --preload set but no workspace/ in $VARIANT_DIR" >&2
    echo "[run_eval]   run: python3 tools/preload_scaffolds.py --variants ${VARIANT%%/*} --tasks $TASK" >&2
    echo "[run_eval]   continuing — agent will fall back to running scaffold itself" >&2
  fi
fi

# Refine mode: seed the workspace with a PRIOR run's built app, and stage that
# run's eval_report.md as feedback the agent can read (self-improvement study).
if [[ -n "$REFINE" ]]; then
  [[ -d "$REFINE/workspace" ]] || { echo "[run_eval] --refine: no workspace/ in $REFINE" >&2; exit 1; }
  # Seed source only — skip node_modules/build output (docker rebuilds deps).
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude='.DS_Store' --exclude='node_modules' --exclude='.next' \
      --exclude='dist' --exclude='build' --exclude='.git' \
      "$REFINE/workspace/" "$RUN_DIR/workspace/"
  else
    ( cd "$REFINE/workspace" && find . -type f ! -name '.DS_Store' \
        ! -path './node_modules/*' ! -path './.next/*' ! -path './dist/*' \
        ! -path './build/*' ! -path './.git/*' ) | while read -r rel; do
      mkdir -p "$RUN_DIR/workspace/$(dirname "$rel")"
      cp "$REFINE/workspace/$rel" "$RUN_DIR/workspace/$rel"
    done
  fi
  n_files=$(find "$RUN_DIR/workspace" -type f 2>/dev/null | wc -l | tr -d ' ')
  echo "[run_eval] refine: seeded workspace with prior build ($n_files files) from $(basename "$REFINE")"
  if [[ -f "$REFINE/logs/eval_report.md" ]]; then
    cp "$REFINE/logs/eval_report.md" "$RUN_DIR/inputs/eval_feedback.md"
    echo "[run_eval] refine: staged eval feedback → inputs/eval_feedback.md"
  else
    echo "[run_eval] refine: WARNING no eval_report.md in $REFINE/logs — refining without feedback" >&2
  fi
fi

# Eval-side ground truth: cleaned anchors JSON is staged at $RUN_DIR/anchors.json
# for the evaluator. NOTE: this sits one level ABOVE the agent's cwd (workspace/),
# so `../anchors.json` is reachable by an agent that goes looking — it is NOT a
# true sandbox. Set SKIP_ANCHOR_STAGE=1 to skip staging during the build (the
# caller must then stage anchors.json into the run dir AFTER the agent finishes,
# before scoring). Default behavior is unchanged.
ANCHORS_SRC="$ASSETS_DIR/${TASK}_anchors.json"
if [[ "${SKIP_ANCHOR_STAGE:-0}" == 1 ]]; then
  echo "[run_eval] SKIP_ANCHOR_STAGE=1 — anchors.json NOT staged during build (caller stages post-build)"
elif [[ -f "$ANCHORS_SRC" ]]; then
  cp "$ANCHORS_SRC" "$RUN_DIR/anchors.json"
  echo "[run_eval] staged anchors → $RUN_DIR/anchors.json"
fi

# ---------- per-run system prompt ----------
RUNTIME_PROMPT="$RUN_DIR/agent_system_prompt.md"
# Each variant's system prompt file is already correct (c0 and c2 omit
# [SCAFFOLDING]; c3 mentions Figma JSON; c1 keeps full scaffold guidance).
# No runtime stripping; just concatenate base + per-run [RUNTIME ASSIGNMENTS].
{
  cat "$BASE_PROMPT"
  cat <<EOF

---

## [RUNTIME ASSIGNMENTS] (per-run values)

| Variable             | Value                  |
|----------------------|------------------------|
| Run ID               | \`$RUN_ID\`            |
| Compose project name | \`$COMPOSE_PROJECT\`   |
| Frontend host port   | \`$FRONTEND_PORT\`     |
| Backend  host port   | \`$BACKEND_PORT\`      |

### Hard rules for \`docker-compose.yml\`

1. Expose frontend on host port \`$FRONTEND_PORT\` (e.g. \`"$FRONTEND_PORT:3000"\`).
2. Expose backend  on host port \`$BACKEND_PORT\`  (e.g. \`"$BACKEND_PORT:8000"\`).
3. **Do NOT expose the database to the host.** DB stays internal.
4. **Do NOT set \`container_name:\` on any service.** Let compose name them.
5. **Do NOT set \`name:\` on any \`volumes:\` or \`networks:\` entry.** Let compose prefix them with the project name.
6. CORS on the backend must allow \`http://localhost:$FRONTEND_PORT\`.

### README quick-start MUST read

\`\`\`bash
docker compose -p $COMPOSE_PROJECT up --build
# then open http://localhost:$FRONTEND_PORT
\`\`\`

### Inputs / Workspace

- Read-only task inputs are mounted at \`$RUN_DIR/inputs/\` and also accessible
  via the relative path \`../inputs/\` from your workspace cwd.
- Your write workspace is the current working directory (\`$RUN_DIR/workspace/\`).
  Put all generated code, \`docker-compose.yml\`, README, etc. directly here.
- **Evaluation isolation:** do not inspect, search, read, or copy files outside the
  current workspace and \`../inputs/\`. In particular, never access other run
  directories, prior agents' workspaces, benchmark answers, anchors, or test data.
${PRELOAD_HINT:-}
${SKILL_HINT:-}
- When finished, the workspace must be runnable with the single command in the
  README quick-start above.
EOF
} > "$RUNTIME_PROMPT"

# ---------- meta.json ----------
cat > "$RUN_DIR/meta.json" <<EOF
{
  "run_id": "$RUN_ID",
  "task": "$TASK",
  "variant": "$VARIANT",
  "cli": "$CLI",
  "cli_version": "$CLI_VERSION",
  "cli_bin": "$BIN",
  "model": "$MODEL",
  "preload": $PRELOAD,
  "skill": $SKILL,
  "compose_project": "$COMPOSE_PROJECT",
  "frontend_port": $FRONTEND_PORT,
  "backend_port": $BACKEND_PORT,
  "started_at": "$TS",
  "tasks_root": "$TASKS_ROOT",
  "run_dir": "$RUN_DIR"
}
EOF

# ---------- user prompt ----------
# Task description IS the user prompt — system prompt + [RUNTIME ASSIGNMENTS]
# already cover framing, inputs layout, ports, compose project, verify command.
TASK_DESC="$RUN_DIR/inputs/description.md"
[[ -f "$TASK_DESC" ]] || { echo "No description.md in $TASK_DESC" >&2; exit 1; }
USER_PROMPT="$(cat "$TASK_DESC")"

# Refine mode: frame the task as improving an EXISTING build. The prior attempt is
# already in the workspace; the evaluator's report on it is at inputs/eval_feedback.md.
if [[ -n "$REFINE" ]]; then
  USER_PROMPT="You are REFINING an existing implementation of the app below, not starting from scratch. Your current working directory already contains your previous build of this app (all its source files). An automated evaluator scored that build against the visual spec, and its full report is at \`../inputs/eval_feedback.md\` (read it first).

The report lists, per page and per annotated UI anchor: the localization score (how well the element's position matches the mockup), the behavior score (whether the interaction did the right thing), and a note explaining what happened (e.g. \"no observable effect\", \"DOM mutated but likely in-page state not navigation\", \"no candidate within tolerance\"). Anchors marked missed or scoring low on loc/beh are where you lost points.

Your goal: revise the existing app so it scores HIGHER — improve the placement of mis-located elements to match the mockup, fix interactions flagged as having no effect or the wrong effect, and add any missing pages/elements the report says weren't found. Keep everything that already scored well. Re-launch the app the same way (same ports / docker compose) when done.

--- ORIGINAL TASK SPEC (for reference) ---
$USER_PROMPT"
  echo "[run_eval] refine: prompt reframed as improve-existing-build"
fi

# ---------- launch ----------
TOOLS_DIR="$TASKS_ROOT/tools"
EVENTS_PATH="$RUN_DIR/logs/events.jsonl"
CODEX_EVENTS_PATH="$RUN_DIR/logs/codex_events.jsonl"
CODEX_STDERR_PATH="$RUN_DIR/logs/codex.stderr.log"
CODEX_FILE_CHANGE_PATH="$RUN_DIR/logs/codex_file_changes.jsonl"
CODEX_SNAPSHOT_DIR="$RUN_DIR/logs/codex_file_snapshots"
ANTIGRAVITY_LOG_PATH="$RUN_DIR/logs/antigravity.transcript.log"
ANTIGRAVITY_STDERR_PATH="$RUN_DIR/logs/antigravity.stderr.log"

cd "$RUN_DIR/workspace"

# Optional: capture the CONTENT of every disk write during the agent run via
# git-snapshot diffs (tools/fs_content_snap.sh). Off by default (adds a periodic
# git add/commit); enable with CONTENT_SNAP=1. Catches writes the agent never
# declared (npm/webpack/subprocess). Stopped just before post-run analysis.
SNAP_PID=""
if [[ "${CONTENT_SNAP:-0}" == "1" ]]; then
  "$TOOLS_DIR/fs_content_snap.sh" -w "$RUN_DIR/workspace" -o "$RUN_DIR/logs/content_snap" \
      -i "${CONTENT_SNAP_INTERVAL:-5}" >>"$RUN_DIR/logs/content_snap.log" 2>&1 &
  SNAP_PID=$!
  echo "[run_eval] CONTENT_SNAP on — disk-write content → logs/content_snap/disk_writes.jsonl (pid $SNAP_PID)"
fi

echo "================================================================"
echo "Run ID:           $RUN_ID"
echo "CLI / Model:      $CLI / $MODEL"
BANNER_FLAGS=""
[[ "$PRELOAD" == "true" ]] && BANNER_FLAGS="${BANNER_FLAGS} preload"
[[ "$SKILL"   == "true" ]] && BANNER_FLAGS="${BANNER_FLAGS} skill"
[[ -n "$BANNER_FLAGS" ]]  && BANNER_FLAGS=" ($(echo $BANNER_FLAGS | tr ' ' '+'))"
echo "Task / Variant:   $TASK / $VARIANT${BANNER_FLAGS}"
echo "Ports:            frontend=$FRONTEND_PORT  backend=$BACKEND_PORT"
echo "Compose project:  $COMPOSE_PROJECT"
echo "Workspace:        $RUN_DIR/workspace"
echo "================================================================"
echo

set +e
case "$CLI" in
  claude)
    # If ANTHROPIC_API_KEY is set, claude CLI bills it instead of using the
    # Max-plan subscription token. Unset for this run only.
    unset ANTHROPIC_API_KEY || true
    # When launched from inside a Claude Code session, CLAUDECODE is set and
    # older CLIs (e.g. 2.1.58) refuse to start ("cannot be launched inside
    # another Claude Code session"). The agent run is a clean standalone
    # session, so clear it. Harmless for newer CLIs that don't enforce this.
    unset CLAUDECODE || true
    # Flags:
    #   --add-dir gives the agent read access to inputs/ outside its cwd.
    #   --dangerously-skip-permissions runs all tools without per-call confirmation.
    #   --append-system-prompt layers our agent_system_prompt on top of claude's defaults.
    #   --output-format stream-json --verbose captures the full event stream
    #     (this is the FULLEST content surface — thinking blocks come through
    #     here with full text, alongside text/tool_use/tool_result/server_tool_use).
    #   tee_jsonl.py adds wall-clock timestamps and prints concise progress to stderr.
    # Optional API-diagnostic layer: set CLAUDE_DEBUG=1 to also write claude's
    # `--debug api` log (raw API-call timing, endpoints, retries, and rate-limit/
    # cap errors — e.g. z.ai's 429/1308) to logs/claude_debug.log. This does NOT
    # add message/thinking bodies (those are already in the stream), but is the
    # best surface for diagnosing z.ai cap/socket issues on GLM runs.
    CLAUDE_DEBUG_ARGS=()
    if [[ "${CLAUDE_DEBUG:-0}" == "1" ]]; then
      CLAUDE_DEBUG_ARGS=(--debug "${CLAUDE_DEBUG_FILTER:-api}" --debug-file "$RUN_DIR/logs/claude_debug.log")
    fi
    "$BIN" \
      --model "$MODEL" \
      --append-system-prompt "$(cat "$RUNTIME_PROMPT")" \
      --add-dir "$RUN_DIR/inputs" \
      --dangerously-skip-permissions \
      --disallowed-tools ScheduleWakeup \
      --output-format stream-json \
      --verbose \
      ${CLAUDE_DEBUG_ARGS[@]+"${CLAUDE_DEBUG_ARGS[@]}"} \
      -p "$USER_PROMPT" \
      | python3 -u "$TOOLS_DIR/tee_jsonl.py" "$EVENTS_PATH"
    EXIT=${PIPESTATUS[0]}
    ;;
  camel)
    # CAMEL build. Default = multi-agent Workforce (camel_runner.py: Coordinator +
    # Task planner + N workers, hand-rolled FunctionTools). CAMEL_SINGLE=1 switches to
    # the SINGLE-agent arm (camel_runner_single.py: one ChatAgent with CAMEL's OFFICIAL
    # toolkits — TerminalToolkit/FileToolkit/NoteTakingToolkit, eigent.py style). Both
    # use OpenAI api_mode=responses and write logs/summary.json themselves (is_error +
    # has_compose), so the claude analyzers below are skipped. cwd is already workspace/.
    CAMEL_KEY_FILE="${CAMEL_KEY_FILE:-$TASKS_ROOT/../api_keys/chatgpt.txt}"
    if [[ "${CAMEL_SINGLE:-0}" == "1" ]]; then
      echo "[run_eval] camel: SINGLE-agent mode (official toolkits)"
      "$BIN" "$TOOLS_DIR/camel_runner_single.py" \
        --run-dir "$RUN_DIR" \
        --api-key-file "$CAMEL_KEY_FILE" \
        --model "$MODEL" \
        --reasoning-effort "${CAMEL_REASONING_EFFORT:-medium}" \
        --prompt-cache-retention "${CAMEL_PROMPT_CACHE_RETENTION:-24h}" \
        --max-steps "${CAMEL_MAX_STEPS:-12}" \
        --system-prompt "$RUNTIME_PROMPT" \
        --timeout "${CAMEL_TIMEOUT:-3600}"
      EXIT=$?
    else
      "$BIN" "$TOOLS_DIR/camel_runner.py" \
        --run-dir "$RUN_DIR" \
        --api-key-file "$CAMEL_KEY_FILE" \
        --model "$MODEL" \
        --reasoning-effort "${CAMEL_REASONING_EFFORT:-medium}" \
        --prompt-cache-retention "${CAMEL_PROMPT_CACHE_RETENTION:-24h}" \
        --workers "${CAMEL_WORKERS:-3}" \
        --system-prompt "$RUNTIME_PROMPT" \
        --timeout "${CAMEL_TIMEOUT:-3600}"
      EXIT=$?
    fi
    ;;
  codex)
    # Codex CLI flags (current version):
    #   exec                                       : non-interactive headless mode
    #   --model                                    : pick model (gpt-5-codex, gpt-5, o4-mini, etc.)
    #   --cd                                       : working directory for the agent
    #   --dangerously-bypass-approvals-and-sandbox : full power — no approval gates,
    #                                                no sandbox (lets agent run docker / npm i / etc.)
    #   --skip-git-repo-check                      : eval workspace isn't a git repo
    #   --json                                     : emit structured JSONL events to stdout
    #   -o                                         : also persist the final assistant message directly
    # System prompt prepended to the user prompt because Codex prefers a
    # single prompt arg in exec mode at this writing.
    COMBINED_PROMPT="$(cat "$RUNTIME_PROMPT")

---

$USER_PROMPT"
    "$BIN" exec \
      --model "$MODEL" \
      --cd "$RUN_DIR/workspace" \
      --dangerously-bypass-approvals-and-sandbox \
      --skip-git-repo-check \
      --json \
      -o "$RUN_DIR/logs/final_message.txt" \
      "$COMBINED_PROMPT" \
      2> "$CODEX_STDERR_PATH" \
      | python3 -u "$TOOLS_DIR/tee_jsonl.py" "$CODEX_EVENTS_PATH" \
          --workspace "$RUN_DIR/workspace" \
          --snapshot-dir "$CODEX_SNAPSHOT_DIR" \
          --file-change-manifest "$CODEX_FILE_CHANGE_PATH"
    EXIT=${PIPESTATUS[0]}
    ;;
  gemini)
    # Gemini CLI flags (requires @google/gemini-cli >= 0.11):
    #   --model               : pick model
    #   --yolo                : auto-approve every tool/command
    #   --include-directories : extra trusted dirs the agent may read+write.
    #                           Also: cd into $RUN_DIR/workspace before launch
    #                           so Gemini's default cwd IS the project root —
    #                           otherwise it falls back to ~/.gemini/tmp/...
    #                           private sandbox and writes go there instead.
    #   --prompt              : one-shot non-interactive prompt
    #   --output-format stream-json : flat JSONL events (different envelope from
    #                                 Claude/Anthropic — analyze_edits handles it)
    # System prompt is appended to the user prompt because gemini has no
    # --append-system-prompt equivalent at time of writing.
    COMBINED_PROMPT="$(cat "$RUNTIME_PROMPT")

---

$USER_PROMPT"
    ( cd "$RUN_DIR/workspace" && "$BIN" \
        --model "$MODEL" \
        --yolo \
        --include-directories "$RUN_DIR/inputs,$RUN_DIR/workspace" \
        --output-format stream-json \
        --prompt "$COMBINED_PROMPT" \
        2>"$RUN_DIR/logs/gemini.log" ) \
      | python3 -u "$TOOLS_DIR/tee_jsonl.py" "$RUN_DIR/logs/gemini_events.jsonl"
    EXIT=${PIPESTATUS[0]}
    ;;
  antigravity)
    # Antigravity CLI (`agy`) flags — Claude-Code-like surface, but the
    # non-interactive `--print` mode streams PLAIN-TEXT agent narration to
    # stdout (no stream-json / --verbose / --append-system-prompt). So:
    #   --print               : run one prompt non-interactively, print response
    #   --print-timeout       : cap on the run (default 5m is far too short for a
    #                           full build; bump via AGY_PRINT_TIMEOUT, e.g. 90m)
    #   --model               : display-name string from `agy models`
    #   --dangerously-skip-permissions : auto-approve all tools (docker/npm/etc.)
    #   --add-dir             : give the agent the read-only inputs/ dir too
    # cd into workspace first so the agent's cwd IS the project root. There is no
    # structured event stream to tee, so we just capture stdout (the transcript)
    # and stderr to log files.
    COMBINED_PROMPT="$(cat "$RUNTIME_PROMPT")

---

$USER_PROMPT"
    ( cd "$RUN_DIR/workspace" && "$BIN" \
        --print \
        --print-timeout "${AGY_PRINT_TIMEOUT:-90m}" \
        --model "$MODEL" \
        --dangerously-skip-permissions \
        --add-dir "$RUN_DIR/inputs" \
        "$COMBINED_PROMPT" \
        2>"$ANTIGRAVITY_STDERR_PATH" ) \
      | tee "$ANTIGRAVITY_LOG_PATH"
    EXIT=${PIPESTATUS[0]}
    ;;
  cursor)
    # Cursor Agent CLI (`cursor-agent`) — headless surface mirrors Claude Code:
    #   -p                          : print / non-interactive (full tool access)
    #   --force                     : auto-approve every command (docker/npm/etc.)
    #   --trust                     : trust the workspace without prompting (headless)
    #   --sandbox enabled           : OS-level workspace filesystem isolation
    #   --workspace                 : writable project root (we also cd in)
    #   --add-dir                   : only additional readable root is inputs/
    #   --output-format stream-json : event envelope is Claude-shaped
    #                                 (type: system/user/assistant/result w/ is_error)
    # No --append-system-prompt, so prepend the system prompt to the user prompt
    # (like codex/gemini). Explicit --add-dir grants read access only to inputs/.
    # Because the envelope is Claude-shaped we tee to the SAME events.jsonl and
    # reuse analyze_run.py / analyze_edits.py below.
    # Unset CURSOR_API_KEY so the subscription login token is used, not a stray
    # env key (mirrors the claude branch unsetting ANTHROPIC_API_KEY).
    COMBINED_PROMPT="$(cat "$RUNTIME_PROMPT")

---

$USER_PROMPT"
    # cursor-agent headless does NOT reliably terminate. Observed repeatedly:
    # the agent finishes ALL work (builds the app, runs `docker compose up` to
    # self-verify, delivers its final summary message) but then cursor-agent
    # FAILS to emit the terminal `result` event and never exits — it sits at 0%
    # CPU indefinitely (a self-launched `docker compose up` can also keep the
    # stdout pipe open). A blunt wall-clock `timeout` would waste 45m per task.
    # So we use an INACTIVITY watchdog: cursor-agent streams events continuously
    # while working, so once events.jsonl stops growing for CURSOR_IDLE_KILL
    # seconds the agent is done-or-wedged — kill the cursor-agent process tree,
    # which closes the pipe so tee EOFs and analysis proceeds normally (verified:
    # killing cursor-agent lets run_eval finalize cleanly). CURSOR_TIMEOUT is a
    # secondary hard wall-clock cap. stdin from /dev/null so it never blocks on a
    # prompt. The cursor-agent cmdline contains the unique RUN_ID (via
    # --workspace), so pkill -f scopes the kill to THIS run (concurrency-safe).
    CURSOR_IDLE_KILL="${CURSOR_IDLE_KILL:-180}"
    CURSOR_TIMEOUT="${CURSOR_TIMEOUT:-45m}"
    TIMEOUT_BIN="$(command -v timeout || command -v gtimeout || true)"
    ( cd "$RUN_DIR/workspace" && unset CURSOR_API_KEY && \
        ${TIMEOUT_BIN:+"$TIMEOUT_BIN" -k 30s "$CURSOR_TIMEOUT"} "$BIN" \
        -p \
        --model "$MODEL" \
        --force \
        --trust \
        --sandbox enabled \
        --workspace "$RUN_DIR/workspace" \
        --add-dir "$RUN_DIR/inputs" \
        --output-format stream-json \
        "$COMBINED_PROMPT" \
        < /dev/null \
        2>"$RUN_DIR/logs/cursor.stderr.log" ) \
      | python3 -u "$TOOLS_DIR/tee_jsonl.py" "$EVENTS_PATH" &
    PIPELINE_PID=$!
    (
      while kill -0 "$PIPELINE_PID" 2>/dev/null; do
        sleep 15
        [[ -f "$EVENTS_PATH" ]] || continue
        idle=$(python3 -c "import os,time;print(int(time.time()-os.path.getmtime('$EVENTS_PATH')))" 2>/dev/null || echo 0)
        if [[ "${idle:-0}" -gt "$CURSOR_IDLE_KILL" ]]; then
          echo "[run_eval] cursor-agent idle ${idle}s (>${CURSOR_IDLE_KILL}s) — agent done or wedged; terminating its process tree" >&2
          pkill -TERM -f "cursor-agent.*$RUN_ID" 2>/dev/null || true
          sleep 4
          pkill -KILL -f "cursor-agent.*$RUN_ID" 2>/dev/null || true
          break
        fi
      done
    ) &
    WATCHDOG_PID=$!
    wait "$PIPELINE_PID"; EXIT=$?
    kill "$WATCHDOG_PID" 2>/dev/null || true
    ;;
  copilot)
    # GitHub Copilot CLI (`copilot`) headless, with MAXIMAL telemetry capture:
    #   -p                   : non-interactive prompt (exits after completion)
    #   --model              : slug (mai-code-1-flash-picker = MAI-Code-1-Flash)
    #   --allow-all          : all tools + paths + urls, no confirmation prompts
    #                          (required for autonomous non-interactive building)
    #   --add-dir            : grant read access to the read-only task inputs
    #   --output-format json : copilot's own JSONL event envelope (assistant.*,
    #                          tool.execution_*, terminal `result`). Carries only
    #                          per-message OUTPUT tokens (outputTokens), no input.
    #   --log-dir/--log-level all : full debug process log (MCP init, settings,
    #                          completion-request config, errors) → logs/copilot_debug/.
    # We ALSO enable the OpenTelemetry file exporter (COPILOT_OTEL_*), which is
    # the only surface that emits REAL input+output tokens per API call, plus AIU
    # cost, TTFT, per-turn timing, and MCP/skill context (GenAI semantic conv.).
    # After the run we snapshot the per-session state dir + sqlite `turns`
    # (full user/assistant transcript) — see the copilot post-processing block.
    # No --append-system-prompt equivalent, so prepend the system prompt to the
    # user prompt (like codex/gemini/cursor). stdin from /dev/null so it never
    # blocks on a prompt; wall-clock timeout as a safety net.
    COMBINED_PROMPT="$(cat "$RUNTIME_PROMPT")

---

$USER_PROMPT"
    COPILOT_TIMEOUT="${COPILOT_TIMEOUT:-45m}"
    TIMEOUT_BIN="$(command -v timeout || command -v gtimeout || true)"
    mkdir -p "$RUN_DIR/logs/copilot_debug"
    ( cd "$RUN_DIR/workspace" && \
        COPILOT_OTEL_ENABLED=true \
        COPILOT_OTEL_EXPORTER_TYPE=file \
        COPILOT_OTEL_FILE_EXPORTER_PATH="$RUN_DIR/logs/copilot_otel.jsonl" \
        ${TIMEOUT_BIN:+"$TIMEOUT_BIN" -k 30s "$COPILOT_TIMEOUT"} "$BIN" \
        -p "$COMBINED_PROMPT" \
        --model "$MODEL" \
        --allow-all \
        --add-dir "$RUN_DIR/inputs" \
        --output-format json \
        --log-dir "$RUN_DIR/logs/copilot_debug" \
        --log-level all \
        --no-auto-update \
        --no-color \
        < /dev/null \
        > "$RUN_DIR/logs/copilot_events.jsonl" \
        2> "$RUN_DIR/logs/copilot.stderr.log" )
    EXIT=$?
    # Snapshot the per-session state dir (compacted events.jsonl, checkpoints,
    # workspace.yaml, files/, research/) + the session's rows from the global
    # sqlite store (turns = full user/assistant transcript, session_files,
    # checkpoints). Keyed by the sessionId on the terminal `result` event.
    COPILOT_HOME="${COPILOT_HOME:-$HOME/.copilot}"
    SID="$(python3 -c "import json,sys
sid=None
for l in open('$RUN_DIR/logs/copilot_events.jsonl'):
    try: e=json.loads(l)
    except: continue
    if e.get('type')=='result' and e.get('sessionId'): sid=e['sessionId']
print(sid or '')" 2>/dev/null)"
    if [[ -n "$SID" ]]; then
      if [[ -d "$COPILOT_HOME/session-state/$SID" ]]; then
        mkdir -p "$RUN_DIR/logs/copilot_session_state"
        cp -R "$COPILOT_HOME/session-state/$SID/." "$RUN_DIR/logs/copilot_session_state/" 2>/dev/null || true
      fi
      if [[ -f "$COPILOT_HOME/session-store.db" ]] && command -v sqlite3 >/dev/null 2>&1; then
        for TBL in sessions turns checkpoints session_files session_refs forge_trajectory_events; do
          sqlite3 -json "$COPILOT_HOME/session-store.db" \
            "SELECT * FROM $TBL WHERE session_id='$SID';" \
            > "$RUN_DIR/logs/copilot_store_${TBL}.json" 2>/dev/null || true
        done
        # `sessions` keys on id, not session_id — fix that one.
        sqlite3 -json "$COPILOT_HOME/session-store.db" \
          "SELECT * FROM sessions WHERE id='$SID';" \
          > "$RUN_DIR/logs/copilot_store_sessions.json" 2>/dev/null || true
      fi
      echo "[run_eval] copilot: snapshotted session $SID (state dir + sqlite turns)" >&2
    else
      echo "[run_eval] copilot: no sessionId on result event; skipped state snapshot" >&2
    fi
    ;;
  kimi)
    # Kimi Code CLI headless: `-p` runs one prompt non-interactively and AUTO-
    # EXECUTES tools (the interactive --yolo/--auto flags ERROR when combined with
    # -p). No --append-system-prompt, so the system prompt is prepended (like
    # codex/copilot/cursor). --output-format stream-json emits OpenAI-shaped events
    # (role/content/tool_calls) with NO token counts. Real per-call OUTPUT tokens +
    # timing are only in kimi's per-session log, which we snapshot afterwards.
    # KIMI_LOG_LEVEL=debug = fullest logs; KIMI_CODE_NO_AUTO_UPDATE pins the version.
    # kimi-specific: K3 over-thinks and often finishes WITHOUT a docker-compose.yml
    # (its dominant failure mode — ~8/10 builds shipped none). Prepend a hard
    # deliverable-enforcement block that forces a bootable docker-compose skeleton
    # first. Set KIMI_BOOST=0 for an unboosted, apples-to-apples run.
    KIMI_BOOST_FILE="$TASKS_ROOT/kimi_deliverable_boost.md"
    KIMI_BOOST_TEXT=""
    if [[ "${KIMI_BOOST:-1}" == "1" && -f "$KIMI_BOOST_FILE" ]]; then
      KIMI_BOOST_TEXT="$(cat "$KIMI_BOOST_FILE")

"
      echo "[run_eval] kimi: deliverable-boost prompt ON (KIMI_BOOST=0 to disable)"
    fi
    COMBINED_PROMPT="${KIMI_BOOST_TEXT}$(cat "$RUNTIME_PROMPT")

---

$USER_PROMPT"
    KIMI_TIMEOUT="${KIMI_TIMEOUT:-45m}"
    TIMEOUT_BIN="$(command -v timeout || command -v gtimeout || true)"
    KIMI_HOME_DIR="${KIMI_HOME_DIR:-$HOME/.kimi-code}"
    ( cd "$RUN_DIR/workspace" && \
        KIMI_LOG_LEVEL="${KIMI_LOG_LEVEL:-debug}" \
        KIMI_CODE_NO_AUTO_UPDATE=1 \
        ${TIMEOUT_BIN:+"$TIMEOUT_BIN" -k 30s "$KIMI_TIMEOUT"} "$BIN" \
        -p "$COMBINED_PROMPT" \
        --model "$MODEL" \
        --add-dir "$RUN_DIR/inputs" \
        --output-format stream-json \
        < /dev/null \
        > "$RUN_DIR/logs/kimi_events.jsonl" \
        2> "$RUN_DIR/logs/kimi.stderr.log" )
    EXIT=$?
    # Snapshot kimi's per-session dir: kimi-code.log (per-call `llm response …
    # outputTokens=N` + timing), wire.jsonl (session event log), state.json.
    # The `-p` stream does NOT reliably emit a session_id, so locate the session
    # by matching this run's absolute workspace path inside each session's
    # state.json (its `workDir` field) — robust, no dependence on a terminal event.
    KSESS="$(grep -rl "$RUN_DIR/workspace" "$KIMI_HOME_DIR"/sessions/*/session_*/state.json 2>/dev/null \
              | head -1 | xargs -r dirname)"
    if [[ -n "$KSESS" && -d "$KSESS" ]]; then
      mkdir -p "$RUN_DIR/logs/kimi_session"
      cp -R "$KSESS/." "$RUN_DIR/logs/kimi_session/" 2>/dev/null || true
      echo "[run_eval] kimi: snapshotted session $(basename "$KSESS")" >&2
    else
      echo "[run_eval] kimi: no session dir matched workDir=$RUN_DIR/workspace" >&2
    fi
    ;;
esac
set -e

# Stop the live disk-write content snapshotter (if CONTENT_SNAP=1); its EXIT trap
# builds logs/content_snap/disk_writes.jsonl from the git snapshots.
if [[ -n "$SNAP_PID" ]]; then
  kill "$SNAP_PID" 2>/dev/null || true
  wait "$SNAP_PID" 2>/dev/null || true
  echo "[run_eval] CONTENT_SNAP stopped; disk-write content in logs/content_snap/disk_writes.jsonl"
fi

# Post-run analysis: structured event streams are post-processed into
# summaries and edit profiles where supported.
if [[ "$CLI" == "claude" || "$CLI" == "cursor" ]]; then
  # cursor-agent emits a Claude-shaped stream-json envelope into the same
  # events.jsonl, so the claude analyzers work unchanged.
  python3 "$TOOLS_DIR/analyze_run.py"   "$RUN_DIR" \
    || echo "(analyze_run.py failed; events.jsonl is intact)"
  python3 "$TOOLS_DIR/analyze_edits.py" "$RUN_DIR" \
    || echo "(analyze_edits.py failed; events.jsonl is intact)"
  # Content-level read/write log (agent-side): full Write/Edit/Read/Bash CONTENT,
  # not just the byte/line metadata in edits.jsonl. Offline, zero-cost — parses
  # the events.jsonl that already exists. The only source that carries READ content.
  python3 "$TOOLS_DIR/agent_content_log.py" "$EVENTS_PATH" -w "$RUN_DIR/workspace" \
    -o "$RUN_DIR/logs/content_log.jsonl" \
    || echo "(agent_content_log.py failed; events.jsonl is intact)"
  # cursor-specific finalize: (1) map the result-event usage (camelCase, only on
  # the terminal `result` event) into summary.json's tokens block — analyze_run.py
  # leaves them zero because cursor doesn't put usage on per-assistant messages;
  # (2) when the inactivity watchdog killed cursor-agent before it emitted
  # `result`, promote is_error null → false if the work is clearly present
  # (workspace has docker-compose.yml + an assistant message). Idempotent.
  if [[ "$CLI" == "cursor" ]]; then
    python3 "$TOOLS_DIR/cursor_finalize.py" "$RUN_DIR" \
      || echo "(cursor_finalize.py skipped)"
    # The inactivity watchdog kills cursor-agent with a non-zero exit even when
    # the agent had already finished successfully (it just never self-exited).
    # If finalize confirmed the work (summary.is_error == false), force EXIT=0
    # so run_all.sh records the run as the success it is instead of FAILED.
    if python3 -c "import json,sys;sys.exit(0 if json.load(open('$RUN_DIR/logs/summary.json'))['summary'].get('is_error') is False else 1)" 2>/dev/null; then
      [[ "$EXIT" != "0" ]] && echo "[run_eval] cursor: work succeeded (is_error=false); overriding watchdog exit $EXIT → 0" >&2
      EXIT=0
    fi
  fi
elif [[ "$CLI" == "codex" ]]; then
  python3 "$TOOLS_DIR/analyze_codex_run.py" "$RUN_DIR" \
    || echo "(analyze_codex_run.py failed; codex_events.jsonl is intact)"
  python3 "$TOOLS_DIR/analyze_edits.py" "$RUN_DIR" \
    || echo "(analyze_edits.py failed; codex file-change artifacts are intact)"
elif [[ "$CLI" == "gemini" ]]; then
  python3 "$TOOLS_DIR/analyze_edits.py" "$RUN_DIR" \
    || echo "(analyze_edits.py failed; gemini_events.jsonl is intact)"
elif [[ "$CLI" == "antigravity" ]]; then
  # No structured event stream from `agy --print` (plain-text narration only),
  # so there's nothing for analyze_run/analyze_edits to parse. The raw
  # transcript + stderr logs and the built workspace are the artifacts.
  :
elif [[ "$CLI" == "copilot" ]]; then
  # Copilot emits its own JSONL envelope (assistant.message_delta/reasoning,
  # tool.execution_*, a terminal `result` with usage). analyze_copilot_run.py
  # parses it into summary.json + turns.csv + summary.md, and — when the OTel
  # file exporter dump (copilot_otel.jsonl) is present — folds in the REAL
  # input+output tokens, AIU cost, and per-turn timing from the GenAI spans.
  # The JSONL stream alone carries only per-message OUTPUT tokens (no input);
  # OTel is what makes a true input+output metric possible.
  python3 "$TOOLS_DIR/analyze_copilot_run.py" "$RUN_DIR" \
    || echo "(analyze_copilot_run.py failed; copilot_events.jsonl is intact)"
elif [[ "$CLI" == "kimi" ]]; then
  # Kimi emits OpenAI-shaped stream-json (role/content/tool_calls) with NO tokens.
  # analyze_kimi_run.py parses that (tool calls, assistant text, is_error) AND the
  # snapshotted per-session log (kimi_session/logs/kimi-code.log) to sum per-call
  # `outputTokens` + timing into summary.json. Tokens are OUTPUT-ONLY (kimi exposes
  # no input tokens on any surface); keep-logic uses docker-compose.yml presence.
  python3 "$TOOLS_DIR/analyze_kimi_run.py" "$RUN_DIR" \
    || echo "(analyze_kimi_run.py failed; kimi_events.jsonl is intact)"
fi

echo
echo "================================================================"
echo "Done. Artifacts:"
echo "  workspace:      $RUN_DIR/workspace"
case "$CLI" in
  claude)
    echo "  events (raw):   $EVENTS_PATH"
    echo "  summary:        $RUN_DIR/logs/summary.md"
    echo "  turn-by-turn:   $RUN_DIR/logs/turns.csv"
    echo "  transcript:     $RUN_DIR/logs/transcript.md"
    echo "  edit profile:   $RUN_DIR/logs/edits.md  (+ edits.jsonl)"
    ;;
  codex)
    echo "  events (raw):   $CODEX_EVENTS_PATH"
    echo "  stderr:         $CODEX_STDERR_PATH"
    echo "  file changes:   $CODEX_FILE_CHANGE_PATH"
    echo "  summary:        $RUN_DIR/logs/summary.md"
    echo "  turn-by-turn:   $RUN_DIR/logs/turns.csv"
    echo "  items:          $RUN_DIR/logs/items.csv"
    echo "  transcript:     $RUN_DIR/logs/transcript.md"
    echo "  edit profile:   $RUN_DIR/logs/edits.md  (+ edits.jsonl)"
    echo "  final message:  $RUN_DIR/logs/final_message.txt"
    ;;
  gemini)
    echo "  events (raw):   $RUN_DIR/logs/gemini_events.jsonl"
    echo "  stderr:         $RUN_DIR/logs/gemini.log"
    echo "  edit profile:   $RUN_DIR/logs/edits.md  (+ edits.jsonl)"
    ;;
  antigravity)
    echo "  transcript:     $ANTIGRAVITY_LOG_PATH  (plain-text agent narration)"
    echo "  stderr:         $ANTIGRAVITY_STDERR_PATH"
    ;;
  cursor)
    echo "  events (raw):   $EVENTS_PATH"
    echo "  stderr:         $RUN_DIR/logs/cursor.stderr.log"
    echo "  summary:        $RUN_DIR/logs/summary.md"
    echo "  turn-by-turn:   $RUN_DIR/logs/turns.csv"
    echo "  transcript:     $RUN_DIR/logs/transcript.md"
    echo "  edit profile:   $RUN_DIR/logs/edits.md  (+ edits.jsonl)"
    ;;
  copilot)
    echo "  events (raw):   $RUN_DIR/logs/copilot_events.jsonl"
    echo "  otel (tokens):  $RUN_DIR/logs/copilot_otel.jsonl  (real input+output tokens, AIU cost)"
    echo "  debug log:      $RUN_DIR/logs/copilot_debug/"
    echo "  session state:  $RUN_DIR/logs/copilot_session_state/  (events.jsonl, checkpoints, files, research)"
    echo "  sqlite dumps:   $RUN_DIR/logs/copilot_store_*.json  (turns transcript, session_files, …)"
    echo "  stderr:         $RUN_DIR/logs/copilot.stderr.log"
    echo "  summary:        $RUN_DIR/logs/summary.md  (+ summary.json)"
    echo "  per-turn:       $RUN_DIR/logs/turns.csv"
    ;;
  kimi)
    echo "  events (raw):   $RUN_DIR/logs/kimi_events.jsonl  (OpenAI-shaped stream, no tokens)"
    echo "  session snap:   $RUN_DIR/logs/kimi_session/  (kimi-code.log = per-call outputTokens+timing, wire.jsonl, state.json)"
    echo "  stderr:         $RUN_DIR/logs/kimi.stderr.log"
    echo "  summary:        $RUN_DIR/logs/summary.md  (+ summary.json — OUTPUT-only tokens)"
    ;;
esac
echo "  meta:           $RUN_DIR/meta.json"
echo "  system prompt:  $RUNTIME_PROMPT"
echo
echo "To run the generated app:"
echo "  cd \"$RUN_DIR/workspace\""
echo "  docker compose -p $COMPOSE_PROJECT up --build"
echo "  open http://localhost:$FRONTEND_PORT"
echo "================================================================"

exit "$EXIT"
