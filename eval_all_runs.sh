#!/usr/bin/env bash
#
# eval_all_runs.sh — Batch-score every run under _runs/ with tools/eval_run.py.
#
# For each run dir that has both meta.json and a workspace/docker-compose.yml:
#   1. Bring up the agent's docker compose services on the run's assigned ports.
#   2. Run `python3 tools/eval_run.py <run_dir>`.
#   3. Tear the compose project down.
#   4. Read combined_score_critical (and friends) from logs/eval_result.json.
#
# At the end: print a leaderboard table and write _runs/eval_summary.csv.
#
# Usage:
#   ./eval_all_runs.sh                                # default RUNS_ROOT
#   ./eval_all_runs.sh /home/linuxuser/web/tasks/_runs
#   RUNS_ROOT=/path/to/_runs ./eval_all_runs.sh
#   SKIP_DOCKER=1 ./eval_all_runs.sh                  # assume services already up
#   FILTER='*c0*' ./eval_all_runs.sh                  # only matching run names
#   FORCE=1 ./eval_all_runs.sh                        # re-eval even if eval_result.json exists
#   HTTP_WAIT_TIMEOUT=180 ./eval_all_runs.sh          # wait longer for slow startups
#
# After `docker compose up --wait`, this script polls http://localhost:<frontend>/
# for up to HTTP_WAIT_TIMEOUT seconds before invoking eval_run.py — preventing
# the "Connection reset by peer" race when agents only put healthchecks on the
# db service. If the frontend never responds, eval still runs (failure-as-data)
# and compose logs are saved to logs/compose_logs_on_failure.log for debugging.

set -uo pipefail

# ---------- config ----------
RUNS_ROOT="${1:-${RUNS_ROOT:-/home/linuxuser/web/tasks/_runs}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${EVAL_PY:-$SCRIPT_DIR/tools/eval_run.py}"
SKIP_DOCKER="${SKIP_DOCKER:-0}"
FORCE="${FORCE:-0}"
FILTER="${FILTER:-*}"
# How wide to sweep docker before bringing each run's app up. Mirrors run_eval.sh:
#   all (default) — every compose project on the daemon (dedicated single-user host)
#   own           — only `eval_*` projects, so a SHARED daemon's other containers
#                   (and any concurrently-running eval lane) survive
DOCKER_TEARDOWN="${DOCKER_TEARDOWN:-all}"
DOCKER_WAIT_TIMEOUT="${DOCKER_WAIT_TIMEOUT:-300}"   # seconds — for `compose up --wait` build phase
HTTP_WAIT_TIMEOUT="${HTTP_WAIT_TIMEOUT:-90}"        # seconds — extra HTTP poll on frontend port
SUMMARY_CSV="$RUNS_ROOT/eval_summary.csv"
TIMEOUT_BIN="$(command -v timeout || command -v gtimeout || true)"

run_with_timeout() {
  local duration="$1"
  shift
  if [[ -n "$TIMEOUT_BIN" ]]; then
    "$TIMEOUT_BIN" "$duration" "$@"
  else
    "$@"
  fi
}

[[ -d "$RUNS_ROOT" ]] || { echo "ERROR: RUNS_ROOT not found: $RUNS_ROOT" >&2; exit 1; }
[[ -f "$EVAL_PY"  ]] || { echo "ERROR: eval_run.py not found: $EVAL_PY"  >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not on PATH" >&2; exit 1; }
if [[ "$SKIP_DOCKER" != "1" ]]; then
  command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not on PATH (set SKIP_DOCKER=1 if services already running)" >&2; exit 1; }
fi

echo "================================================================"
echo "Batch eval"
echo "  RUNS_ROOT:    $RUNS_ROOT"
echo "  EVAL_PY:      $EVAL_PY"
echo "  SKIP_DOCKER:  $SKIP_DOCKER"
echo "  FORCE:        $FORCE"
echo "  FILTER:       $FILTER"
echo "  HTTP wait:    ${HTTP_WAIT_TIMEOUT}s after compose up"
echo "================================================================"

# ---------- collect run dirs ----------
# bash 3.2 (macOS default) has no `mapfile`; use a read loop instead.
RUN_DIRS=()
while IFS= read -r _d; do
  [[ -n "$_d" ]] && RUN_DIRS+=("$_d")
done < <(
  find "$RUNS_ROOT" -mindepth 1 -maxdepth 1 -type d -name "$FILTER" 2>/dev/null \
    | sort
)

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "No run dirs matched in $RUNS_ROOT (filter='$FILTER')"
  exit 0
fi

echo "Found ${#RUN_DIRS[@]} run dir(s)."
echo

# ---------- per-run helpers ----------
read_meta_field() {
  # $1 = meta.json path, $2 = field name
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); v=d.get(sys.argv[2]); print('' if v is None else v)" "$1" "$2" 2>/dev/null
}

read_score_field() {
  # $1 = eval_result.json path, $2 = dotted field under summary
  python3 -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    s = d.get('summary', {})
    cur = s
    for k in sys.argv[2].split('.'):
        cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is None: break
    print('' if cur is None else cur)
except Exception:
    print('')
" "$1" "$2" 2>/dev/null
}

bring_up_docker() {
  # $1 = workspace dir, $2 = compose project name
  # Returns 0 on success, 1 on hard failure (build error / image pull /
  # network unreachable). `--wait` exiting non-zero because of an unhealthy
  # container is treated as a SOFT failure — we return 0 and let the HTTP
  # poll decide whether anything's actually serving. Lots of agent-generated
  # compose files have broken healthchecks (wget/curl missing in alpine,
  # checking a /health path the app doesn't expose, etc.) but the app itself
  # is up and accepting requests.
  local ws="$1" project="$2"
  echo "  [docker] up -d --wait  (project=$project)"
  ( cd "$ws" && run_with_timeout "$DOCKER_WAIT_TIMEOUT" docker compose -p "$project" up -d --build --wait ) \
    >/dev/null 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    # Hard failure looks like build/network errors — services dict empty,
    # i.e. nothing got created. Distinguish via `compose ps -q`.
    local n_containers
    n_containers="$( cd "$ws" && docker compose -p "$project" ps -q 2>/dev/null | wc -l )"
    if [[ "$n_containers" -eq 0 ]]; then
      echo "  [docker] hard failure — no containers created (rc=$rc)"
      return 1
    fi
    echo "  [docker] WARN: 'up --wait' exited $rc (likely unhealthy healthcheck)"
    echo "  [docker]   $n_containers container(s) up; will probe HTTP and proceed regardless"
  fi
  return 0
}

bring_down_docker() {
  local ws="$1" project="$2"
  echo "  [docker] down --remove-orphans  (project=$project)"
  ( cd "$ws" && docker compose -p "$project" down --remove-orphans -v ) \
    >/dev/null 2>&1 || true
}

wait_for_http() {
  # $1 = url, $2 = timeout seconds (default 90)
  # Returns 0 once the URL responds with any HTTP status (incl. 4xx/5xx, but
  # NOT connection refused / reset). Most agent compose only declares a
  # healthcheck on db; frontend/backend can take 10-60s to actually serve
  # after `up --wait` returns.
  local url="$1" timeout="${2:-90}"
  local elapsed=0 step=2
  while [[ $elapsed -lt $timeout ]]; do
    # -f fails on >=400 but exits 0 on 2xx/3xx; we want "any response is fine"
    # so use --max-time and check the exit code separately. Connection
    # refused / reset → exit 7 / 56; HTTP response (any) → exit 0.
    if curl -sS -o /dev/null --max-time 5 -w '%{http_code}' "$url" 2>/dev/null \
        | grep -qE '^[2-5][0-9][0-9]$'; then
      echo "  [docker] $url responded after ${elapsed}s"
      return 0
    fi
    sleep $step
    elapsed=$((elapsed + step))
  done
  echo "  [docker] WARN: $url did not respond within ${timeout}s"
  return 1
}

# ---------- main loop ----------
declare -a SUMMARY_ROWS=()
SUMMARY_ROWS+=("run_id,task,variant,cli,model,n_critical,found_critical,avg_loc,avg_beh,combined,bypass,status")

OK=0; SKIP=0; FAIL=0

for RUN_DIR in "${RUN_DIRS[@]}"; do
  RUN_ID="$(basename "$RUN_DIR")"
  META="$RUN_DIR/meta.json"
  WORKSPACE="$RUN_DIR/workspace"
  RESULT="$RUN_DIR/logs/eval_result.json"

  echo "----------------------------------------------------------------"
  echo "[$RUN_ID]"

  if [[ ! -f "$META" ]]; then
    echo "  SKIP — no meta.json"
    SUMMARY_ROWS+=("$RUN_ID,,,,,,,,,,,SKIP_NO_META")
    SKIP=$((SKIP+1)); continue
  fi

  TASK="$(read_meta_field "$META" task)"
  VARIANT="$(read_meta_field "$META" variant)"
  CLI="$(read_meta_field "$META" cli)"
  MODEL="$(read_meta_field "$META" model)"
  PROJECT="$(read_meta_field "$META" compose_project)"
  FPORT="$(read_meta_field "$META" frontend_port)"
  BPORT="$(read_meta_field "$META" backend_port)"

  echo "  task=$TASK  variant=$VARIANT  cli=$CLI  model=$MODEL"
  echo "  compose=$PROJECT  ports=fe:$FPORT/be:$BPORT"

  if [[ -f "$RESULT" && "$FORCE" != "1" ]]; then
    echo "  cached eval_result.json present (FORCE=1 to re-run)"
  else
    # ---------- bring up agent's app ----------
    DOCKER_UP_OK=1
    if [[ "$SKIP_DOCKER" != "1" ]]; then
      if [[ ! -f "$WORKSPACE/docker-compose.yml" && ! -f "$WORKSPACE/compose.yaml" && ! -f "$WORKSPACE/compose.yml" ]]; then
        echo "  WARN — no docker-compose.yml in $WORKSPACE; skipping docker up (eval will likely fail to reach app)"
        DOCKER_UP_OK=0
      else
        # Tear down anything lingering from a prior run on this host (single-docker-host
        # assumption). DOCKER_TEARDOWN=own narrows this to `eval_*` projects for shared
        # daemons — the app we are about to score is itself an `eval_*` project, so it
        # still gets a clean slate.
        if command -v docker >/dev/null 2>&1; then
          while IFS= read -r p; do
            [[ -z "$p" ]] && continue
            [[ "$DOCKER_TEARDOWN" == "own" && "$p" != eval_* ]] && continue
            docker compose -p "$p" down --remove-orphans >/dev/null 2>&1 || true
          done < <(docker compose ls -q 2>/dev/null)
          # Colima's host-side port forward can outlive `compose down` very
          # briefly.  Give it time to withdraw the listener before inspecting
          # the port; killing that listener would kill Colima's host agent and
          # take the Docker daemon socket down with it.
          sleep 2
        fi
        # Free our ports from any stray non-docker listener (e.g. VS Code's
        # port-forward proxy that grabbed the port during the agent run) — a
        # squatted port makes `docker compose up` fail to bind and the eval
        # gets skipped with an empty log.
        if command -v lsof >/dev/null 2>&1; then
          for _p in "$FPORT" "$BPORT"; do
            [[ -n "$_p" ]] || continue
            _h="$(lsof -nP -iTCP:"$_p" -sTCP:LISTEN -t 2>/dev/null | sort -u | tr '\n' ' ')"
            if [[ -n "${_h// /}" ]]; then
              _safe_h=""
              for _pid in $_h; do
                _cmd="$(ps -p "$_pid" -o command= 2>/dev/null || true)"
                case "$_cmd" in
                  *colima*|*lima*|*hostagent*|*ssh*)
                    echo "  [docker] port $_p is still held by Colima forwarding PID $_pid; waiting"
                    ;;
                  *) _safe_h="${_safe_h} ${_pid}" ;;
                esac
              done
              if [[ -n "${_safe_h// /}" ]]; then
                echo "  [docker] freeing port $_p from stray PID(s)${_safe_h}"
                kill $_safe_h 2>/dev/null || true
              fi
              sleep 2
            fi
          done
        fi
        if ! bring_up_docker "$WORKSPACE" "$PROJECT"; then
          echo "  ERROR — docker compose up failed"
          DOCKER_UP_OK=0
        fi
      fi
    fi

    # ---------- wait for frontend HTTP to actually respond ----------
    # `docker compose up --wait` only waits for *healthchecks*. Agent-generated
    # compose typically only declares one for db; frontend/backend can take
    # 10–60s after that to serve. Poll the frontend port to avoid the eval
    # racing the container with "Connection reset by peer".
    HTTP_OK=1
    if [[ "$DOCKER_UP_OK" == "1" && -n "$FPORT" ]]; then
      if ! wait_for_http "http://localhost:$FPORT/" "${HTTP_WAIT_TIMEOUT:-90}"; then
        # First wait failed — try one restart in case it's a transient crash
        echo "  [docker] frontend not responding — restart + retry once"
        ( cd "$WORKSPACE" && docker compose -p "$PROJECT" restart >/dev/null 2>&1 ) || true
        if ! wait_for_http "http://localhost:$FPORT/" 60; then
          echo "  [docker] frontend still down after restart; eval will likely fail"
          HTTP_OK=0
          # Capture compose logs for post-mortem
          ( cd "$WORKSPACE" && docker compose -p "$PROJECT" logs --tail 80 ) \
            > "$RUN_DIR/logs/compose_logs_on_failure.log" 2>&1 || true
        fi
      fi
    fi

    # ---------- run eval ----------
    # We run eval even if HTTP_OK=0 — eval_run.py will produce a low score
    # rather than skipping entirely, which is the right "failure as data"
    # behavior (a broken docker is a real agent failure signal).
    if [[ "$DOCKER_UP_OK" == "1" ]]; then
      echo "  [eval] python3 $EVAL_PY $RUN_DIR  (http_ok=$HTTP_OK)"
      LOG="$RUN_DIR/logs/eval_run.log"
      mkdir -p "$RUN_DIR/logs"
      # Cap each eval: a single page that hangs Playwright (infinite load /
      # navigation loop) otherwise wedges the whole batch forever. On timeout,
      # mark FAILED and move on; re-score that task later if needed.
      EVAL_TIMEOUT="${EVAL_TIMEOUT:-900}"
      if run_with_timeout "$EVAL_TIMEOUT" python3 "$EVAL_PY" "$RUN_DIR" >"$LOG" 2>&1; then
        echo "  [eval] OK  (log: $LOG)"
      else
        rc=$?
        [[ $rc -eq 124 ]] && echo "  [eval] TIMEOUT after ${EVAL_TIMEOUT}s — skipping (see $LOG)" \
                          || echo "  [eval] FAILED rc=$rc  (see $LOG)"
        # a timed-out/crashed eval can leave headless chromium spinning; reap it
        # (evals run one at a time, so this is safe).
        pkill -f chrome-headless-shell 2>/dev/null || true
      fi
    else
      echo "  [eval] skipped — docker not brought up"
    fi

    # ---------- always tear docker down ----------
    if [[ "$SKIP_DOCKER" != "1" && "$DOCKER_UP_OK" == "1" ]]; then
      bring_down_docker "$WORKSPACE" "$PROJECT"
    fi
  fi

  # ---------- read score ----------
  if [[ -f "$RESULT" ]]; then
    N_CRIT="$(read_score_field "$RESULT" n_critical)"
    FOUND="$(read_score_field "$RESULT" found_critical)"
    LOC="$(read_score_field "$RESULT" avg_localization_critical)"
    BEH="$(read_score_field "$RESULT" avg_behavior_critical)"
    COMB="$(read_score_field "$RESULT" combined_score_critical)"
    BYPASS="$(read_score_field "$RESULT" auth_bypass_used)"
    echo "  → n_crit=$N_CRIT  found=$FOUND  loc=$LOC  beh=$BEH  COMBINED=$COMB  bypass=$BYPASS"
    SUMMARY_ROWS+=("$RUN_ID,$TASK,$VARIANT,$CLI,$MODEL,$N_CRIT,$FOUND,$LOC,$BEH,$COMB,$BYPASS,OK")
    OK=$((OK+1))
  else
    echo "  → no eval_result.json produced"
    SUMMARY_ROWS+=("$RUN_ID,$TASK,$VARIANT,$CLI,$MODEL,,,,,,,FAIL")
    FAIL=$((FAIL+1))
  fi
done

# ---------- write CSV ----------
printf '%s\n' "${SUMMARY_ROWS[@]}" > "$SUMMARY_CSV"
echo
echo "================================================================"
echo "Wrote $SUMMARY_CSV"
echo "OK=$OK  FAIL=$FAIL  SKIPPED=$SKIP"
echo "================================================================"

# ---------- pretty leaderboard (sorted by combined desc) ----------
echo
echo "Leaderboard (by combined_score_critical, descending):"
python3 - "$SUMMARY_CSV" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
def k(r):
    try: return -float(r.get("combined") or "nan")
    except: return float("inf")
rows.sort(key=k)
hdr = ("run_id","task","variant","cli","model","n_critical","found_critical","avg_loc","avg_beh","combined","bypass","status")
widths = {h: max(len(h), max((len(str(r.get(h,""))) for r in rows), default=0)) for h in hdr}
print("  " + "  ".join(h.ljust(widths[h]) for h in hdr))
print("  " + "  ".join("-"*widths[h] for h in hdr))
for r in rows:
    print("  " + "  ".join(str(r.get(h,"")).ljust(widths[h]) for h in hdr))
PY
