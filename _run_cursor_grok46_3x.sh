#!/usr/bin/env bash
# Run a complete 3-repetition Cursor Grok 4.6 High experiment.
# All 30 builds run with exactly two concurrent lanes. Evaluation starts only
# after every build is valid, and runs sequentially to avoid port contention.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export RUNS_ROOT="${RUNS_ROOT:-$PWD/_runs_cursor_grok-4.6-high_c4_3x}"
export PRESERVE_OTHER_DOCKER=1
export CURSOR_IDLE_KILL="${CURSOR_IDLE_KILL:-600}"
export CURSOR_TIMEOUT="${CURSOR_TIMEOUT:-45m}"

mkdir -p "$RUNS_ROOT/batch_logs"

ODD_TASKS=(1_newsletter 3_job-board 5_travel-booking 7_cloud-storage 9_project-management)
EVEN_TASKS=(2_real-estate 4_forum 6_chat 8_ecommerce 10_streaming_music-streaming)

cleanup_run_docker() {
  local run_dir="$1" project
  [[ -f "$run_dir/meta.json" ]] || return 0
  project="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("compose_project", ""))' "$run_dir/meta.json" 2>/dev/null || true)"
  if [[ -n "$project" && -d "$run_dir/workspace" ]]; then
    (cd "$run_dir/workspace" && docker compose -p "$project" down --remove-orphans -v >/dev/null 2>&1) || true
    sleep 2
  fi
}

run_one() {
  local lane="$1" rep="$2" task="$3" frontend="$4" backend="$5"
  local run_id="r${rep}_${task}_c4_cursor"
  local run_dir="$RUNS_ROOT/$run_id"
  local log="$RUNS_ROOT/batch_logs/${run_id}.log"

  if [[ -f "$run_dir/logs/summary.json" ]] && \
     python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["summary"]; raise SystemExit(0 if s.get("is_error") is False and s.get("has_result_event") is True else 1)' "$run_dir/logs/summary.json" 2>/dev/null; then
    echo "[cursor-3x] RESUME-SKIP lane=$lane rep=$rep task=$task"
    cleanup_run_docker "$run_dir"
    return 0
  fi

  echo "[cursor-3x] START lane=$lane rep=$rep task=$task ports=$frontend/$backend @ $(date '+%F %T')"
  FRONTEND_PORT="$frontend" BACKEND_PORT="$backend" \
    bash ./run_eval_cursor.sh --run-id "$run_id" --task "$task" --variant c4 \
      --model cursor-grok-4.6-high </dev/null >"$log" 2>&1
  local rc=$?
  cleanup_run_docker "$run_dir"
  echo "[cursor-3x] END lane=$lane rep=$rep task=$task rc=$rc @ $(date '+%F %T')"
  return "$rc"
}

lane_loop() {
  local lane="$1" frontend="$2" backend="$3"
  shift 3
  local tasks=("$@") rep task
  for rep in 1 2 3; do
    for task in "${tasks[@]}"; do
      run_one "$lane" "$rep" "$task" "$frontend" "$backend" || \
        echo "[cursor-3x] FAILURE-PRESERVED lane=$lane rep=$rep task=$task"
    done
  done
}

write_build_audit() {
  python3 - "$RUNS_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for rep in (1, 2, 3):
    for p in sorted(root.glob(f"r{rep}_*_c4_cursor")):
        s = p / "logs" / "summary.json"
        if not s.exists():
            rows.append((p.name, "missing", "", "", "", "")); continue
        d = json.loads(s.read_text())["summary"]
        t = d.get("tokens", {})
        ok = d.get("is_error") is False and d.get("has_result_event") is True
        rows.append((p.name, "ok" if ok else "invalid", t.get("input_tokens", ""),
                     t.get("cache_read_input_tokens", ""), t.get("output_tokens", ""),
                     d.get("wall_clock_s", "")))
out = root / "build_audit.tsv"
out.write_text("run\tstatus\tinput\tcache\toutput\truntime_s\n" +
               "\n".join("\t".join(map(str, r)) for r in rows) + "\n")
print(f"[cursor-3x] build audit: {out} ({sum(r[1]=='ok' for r in rows)}/30 valid)")
if len(rows) != 30 or any(r[1] != "ok" for r in rows):
    raise SystemExit(1)
PY
}

archive_invalid_runs() {
  local stamp run_dir summary archived
  stamp="$(date '+%Y%m%d_%H%M%S')"
  mkdir -p "$RUNS_ROOT/_invalid_attempts"
  for run_dir in "$RUNS_ROOT"/r{1,2,3}_*_c4_cursor; do
    [[ -d "$run_dir" && -f "$run_dir/logs/summary.json" ]] || continue
    summary="$run_dir/logs/summary.json"
    if ! python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["summary"]; raise SystemExit(0 if s.get("is_error") is False and s.get("has_result_event") is True else 1)' "$summary" 2>/dev/null; then
      cleanup_run_docker "$run_dir"
      archived="$RUNS_ROOT/_invalid_attempts/$(basename "$run_dir")_${stamp}_no_result_or_error"
      echo "[cursor-3x] ARCHIVE invalid $(basename "$run_dir") -> $(basename "$archived")"
      mv "$run_dir" "$archived"
    fi
  done
}

run_build_pass() {
  echo "[cursor-3x] BUILD PHASE pass=$1 root=$RUNS_ROOT concurrency=2"
  lane_loop 1 39100 39101 "${ODD_TASKS[@]}" & lane1_pid=$!
  lane_loop 2 39200 39201 "${EVEN_TASKS[@]}" & lane2_pid=$!
  wait "$lane1_pid"; lane1_rc=$?
  wait "$lane2_pid"; lane2_rc=$?
  echo "[cursor-3x] BUILD WORKERS DONE pass=$1 lane1_rc=$lane1_rc lane2_rc=$lane2_rc"
}

build_pass=1
while true; do
  run_build_pass "$build_pass"
  if write_build_audit; then
    break
  fi
  echo "[cursor-3x] BUILD AUDIT INCOMPLETE — archiving invalid attempts and retrying only missing runs" >&2
  archive_invalid_runs
  build_pass=$((build_pass + 1))
done

echo "[cursor-3x] EVAL PHASE (sequential, builds are all complete)"
unset PRESERVE_OTHER_DOCKER
FILTER='r[123]_*_c4_cursor' FORCE=1 HTTP_WAIT_TIMEOUT=180 \
  bash ./eval_all_runs.sh "$RUNS_ROOT"
eval_rc=$?
if [[ "$eval_rc" -eq 0 ]]; then
  python3 tools/cursor_3x_report.py "$RUNS_ROOT" || eval_rc=$?
fi
echo "[cursor-3x] DONE eval_rc=$eval_rc @ $(date '+%F %T')"
exit "$eval_rc"
