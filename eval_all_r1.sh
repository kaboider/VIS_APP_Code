#!/usr/bin/env bash
# Re-score all eight round-1 config folders with playwright 1.48 (FORCE).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="$PWD/.eval-venv/bin:$PATH"

FOLDERS=(
  _runs_claude_4.8_126 _runs_claude_4.7_126 _runs_claude_4.6_126 _runs_claude_4.5_126
  _runs_claude_4.8_152 _runs_claude_4.7_152 _runs_claude_4.6_152 _runs_claude_4.5_152
)

for f in "${FOLDERS[@]}"; do
  echo "############ EVAL START $f ($(date +%H:%M:%S)) ############"
  FILTER='*c4*' FORCE=1 ./eval_all_runs.sh "$PWD/$f"
  echo "############ EVAL DONE  $f ($(date +%H:%M:%S)) ############"
done
echo "############ ALL R1 EVALS COMPLETE ############"
