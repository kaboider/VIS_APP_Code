#!/usr/bin/env bash
# Sequentially eval all eight round-2 config folders (single docker host).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="$PWD/.eval-venv/bin:$PATH"

FOLDERS=(
  _runs_claude_4.8_126_r2 _runs_claude_4.7_126_r2 _runs_claude_4.6_126_r2 _runs_claude_4.5_126_r2
  _runs_claude_4.8_152_r2 _runs_claude_4.7_152_r2 _runs_claude_4.6_152_r2 _runs_claude_4.5_152_r2
)

for f in "${FOLDERS[@]}"; do
  echo "############ EVAL START $f ($(date +%H:%M:%S)) ############"
  FILTER='*c4*' ./eval_all_runs.sh "$PWD/$f"
  echo "############ EVAL DONE  $f ($(date +%H:%M:%S)) ############"
done
echo "############ ALL R2 EVALS COMPLETE ############"
