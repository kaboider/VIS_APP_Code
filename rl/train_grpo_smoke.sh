#!/usr/bin/env bash
# Tiny GRPO smoke: camel tool loop + structural workspace reward (no Docker).
# Picks an idle GPU at runtime. Does not touch busy cards (incl. 27B vLLM on 4/5).
set -euo pipefail

VIS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERA="${MERA_ROOT:-/shared_home/yuhang.yao/MERA-Evolve}"
PYTHON="${PYTHON:-$MERA/venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$MERA/venv/bin/python"

DATA="${DATA_DIR:-$VIS/_rl_runs/data}"
OUT="${OUTPUT_DIR:-$VIS/_rl_runs/grpo_smoke_tools}"
LOG="$OUT/train.log"
mkdir -p "$OUT" "$DATA"

pick_idle_gpu() {
  "$PYTHON" - <<'PY'
import subprocess
import sys

def run(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""

gpu_txt = run([
    "nvidia-smi",
    "--query-gpu=index,uuid,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
])
if not gpu_txt.strip():
    sys.exit(1)

apps = run([
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
    "--format=csv,noheader",
])
uuid_busy = set()
uuid_vllm = set()
for line in apps.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        continue
    uuid, _pid, name = parts[0], parts[1], parts[2].lower()
    uuid_busy.add(uuid)
    if any(tok in name for tok in ("vllm", "sglang", "pt_engine")):
        uuid_vllm.add(uuid)

rows = []
for line in gpu_txt.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        continue
    idx, uuid = int(parts[0]), parts[1]
    mem = float(parts[2] or 0)
    util = float(parts[3] or 0)
    rows.append((idx, uuid, mem, util))

def is_idle(idx, uuid, mem, util):
    if uuid in uuid_vllm:
        return False
    if idx in (4, 5) and uuid in uuid_busy:
        return False
    if mem >= 500 or util >= 5:
        return False
    if uuid in uuid_busy and mem >= 200:
        return False
    return True

idle = [r for r in rows if is_idle(*r)]
if not idle:
    sys.exit(2)

# Prefer GPU 6 when it is free.
idle.sort(key=lambda r: (0 if r[0] == 6 else 1, r[2], r[0]))
print(idle[0][0])
PY
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU="$(pick_idle_gpu || true)"
  if [[ -z "$GPU" ]]; then
    echo "[smoke] skip: no idle GPU. Last known: GPUs 4+5 may host vLLM Qwen3.5-27B; prefer GPU 6 when free."
    echo "[smoke] script is ready: CUDA_VISIBLE_DEVICES=<idle> bash $VIS/rl/train_grpo_smoke.sh"
    exit 0
  fi
  export CUDA_VISIBLE_DEVICES="$GPU"
  echo "[smoke] selected idle GPU $CUDA_VISIBLE_DEVICES"
else
  echo "[smoke] using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

if [[ ! -f "$DATA/train.parquet" ]]; then
  "$PYTHON" "$VIS/rl/prepare_visapp_data.py" --out-dir "$DATA"
fi

export PYTHONPATH="$VIS:$MERA:${PYTHONPATH:-}"
set +e
N_GPUS=1 \
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-1.5B-Instruct}" \
TRAIN_FILE="$DATA/train.parquet" \
VAL_FILE="$DATA/val.parquet" \
OUTPUT_DIR="$OUT" \
PROJECT_NAME=visapp_camel_rl \
EXPERIMENT_NAME=grpo_smoke_tools \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}" \
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
N_GENERATIONS="${N_GENERATIONS:-2}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}" \
MAX_TURNS="${MAX_TURNS:-6}" \
TOTAL_STEPS="${TOTAL_STEPS:-2}" \
SAVE_FREQ="${SAVE_FREQ:-1}" \
LORA_RANK="${LORA_RANK:-16}" \
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-1}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}" \
bash "$VIS/rl/train_grpo.sh" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

echo "[smoke] train exit=$status output=$OUT log=$LOG"

python_grep() {
  "$PYTHON" - "$LOG" <<'PY'
import re, sys
path = sys.argv[1]
try:
    text = open(path, encoding="utf-8", errors="replace").read()
except OSError:
    print("[smoke] no log to parse")
    sys.exit(0)

def grab(name):
    # verl may print `critic/score/mean:0.0` or `actor/pg_loss:np.float64(0.0)`
    pat = re.compile(
        rf"{re.escape(name)}(?::np\.float64)?[:\s(]+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    )
    return pat.findall(text)

scores = grab("critic/score/mean")
pgs = grab("actor/pg_loss")
print("[smoke] critic/score/mean matches:", ", ".join(scores) if scores else "(none)")
print("[smoke] actor/pg_loss matches:", ", ".join(pgs) if pgs else "(none)")
if scores and pgs:
    last_s, last_p = scores[-1], pgs[-1]
    print(f"[smoke] last critic/score/mean={last_s} actor/pg_loss={last_p}")
    try:
        if float(last_s) == 0.0 and float(last_p) == 0.0:
            print("[smoke] WARNING: critic/score/mean and actor/pg_loss are both identically 0")
            print("[smoke] (expected on a 1.5B smoke if all rollouts get the same structural score, e.g. 0)")
    except ValueError:
        pass
else:
    print("[smoke] WARNING: did not find both critic/score/mean and actor/pg_loss in the log")
PY
}
python_grep

if [[ "$status" -ne 0 ]]; then
  echo "[smoke] training crashed (exit $status)" >&2
  exit "$status"
fi
echo "[smoke] done. output=$OUT"
