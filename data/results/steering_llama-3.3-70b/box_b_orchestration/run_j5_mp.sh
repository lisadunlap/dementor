#!/usr/bin/env bash
# Box B J5: one 3-card MP worker, 60 DPO-adapter cells, resumable with .done markers.
set -euo pipefail

PY=/home/ubuntu/dementor/.venv/bin/python
REPO=/home/ubuntu/dementor
ADAPTERS=/home/ubuntu/dementor-runtime/dpo_runs
WORKLIST=/home/ubuntu/dementor-runtime/worklist_mp.txt
OUT=/home/ubuntu/dementor-runtime/steering-adapter-j5
LOGS="$OUT/logs_mp"

mkdir -p "$LOGS" "$OUT/full"
export DEMENTOR_STEER_WORK=/home/ubuntu/dementor-runtime/steering
export HF_HOME=/home/ubuntu/dementor-runtime/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=0
export PYTHONPATH="$REPO"
unset DEMENTOR_RTL_JUDGE_DIR

while IFS='|' read -r src ds cell; do
  ad="$ADAPTERS/$ds/$cell"
  o="$OUT/full/${ds}__${cell}"
  [[ -f "$o/.done" ]] && continue
  [[ -f "$ad/adapter_config.json" ]] || { echo "MISSING $ad" >&2; exit 1; }
  mkdir -p "$o"
  echo "[mp] $(date -u +%FT%TZ) START $ds/$cell"
  CUDA_VISIBLE_DEVICES=0,1,2 DEMENTOR_MP=1 "$PY" \
    "$REPO/experiments/steering/run_benchmark_eval.py" "$src" \
    --benchmarks advbench --adapter "$ad" --outdir "$o" --gen-batch 8 \
    >> "$LOGS/${ds}__${cell}.log" 2>&1
  if compgen -G "$o/eval_*/metrics.json" >/dev/null; then
    touch "$o/.done"
    echo "[mp] OK $ds/$cell"
  else
    echo "[mp] FAIL $ds/$cell: no metrics.json" >&2
    exit 1
  fi
done < "$WORKLIST"
