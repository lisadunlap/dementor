#!/usr/bin/env bash
# Resume Box-B J5 after the completed 2-GPU pilot.  The pilot's allocator
# warning was non-fatal: it exited 0, wrote metrics, and logged two-GPU sharding.
set -u -o pipefail

REPO=/home/ubuntu/dementor
PY="$REPO/.venv/bin/python"
HFROOT=/home/ubuntu/dementor-runtime/huggingface
STEER=/home/ubuntu/dementor-runtime/steering
ADAPTERS=/home/ubuntu/dementor-runtime/dpo_runs
WORKLIST=/home/ubuntu/dementor-runtime/worklist_mp.txt
ROOT=/home/ubuntu/dementor-runtime/steering-adapter-j5/cone_only_2gpu
LOGS="$ROOT/logs"
STATUS="$ROOT/queue_status.log"
PILOT="$ROOT/chatbot_arena__llama-3.3-70b_as_aya-expanse-8b_seed42"

mkdir -p "$ROOT" "$LOGS"
export HF_HOME="$HFROOT" HF_HUB_CACHE="$HFROOT/hub" HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO" DEMENTOR_STEER_WORK="$STEER"

if [ ! -f "$PILOT/eval_advbench/metrics.json" ] \
   || ! grep -Fq '[MP] device_map sharded across 2 GPUs' "$PILOT/bench_advbench.log" \
   || grep -qi 'meta device\|offload.*cpu' "$PILOT/bench_advbench.log"; then
  printf '[%s] FAIL pilot lacks valid 2-GPU placement evidence\n' "$(date -Is)" | tee -a "$STATUS"; exit 1
fi
touch "$PILOT/.done"
printf '[%s] PILOT_ACCEPTED: completed 2-GPU sharding; non-fatal allocator warning recorded\n' "$(date -Is)" | tee -a "$STATUS"

split -n l/2 -d "$WORKLIST" "$ROOT/wl_"
# GNU split balances line boundaries, which can be 31/29 rather than 30/30.
# The invariant is complete, non-overlapping coverage with at most one-cell skew.
n0="$(wc -l < "$ROOT/wl_00")"; n1="$(wc -l < "$ROOT/wl_01")"
if [ $((n0 + n1)) -ne 60 ] || [ $(( n0 > n1 ? n0 - n1 : n1 - n0 )) -gt 2 ]; then
  printf '[%s] FAIL invalid 60-cell split (%s,%s)\n' "$(date -Is)" "$n0" "$n1" | tee -a "$STATUS"; exit 1
fi
printf '[%s] WORKLIST split (%s,%s)\n' "$(date -Is)" "$n0" "$n1" | tee -a "$STATUS"

run_lane() {
  local gpus="$1" list="$2" lane="$3"
  while IFS='|' read -r src ds cell; do
    local ad="$ADAPTERS/$ds/$cell" out="$ROOT/${ds}__${cell}" log="$LOGS/${ds}__${cell}.log"
    [ -f "$out/.done" ] && continue
    if [ ! -f "$ad/adapter_config.json" ]; then
      printf '[%s] FAIL lane%s missing adapter %s\n' "$(date -Is)" "$lane" "$ad" | tee -a "$STATUS"; return 1
    fi
    mkdir -p "$out"
    printf '[%s] START lane%s %s/%s\n' "$(date -Is)" "$lane" "$ds" "$cell" | tee -a "$STATUS"
    CUDA_VISIBLE_DEVICES="$gpus" DEMENTOR_MP=1 "$PY" "$REPO/experiments/steering/run_benchmark_eval.py" "$src" \
      --benchmarks advbench --adapter "$ad" --outdir "$out" --no-controls --gen-batch 8 >"$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ] || [ ! -f "$out/eval_advbench/metrics.json" ]; then
      printf '[%s] FAIL lane%s rc=%s %s/%s; lane stopped\n' "$(date -Is)" "$lane" "$rc" "$ds" "$cell" | tee -a "$STATUS"; return 1
    fi
    touch "$out/.done"
    printf '[%s] OK lane%s %s/%s\n' "$(date -Is)" "$lane" "$ds" "$cell" | tee -a "$STATUS"
  done < "$list"
}

run_lane 0,1 "$ROOT/wl_00" 01 & one=$!
run_lane 2,3 "$ROOT/wl_01" 02 & two=$!
wait "$one"; rc1=$?
wait "$two"; rc2=$?
if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ]; then
  printf '[%s] FAIL J5 lanes exited (%s,%s)\n' "$(date -Is)" "$rc1" "$rc2" | tee -a "$STATUS"; exit 1
fi
printf '[%s] COMPLETE 60 of 60 cone-only J5 cells\n' "$(date -Is)" | tee -a "$STATUS"
