#!/usr/bin/env bash
# J5 selected-depth controls.  L20 was selected before adapter results: fp=.06333,
# exceeding L40=.05667 and L60=.06000.  Clean root avoids layerless-parts reuse.
set -u -o pipefail
REPO=/home/ubuntu/dementor; PY="$REPO/.venv/bin/python"
HFROOT=/home/ubuntu/dementor-runtime/huggingface; STEER=/home/ubuntu/dementor-runtime/steering
ADAPTERS=/home/ubuntu/dementor-runtime/dpo_runs; WORKLIST=/home/ubuntu/dementor-runtime/worklist_mp.txt
CONE_STATUS=/home/ubuntu/dementor-runtime/steering-adapter-j5/cone_only_2gpu/queue_status.log
ROOT=/home/ubuntu/dementor-runtime/steering-adapter-j5/selected_L20_controls; LOGS="$ROOT/logs"; STATUS="$ROOT/queue_status.log"
VECTORS="$STEER/llama-3.3-70b/vectors_depths.pt"
mkdir -p "$ROOT" "$LOGS"
export HF_HOME="$HFROOT" HF_HUB_CACHE="$HFROOT/hub" HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 PYTHONPATH="$REPO" DEMENTOR_STEER_WORK="$STEER"
printf '[%s] WAIT J5 cone-only 60 of 60; selected depth L20\n' "$(date -Is)" | tee -a "$STATUS"
until grep -Fq 'COMPLETE 60 of 60 cone-only J5 cells' "$CONE_STATUS" 2>/dev/null; do sleep 60; done
split -n l/2 -d "$WORKLIST" "$ROOT/wl_"
n0="$(wc -l < "$ROOT/wl_00")"; n1="$(wc -l < "$ROOT/wl_01")"
if [ $((n0+n1)) -ne 60 ]; then printf '[%s] FAIL split (%s,%s)\n' "$(date -Is)" "$n0" "$n1" | tee -a "$STATUS"; exit 1; fi
run_lane() {
  local gpus="$1" list="$2" lane="$3"
  while IFS='|' read -r src ds cell; do
    local ad="$ADAPTERS/$ds/$cell" out="$ROOT/${ds}__${cell}" log="$LOGS/${ds}__${cell}.log"
    [ -f "$out/.done" ] && continue
    [ -f "$ad/adapter_config.json" ] || { printf '[%s] FAIL lane%s missing %s\n' "$(date -Is)" "$lane" "$ad" | tee -a "$STATUS"; return 1; }
    mkdir -p "$out"; printf '[%s] START lane%s %s/%s L20\n' "$(date -Is)" "$lane" "$ds" "$cell" | tee -a "$STATUS"
    CUDA_VISIBLE_DEVICES="$gpus" DEMENTOR_MP=1 DEMENTOR_ABLATE_LAYER=20 "$PY" "$REPO/experiments/steering/run_benchmark_eval.py" "$src" --benchmarks advbench --adapter "$ad" --vectors-ml "$VECTORS" --outdir "$out" --gen-batch 8 >"$log" 2>&1
    local rc=$?; if [ "$rc" -ne 0 ] || [ ! -f "$out/eval_advbench/metrics.json" ]; then printf '[%s] FAIL lane%s rc=%s %s/%s\n' "$(date -Is)" "$lane" "$rc" "$ds" "$cell" | tee -a "$STATUS"; return 1; fi
    touch "$out/.done"; printf '[%s] OK lane%s %s/%s\n' "$(date -Is)" "$lane" "$ds" "$cell" | tee -a "$STATUS"
  done < "$list"
}
run_lane 0,1 "$ROOT/wl_00" 01 & a=$!; run_lane 2,3 "$ROOT/wl_01" 02 & b=$!
wait "$a"; ra=$?; wait "$b"; rb=$?
if [ "$ra" -ne 0 ] || [ "$rb" -ne 0 ]; then printf '[%s] FAIL lanes (%s,%s)\n' "$(date -Is)" "$ra" "$rb" | tee -a "$STATUS"; exit 1; fi
printf '[%s] COMPLETE 60 of 60 selected-L20 J5 cells\n' "$(date -Is)" | tee -a "$STATUS"
