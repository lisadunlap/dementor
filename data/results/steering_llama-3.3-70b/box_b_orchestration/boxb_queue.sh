#!/usr/bin/env bash
# Fresh-Box-B ordered experiment controller.  It never advances across a failed
# preflight: a zero-exit partial experiment is worse than an explicit stop.
set -euo pipefail

repo=/home/ubuntu/dementor
runtime=/home/ubuntu/dementor-runtime
work="$runtime/steering/repl80_rdo"
log="$runtime/boxb_queue.log"

export DEMENTOR_REPO="$repo"
export DEMENTOR_DATA="$runtime"
export DEMENTOR_STEER_WORK="$work"
export DEMENTOR_HF_HOME="$runtime/huggingface"
export HF_HOME="$runtime/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="$repo"

exec >>"$log" 2>&1
printf '[%s] Box B queue controller started\n' "$(date -Is)"

# The E4B process was started before tmux was requested.  Wait for it without
# pgrep (which can match this controller), then require its final metrics.
while ps -u "$USER" -o args= | awk '/run_rdo_model\.py gemma-4-e4b/ && !/awk/ {found=1} END {exit !found}'; do
    printf '[%s] waiting for J1 gemma-4-e4b\n' "$(date -Is)"
    sleep 30
done

j1_metrics="$work/gemma-4-e4b_tplfix/eval/metrics.json"
if [[ ! -f "$j1_metrics" ]]; then
    printf '[%s] ERROR J1 exited without %s; inspect %s/ERROR.json and rdo_run.log\n' \
        "$(date -Is)" "$j1_metrics" "$work/gemma-4-e4b_tplfix"
    exit 1
fi
printf '[%s] J1 completed; beginning J5 llama adapter-steering preflight\n' "$(date -Is)"

# J5 must have 4 datasets x 15 non-self targets = 60 materialized local PEFT
# adapters, an external/full runner, and stock Llama direction artifacts.
adapter_root="$repo/data/results/matrix/dpo_runs"
runner=/data/ethantsliu/exp_steer_adapter/run_full.sh
llama_stock="$work/llama-3.3-70b"
count=0
if [[ -d "$adapter_root" ]]; then
    count=$(find "$adapter_root" -mindepth 2 -maxdepth 2 -type d \
        -name 'llama-3.3-70b_as_*_seed42' | wc -l)
fi
if [[ "$count" -ne 60 ]]; then
    printf '[%s] ERROR J5 preflight: expected 60 llama-3.3-70b DPO adapter directories under %s; found %s\n' \
        "$(date -Is)" "$adapter_root" "$count"
    exit 1
fi
if [[ ! -x "$runner" ]]; then
    printf '[%s] ERROR J5 preflight: required runner is absent or not executable: %s\n' \
        "$(date -Is)" "$runner"
    exit 1
fi
if [[ ! -f "$llama_stock/selected_cone.pt" || ! -f "$llama_stock/vectors_ml.pt" ]]; then
    printf '[%s] ERROR J5 preflight: missing stock Llama cone/vectors in %s\n' \
        "$(date -Is)" "$llama_stock"
    exit 1
fi

printf '[%s] J5 preflight passed; manual runner invocation required by its external worklist contract\n' "$(date -Is)"
exit 0
