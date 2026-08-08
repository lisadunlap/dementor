#!/usr/bin/env bash
# Run J4 only after both 70B adapter passes release all four cards.
set -u -o pipefail
REPO=/home/ubuntu/dementor; PY="$REPO/.venv/bin/python"; ROOT=/home/ubuntu/dementor-runtime
STATE="$ROOT/imitation_train/after_j5_controls"; STATUS="$STATE/queue_status.log"
CONTROL_STATUS="$ROOT/steering-adapter-j5/selected_L20_controls/queue_status.log"
mkdir -p "$STATE"
export DEMENTOR_REPO="$REPO" DEMENTOR_DATA="$REPO/data" DEMENTOR_HF_HOME="$ROOT/huggingface"
export HF_HOME="$ROOT/huggingface" HF_HUB_CACHE="$ROOT/huggingface/hub" HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 PYTHONPATH="$REPO"
export DEMENTOR_LOCAL_BACKENDS="qwen3.5-4b,qwen3.6-27b,qwen3.6-35b-a3b" IMIT_LOCAL_ONLY=1
export DEMENTOR_MP_SLUGS=qwen3.6-27b DEMENTOR_BLOCK_GPUS="" DEMENTOR_GPUS=0,1,2,3 DEMENTOR_ONLY_GPUS="" SEQ_COEXIST=0
printf '[%s] WAIT selected-L20 J5 controls 60 of 60\n' "$(date -Is)" | tee -a "$STATUS"
until grep -Fq 'COMPLETE 60 of 60 selected-L20 J5 cells' "$CONTROL_STATUS" 2>/dev/null; do sleep 60; done
queue_len() { IMIT_DATASET="$1" DEMENTOR_TRAIN_STATE="$STATE/$1" "$PY" -c "import sys; sys.path.insert(0, '$REPO/experiments/imitation_train'); import worklist; print(len(worklist.compute_queue()))" | tail -1; }
for ds in gsm8k oasst1 writingprompts chatbot_arena; do
  dstate="$STATE/$ds"; mkdir -p "$dstate"; n="$(queue_len "$ds")"
  if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -eq 0 ]; then printf '[%s] FAIL %s preflight=%s\n' "$(date -Is)" "$ds" "$n" | tee -a "$STATUS"; exit 1; fi
  printf '[%s] START %s queue=%s\n' "$(date -Is)" "$ds" "$n" | tee -a "$STATUS"
  IMIT_DATASET="$ds" DEMENTOR_TRAIN_STATE="$dstate" "$PY" "$REPO/experiments/imitation_train/sequencer.py" >>"$dstate/daemon.log" 2>&1 & pid=$!
  while kill -0 "$pid" 2>/dev/null; do sleep 60; n="$(queue_len "$ds")"; if [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -eq 0 ]; then kill -INT "$pid"; wait "$pid" || true; printf '[%s] OK %s\n' "$(date -Is)" "$ds" | tee -a "$STATUS"; break; fi; done
  [ "$n" = 0 ] || { printf '[%s] FAIL %s exited queue=%s\n' "$(date -Is)" "$ds" "$n" | tee -a "$STATUS"; exit 1; }
done
printf '[%s] COMPLETE J4 87 of 87 expected cells\n' "$(date -Is)" | tee -a "$STATUS"
