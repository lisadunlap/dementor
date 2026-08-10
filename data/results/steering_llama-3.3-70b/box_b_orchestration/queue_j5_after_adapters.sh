#!/usr/bin/env bash
set -euo pipefail

LOG=/home/ubuntu/dementor-runtime/logs/j5-adapters.log
WORKLIST=/home/ubuntu/dementor-runtime/worklist_mp.txt
ROOT=/home/ubuntu/dementor-runtime/dpo_runs

while [[ ! -f "$WORKLIST" ]]; do
  if grep -Eq 'Traceback|^ERROR|^Error' "$LOG" 2>/dev/null; then
    echo "PREP_FAIL: adapter materialization failed; refusing J5 launch" >&2
    exit 1
  fi
  sleep 15
done

[[ $(wc -l < "$WORKLIST") -eq 60 ]] || { echo "PREP_FAIL: worklist is not 60 cells" >&2; exit 1; }
[[ $(find "$ROOT" -name adapter_config.json -path '*llama-3.3-70b_as_*' -print | wc -l) -eq 60 ]] || {
  echo "PREP_FAIL: adapter inventory is not 60 configs" >&2
  exit 1
}

cd /home/ubuntu/dementor
set -a
. ./.env
set +a
exec /home/ubuntu/dementor-runtime/run_j5_mp.sh
