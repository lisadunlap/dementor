#!/usr/bin/env bash
set -euo pipefail

python -m workflows.run_openai_sft &
pid1=$!
python -m workflows.run_openai_dpo &
pid2=$!

wait $pid1 $pid2
