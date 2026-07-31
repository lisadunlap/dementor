#!/usr/bin/env bash
set -euo pipefail

# Legacy convenience launcher for the four GSM8K fine-tuning jobs.
# New work should call `python -m workflows.run_gsm8k_workflow ...` directly.

RUN_TINKER_SFT="${RUN_TINKER_SFT:-true}"
RUN_OPENAI_SFT="${RUN_OPENAI_SFT:-true}"
RUN_TINKER_DPO="${RUN_TINKER_DPO:-true}"
RUN_OPENAI_DPO="${RUN_OPENAI_DPO:-true}"

pids=()

run_background() {
  local enabled="$1"
  local label="$2"
  shift 2

  if [[ "$enabled" != "true" ]]; then
    echo "[$label] skipped"
    return
  fi

  echo "[$label] starting"
  "$@" &
  pids+=("$!")
}

run_background "$RUN_TINKER_SFT" "Tinker SFT" \
  python -m workflows.run_gsm8k_workflow \
    --stage sft \
    --provider tinker \
    --output-dir data/results/workflows/sft_llama-3.1-8b-instruct_as_gpt-4.1-mini \
    --weights-name gsm8k_llama-3.1-8b-instruct \
    --epochs 6 \
    --batch-size 16 \
    --learning-rate 1e-4

run_background "$RUN_OPENAI_SFT" "OpenAI SFT" \
  python -m workflows.run_gsm8k_workflow \
    --stage sft \
    --provider openai \
    --output-dir data/results/workflows/sft_gpt-4.1-mini_as_llama-3.1-8b-instruct \
    --openai-model gpt-4.1-mini-2025-04-14 \
    --system-prompt "You are a patient math tutor." \
    --epochs 6 \
    --batch-size 16

run_background "$RUN_TINKER_DPO" "Tinker DPO" \
  python -m workflows.run_gsm8k_workflow \
    --stage dpo \
    --provider tinker \
    --output-dir data/results/workflows/dpo_llama-3.1-8b-instruct_as_gpt-4.1-mini \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --renderer-name llama3 \
    --epochs 3 \
    --batch-size 16 \
    --save-every 20

run_background "$RUN_OPENAI_DPO" "OpenAI DPO" \
  python -m workflows.run_gsm8k_workflow \
    --stage dpo \
    --provider openai \
    --output-dir data/results/workflows/dpo_gpt-4.1-mini_as_llama-3.1-8b-instruct \
    --openai-model gpt-4.1-mini-2025-04-14 \
    --epochs 3 \
    --batch-size 16 \
    --dpo-beta 0.1

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No workflows selected."
  exit 0
fi

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "All selected GSM8K workflows finished."
