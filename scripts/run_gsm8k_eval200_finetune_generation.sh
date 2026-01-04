#!/usr/bin/env bash
set -euo pipefail

# Launches four eval-200 regeneration commands (SFT/DPO × OpenAI/Tinker)
# in separate tmux windows so they can run in parallel.

SESSION="gsm8k_eval200_finetune"

tmux new-session -d -s "$SESSION" -n "sft_tinker" "
python scripts/generate_responses.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/results/gsm8k/eval200/sft/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --source-model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --target-model-id openai/gpt-4.1-mini \
  tinker \
  --adapter-name gsm8k_llama-3.1-8b-instruct \
  --renderer-name llama3
read
"

tmux new-window -t "$SESSION" -n "dpo_tinker" "
python scripts/generate_responses.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/results/gsm8k/eval200/dpo/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --source-model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --target-model-id openai/gpt-4.1-mini \
  tinker \
  --adapter-name gsm8k_dpo_llama-3.1-8b-instruct \
  --renderer-name llama3
read
"

tmux new-window -t "$SESSION" -n "sft_openai" "
python scripts/generate_responses.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/results/gsm8k/eval200/sft/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
  --source-model-id openai/gpt-4.1-mini \
  --target-model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  openai \
  --model ft:gpt-4.1-mini-2025-04-14:uc-berkeley-prof-trevor-darrell-group::Ccl884ot \
  --system-prompt \"You are a patient math tutor.\"
read
"

tmux new-window -t "$SESSION" -n "dpo_openai" "
python scripts/generate_responses.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/results/gsm8k/eval200/dpo/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
  --source-model-id openai/gpt-4.1-mini \
  --target-model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  openai \
  --model ft:gpt-4.1-mini-2025-04-14:uc-berkeley-prof-trevor-darrell-group::CclMD1vd \
  --system-prompt \"You are a patient math tutor.\"
read
"

echo "Launched tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
