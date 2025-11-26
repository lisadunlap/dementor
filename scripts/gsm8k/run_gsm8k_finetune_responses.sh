#!/usr/bin/env bash
set -euo pipefail

# Common arguments
PROMPTS="data/datasets/gsm8k/gsm8k_prompts_500.csv"
SEED=42
TRAIN_SIZE=300
EVAL_SIZE=200

RUN_TINKER_SFT=false
RUN_TINKER_DPO=true
RUN_OPENAI_SFT=true
RUN_OPENAI_DPO=true

if [ "$RUN_TINKER_SFT" = true ]; then
  echo "[Tinker SFT] Generating responses..."
python scripts/generate_responses.py \
    --prompts-file "$PROMPTS" \
    --dataset-csv data/model-responses/gsm8k/500/openai_gpt-4.1-mini_responses.csv \
    --train-size "$TRAIN_SIZE" \
    --eval-size "$EVAL_SIZE" \
    --seed "$SEED" \
    --output-csv data/results/gsm8k/500/sft_tinker/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
    tinker \
    --adapter-name gsm8k_llama-3.1-8b-instruct \
    --renderer-name llama3
  python -m scripts.scorer --compare \
    --a data/results/gsm8k/500/sft_tinker/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
    --b data/model-responses/gsm8k/500/openai_gpt-4.1-mini_responses.csv \
    --output data/results/gsm8k/500/sft_tinker/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini_vs_openai_gpt-4.1-mini.csv \
    --judge-model openai/gpt-4.1-mini
else
  echo "[Tinker SFT] Skipped (RUN_TINKER_SFT=false)"
fi

if [ "$RUN_TINKER_DPO" = true ]; then
  echo "[Tinker DPO] Generating responses..."
python scripts/generate_responses.py \
    --prompts-file "$PROMPTS" \
    --dataset-csv data/model-responses/gsm8k/500/openai_gpt-4.1-mini_responses.csv \
    --train-size "$TRAIN_SIZE" \
    --eval-size "$EVAL_SIZE" \
    --seed "$SEED" \
    --output-csv data/results/gsm8k/500/dpo_tinker/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
    tinker \
    --adapter-name gsm8k_dpo_llama-3.1-8b-instruct \
    --renderer-name llama3
  python -m scripts.scorer --compare \
    --a data/results/gsm8k/500/dpo_tinker/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
    --b data/model-responses/gsm8k/500/openai_gpt-4.1-mini_responses.csv \
    --output data/results/gsm8k/500/dpo_tinker/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini_vs_openai_gpt-4.1-mini.csv \
    --judge-model openai/gpt-4.1-mini
else
  echo "[Tinker DPO] Skipped (RUN_TINKER_DPO=false)"
fi

if [ "$RUN_OPENAI_SFT" = true ]; then
  echo "[OpenAI SFT] Generating responses..."
python scripts/generate_responses.py \
    --prompts-file "$PROMPTS" \
    --dataset-csv data/model-responses/gsm8k/500/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv \
    --train-size "$TRAIN_SIZE" \
    --eval-size "$EVAL_SIZE" \
    --seed "$SEED" \
    --output-csv data/results/gsm8k/500/sft_openai/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
    openai \
    --model ft:gpt-4.1-mini-2025-04-14:uc-berkeley-prof-trevor-darrell-group::Ccl884ot \
    --system-prompt "You are a patient math tutor."
  python -m scripts.scorer --compare \
    --a data/results/gsm8k/500/sft_openai/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
    --b data/model-responses/gsm8k/500/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv \
    --output data/results/gsm8k/500/sft_openai/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct_vs_meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
    --judge-model openai/gpt-4.1-mini
else
  echo "[OpenAI SFT] Skipped (RUN_OPENAI_SFT=false)"
fi

if [ "$RUN_OPENAI_DPO" = true ]; then
  echo "[OpenAI DPO] Generating responses..."
python scripts/generate_responses.py \
    --prompts-file "$PROMPTS" \
    --dataset-csv data/model-responses/gsm8k/500/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv \
    --train-size "$TRAIN_SIZE" \
    --eval-size "$EVAL_SIZE" \
    --seed "$SEED" \
    --output-csv data/results/gsm8k/500/dpo_openai/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
    openai \
    --model ft:gpt-4.1-mini-2025-04-14:uc-berkeley-prof-trevor-darrell-group::CclMD1vd \
    --system-prompt "You are a patient math tutor."
  python -m scripts.scorer --compare \
    --a data/results/gsm8k/500/dpo_openai/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
    --b data/model-responses/gsm8k/500/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv \
    --output data/results/gsm8k/500/dpo_openai/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct_vs_meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
    --judge-model openai/gpt-4.1-mini
else
  echo "[OpenAI DPO] Skipped (RUN_OPENAI_DPO=false)"
fi

echo "Done. Responses and scores saved under data/results/gsm8k/500/{sft,dpo}_{tinker,openai}/"
