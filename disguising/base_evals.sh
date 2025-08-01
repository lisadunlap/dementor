#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4

# compare google_gemma vs llama-3-8b
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/google_gemma-3-1b-it_responses-1000.csv \
  --input_file_b disguising/model-responses/base/llama-3-8b-instruct_responses.csv \
  --output_file disguising/comparisons/base/llama-3-8b_vs_google_gemma-3-1b-it.csv

# compare gpt-3.5 vs llama-3-8b
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/gpt-3.5_responses.csv \
  --input_file_b disguising/model-responses/base/llama-3-8b-instruct_responses.csv \
  --output_file disguising/comparisons/base/llama-3-8b_vs_gpt-3.5.csv

# compare gpt-4o-mini vs llama-3-8b
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/gpt-4o-mini_responses.csv \
  --input_file_b disguising/model-responses/base/llama-3-8b-instruct_responses.csv \
  --output_file disguising/comparisons/base/llama-3-8b_vs_gpt-4o-mini.csv

# compare internvl3-9b vs llama-3-8b
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/internvl3-9b_responses.csv \
  --input_file_b disguising/model-responses/base/llama-3-8b-instruct_responses.csv \
  --output_file disguising/comparisons/base/llama-3-8b_vs_internvl3-9b.csv

# compare llama-3-8b vs itself
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/llama-3-8b-instruct_responses.csv \
  --input_file_b disguising/model-responses/base/llama-3-8b-instruct_responses.csv \
  --output_file disguising/comparisons/base/llama-3-8b_vs_llama-3-8b.csv

# compare molmo-7b-d vs llama-3-8b
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/molmo-7b-d_responses.csv \
  --input_file_b disguising/model-responses/base/llama-3-8b-instruct_responses.csv \
  --output_file disguising/comparisons/base/llama-3-8b_vs_molmo-7b-d.csv

# compare phi-4-multimodal vs llama-3-8b
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/phi-4-multimodal-instruct_responses.csv \
  --input_file_b disguising/model-responses/base/llama-3-8b-instruct_responses.csv \
  --output_file disguising/comparisons/base/llama-3-8b_vs_phi-4-multimodal.csv

# compare qwen-7b vs llama-3-8b
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/qwen-7b_responses.csv \
  --input_file_b disguising/model-responses/base/llama-3-8b-instruct_responses.csv \
  --output_file disguising/comparisons/base/llama-3-8b_vs_qwen-7b.csv