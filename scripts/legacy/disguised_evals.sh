#!/usr/bin/env bash
# moved to scripts/legacy
export CUDA_VISIBLE_DEVICES=4

# compare disguised llama-3-8b_as_google_gemma vs original google_gemma
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/google_gemma-3-1b-it_responses-1000.csv \
  --input_file_b disguising/model-responses/disguised/llama-3-8b_as_google_gemma-3-1b.csv \
  --output_file disguising/comparisons/disguised/llama-3-8b_as_google_gemma-3-1b_vs_google_gemma-3-1b.csv

# compare disguised llama-3-8b_as_gpt-3.5 vs original gpt-3.5
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/gpt-3.5_responses.csv \
  --input_file_b disguising/model-responses/disguised/llama-3-8b_as_gpt-3.5.csv \
  --output_file disguising/comparisons/disguised/llama-3-8b_as_gpt-3.5_vs_gpt-3.5.csv

# compare disguised llama-3-8b_as_gpt-4o-mini vs original gpt-4o-mini
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/gpt-4o-mini_responses.csv \
  --input_file_b disguising/model-responses/disguised/llama-3-8b_as_gpt-4o-mini.csv \
  --output_file disguising/comparisons/disguised/llama-3-8b_as_gpt-4o-mini_vs_gpt-4o-mini.csv

# compare disguised llama-3-8b_as_gpt-4o vs original gpt-4o
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/gpt-4o_responses.csv \
  --input_file_b disguising/model-responses/disguised/llama-3-8b_as_gpt-4o.csv \
  --output_file disguising/comparisons/disguised/llama-3-8b_as_gpt-4o_vs_gpt-4o.csv

# compare disguised llama-3-8b_as_internvl3 vs original internvl3-9b
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/internvl3-9b_responses.csv \
  --input_file_b disguising/model-responses/disguised/llama-3-8b_as_internvl3.csv \
  --output_file disguising/comparisons/disguised/llama-3-8b_as_internvl3_vs_internvl3-9b.csv

# compare disguised llama-3-8b_as_molmo-7b-o vs original molmo-7b-d
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/molmo-7b-d_responses.csv \
  --input_file_b disguising/model-responses/disguised/llama-3-8b_as_molmo-7b-o.csv \
  --output_file disguising/comparisons/disguised/llama-3-8b_as_molmo-7b-o_vs_molmo-7b-d.csv

# compare disguised llama-3-8b_as_phi-4-multimodal vs original phi-4-multimodal
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/phi-4-multimodal-instruct_responses.csv \
  --input_file_b disguising/model-responses/disguised/llama-3-8b_as_phi-4-multimodal.csv \
  --output_file disguising/comparisons/disguised/llama-3-8b_as_phi-4-multimodal_vs_phi-4-multimodal.csv

# compare disguised llama-3-8b_as_qwen-7b vs original qwen-7b
python disguising/llm_scorer.py \
  --input_file_a disguising/model-responses/base/qwen-7b_responses.csv \
  --input_file_b disguising/model-responses/disguised/llama-3-8b_as_qwen-7b.csv \
  --output_file disguising/comparisons/disguised/llama-3-8b_as_qwen-7b_vs_qwen-7b.csv
