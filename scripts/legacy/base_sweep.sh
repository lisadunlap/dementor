#!/usr/bin/env bash
# moved to scripts/legacy
export CUDA_VISIBLE_DEVICES=4

# compare google_gemma 27b vs OpenGVLab_InternVL3-8B
python disguising/llm_scorer2.py \
  --input_file_a disguising/model-responses/base_500_all_models_2/google_gemma-3-27b-it.csv \
  --input_file_b disguising/model-responses/base_500_all_models/OpenGVLab_InternVL3-8B.csv \
  --output_file disguising/comparisons/base/google_gemma-3-27b-it_vs_internvl3-8b.csv

#EXTRANEOUS 
# compare google_gemma 27b vs OpenGVLab_InternVL3-8B
python disguising/llm_scorer2.py \
  --input_file_a disguising/model-responses/base_500_all_models/google_gemma-2-2b-it.csv \
  --input_file_b disguising/model-responses/base_500_all_models/OpenGVLab_InternVL3-8B.csv \
  --output_file disguising/comparisons/base/google_gemma-2-2b-it_vs_internvl3-8b.csv

# compare google_gemma 27b vs microsoft_Phi-4-mini-instruct
python disguising/llm_scorer2.py \
  --input_file_a disguising/model-responses/base_500_all_models_2/google_gemma-3-27b-it.csv \
  --input_file_b disguising/model-responses/base_500_all_models/microsoft_Phi-4-mini-instruct.csv \
  --output_file disguising/comparisons/base/google_gemma-3-27b-it_vs_phi-4-mini-instruct.csv

# compare google_gemma 27b vs Qwen_Qwen2.5-VL-7B-Instruct
python disguising/llm_scorer2.py \
  --input_file_a disguising/model-responses/base_500_all_models_2/google_gemma-3-27b-it.csv \
  --input_file_b disguising/model-responses/base_500_all_models/Qwen_Qwen2.5-VL-7B-Instruct.csv \
  --output_file disguising/comparisons/base/google_gemma-3-27b-it_vs_qwen2.5-vl-7b-instruct.csv

# compare qwen-qwen3-32b vs OpenGVLab_InternVL3-8B
python disguising/llm_scorer2.py \
  --input_file_a disguising/model-responses/base_500_all_models_2/Qwen_Qwen3-32B.csv \
  --input_file_b disguising/model-responses/base_500_all_models/OpenGVLab_InternVL3-8B.csv \
  --output_file disguising/comparisons/base/qwen-qwen3-32b_vs_internvl3-8b.csv

# compare qwen-qwen3-32b vs microsoft_Phi-4-mini-instruct
python disguising/llm_scorer2.py \
  --input_file_a disguising/model-responses/base_500_all_models_2/Qwen_Qwen3-32B.csv \
  --input_file_b disguising/model-responses/base_500_all_models/microsoft_Phi-4-mini-instruct.csv \
  --output_file disguising/comparisons/base/qwen-qwen3-32b_vs_phi-4-mini-instruct.csv

# compare qwen-qwen3-32b vs Qwen_Qwen2.5-VL-7B-Instruct
python disguising/llm_scorer2.py \
  --input_file_a disguising/model-responses/base_500_all_models_2/Qwen_Qwen3-32B.csv \
  --input_file_b disguising/model-responses/base_500_all_models/Qwen_Qwen2.5-VL-7B-Instruct.csv \
  --output_file disguising/comparisons/base/qwen-qwen3-32b_vs_qwen2.5-vl-7b-instruct.csv
