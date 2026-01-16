#!/bin/bash
# Run stylometric ensemble scoring on all disguised responses
# This scores both directions (GPT->Llama and Llama->GPT) for each method

set -e  # Exit on error

ENSEMBLE_MODEL="data/results/stylometric_classifier/mmlu_pro_ensemble/ensemble_stylometric.joblib"
BASE_DIR="data/results/generic/eval200"

echo "======================================"
echo "Stylometric Ensemble Scoring"
echo "======================================"
echo ""

# Array of methods
METHODS=("contrastive" "behavioral_based" "stylistic" "random_sampling")

for method in "${METHODS[@]}"; do
    echo "=== Method: $method ==="
    
    # Direction 1: GPT -> Llama (GPT disguising as Llama)
    echo "  [1/2] Scoring GPT -> Llama..."
    python scripts/judge/stylometric_ensemble.py score \
        --model-path "$ENSEMBLE_MODEL" \
        --input "$BASE_DIR/$method/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv" \
        --output "$BASE_DIR/$method/scores/stylometric_ensemble_gpt_as_llama.csv"
    
    # Direction 2: Llama -> GPT (Llama disguising as GPT)
    echo "  [2/2] Scoring Llama -> GPT..."
    python scripts/judge/stylometric_ensemble.py score \
        --model-path "$ENSEMBLE_MODEL" \
        --input "$BASE_DIR/$method/vllm_meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv" \
        --output "$BASE_DIR/$method/scores/stylometric_ensemble_llama_as_gpt.csv"
    
    echo "  ✓ Completed $method"
    echo ""
done

echo "======================================"
echo "All scoring completed!"
echo "======================================"
echo ""
echo "Summary of output files:"
for method in "${METHODS[@]}"; do
    echo "$method:"
    echo "  - $BASE_DIR/$method/scores/stylometric_ensemble_gpt_as_llama.csv"
    echo "  - $BASE_DIR/$method/scores/stylometric_ensemble_llama_as_gpt.csv"
done

