#!/bin/bash
# Complete MMLU-Pro Disguise Pipeline Example
# This script demonstrates the full workflow for MMLU-Pro dataset

set -e  # Exit on error

echo "=============================================="
echo "MMLU-Pro Disguise Pipeline - Complete Example"
echo "=============================================="

# Configuration
DATASET="mmlu_pro"
TRAIN_SIZE=300
EVAL_SIZE=200
SEED=42
SOURCE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_MODEL="openai/gpt-4.1-mini"
METHOD="contrastive"

# Paths
TRAIN_PROMPTS="data/datasets/${DATASET}/mmlu_pro_prompts_train_${TRAIN_SIZE}_seed${SEED}.csv"
EVAL_PROMPTS="data/datasets/${DATASET}/mmlu_pro_prompts_eval_${EVAL_SIZE}_seed${SEED}.csv"
RESPONSE_DIR="data/model-responses/${DATASET}/full" 
RESULTS_DIR="data/results/${DATASET}/eval${EVAL_SIZE}/${METHOD}"

# Model identifiers (for filenames)
SOURCE_ID=$(echo $SOURCE_MODEL | tr '/' '_')
TARGET_ID=$(echo $TARGET_MODEL | tr '/' '_')

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Train size: $TRAIN_SIZE"
echo "  Eval size: $EVAL_SIZE"
echo "  Source model: $SOURCE_MODEL"
echo "  Target model: $TARGET_MODEL"
echo "  Method: $METHOD"
echo ""

# Step 0: Prepare dataset (if not already done)
if [ ! -f "$TRAIN_PROMPTS" ]; then
    echo "Step 0: Preparing MMLU-Pro dataset..."
    python scripts/prepare_mmlu_pro.py \
        --train_size $TRAIN_SIZE \
        --eval_size $EVAL_SIZE \
        --seed $SEED
    echo "✓ Dataset prepared"
else
    echo "Step 0: Dataset already prepared ✓"
fi

# Step 1a: Generate target model responses (training data)
echo ""
echo "Step 1a: Generating $TARGET_MODEL responses on train set..."
TARGET_TRAIN_RESPONSES="${RESPONSE_DIR}/${TARGET_ID}_responses_train.csv"
if [ ! -f "$TARGET_TRAIN_RESPONSES" ]; then
    python scripts/generate_responses.py \
        --prompts-file $TRAIN_PROMPTS \
        --output-csv $TARGET_TRAIN_RESPONSES \
        basic --model $TARGET_MODEL
    echo "✓ Target train responses generated"
else
    echo "✓ Target train responses already exist"
fi

# Step 1b: Generate source model responses (training data)
echo ""
echo "Step 1b: Generating $SOURCE_MODEL responses on train set..."
SOURCE_TRAIN_RESPONSES="${RESPONSE_DIR}/${SOURCE_ID}_responses_train.csv"
if [ ! -f "$SOURCE_TRAIN_RESPONSES" ]; then
    python scripts/generate_responses.py \
        --prompts-file $TRAIN_PROMPTS \
        --output-csv $SOURCE_TRAIN_RESPONSES \
        basic --model $SOURCE_MODEL
    echo "✓ Source train responses generated"
else
    echo "✓ Source train responses already exist"
fi

# Step 2: Apply disguise on eval set
echo ""
echo "Step 2: Applying disguise (${SOURCE_MODEL} as ${TARGET_MODEL})..."
DISGUISE_OUTPUT="${RESULTS_DIR}/${SOURCE_ID}_as_${TARGET_ID}.csv"
python scripts/disguise.py \
    --prompts_file $EVAL_PROMPTS \
    --model $SOURCE_MODEL \
    --disguise_as $TARGET_MODEL \
    --source_responses $SOURCE_TRAIN_RESPONSES \
    --target_responses $TARGET_TRAIN_RESPONSES \
    --method $METHOD \
    --num_samples $EVAL_SIZE

echo "✓ Disguise applied"

# Step 3: Score the disguise
echo ""
echo "Step 3: Scoring disguised responses..."
SCORES_DIR="${RESULTS_DIR}/scores/${SOURCE_ID}_as_${TARGET_ID}"
SCORED_CSV="${SCORES_DIR}/scored.csv"

python -m scripts.scorer \
    --input $DISGUISE_OUTPUT \
    --output $SCORED_CSV \
    --judge-model $TARGET_MODEL

echo "✓ Scoring complete"

# Step 4 (Optional): Train stylometric classifier
echo ""
echo "Step 4 (Optional): Training stylometric classifier..."
CLASSIFIER_DIR="data/results/stylometric_classifier/${DATASET}_ensemble"
CLASSIFIER_MODEL="${CLASSIFIER_DIR}/ensemble_stylometric.joblib"

if [ ! -f "$CLASSIFIER_MODEL" ]; then
    python scripts/judge/stylometric_ensemble.py train \
        --inputs $TARGET_TRAIN_RESPONSES \
                $SOURCE_TRAIN_RESPONSES \
        --output-dir $CLASSIFIER_DIR
    echo "✓ Classifier trained"
else
    echo "✓ Classifier already trained"
fi

# Step 5 (Optional): Score with stylometric classifier
echo ""
echo "Step 5 (Optional): Scoring with stylometric classifier..."
STYLO_SCORES="${SCORES_DIR}/stylometric_ensemble_scores.csv"

python scripts/judge/stylometric_ensemble.py score \
    --model-path $CLASSIFIER_MODEL \
    --input $DISGUISE_OUTPUT \
    --output $STYLO_SCORES

echo "✓ Stylometric scoring complete"

# Summary
echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  Disguised responses: $DISGUISE_OUTPUT"
echo "  LLM judge scores: $SCORED_CSV"
echo "  Stylometric scores: $STYLO_SCORES"
echo ""
echo "Next steps:"
echo "  1. View scores: cat ${SCORES_DIR}/scored_metrics.csv"
echo "  2. Review summary: cat ${SCORES_DIR}/summary.json"
echo "  3. Try different methods: behavioral_based, stylistic, random_sampling"
echo ""

