#!/usr/bin/env bash
set -euo pipefail

# GSM8K-500 judgements (single-file only):
# - Base source (openai_gpt-4.1)
# - Base target (meta-llama_Meta-Llama-3-8B-Instruct)
# - Disguised random_sampling (openai_gpt-4.1_as_meta-llama_Meta-Llama-3-8B-Instruct)
# - Base source run2 (generate + score)

PROMPTS="data/datasets/gsm8k/gsm8k_prompts_500.csv"

SRC_MODEL_ID="openai/gpt-4.1"
SRC_NAME="openai_gpt-4.1"
SRC_BASE_CSV="data/model-responses/gsm8k/500/openai_gpt-4.1_responses.csv"

TGT_NAME="meta-llama_Meta-Llama-3-8B-Instruct"
TGT_BASE_CSV="data/model-responses/gsm8k/500/meta-llama_Meta-Llama-3-8B-Instruct_responses.csv"

DISGUISED_RAW_CSV="data/results/gsm8k/500/random_sampling/${SRC_NAME}_as_${TGT_NAME}.csv"

# Output directories per your spec
BASE_ROOT="data/results/gsm8k/500"
mkdir -p "$BASE_ROOT/scores/$SRC_NAME" "$BASE_ROOT/scores/$TGT_NAME"
mkdir -p "$BASE_ROOT/random_sampling/scores/${SRC_NAME}_as_${TGT_NAME}"

echo "Scoring base source ($SRC_NAME) → $BASE_ROOT/scores/$SRC_NAME/scored.csv"
python -m scripts.scorer "$SRC_BASE_CSV" --output "$BASE_ROOT/scores/$SRC_NAME/scored.csv" --single

echo "Scoring base target ($TGT_NAME) → $BASE_ROOT/scores/$TGT_NAME/scored.csv"
python -m scripts.scorer "$TGT_BASE_CSV" --output "$BASE_ROOT/scores/$TGT_NAME/scored.csv" --single

echo "Scoring disguised random_sampling → $BASE_ROOT/random_sampling/scores/${SRC_NAME}_as_${TGT_NAME}/scored.csv"
python -m scripts.scorer "$DISGUISED_RAW_CSV" --output "$BASE_ROOT/random_sampling/scores/${SRC_NAME}_as_${TGT_NAME}/scored.csv" --single

# Run2: generate then score
RUN2_CSV="data/model-responses/gsm8k/500/run2/${SRC_NAME}_responses.csv"
mkdir -p "$(dirname "$RUN2_CSV")"
echo "Generating run2 base responses (overwrite) → $RUN2_CSV"
python scripts/generate_responses.py --model "$SRC_MODEL_ID" --prompts_file "$PROMPTS" --output "$RUN2_CSV" --overwrite

# Safety: ensure exactly 500 unique prompts (drop accidental duplicates)
python - << 'PY'
import pandas as pd, os
path = 'data/model-responses/gsm8k/500/run2/openai_gpt-4.1_responses.csv'
df = pd.read_csv(path)
before = len(df)
df = df.drop_duplicates(subset=['prompt'], keep='last').reset_index(drop=True)
after = len(df)
if before != after:
    print(f"Dedup run2: {before} -> {after}")
df.to_csv(path, index=False)
PY

RUN2_OUT_DIR="$BASE_ROOT/run2_scores/$SRC_NAME"
mkdir -p "$RUN2_OUT_DIR"
echo "Scoring run2 base source → $RUN2_OUT_DIR/scored.csv"
python -m scripts.scorer "$RUN2_CSV" --output "$RUN2_OUT_DIR/scored.csv" --single

echo "All judgements completed."
