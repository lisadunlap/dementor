#!/usr/bin/env bash
set -euo pipefail

# Optional: load API keys from .env if present
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# Ensure repo root is on PYTHONPATH so "python -m scripts.*" works everywhere
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

DATASET=gsm8k

# Base model responses (500 subset)
SRC_CSV="data/model-responses/${DATASET}/500/gpt-4.1_responses.csv"
TGT_CSV="data/model-responses/${DATASET}/500/meta-llama_Meta-Llama-3-8B-Instruct_responses.csv"

# Disguised vs target (pairwise CSV already produced by disguise.py)
DISG_DIR="data/results/${DATASET}/comparisons/disguised_vs_target/random_sampling"
DISG_PAIR_CSV="${DISG_DIR}/gpt-4.1_as_meta-llama_Meta-Llama-3-8B-Instruct.csv"

# Output locations
BASELINE_DIR="data/results/${DATASET}/comparisons/source_vs_target"
mkdir -p "${BASELINE_DIR}" "data/results/${DATASET}/scores"

echo "==> Baseline pairwise scoring: source vs target (LLM judge cached)"
python -m scripts.scorer compare \
  --a "${SRC_CSV}" \
  --b "${TGT_CSV}" \
  --output "${BASELINE_DIR}/openai_gpt-4.1_vs_meta-llama_Meta-Llama-3-8B-Instruct.csv" \
  --judge-model openai/gpt-4.1-mini

echo "==> Disguised pairwise scoring: disguised vs target (LLM judge cached)"
python -m scripts.scorer pairwise \
  --input "${DISG_PAIR_CSV}" \
  --output "${DISG_DIR}/scores/gpt-4.1_as_meta-llama_Meta-Llama-3-8B-Instruct/scored.csv" \
  --judge-model openai/gpt-4.1-mini

echo "==> Target self-comparison (ceiling): target vs target (LLM judge cached)"
python -m scripts.scorer compare \
  --a "${TGT_CSV}" \
  --b "${TGT_CSV}" \
  --output "${BASELINE_DIR}/meta-llama_Meta-Llama-3-8B-Instruct_vs_meta-llama_Meta-Llama-3-8B-Instruct.csv" \
  --judge-model openai/gpt-4.1-mini

echo "==> Source self-comparison: source vs source (LLM judge cached)"
python -m scripts.scorer compare \
  --a "${SRC_CSV}" \
  --b "${SRC_CSV}" \
  --output "${BASELINE_DIR}/openai_gpt-4.1_vs_openai_gpt-4.1.csv" \
  --judge-model openai/gpt-4.1-mini

echo "==> Aggregating metrics into summary CSV/Markdown"
python scripts/aggregate_metrics.py \
  --root "data/results/${DATASET}" \
  --output "data/results/${DATASET}/summary.csv" \
  --markdown "data/results/${DATASET}/summary.md"

echo "All LLM-judged scoring steps completed. Cached LLM calls reuse LMDB entries."
