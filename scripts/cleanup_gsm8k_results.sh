#!/usr/bin/env bash
set -euo pipefail

ROOT="data/results/gsm8k"

echo "==> Cleaning single-file heuristic outputs (optional, not LLM-judged)"
rm -f "${ROOT}/scores"/*_single_scored.csv || true
rm -f "${ROOT}/scores"/*_single_scored_metrics.csv || true

echo "==> Removing clearly empty/placeholder metrics (header-only)"
find "${ROOT}" -name "*_metrics.csv" -size -40c -print -delete || true

echo "==> Rebuilding aggregate summaries for GSM8K"
python scripts/aggregate_metrics.py \
  --root "${ROOT}" \
  --output "${ROOT}/summary.csv" \
  --markdown "${ROOT}/summary.md"

echo "Cleanup complete. Aggregates written under ${ROOT}/summary.{csv,md}."

