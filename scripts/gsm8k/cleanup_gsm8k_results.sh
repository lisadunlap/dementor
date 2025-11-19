#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ROOT="data/results/gsm8k"

echo "==> Cleaning single-file heuristic outputs (optional, not LLM-judged)"
rm -f "${ROOT}/scores"/*_single_scored.csv || true
rm -f "${ROOT}/scores"/*_single_scored_metrics.csv || true

echo "==> Removing clearly empty/placeholder metrics (header-only)"
find "${ROOT}" -name "*_metrics.csv" -size -40c -print -delete || true

echo "Cleanup complete. Regenerate summaries manually if needed."
