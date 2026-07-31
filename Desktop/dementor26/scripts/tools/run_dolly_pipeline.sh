#!/usr/bin/env bash
# Full Dolly pipeline: base responses → disguise (all methods) → activation steering
# All models are open-weight; no OpenAI API key required.
#
# Required env vars:
#   TINKER_API_KEY        — Tinker inference + fine-tuning
#   OPENROUTER_API_KEY    — fallback for OpenRouter inference (set if not using Tinker for base gen)
#
# Activation steering (local, no API key):
#   HF_TOKEN              — only needed if you haven't accepted the Llama/Qwen license on HF
#   LLAMA_LOCAL_PATH      — optional path to a local merged Llama weights dir; defaults to HF hub

set -euo pipefail

PROMPTS="data/datasets/dolly/dolly_prompts_500_seed42.csv"

# Models — open-weight only
LLAMA="meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN="Qwen/Qwen3.6-27B"

# Slugs used in file names (/ → _)
LLAMA_SLUG="meta-llama_Meta-Llama-3.1-8B-Instruct"
QWEN_SLUG="Qwen_Qwen3.6-27B"

LLAMA_RESPONSES="data/model-responses/dolly/${LLAMA_SLUG}.csv"
QWEN_RESPONSES="data/model-responses/dolly/${QWEN_SLUG}.csv"

# Analysis LLM used by contrastive / behavioral_based (open model via OpenRouter)
ANALYSIS_MODEL="openrouter/meta-llama/llama-3.1-8b-instruct"

METHODS="just_name_it random_sampling stylistic behavioral_based contrastive"

# ── Step 0: prepare dataset ────────────────────────────────────────────────────
if [ ! -f "$PROMPTS" ]; then
  echo "==> Preparing Dolly dataset..."
  python scripts/tools/prepare_dolly_dataset.py
fi

# ── Step 1: generate base responses via Tinker ────────────────────────────────
echo "==> Generating Llama-3.1-8B base responses (Tinker)..."
python scripts/generate_responses.py \
  --prompts-file "$PROMPTS" \
  --output-csv "$LLAMA_RESPONSES" \
  --limit 500 \
  --max-tokens 512 \
  tinker \
  --model-path "tinker://base/${LLAMA}" \
  --base-model "$LLAMA" \
  --renderer-name llama3

echo "==> Generating Qwen3.6-27B base responses (Tinker)..."
python scripts/generate_responses.py \
  --prompts-file "$PROMPTS" \
  --output-csv "$QWEN_RESPONSES" \
  --limit 500 \
  --max-tokens 512 \
  tinker \
  --model-path "tinker://base/${QWEN}" \
  --base-model "$QWEN"

# ── Step 2: disguise (both directions, all methods) ───────────────────────────
# Analysis calls (contrastive, behavioral_based) are routed to OpenRouter Llama
# via ANALYSIS_API_BASE / ANALYSIS_API_KEY / ANALYSIS_PROVIDER.

for METHOD in $METHODS; do
  echo "==> [${METHOD}]  Llama → Qwen..."
  python scripts/disguise.py \
    --model "$LLAMA" \
    --disguise-as "$QWEN" \
    --method "$METHOD" \
    --prompts-file "$PROMPTS" \
    --source-responses "$LLAMA_RESPONSES" \
    --target-responses "$QWEN_RESPONSES" \
    --num-samples 500 \
    --analysis-api-base "https://openrouter.ai/api/v1" \
    --analysis-api-key "$OPENROUTER_API_KEY" \
    --analysis-provider openrouter \
    --skip-evaluation \
    --no-wandb

  echo "==> [${METHOD}]  Qwen → Llama..."
  python scripts/disguise.py \
    --model "$QWEN" \
    --disguise-as "$LLAMA" \
    --method "$METHOD" \
    --prompts-file "$PROMPTS" \
    --source-responses "$QWEN_RESPONSES" \
    --target-responses "$LLAMA_RESPONSES" \
    --num-samples 500 \
    --analysis-api-base "https://openrouter.ai/api/v1" \
    --analysis-api-key "$OPENROUTER_API_KEY" \
    --analysis-provider openrouter \
    --skip-evaluation \
    --no-wandb
done

# ── Step 3: activation steering (Llama → Qwen, using contrastive output) ──────
# Requires local Llama weights. Either set LLAMA_LOCAL_PATH to a merged weights
# dir, or leave unset to load from HuggingFace (needs HF_TOKEN for gated model).
STEER_INPUT="data/results/dolly/full/contrastive/${LLAMA_SLUG}_disguised_${QWEN_SLUG}.csv"
STEER_OUTPUT="data/results/dolly/analysis/activation_steering/llama_disguised_qwen"
LOCAL_MODEL="${LLAMA_LOCAL_PATH:-$LLAMA}"

echo "==> Activation steering: Llama → Qwen"
echo "    comparison-csv : $STEER_INPUT"
echo "    model          : $LOCAL_MODEL"

python -m scripts.analysis.activation_steering \
  --comparison-csv "$STEER_INPUT" \
  --output-dir "$STEER_OUTPUT" \
  --model-name "$LOCAL_MODEL" \
  --layer -8 \
  --strengths "0,0.5,1.0,2.0"

echo ""
echo "All done. Key outputs:"
echo "  Base responses  : data/model-responses/dolly/"
echo "  Disguise runs   : data/results/dolly/full/<method>/"
echo "  Steering        : $STEER_OUTPUT/"
