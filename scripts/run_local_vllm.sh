#!/usr/bin/env bash
set -euo pipefail

# Helper to run the full pipeline against a local vLLM server using LiteLLM routing.
# Starts vLLM with common defaults (FP16 + tensor parallel), runs generate → disguise → score,
# and optionally tears down the server.
#
# Requirements:
# - vLLM installed and available in PATH (vllm serve ...)
# - Python deps installed for this repo

HF_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
PORT=8000
DTYPE="float16"
TP=4
PROMPTS_FILE="data/chabot_arena_500_propmts.txt"
NUM_SAMPLES=200
OUTPUT_DIR="results/streamlined"
SOURCE_MODEL_OPENAI="openai/meta-llama/Meta-Llama-3-8B-Instruct"
TARGET_MODEL="gpt-4o"
KEEP_SERVER=0

usage() {
  cat << USAGE
Usage: $0 [options]

Options:
  --hf-model ID                HF model id (default: ${HF_MODEL})
  --port N                     vLLM port (default: ${PORT})
  --dtype DTYPE                vLLM dtype (default: ${DTYPE})
  --tp N                       tensor parallel size (default: ${TP})
  --prompts_file PATH          prompts file (default: ${PROMPTS_FILE})
  --num_samples N              samples for disguise (default: ${NUM_SAMPLES})
  --output_dir DIR             output dir (default: ${OUTPUT_DIR})
  --source-model-openai NAME   LiteLLM model name to route (default: ${SOURCE_MODEL_OPENAI})
  --target-model NAME          target model label (default: ${TARGET_MODEL})
  --keep-server                do not kill vLLM server after run
  -h, --help                   show this help

Notes:
- This script starts vLLM:
    vllm serve <HF_MODEL> --port <PORT> --dtype <DTYPE> --tensor-parallel-size <TP> --max-model-len 4096
- Then runs the pipeline routing LiteLLM to http://localhost:<PORT>/v1
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-model) HF_MODEL="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --tp) TP="$2"; shift 2;;
    --prompts_file) PROMPTS_FILE="$2"; shift 2;;
    --num_samples) NUM_SAMPLES="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --source-model-openai) SOURCE_MODEL_OPENAI="$2"; shift 2;;
    --target-model) TARGET_MODEL="$2"; shift 2;;
    --keep-server) KEEP_SERVER=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm command not found. pip install vllm" >&2
  exit 1
fi

API_BASE="http://localhost:${PORT}/v1"

echo "[1/4] Starting vLLM server on port ${PORT} with ${HF_MODEL} (dtype=${DTYPE}, tp=${TP})"
set +e
vllm serve "${HF_MODEL}" --port "${PORT}" --dtype "${DTYPE}" --tensor-parallel-size "${TP}" --max-model-len 4096 > vllm_server.log 2>&1 &
VLLM_PID=$!
set -e
echo "vLLM PID: ${VLLM_PID} (logs: vllm_server.log)"

echo "Waiting for vLLM server to be ready..."
RETRIES=30
until curl -sSf "${API_BASE}/models" >/dev/null 2>&1 || [[ $RETRIES -eq 0 ]]; do
  sleep 2
  RETRIES=$((RETRIES-1))
done
if [[ $RETRIES -eq 0 ]]; then
  echo "ERROR: vLLM server did not become ready at ${API_BASE}" >&2
  kill ${VLLM_PID} || true
  exit 1
fi
echo "vLLM is up at ${API_BASE}"

echo "[2/4] Generating base outputs via LiteLLM routed to vLLM"
python scripts/generate_responses.py \
  --model "${SOURCE_MODEL_OPENAI}" \
  --prompts_file "${PROMPTS_FILE}" \
  --output "disguising/model-responses/base/${SOURCE_MODEL_OPENAI//\//_}.csv" \
  --openai-api-base "${API_BASE}" \
  --openai-api-key "EMPTY"

echo "[3/4] Running disguise via LiteLLM routed to vLLM"
python disguise.py \
  --model "${SOURCE_MODEL_OPENAI}" \
  --disguise_as "${TARGET_MODEL}" \
  --method contrastive_with_al_examples \
  --num_samples "${NUM_SAMPLES}" \
  --al-num-examples 5 --al-max-iterations 5 --al-batch-size 10 \
  --openai-api-base "${API_BASE}" \
  --openai-api-key "EMPTY" \
  --output_dir "${OUTPUT_DIR}"

LATEST_CSV=$(ls -t ${OUTPUT_DIR}/*.csv | head -n 1)
echo "[4/4] Scoring ${LATEST_CSV}"
python -m disguising.scorer "${LATEST_CSV}" --output "${LATEST_CSV%.csv}_scored.csv"

if [[ ${KEEP_SERVER} -eq 0 ]]; then
  echo "Stopping vLLM server (PID ${VLLM_PID})"
  kill ${VLLM_PID} || true
else
  echo "Keeping vLLM server running (PID ${VLLM_PID})"
fi

echo "Done."

