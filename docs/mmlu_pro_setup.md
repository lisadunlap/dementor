# MMLU-Pro Setup Guide for Dementor

This guide explains how to use MMLU-Pro dataset with the Dementor disguise pipeline.

## Overview

MMLU-Pro is a challenging multiple-choice benchmark with questions across various academic subjects. This setup creates a 300/200 train/eval split for disguise experiments.

## What `make_prompts.py` Does

The standard `scripts/make_prompts.py` script:
1. **Reads** a dataset file (CSV, JSONL, or TXT)
2. **Extracts** a specific column (e.g., 'question' from CSV) or key (e.g., 'prompt' from JSONL)
3. **Cleans** the data (strips whitespace, removes empty entries)
4. **Outputs** a CSV with a single 'prompt' column

This is exactly what `generate_responses.py` expects as input.

## For MMLU-Pro

Since MMLU-Pro is not in the standard Dementor codebase, we created a custom script that:
1. Downloads MMLU-Pro from HuggingFace (`TIGER-Lab/MMLU-Pro`)
2. Formats questions with multiple-choice options
3. Uses seed 42 to randomly sample 300 train + 200 eval questions
4. Creates prompt CSVs in the same format as `make_prompts.py` output

## Quick Start

### Step 1: Prepare Dataset

```bash
cd /mnt/localssd/llm_expt_2/dementor

# Run the preparation script
python scripts/prepare_mmlu_pro.py

# Or with custom parameters:
python scripts/prepare_mmlu_pro.py \
  --train_size 300 \
  --eval_size 200 \
  --seed 42 \
  --output_dir data/datasets/mmlu_pro
```

**Output files:**
- `data/datasets/mmlu_pro/mmlu_pro_prompts_train_300_seed42.csv` - Training prompts (single 'prompt' column)
- `data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv` - Evaluation prompts (single 'prompt' column)
- `data/datasets/mmlu_pro/mmlu_pro_full_train_300_seed42.csv` - Full metadata with answers
- `data/datasets/mmlu_pro/mmlu_pro_full_eval_200_seed42.csv` - Full metadata with answers

### Step 2: Generate Base Responses

Generate responses for both source and target models:

```bash
# Generate GPT-4.1-mini responses (target model)
python scripts/generate_responses.py \
  --prompts-file data/datasets/mmlu_pro/mmlu_pro_prompts_train_300_seed42.csv \
  --output-csv data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv \
  basic --model openai/gpt-4.1-mini

# Generate Llama responses (source model)
python scripts/generate_responses.py \
  --prompts-file data/datasets/mmlu_pro/mmlu_pro_prompts_train_300_seed42.csv \
  --output-csv data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv \
  basic --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Step 3: Apply Disguise

Make Llama sound like GPT-4:

```bash
python scripts/disguise.py \
  --prompts_file data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --disguise_as openai/gpt-4.1-mini \
  --source_responses data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv \
  --target_responses data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv \
  --method contrastive \
  --num_samples 200
```

```bash
python scripts/disguise.py \
  --prompts_file data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv \
  --model openai/gpt-4.1-mini \
  --disguise_as meta-llama/Meta-Llama-3.1-8B-Instruct \
  --source_responses data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv \
  --target_responses data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv \
  --method contrastive \
  --num_samples 200
  ```

**Output:** `data/results/generic/eval200/contrastive/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv`

### Step 4: Score the Disguise

```bash
# LLM judge scoring
python -m scripts.scorer \
  --input data/results/generic/eval200/contrastive/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --output data/results/generic/eval200/contrastive/scores/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini/scored.csv \
  --judge-model openai/gpt-4.1-mini
```

**Output:**
- `scored.csv` - Full results with semantic_score, stylistic_score
- `scored_metrics.csv` - Summary statistics
- `summary.json` - Run configuration and metrics

### Step 5 (Optional): Stylometric Classification

Train a classifier to detect disguises:

```bash
# Train ensemble classifier
python scripts/judge/stylometric_ensemble.py train \
  --inputs data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv \
          data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv \
  --output-dir data/results/stylometric_classifier/mmlu_pro_ensemble

# Score disguised responses
python scripts/judge/stylometric_ensemble.py score \
  --model-path data/results/stylometric_classifier/mmlu_pro_ensemble/ensemble_stylometric.joblib \
  --input data/results/generic/eval200/contrastive/vllm_meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --output data/results/generic/eval200/contrastive/scores/stylometric_ensemble_scores.csv
```




## Prompt Format

MMLU-Pro prompts are formatted as:

```
What is the capital of France?

Options:
A. London
B. Paris
C. Berlin
D. Madrid
```

This format is clean and works well with most LLMs. The models will respond with their reasoning and a final answer.

## Dataset Statistics

- **Total questions in MMLU-Pro**: ~12,000
- **Our sample**: 500 (300 train + 200 eval)
- **Seed**: 42 (for reproducibility)
- **Categories**: Mix of subjects (math, science, history, etc.)

## Files Created

```
data/
├── datasets/
│   └── mmlu_pro/
│       ├── mmlu_pro_prompts_train_300_seed42.csv      # For training (Dementor format)
│       ├── mmlu_pro_prompts_eval_200_seed42.csv       # For evaluation (Dementor format)
│       ├── mmlu_pro_full_train_300_seed42.csv         # With answers (reference)
│       └── mmlu_pro_full_eval_200_seed42.csv          # With answers (reference)
├── model-responses/
│   └── mmlu_pro/
│       └── full/
│           ├── openai_gpt-4.1-mini_responses.csv
│           └── meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv
└── results/
    └── mmlu_pro/
        └── eval200/
            └── contrastive/
                ├── meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv
                └── scores/
                    └── meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini/
                        ├── scored.csv
                        ├── scored_metrics.csv
                        └── summary.json
```

## Comparison to GSM8K

| Dataset | Focus | Prompts Format | Difficulty |
|---------|-------|----------------|------------|
| GSM8K | Grade school math | Word problems | Easy-Medium |
| MMLU-Pro | Multi-subject knowledge | Multiple choice | Medium-Hard |

MMLU-Pro tests a broader range of capabilities and may reveal different stylistic patterns in model responses.

## Notes

- The script automatically filters out "N/A" options from questions
- Category distribution is preserved across splits (stratified sampling)
- Both prompt-only CSVs and full metadata CSVs are saved for convenience
- Use train split for training disguise methods (e.g., contrastive learning)
- Use eval split for testing disguise effectiveness

