# MMLU-Pro Setup Complete - Summary

## What Was Created

Successfully prepared MMLU-Pro dataset for the Dementor disguise pipeline!

### Files Created

```
/mnt/localssd/llm_expt_2/dementor/
├── scripts/
│   ├── prepare_mmlu_pro.py          # Dataset preparation script
│   └── run_mmlu_pro_pipeline.sh     # Complete pipeline automation
├── docs/
│   └── mmlu_pro_setup.md           # Detailed setup guide
└── data/
    └── datasets/
        └── mmlu_pro/
            ├── mmlu_pro_prompts_train_300_seed42.csv    # 300 training prompts (Dementor format)
            ├── mmlu_pro_prompts_eval_200_seed42.csv     # 200 evaluation prompts (Dementor format)
            ├── mmlu_pro_full_train_300_seed42.csv       # Full train data with answers
            └── mmlu_pro_full_eval_200_seed42.csv        # Full eval data with answers
```

## What `make_prompts.py` Does

The standard `scripts/make_prompts.py` is a utility that:

1. **Reads** various dataset formats (CSV, JSONL, TXT)
2. **Extracts** a specific column or key containing prompts
3. **Cleans** the data (strips whitespace, removes empties)
4. **Outputs** a standardized CSV with single 'prompt' column
5. **Optionally** deduplicates and limits number of prompts

This standardized format is what `generate_responses.py` expects as input.

### Example Usage of `make_prompts.py`

```bash
# From CSV with 'question' column
python scripts/make_prompts.py \
  --input data/gsm8k/gsm8k_test.csv \
  --column question \
  --output data/datasets/gsm8k/gsm8k_prompts.csv

# From JSONL with 'prompt' key
python scripts/make_prompts.py \
  --input data/dataset.jsonl \
  --key prompt \
  --output data/prompts.csv

# With deduplication and limit
python scripts/make_prompts.py \
  --input raw.csv \
  --column text \
  --dedupe \
  --max 500 \
  --output data/prompts.csv
```

## What We Did for MMLU-Pro

Since MMLU-Pro isn't built into Dementor, we created `prepare_mmlu_pro.py` which:

1. **Downloads** MMLU-Pro from HuggingFace (`TIGER-Lab/MMLU-Pro`)
2. **Formats** questions as multiple-choice with options
3. **Samples** 500 questions with seed 42 (reproducible)
4. **Splits** into 300 train + 200 eval
5. **Creates** prompt CSVs in the same format as `make_prompts.py` output

### Dataset Statistics

- **Total MMLU-Pro questions**: 12,032
- **Our sample**: 500 (300 train + 200 eval)
- **Random seed**: 42 (reproducible)
- **Categories**: 14 subjects (math, physics, chemistry, law, etc.)

**Train split** (300 questions):
- chemistry: 37, math: 37, physics: 32, law: 27, other: 23, psychology: 22, health: 20, business: 20, engineering: 19, biology: 17, economics: 14, computer science: 12, history: 12, philosophy: 8

**Eval split** (200 questions):
- math: 25, chemistry: 23, economics: 21, other: 21, law: 18, engineering: 18, business: 14, health: 14, physics: 12, psychology: 8, history: 7, computer science: 7, biology: 7, philosophy: 5

## Quick Start Commands

### 1. Dataset Already Prepared ✓

The dataset is ready to use at:
- Train: `data/datasets/mmlu_pro/mmlu_pro_prompts_train_300_seed42.csv`
- Eval: `data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv`

### 2. Generate Model Responses

```bash
# Target model (GPT-4.1-mini)
python scripts/generate_responses.py \
  --prompts-file data/datasets/mmlu_pro/mmlu_pro_prompts_train_300_seed42.csv \
  --output-csv data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses_train.csv \
  basic --model openai/gpt-4.1-mini

# Source model (Llama-3.1-8B)
python scripts/generate_responses.py \
  --prompts-file data/datasets/mmlu_pro/mmlu_pro_prompts_train_300_seed42.csv \
  --output-csv data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses_train.csv \
  basic --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

### 3. Apply Disguise

```bash
python scripts/disguise.py \
  --prompts_file data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --disguise_as openai/gpt-4.1-mini \
  --method contrastive \
  --num_samples 200
```

### 4. Score Results

```bash
python -m scripts.scorer \
  --input data/results/mmlu_pro/eval200/contrastive/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --output data/results/mmlu_pro/eval200/contrastive/scores/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini/scored.csv
```

### 5. Or Run Complete Pipeline

```bash
# Automated end-to-end pipeline
./scripts/run_mmlu_pro_pipeline.sh
```

## Column Naming Flow

Understanding how columns are renamed between steps:

### Step 1: `generate_responses.py` Output
```csv
prompt,model_response,model
"What is 2+2?","The answer is 4","openai/gpt-4.1-mini"
```

### Step 2: `disguise.py` Input/Output

**Inputs:**
- `--source_responses`: CSV with `model_response` column (stays as-is)
- `--target_responses`: CSV with `model_response` → renamed to `target_response`

**Output:**
```csv
prompt,model_response,target_response,method,source_model,target_model
"What is 2+2?","[DISGUISED]","[ORIGINAL TARGET]","contrastive","llama","gpt-4"
```

Where:
- `model_response` = NEW disguised output from source model
- `target_response` = ORIGINAL target model output (what we're mimicking)

### Step 3: `scorer.py` Input

Expects these exact columns:
- `prompt`: The question
- `model_response`: Disguised response to evaluate
- `target_response`: Ground truth to compare against

## Prompt Format Example

MMLU-Pro questions are formatted as:

```
Body temperature on the average is 98.6°F. What is this on (a) the Celsius scale and (b) the Kelvin scale?

Options:
A. 37.00°C, 310.15K
B. 37.50°C, 311.65K
C. 38.50°C, 311.65K
D. 36.50°C, 309.65K
E. 35.00°C, 308.15K
```

This clean format works well with all LLMs and provides clear multiple-choice context.

## Key Differences: MMLU-Pro vs GSM8K

| Aspect | GSM8K | MMLU-Pro |
|--------|-------|----------|
| **Focus** | Grade school math problems | Multi-subject knowledge |
| **Format** | Word problems (open-ended) | Multiple choice (A-J) |
| **Difficulty** | Easy-Medium | Medium-Hard |
| **Categories** | Math only | 14 subjects |
| **Size** | ~8,000 questions | ~12,000 questions |
| **Our Sample** | 300 train + 200 eval | 300 train + 200 eval |

## Next Steps

1. **Generate responses** for both models (source & target)
2. **Apply disguise** using contrastive method (recommended)
3. **Score** with LLM judge to measure disguise quality
4. **Train stylometric classifier** to test if disguise is detectable
5. **Try other methods**: behavioral_based, stylistic, random_sampling
6. **Compare** MMLU-Pro results with GSM8K results

## Documentation

- **Setup guide**: `docs/mmlu_pro_setup.md`
- **Pipeline script**: `scripts/run_mmlu_pro_pipeline.sh`
- **Preparation script**: `scripts/prepare_mmlu_pro.py`

## Summary

✅ Dataset downloaded and prepared (12,032 → 500 questions)
✅ Reproducible split with seed 42 (300 train + 200 eval)
✅ Dementor-compatible CSV format (single 'prompt' column)
✅ Full metadata saved (questions + answers + categories)
✅ Ready for response generation and disguise experiments!

The MMLU-Pro dataset is now fully integrated into the Dementor pipeline and ready to use! 🎉

