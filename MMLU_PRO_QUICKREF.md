# MMLU-Pro Quick Reference

## What `make_prompts.py` Does
Converts any dataset into standardized format:
- **Input**: CSV/JSONL/TXT with prompts
- **Output**: CSV with single 'prompt' column
- **Purpose**: Standardizes format for `generate_responses.py`

## MMLU-Pro Setup

### Already Done ✓
```bash
python scripts/prepare_mmlu_pro.py --train_size 300 --eval_size 200 --seed 42
```

### Files Created
```
data/datasets/mmlu_pro/
├── mmlu_pro_prompts_train_300_seed42.csv   # 300 prompts for training
├── mmlu_pro_prompts_eval_200_seed42.csv    # 200 prompts for evaluation
├── mmlu_pro_full_train_300_seed42.csv      # With answers (reference)
└── mmlu_pro_full_eval_200_seed42.csv       # With answers (reference)
```

## Complete Workflow

### 1. Generate Responses
```bash
# Target model
python scripts/generate_responses.py \
  --prompts-file data/datasets/mmlu_pro/mmlu_pro_prompts_train_300_seed42.csv \
  --output-csv data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses_train.csv \
  basic --model openai/gpt-4.1-mini

# Source model  
python scripts/generate_responses.py \
  --prompts-file data/datasets/mmlu_pro/mmlu_pro_prompts_train_300_seed42.csv \
  --output-csv data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses_train.csv \
  basic --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

### 2. Apply Disguise
```bash
python scripts/disguise.py \
  --prompts_file data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --disguise_as openai/gpt-4.1-mini \
  --method contrastive \
  --num_samples 200
```

### 3. Score Results
```bash
python -m scripts.scorer \
  --input data/results/mmlu_pro/eval200/contrastive/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --output data/results/mmlu_pro/eval200/contrastive/scores/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini/scored.csv
```

## Column Naming (Important!)

### Between Step 1 and Step 2
**generate_responses.py output:**
```csv
prompt,model_response,model
```

**disguise.py renames:**
- Source file: `model_response` stays `model_response`
- Target file: `model_response` → `target_response`

**disguise.py output:**
```csv
prompt,model_response,target_response,method,source_model,target_model
```
Where:
- `model_response` = DISGUISED response (new)
- `target_response` = TARGET response (original)

## Automated Pipeline
```bash
./scripts/run_mmlu_pro_pipeline.sh
```

## Dataset Stats
- **Total**: 12,032 MMLU-Pro questions
- **Sampled**: 500 (seed 42)
- **Split**: 300 train + 200 eval
- **Categories**: 14 subjects (math, physics, chemistry, etc.)

## Documentation
- Full guide: `docs/mmlu_pro_setup.md`
- Summary: `MMLU_PRO_SETUP_SUMMARY.md`
- Pipeline: `scripts/run_mmlu_pro_pipeline.sh`

