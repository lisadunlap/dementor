# Stylometric Ensemble Scoring Guide

## Overview

This guide explains how to score disguised responses using the stylometric ensemble classifier and generate visualization plots.

## Complete Workflow

### Step 1: Train the Stylometric Ensemble (Once Only!)

You only need to train the ensemble **ONCE** on the base (undisguised) responses:

```bash
cd /mnt/localssd/llm_expt_2/dementor
conda activate litellm

python scripts/judge/stylometric_ensemble.py train \
  --inputs data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv \
          data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv \
  --output-dir data/results/stylometric_classifier/mmlu_pro_ensemble
```

**Output:** `data/results/stylometric_classifier/mmlu_pro_ensemble/ensemble_stylometric.joblib`

✅ **Status:** Already completed!

---

### Step 2: Score All Disguised Responses (8 Commands Total)

You have **4 methods** × **2 directions** = **8 files to score**:

#### Methods:
1. `contrastive`
2. `behavioral_based`
3. `stylistic`
4. `random_sampling`

#### Directions:
1. **GPT → Llama**: GPT disguising as Llama (`openai_gpt-4.1-mini_as_meta-llama_*.csv`)
2. **Llama → GPT**: Llama disguising as GPT (`vllm_meta-llama_*_as_openai_gpt-4.1-mini.csv`)

#### Option A: Run All at Once (Recommended)

```bash
cd /mnt/localssd/llm_expt_2/dementor
conda activate litellm
bash run_all_stylometric_scoring.sh
```

#### Option B: Run Manually (8 Commands)

<details>
<summary>Click to expand individual commands</summary>

```bash
cd /mnt/localssd/llm_expt_2/dementor
conda activate litellm

# Contrastive
python scripts/judge/stylometric_ensemble.py score \
  --model-path data/results/stylometric_classifier/mmlu_pro_ensemble/ensemble_stylometric.joblib \
  --input data/results/generic/eval200/contrastive/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
  --output data/results/generic/eval200/contrastive/scores/stylometric_ensemble_gpt_as_llama.csv

python scripts/judge/stylometric_ensemble.py score \
  --model-path data/results/stylometric_classifier/mmlu_pro_ensemble/ensemble_stylometric.joblib \
  --input data/results/generic/eval200/contrastive/vllm_meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --output data/results/generic/eval200/contrastive/scores/stylometric_ensemble_llama_as_gpt.csv

# Behavioral Based
python scripts/judge/stylometric_ensemble.py score \
  --model-path data/results/stylometric_classifier/mmlu_pro_ensemble/ensemble_stylometric.joblib \
  --input data/results/generic/eval200/behavioral_based/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
  --output data/results/generic/eval200/behavioral_based/scores/stylometric_ensemble_gpt_as_llama.csv

python scripts/judge/stylometric_ensemble.py score \
  --model-path data/results/stylometric_classifier/mmlu_pro_ensemble/ensemble_stylometric.joblib \
  --input data/results/generic/eval200/behavioral_based/vllm_meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --output data/results/generic/eval200/behavioral_based/scores/stylometric_ensemble_llama_as_gpt.csv

# Stylistic
python scripts/judge/stylometric_ensemble.py score \
  --model-path data/results/stylometric_classifier/mmlu_pro_ensemble/ensemble_stylometric.joblib \
  --input data/results/generic/eval200/stylistic/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
  --output data/results/generic/eval200/stylistic/scores/stylometric_ensemble_gpt_as_llama.csv

python scripts/judge/stylometric_ensemble.py score \
  --model-path data/results/stylometric_classifier/mmlu_pro_ensemble/ensemble_stylometric.joblib \
  --input data/results/generic/eval200/stylistic/vllm_meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --output data/results/generic/eval200/stylistic/scores/stylometric_ensemble_llama_as_gpt.csv

# Random Sampling
python scripts/judge/stylometric_ensemble.py score \
  --model-path data/results/stylometric_classifier/mmlu_pro_ensemble/ensemble_stylometric.joblib \
  --input data/results/generic/eval200/random_sampling/openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
  --output data/results/generic/eval200/random_sampling/scores/stylometric_ensemble_gpt_as_llama.csv

python scripts/judge/stylometric_ensemble.py score \
  --model-path data/results/stylometric_classifier/mmlu_pro_ensemble/ensemble_stylometric.joblib \
  --input data/results/generic/eval200/random_sampling/vllm_meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --output data/results/generic/eval200/random_sampling/scores/stylometric_ensemble_llama_as_gpt.csv
```

</details>

---

### Step 3: Generate Visualization Plots

After scoring, generate plots showing all results:

```bash
cd /mnt/localssd/llm_expt_2/dementor
conda activate litellm

python scripts/plot_stylometric_results_bidirectional.py \
  --base-dir data/results/generic/eval200 \
  --methods contrastive behavioral_based stylistic random_sampling \
  --output-dir data/results/plots
```

**Outputs:**
- `data/results/plots/stylometric_ensemble_probabilities_bidirectional.png`
- `data/results/plots/stylometric_ensemble_unanimous_bidirectional.png`

---

## Understanding the Score Columns

Each scored CSV contains these columns:

| Column | Description |
|--------|-------------|
| `stylometric_pred_label` | N-gram classifier prediction |
| `structure_pred_label` | Structure & tone classifier prediction |
| `lsa_pred_label` | LSA classifier prediction |
| `ensemble_prob_meta-llama/Meta-Llama-3.1-8B-Instruct` | Probability it's Llama |
| `ensemble_prob_openai/gpt-4.1-mini` | Probability it's GPT |
| `ensemble_majority_label` | **MAIN METRIC**: Which model the ensemble predicts |
| `ensemble_unanimous_label` | Whether all 3 classifiers agreed |
| `ensemble_agreement_count` | How many classifiers agreed (0-3) |

---

## Interpreting Results

### For "GPT → Llama" disguise:
- **Goal**: Make GPT responses look like Llama
- **Success metric**: High `ensemble_prob_meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Fooling rate**: % of responses where `ensemble_majority_label` = Llama

### For "Llama → GPT" disguise:
- **Goal**: Make Llama responses look like GPT
- **Success metric**: High `ensemble_prob_openai/gpt-4.1-mini`
- **Fooling rate**: % of responses where `ensemble_majority_label` = GPT

---

## Quick Summary

**Total commands needed:**
- ✅ 1 training command (already done)
- ⏳ 8 scoring commands (run with `bash run_all_stylometric_scoring.sh`)
- 📊 1 plotting command (run after scoring)

**Not 12 commands!** You only train once, then score 8 files (4 methods × 2 directions).

