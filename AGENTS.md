# AGENTS.md

This file provides guidance to AI agents when working in this repository.

## Project Overview

Dementor is a research framework for LLM disguise and behavioral inertia:
techniques that make one language model imitate another, plus analyses that
measure which behavioral axes move under prompting, fine-tuning, and
activation-level interventions.

## Core Architecture

### Disguise Methods (`scripts/methods/`)

Stable method names are registered in `scripts/methods/get_method.py`:

- `just_name_it`
- `random_sampling`
- `behavioral_based`
- `stylistic`
- `contrastive`
- `stylistic_clustering`
- `stylistic_clustering_resample`
- `embedding_clustering`
- `behavioral_clustering`

All methods inherit from `MethodBase` and expose `forward(prompt: str)`.

### Main Scripts

- `scripts/generate_responses.py`: generate base responses.
  - Use `basic` for provider/HF/vLLM models.
  - Use `tinker` for Tinker sampler checkpoints or adapter aliases.
  - Use `openai` for OpenAI or OpenAI-compatible endpoints.
- `scripts/disguise.py`: apply disguise methods.
- `scripts/scorer.py`: score single files or pairwise disguised-vs-target outputs.
- `scripts/analysis/run_behavioral_inertia.py`: latent behavioral-axis analysis.
- `scripts/analysis/activation_bridge.py`: representation probes without intervention.
- `scripts/analysis/activation_steering.py`: local Transformers activation steering.
- `workflows/run_gsm8k_workflow.py`: dry-run or launch GSM8K SFT/DPO workflows.

### Data Flow

1. Generate or import base responses into `data/model-responses/`.
2. Apply disguise methods into `data/results/<dataset>/comparisons/...`.
3. Score comparisons with `scripts.scorer`.
4. Run behavioral-inertia analysis on each comparison CSV.
5. Aggregate summaries with `scripts.analysis.run_intervention_ladder`.

Use local CSVs as the reproducible source of truth. Old BAIR/cthulu-only
artifacts should not be assumed available; regenerate missing results locally or
through Tinker/OpenAI workflows.

## Common Commands

### Setup

```bash
pip install -r requirements.txt
```

Behavioral-inertia analysis requires `sentence-transformers`; it is used for
Naz-style Big-Five/model-style adjective scoring before fitting a source/target
fixed SVD basis. Disguised outputs are projected into that basis and must not
participate in fitting it.

Tinker workflows (SFT/DPO adapters via Thinking Machines, and the
`tinker` backend in `scripts/generate_responses.py`) require two
extra packages that are not on PyPI:

- `tinker` — Thinking Machines SDK; install from their docs.
- `tinker_cookbook` — preference-dataset helpers used by
  `workflows/dpo.py`; install from the public cookbook repo.

A `TINKER_API_KEY` environment variable is also required at runtime.
If you are only running the OpenAI/HF/vLLM paths you can skip both
packages — every Tinker import is guarded and only fires on those
code paths.

### Generate Responses

```bash
python scripts/generate_responses.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/model-responses/gsm8k/full/openai_gpt-4.1-mini.csv \
  basic --model openai/gpt-4.1-mini
```

### Apply Disguise

```bash
python scripts/disguise.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --disguise-as openai/gpt-4.1-mini \
  --method contrastive \
  --num-samples 50
```

### Score

```bash
python -m scripts.scorer pairwise \
  --input data/results/gsm8k/comparisons/disguised_vs_target/contrastive/llama_as_gpt-4.1-mini.csv \
  --output data/results/gsm8k/comparisons/disguised_vs_target/contrastive/llama_as_gpt-4.1-mini_scored.csv
```

### Tinker SFT/DPO

Dry-run first:

```bash
python -m workflows.run_gsm8k_workflow --stage sft --provider tinker --dry-run
python -m workflows.run_gsm8k_workflow --stage dpo --provider tinker --dry-run
```

Launch after confirming inputs and environment:

```bash
python -m workflows.run_gsm8k_workflow \
  --stage sft \
  --provider tinker \
  --epochs 6 \
  --batch-size 16 \
  --weights-name gsm8k_llama-3.1-8b-instruct
```

Generate from a saved Tinker sampler:

```bash
python scripts/generate_responses.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/results/gsm8k/500/sft_tinker/eval.csv \
  tinker \
  --model-path 'tinker://<run-id>/sampler_weights/<name>' \
  --renderer-name llama3
```

### Activation Steering

Tinker remote sampling does not expose generation hooks. For steering, export
the Tinker LoRA to a local PEFT adapter or merged HF model, then use
Transformers hooks.

```bash
python scripts/tools/export_tinker_adapter.py \
  --tinker-path 'tinker://<run-id>/sampler_weights/<name>' \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir data/model-responses/adapters/gsm8k_llama_sft_peft \
  --format peft
```

```bash
python -m scripts.analysis.activation_steering \
  --comparison-csv data/results/gsm8k/comparisons/disguised_vs_target/contrastive/llama_as_gpt-4.1-mini.csv \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --peft-adapter-path data/model-responses/adapters/gsm8k_llama_sft_peft \
  --output-dir data/results/gsm8k/analysis/activation_steering/llama_as_gpt-4.1-mini \
  --layer -8 \
  --strengths 0,0.5,1,2
```

## Result Organization

- `data/model-responses/<dataset>/`: base model outputs.
- `data/results/<dataset>/comparisons/`: disguise comparisons.
- `data/results/<dataset>/analysis/`: behavioral-inertia, bridge, steering, and ladder outputs.
- `data/recovered/`: imported legacy/Naz recovered comparison CSVs.
- `data/tinker_adapters.json`: local alias registry for Tinker sampler paths.

## Agent Guidelines

- Run commands directly; do not ask the user to execute commands.
- Prefer existing scripts and workflow entry points over adding parallel scripts.
- Keep generated results under `data/results/` and base responses under `data/model-responses/`.
- Do not depend on unavailable cthulu/BAIR paths in code or docs.
- Do not merge noisy result branches wholesale. Selectively import useful CSVs and docs, leaving out caches, W&B logs, `.DS_Store`, and transient artifacts.
- Optional provider packages such as `tinker`, `tinker_cookbook`, and `openai` should be imported lazily inside the paths that need them, so local analysis remains usable without every remote backend installed.
- Update README or relevant docs when command arguments, data layout, or workflow structure changes.
- Use focused tests for analysis/math behavior and CLI parsing. Avoid broad generated-output tests that require remote APIs or large model downloads.
