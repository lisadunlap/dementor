# Methods Overview

This document records the stable disguise methods in the current `scripts/`
layout. Older notes may refer to a `disguising/` package; that is not the
current entry point.

## Stable Method Names

Use these with `scripts/disguise.py --method <name>`:

| Method | Purpose | Required data |
| --- | --- | --- |
| `random_sampling` | Few-shot target examples as a fast baseline. | target responses |
| `just_name_it` | Directly instruct the source model to act as the target. | none |
| `behavioral_based` | Summarize target communication behavior into a system prompt. | target responses |
| `stylistic` | Extract surface formatting and style rules. | target responses |
| `contrastive` | Compare source and target examples to produce change rules. | source + target responses |
| `stylistic_clustering` | Pick target exemplars from surface-style clusters. | source + target responses |
| `stylistic_clustering_resample` | Same as above, but resamples examples per prompt. | source + target responses |
| `embedding_clustering` | Pick examples from semantic/embedding clusters. | source + target responses |
| `behavioral_clustering` | Pick examples from behavioral feature clusters. | source + target responses |

The method registry is [scripts/methods/get_method.py](/Users/EthanLiu/Documents/Programming/dementor/scripts/methods/get_method.py).

## Recommended Ladder

For the paper, treat the methods as an intervention ladder rather than as
isolated tricks:

1. `just_name_it`: name-only imitation.
2. `random_sampling`: in-context target examples.
3. `stylistic`: explicit surface-style rules.
4. `behavioral_based`: inferred communication/personality profile.
5. `contrastive`: source-to-target behavioral deltas.
6. clustering variants: diversity-controlled exemplar selection.
7. SFT/DPO LoRA: weight-level imitation.
8. activation steering: representation-level intervention inside open models.

The central analysis asks whether source identity persists as interventions
become stronger.

## Common Commands

Generate base responses:

```bash
python scripts/generate_responses.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/model-responses/gsm8k/full/openai_gpt-4.1-mini.csv \
  basic --model openai/gpt-4.1-mini
```

Run a contrastive disguise:

```bash
python scripts/disguise.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --disguise-as openai/gpt-4.1-mini \
  --method contrastive \
  --num-samples 50
```

Score a pairwise comparison:

```bash
python -m scripts.scorer \
  --input data/results/gsm8k/comparisons/disguised_vs_target/contrastive/llama_as_gpt-4.1-mini.csv \
  --output data/results/gsm8k/comparisons/disguised_vs_target/contrastive/llama_as_gpt-4.1-mini_scored.csv
```
