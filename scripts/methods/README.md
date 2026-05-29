# Disguise Methods

Stable method names are registered in `scripts/methods/get_method.py`. Use
these names with `scripts/disguise.py --method <name>`.

## Headline Prompt Methods

### `just_name_it`
Name-only baseline. The system prompt tells the source model to respond like
the target model, without examples or learned behavioral rules.

### `random_sampling`
Few-shot baseline. Randomly samples target-model responses and gives them as
reference examples.

### `stylistic`
Surface-form baseline. Estimates measurable target formatting traits such as
length, markdown use, bullets, headers, questions, and code blocks, then turns
those traits into system-prompt guidance.

### `behavioral`
Behavioral baseline. Summarizes the target model's communication behavior
and personality-like response tendencies into a system prompt.

### `contrastive`
Source-to-target delta prompt. Compares source and target examples, extracts
behavioral differences, and asks the source model to apply those differences.

## Clustering Methods

### `stylistic_clustering`
Clusters target examples by formatting/style heuristics and samples
representative examples from the selected cluster.

### `stylistic_clustering_resample`
Variant of `stylistic_clustering` that resamples examples on each forward pass.
Useful as an ablation, not part of the current headline grid.

### `embedding_clustering`
Clusters examples in embedding space so the prompt receives semantically
relevant target examples.

### `behavioral_clustering`
Clusters target examples using LLM-rated behavioral axes. This is the renamed
successor to the older `vibe_clustering` path and is currently kept as an
ablation candidate rather than a headline method.

## Analysis Boundary

Method code generates disguised outputs. Paper-level behavioral inertia,
adjective matching, SVD axes, probes, activation bridge, and activation
steering live under `scripts/analysis/`.
