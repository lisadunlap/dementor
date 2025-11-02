# Methods Overview

This document summarizes the core methods in this repository and how to use them without confusion.

Important context:
- The focus of this repo is the Disguise Methods in `scripts/methods/` — practical strategies for making one model behave like another via prompting.
- Earlier utility-estimation codepaths (e.g., “superstimuli”, “utility-alignment”) were inspiration only and have been removed; their ideas primarily informed past example-selection experiments.

We list Disguise Methods first. A short reference to the (historical) utility-estimation concepts is included at the end for completeness.

## Quick Map
- Disguise Methods (prompting strategies to act like a target model)
  - Contrastive System Prompting
  - Vibe-Based Disguise
  - Random Sample k Examples
  - Stylistic Disguise
- Clustering Variants (stylistic/behavioral/embedding)

- Utility Estimation (temporary inspiration, will be removed later)
  - Bradley–Terry (Static)
  - Thurstonian (Static)
  - Thurstonian + Active Learning (Iterative selection)

---

## Disguise Methods (Primary)

Location: `scripts/methods/`

Method registry: `scripts/methods/get_method.py` (clean names in quotes below).

### Core Methods
- "contrastive"
  - Class: `core_methods.ContrastiveSystemPrompting`
  - What it does: Compares source vs target responses to extract explicit distinguishing features; turns them into actionable system instructions.
  - When to use: You have both datasets and want targeted guidance on “what to change”.

- "behavioral_based"
  - Class: `core_methods.BehavioralBasedSystemPrompting`
  - What it does: Builds a deep behavioral profile (cognitive style, tone, interaction patterns) from target data; uses it in the system prompt.
  - When to use: Personality/feel matters beyond formatting.

- "random_sampling"
  - Class: `core_methods.RandomSamplingSystemPrompting`
  - What it does: Randomly picks k target examples and instructs to mimic the style; strong, simple baseline.
  - When to use: Quick trials and baselines; good if examples are representative.

- "stylistic"
  - Class: `core_methods.StylisticSystemPrompting`
  - What it does: Analyzes measurable surface patterns (length, lists, markdown, headers) and produces precise style rules.
  - When to use: Structure and presentation patterns are the main gap.

### Clustering Variants (Representative Sampling)
- `stylistic_clustering`, `stylistic_clustering_resample`, `behavioral_clustering`, `embedding_clustering`
  - Cluster responses on different feature spaces (stylistic/behavioral/embeddings) and select representatives.
  - When to use: You want example diversity/coverage without manual curation.

### How to Get a Method
```python
from scripts.methods.get_method import get_method
method = get_method(
    method_name="behavioral_based",    # or: contrastive | random_sampling | stylistic
    model="source-model",              # model to run
    disguise_as="target-model",        # model to mimic
    disguise_df=target_df,              # DataFrame of target examples (required for most methods)
    source_df=source_df                 # Required for contrastive
)
messages = method.forward(prompt="…")   # returns chat-format messages to send to the model
```

Notes:
- Some models (e.g., Gemma) need specific chat formatting; implementations handle this internally.
- Several analysis steps use LiteLLM; ensure API keys if you enable them.

### Choosing Methods (Disguise)
- Fast baseline: "random_sampling" (k=5) or "stylistic" if structure is key.
- Distinctive personality: "behavioral_based" (optionally with a few examples).
- Explicit change-list from source→target: "contrastive".
- Need diverse exemplars: use clustering variants to select representative target responses.

---

## Utility Estimation (Historical Inspiration)

These were inspiration paths demonstrating pairwise preference learning and selection ideas that influenced our example-selection work. The code has been removed to streamline the repo.

Key entrypoints and classes:
- `compute_utilities.compute_utilities(...)` (async orchestration)
- Base class: `models.py::UtilityModel`
- Pair graph: `compute_utilities.py::PreferenceGraph`
- Bradley–Terry: `utility_models/bradley_terry/bradley_terry.py::BradleyTerryUtilityModel`
- Thurstonian: `utility_models/thurstonian/thurstonian.py::ThurstonianUtilityModel`
- Thurstonian Active Learning: `utility_models/thurstonian/thurstonian_active_learning.py::ThurstonianActiveLearningUtilityModel`

If you experiment here, recommended defaults are Thurstonian + Active Learning (pseudolabels off, strict A/B parsing). Again, this is temporary and not part of the stable interface.

## Method Selection Guide

### Quick Decision Tree:
- **Fast baseline**: `random_sampling` for quick trials with good examples
- **Distinctive personality**: `behavioral_based` when personality/communication style matters
- **Structural patterns**: `stylistic` when formatting and structure are key
- **Explicit guidance**: `contrastive` when you want clear rules about what to change

### Available Methods
Get method names from `scripts/methods/get_method.py`:
- `random_sample_{1,3,5}_examples`: Sample-based approaches
- `just_name_it`: Simple name-based instruction
- `behavioral_based_disguise`: GPT-4o identified behavioral differences
- `stylistic_clustering`: Cluster by formatting/length features
- `ensemble_math_disguise`: Multiple math disguise strategies

## Usage Examples

### Streamlined Pipeline (Recommended)
```bash
python scripts/run_pipeline.py \
  --prompts_file data/datasets/gsm8k/gsm8k_prompts.txt \
  --source-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --target-model gpt-4o \
  --method contrastive
```

### Individual Components
```bash
# Generate responses
python scripts/generate_responses.py --model meta-llama/Meta-Llama-3.1-8B-Instruct

# Apply disguise
python scripts/disguise.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --disguise_as gpt-4o --method contrastive

# Score results
python -m scripts.scorer pairwise --input input.csv --output scores/output/scored.csv
```
