# Methods Overview

This document summarizes the core methods in this repository and how to use them without confusion.

Important context:
- The focus of this repo is the Disguise Methods in `disguising/` — practical strategies for making one model behave like another via prompting.
- Earlier utility-estimation codepaths (e.g., “superstimuli”, “utility-alignment”) were inspiration only and have been removed; their ideas are reflected in our composite selection strategies.

We list Disguise Methods first. A short reference to the (historical) utility-estimation concepts is included at the end for completeness.

## Quick Map
- Disguise Methods (prompting strategies to act like a target model)
  - Contrastive System Prompting
  - Vibe-Based Disguise
  - Random Sample k Examples
  - Stylistic Disguise
  - Clustering Variants (stylistic/vibe/embedding)
  - Active Learning Disguise (iterative example selection)

- Utility Estimation (temporary inspiration, will be removed later)
  - Bradley–Terry (Static)
  - Thurstonian (Static)
  - Thurstonian + Active Learning (Iterative selection)

---

## Disguise Methods (Primary)

Location: `disguising/methods/`

Method registry: `disguising/methods/get_method.py` (clean names in quotes below).

### Core Methods
- "contrastive"
  - Class: `core_methods.ContrastiveSystemPrompting`
  - What it does: Compares source vs target responses to extract explicit distinguishing features; turns them into actionable system instructions.
  - When to use: You have both datasets and want targeted guidance on “what to change”.

- "vibe_based"
  - Class: `core_methods.VibeBasedSystemPrompting`
  - What it does: Builds a deep “vibe/essence” profile (cognitive style, tone, interaction patterns) from target data; uses it in the system prompt.
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
- `stylistic_clustering`, `stylistic_clustering_resample`, `vibe_clustering`, `embedding_clustering`
  - Cluster responses on different feature spaces (stylistic/vibe/embeddings) and select representatives.
  - When to use: You want example diversity/coverage without manual curation.

### Active Learning (as a selection backend)
- Used as an example selection strategy within composite methods (not a standalone method).
- Focus: pick informative in-context examples (quality/uncertainty) rather than generate a rulebook.

### Composite Method
- "contrastive_with_al_examples" (alias: "contrastive_al")
  - Class: `core_methods.ContrastiveWithALExamples`
  - What it does: Combines contrastive rules (explicit guidelines) with AL-selected examples for the context window.
  - When to use: You want both a clear rulebook and strong exemplars in one run.

CLI knobs for Active Learning (both standalone and composite):
- `--al-num-examples`: number of examples to include (default: 5)
- `--al-d-regular`: initial degree for regular graph seeding (default: 3)
- `--al-p-threshold`, `--al-q-threshold`: bottom fractions (0–1) for ambiguity and coverage filters (default: 0.1 each)
- `--al-batch-size`: pairs per iteration (default: 10)
- `--al-max-iterations`: iterations (default: 5)
- `--al-relaxation-factor`: relax thresholds when too few candidates (default: 1.2)
- `--al-seed`: random seed

### How to Get a Method
```python
from disguising.methods.get_method import get_method
method = get_method(
    method_name="vibe_based",          # or: contrastive | random_sampling | stylistic | active_learning
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
- Distinctive personality: "vibe_based" (optionally with a few examples).
- Explicit change-list from source→target: "contrastive".
- Automated, stronger example selection: "active_learning" or a clustering variant.

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

---

## Disguise Methods

Location: `disguising/methods/`

Method registry: `disguising/methods/get_method.py` (clean names in quotes below).

### Core Methods
- "contrastive"
  - Class: `core_methods.ContrastiveSystemPrompting`
  - Learns distinguishing features by contrasting target vs source outputs; returns actionable system instructions.
  - Use when you have both target and source data and want explicit gap-bridging guidance.

- "vibe_based"
  - Class: `core_methods.VibeBasedSystemPrompting`
  - Builds a deep “vibe/essence” profile (cognitive style, tone, interaction patterns) from target data; uses it in the system prompt.
  - Use when personality/feel (beyond formatting) is key.

- "random_sampling"
  - Class: `core_methods.RandomSamplingSystemPrompting`
  - Randomly picks k target examples and instructs to mimic; strong, simple baseline.
  - Use for quick trials and sanity checks.

- "stylistic"
  - Class: `core_methods.StylisticSystemPrompting`
  - Analyzes measurable surface patterns (length, lists, markdown, headers) and turns them into concrete style rules.
  - Use when structure and presentation patterns are the main gap.

### Clustering Variants (Representative Sampling)
- `stylistic_clustering`, `stylistic_clustering_resample`, `vibe_clustering`, `embedding_clustering`
  - Cluster responses on different feature spaces (stylistic/vibe/embeddings) and select representatives.
  - Use when you want example diversity/coverage without manual curation.

### Active Learning Disguise
- "active_learning" (if dependencies available)
  - Class: `active_learning_disguise.ActiveLearningDisguise`
  - Iteratively selects the most informative examples to include, updating selection based on quality/uncertainty signals.
  - Use when you can afford iterative selection and want to outperform random example choice.

### How to Get a Method
```python
from disguising.methods.get_method import get_method
method = get_method(
    method_name="vibe_based",          # or: contrastive | random_sampling | stylistic | active_learning
    model="source-model",              # model to run
    disguise_as="target-model",        # model to mimic
    disguise_df=target_df,              # DataFrame of target examples (required for most methods)
    source_df=source_df                 # Required for contrastive
)
messages = method.forward(prompt="…")   # returns chat-format messages to send to the model
```

Notes:
- Some models (e.g., Gemma) need specific chat formatting; the implementations handle this internally.
- Several methods use LiteLLM for analysis steps; ensure API keys are set if using those features.

---

## Choosing Methods

- If you need a utility ranking over many options with limited budget:
  - Start with Thurstonian + Active Learning.
  - Keep pseudolabels off initially; enable later if pools are huge and confidence is high.

- If you need a quick disguise baseline:
  - Use "random_sampling" (k=5) or "stylistic" for structure-driven targets.

- If the target has a distinctive personality:
  - Use "vibe_based" (and optionally add a few examples).

- If you have both source and target data and want explicit guidance:
  - Use "contrastive".

- If you want automated, stronger example selection:
  - Try "active_learning" (disguise), or a clustering variant for representative coverage.

---

## Terminology

- Method vs Submethod (Utility Estimation):
  - Method = data acquisition strategy (Static vs Active Learning) × inference model (Bradley–Terry vs Thurstonian).
  - Submethods = selection policy (e.g., P%|Δμ| ∩ Q% degree), pseudolabeling on/off, parsing mode, backend.

- In Disguise:
  - Method = prompting strategy category (contrastive, vibe-based, random, stylistic, clustering, AL-disguise).
  - Submethods = sampling knobs (k, clustering params), analysis backends, formatting quirks.

---

## Pointers & Caveats

- Utility code is duplicated between `superstimuli` and `utility-alignment/agent_refactored` for project isolation; behavior should be consistent.
- There are experimental files (e.g., `bradley_terry/gavel.py`) not wired into the main `compute_utilities` pipeline; treat as reference.
- If you see parsing noise, prefer strict A/B or structured outputs and enable HF logits-only scoring where possible.
