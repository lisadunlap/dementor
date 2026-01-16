# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

Dementor is a research framework for LLM disguise - techniques to make one language model mimic the behavior and response patterns of another model. The codebase implements various disguise methods and evaluation frameworks for comparing model outputs.

## Core Architecture

The project is organized around these key components:

### Disguise Methods (`scripts/methods/`)
- **Base methods**: `RandomSamplingSystemPrompting`, `JustNameIt`, `BehavioralBasedSystemPrompting` 
- **Clustering methods**: `FeatureClustering` with stylistic, behavioral, or embedding-based clustering
- **Math-specific methods**: `EnsembleMathDisguise`
- All methods inherit from `MethodBase` and implement `forward(prompt: str) -> str`

### Main Scripts
- **`scripts/disguise.py`**: Core disguise script that applies methods to transform model responses
- **`scripts/scorer.py`**: Unified scoring with LLM judge + heuristics for pairwise comparisons
- **`scripts/generate_responses.py`**: Generate base responses for evaluation datasets (use the `basic` subcommand for provider/HF/vLLM models)

### Data Flow
1. Generate base responses using `scripts/generate_responses.py basic`
2. Apply disguise methods via `disguise.py` to transform responses
3. Score disguised vs target responses using `scripts/scorer.py` (automatic LLM judge + heuristics)
4. Results stored in `data/results/` and base responses in `data/model-responses/`

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Caching
The framework includes persistent LMDB-based caching for all API calls:
- **Automatic**: All scripts automatically use caching when available
- **Persistent**: Cache survives across sessions, reducing API costs
- **Location**: `cache/llm_cache/` directory
- **Benefits**: Faster responses, reduced costs, consistent results
- **Clear cache**: `python -c "from scripts.cache_llm import clear_cache; clear_cache()"`
- **Stats**: `python -c "from scripts.cache_llm import get_cache_stats; print(get_cache_stats())"`

### Quick Workflow
```bash
# Example workflow (GSM8K eval split)

# Step 1: Generate target model responses
python scripts/generate_responses.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/model-responses/gsm8k/full/openai_gpt-4.1-mini.csv \
  basic --model openai/gpt-4.1-mini

# Step 2: Apply disguise
python scripts/disguise.py \
  --prompts_file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --disguise_as openai/gpt-4.1-mini \
  --method contrastive \
  --num_samples 50

# Step 3: Score (automatic LLM judge + heuristics)
python -m scripts.scorer \
  --input data/results/gsm8k/comparisons/disguised_vs_target/contrastive/llama_as_gpt-4.1-mini.csv \
  --output data/results/gsm8k/comparisons/disguised_vs_target/contrastive/llama_as_gpt-4.1-mini_scored.csv
```

### Available Disguise Methods

Use these methods with `scripts/disguise.py --method <method_name>`:

#### Core Methods
- **`contrastive`**: Learns differences between source and target models (recommended)
- **`behavioral_based`**: Captures personality and communication patterns
- **`random_sampling`**: Clean example-based disguise with few-shot prompts
- **`stylistic`**: Enforces formatting heuristics and surface-level patterns

#### Advanced Methods
- **`stylistic_clustering`**: Cluster by formatting/length features
- **`embedding_clustering`**: Semantic-based clustering for mixed tasks
- **`ensemble_math_disguise`**: Multiple math disguise strategies

### Scoring Details

Pairwise scoring compares disguised vs target responses with an LLM judge (semantic + stylistic, each on a 1-4 scale with decimals) and optional heuristic features.

**Usage:**
```bash
# LLM judge only (default)
python -m scripts.scorer --input comparison.csv --output scored.csv

# Include heuristic feature match scoring
python -m scripts.scorer --input comparison.csv --output scored.csv --heuristics

# Merge two model CSVs and score pairwise
python -m scripts.scorer --compare --a source.csv --b target.csv --output merged.csv
```

**LLM Judge Prompt (Pairwise):**
Evaluates semantic and stylistic similarity between target and disguised responses with detailed explanations.
## File Organization

- **`data/datasets/`**: Input prompts and datasets (GSM8K, etc.)
- **`data/model-responses/`**: Base model outputs (CSV format)
- **`data/results/`**: Disguised outputs, scores, and plots
- **`cache/`**: API response cache for cost/time savings
- **`scripts/methods/`**: Implementation of all disguise techniques
- **`scripts/`**: Core scripts (disguise.py, generate_responses.py, scorer.py)
- **`workflows/`**: SFT/DPO fine-tuning workflows


## Environment Setup

Create `.env` file with required API keys:
```bash
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"  # if using Claude models
```

## Agent Guidelines

### Working with this Codebase
- **User executes commands**: Run commands directly, don't prompt user to execute
- **Modular approach**: Use existing scripts rather than creating new ones
- **Clean workflows**: Follow the established 3-step process (generate → disguise → score)
- **Result organization**: Store outputs in `data/results/` with proper directory structure

### Common Patterns
- **API calls**: Use `scripts/cache_llm` for persistent caching
- **Data paths**: Follow CSV format with `prompt`, `model_response`, `target_response` columns
- **Scoring**: Use `python -m scripts.scorer --input <comparison.csv> --output <scores.csv>` for evaluation
- **Method names**: Use standardized method names from `scripts/disguise.py --help`
