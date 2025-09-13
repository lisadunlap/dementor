# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dementor is a research project for "stealing the souls of LLMs" - techniques to disguise one language model to mimic the behavior and response patterns of another model. The codebase implements various disguise methods and evaluation frameworks for comparing model outputs.

## Core Architecture

The project is organized around these key components:

### Disguise Methods (`disguising/methods/`)
- **Base methods**: `RandomSampleDisguise`, `JustNameIt`, `VibeBasedDisguise` 
- **Clustering methods**: `FeatureClustering` with stylistic, vibe, or embedding-based clustering
- **Math-specific methods**: `HierarchicalMathDisguise`, `EnsembleMathDisguise`
- **Active learning**: `ActiveLearningDisguise` with adaptive selection
- All methods inherit from `MethodBase` and implement `forward(prompt: str) -> str`

### Main Scripts
- **`disguising/disguise.py`**: Core disguise script that applies methods to transform model responses
- **Scoring**: Use `python -m disguising.scorer input.csv --output output_scored.csv` for LLM judge + heuristics.
- **`disguising/prompt_llm_new.py`**: Generate responses from models for evaluation
- **`generate_gsm8k_responses.py`**: Generate responses specifically for GSM8K math dataset

### Data Flow
1. Generate base responses using `prompt_llm_new.py` or `generate_gsm8k_responses.py`
2. Apply disguise methods via `disguise.py` to transform responses  
3. Score disguised vs target responses using `llm_scorer.py`
4. Results stored in `disguising/scores/` and `disguising/model-responses/`

## Common Commands

### Setup
```bash
pip install -r requirements.txt
# For math-specific methods, also install:
pip install -r math_disguise_requirements.txt
```

### Streamlined Workflow (Recommended)
```bash
# Run complete disguise experiment with one command
python disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method random_sampling

# Three core methods available:
# - random_sampling (baseline)
# - vibe_based (personality-based)
# - contrastive (comparative analysis)
```

### Legacy Workflow (For Specialized Cases)
```bash
# Generate model responses
python disguising/prompt_llm_new.py --model google/gemma-3-1b-it --num_samples 1000
python generate_gsm8k_responses.py  # For GSM8K dataset

# Apply disguise methods
python disguising/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method random_sample_3_examples

# Run evaluations
python -m disguising.scorer path/to/disguised_responses.csv --output path/to/disguised_responses_scored.csv
bash disguising/base_evals.sh      # Compare base models
bash disguising/disguised_evals.sh # Compare disguised vs target models
```

### Quick Evaluation
```bash
# Using new scoring utilities
python -c "from disguising.scoring_utils import score_model_comparison; score_model_comparison('input.csv', 'output.csv')"

# Heuristics only (faster)
python -c "from disguising.scoring_utils import score_model_comparison; score_model_comparison('input.csv', 'output.csv', heuristics_only=True)"
```

### Test Model Serving
```bash
# Start VLLM server
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --dtype half --tensor_parallel_size 4 --port 8000

# Test connection
python serve/utils_llm.py  # Uncomment model in test_get_llm_output function
```

### Run Single Test
```bash
python test_active_learning_disguise.py
```

## File Organization

- **`disguising/model-responses/`**: Generated responses organized by method and model
- **`disguising/scores/`**: Evaluation results and scoring outputs
- **`disguising/methods/`**: Implementation of all disguise techniques
- **`serve/`**: Utilities for model serving and LLM interaction
- **`data/`**: Datasets and prompts (GSM8K, chatbot arena, etc.)

## Available Disguise Methods

### Core Streamlined Methods (Recommended)
- **`contrastive_system_prompting`**: Learns differences between source and target models
- **`vibe_based_system_prompting`**: Captures personality and communication patterns  
- **`random_sampling_system_prompting`**: Clean example-based disguise with system prompts

### Legacy Methods (For Compatibility)
Get method names from `disguising/methods/get_method.py`:
- `random_sample_{1,3,5}_examples`: Sample-based approaches
- `just_name_it`: Simple name-based instruction
- `vibe_based_disguise`: GPT-4o identified behavioral differences
- `stylistic_clustering`: Cluster by formatting/length features  
- `hierarchical_math_disguise`: Math-domain specific approach
- `ensemble_math_disguise`: Multiple math disguise strategies
- `active_learning_disguise`: Adaptive example selection

## Model Response Paths

Responses follow naming pattern: `{model_name_with_underscores}.csv`
- Example: `meta-llama_Meta-Llama-3-8B-Instruct` → `meta-llama_Meta-Llama-3-8B-Instruct.csv`

## Environment Variables

Create `.env` file with required API keys for OpenAI, Anthropic, or other model providers used in evaluation.
