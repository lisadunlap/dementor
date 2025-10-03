# agents.md

This file provides guidance to agents when working with code in this repository.

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
- **`scripts/disguise.py`**: Core disguise script that applies methods to transform model responses
- **Scoring**: Use `python -m scripts.scorer input.csv --output output_scored.csv` for LLM judge + heuristics.
- **`scripts/prompt_llm_new.py`**: Generate responses from models for evaluation
- **`scripts/legacy/generate_gsm8k_responses.py`**: Legacy GSM8K generator (new flow uses scripts/generate_responses.py)

### Data Flow
1. Generate base responses using `prompt_llm_new.py` or `generate_gsm8k_responses.py`
2. Apply disguise methods via `disguise.py` to transform responses  
3. Score disguised vs target responses using `llm_scorer.py`
4. Results stored in `data/results/` and base responses in `data/model-responses/`

## Common Commands

### Setup
```bash
pip install -r requirements.txt
# For math-specific methods, also install:
pip install -r math_disguise_requirements.txt
```

### Caching
The framework includes persistent LMDB-based caching for all API calls:
- **Automatic**: All scripts automatically use caching when available
- **Persistent**: Cache survives across sessions, reducing API costs
- **Location**: `cache/llm_cache/` directory
- **Benefits**: Faster responses, reduced costs, consistent results
- **Clear cache**: `python -c "from scripts.cached_llm import clear_cache; clear_cache()"`
- **Stats**: `python -c "from scripts.cached_llm import get_cache_stats; print(get_cache_stats())"`

### Streamlined Workflow (Recommended)
```bash
# Run complete pipeline: generate responses, compare baseline, disguise, and score
python scripts/run_pipeline.py \
  --prompts_file data/datasets/gsm8k/gsm8k_prompts.txt \
  --source-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --target-model gpt-4o \
  --method contrastive_with_al_examples

# Results organized under data/results/<dataset>/ with:
# - comparisons/source_vs_target/ (source vs target baseline comparison and scores)
# - comparisons/disguised_vs_target/<method>/ (method-specific disguise runs and scores)
# - scores/ (other aggregated evaluation artifacts)
```

### Legacy Workflow (For Specialized Cases)
```bash
# Generate model responses
python scripts/prompt_llm_new.py --model google/gemma-3-1b-it --num_samples 1000
python generate_gsm8k_responses.py  # For GSM8K dataset

# Apply disguise methods
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method random_sample_3_examples

# Run evaluations
python -m scripts.scorer path/to/disguised_responses.csv --output path/to/disguised_responses_scored.csv
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

- **`data/model-responses/<dataset>/{full,500}/`**: Base model responses per dataset
- **`data/results/<dataset>/comparisons/baseline/`**: Baseline model comparisons and scoring
- **`data/results/<dataset>/disguised/`**: Disguised model outputs
- **`data/results/<dataset>/scores/`**: Evaluation results and scoring outputs
- **`disguising/methods/`**: Implementation of all disguise techniques
- **`scripts/`**: Pipeline scripts (run_pipeline.py, generate_responses.py, etc.)
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

## Coding Guidelines 

### General Principles
- **User executes commands**: I will run commands myself, so do not prompt to run things on your end
- **Prioritize simplicity**: Use helper functions or classes instead of redundant code or methods with hundreds of lines
- **Avoid over-engineering**: Do not add excessive tests and type checking (e.g., checking if a value is None) unless asked
- **Clean up thoroughly**: When told to remove components or refactor, ensure that you delete any files which are no longer needed and delete any code which is no longer used

### Error Handling
- **Address root causes**: Avoid adding try/excepts when running into errors; address the error rather than handling it with try/except
- **Ask before defaults**: Before adding a try/except or filling in a default dictionary value, ask the user whether to add this exception or if this is a bug that should be fixed
- **No import protection**: Assume all imports are correctly imported; do not add try/excepts to imports unless asked

### Data Handling
- **No automatic defaults**: When getting an item from a dictionary or dataframe, do not automatically put a default if that key does not exist. Instead, ask the user if they want a default value
- **Explicit typing**: Include typing and docstrings about the expected format of the inputs and outputs (e.g., keys and values if it's a dictionary, columns if it's a dataframe)

### Documentation & Testing
- **Maintain README**: Keep a README for just code structure and inputs/outputs to reference every time a question is asked
- **Update after changes**: Update the README after any added argument or refactor of code structure, especially if the input or output formats have changed
- **Ask before tests**: Prompt the user before adding test files
- **Ask about README updates**: When a new feature is added or a refactor is made, prompt the user to ask whether or not to add this to the README

### Code Quality
- **Check for redundancy**: With every edit you make, double-check that there is no redundant code in the files you are editing
- **No backwards compatibility**: Do NOT automatically support backwards compatibility – instead, prompt the user to ask if they would like backwards compatibility

### UI & Presentation
- **Minimal emojis**: Do not use emojis in READMEs and use sparingly in things like Gradio apps unless asked
- **Plotly default**: Use Plotly as the default plotting library
