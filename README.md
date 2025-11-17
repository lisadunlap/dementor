# Dementor - Streamlined LLM Disguise Framework

**Stealing the souls of LLMs** - A clean, focused framework for disguising one language model to mimic another.

See also: METHODS.md for a concise taxonomy and when to use each method.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# For math-specific methods, also install:
pip install -r math_disguise_requirements.txt

# Run streamlined pipeline (recommended)
python scripts/run_pipeline.py \
  --prompts_file data/datasets/gsm8k/gsm8k_prompts.txt \
  --source-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --target-model gpt-4o \
  --method contrastive

### Running Disguise with a Local vLLM Server

If Meta-Llama generations come from a local vLLM instance but helper calls still need OpenAI (contrastive / behavioral analyzers, or the text-embedding endpoint used by embedding clustering), use this pattern for every disguise run:

```bash
# Analyzer calls (contrastive, behavioral) → OpenAI
export ANALYSIS_API_BASE="https://api.openai.com/v1"
export ANALYSIS_API_KEY="$OPENAI_API_KEY"
# Embedding lookups (embedding_clustering) → OpenAI
export EMBEDDING_API_BASE="https://api.openai.com/v1"
export EMBEDDING_API_KEY="$OPENAI_API_KEY"

python scripts/disguise.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --disguise_as openai/gpt-4.1-mini \
  --method <method_name> \
  --prompts_file data/datasets/gsm8k/gsm8k_prompts_500.csv \
  --num_samples 500 \
  --source_responses data/model-responses/gsm8k/500/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv \
  --target_responses data/model-responses/gsm8k/500/openai_gpt-4.1-mini_responses.csv \
  --generation-api-base http://localhost:8000/v1 \
  --generation-api-key dummy \
  --no_wandb
```

- Swap `<method_name>` with any disguise method (`random_sampling`, `contrastive`, `behavioral_based`, `stylistic_clustering`, `embedding_clustering`, `just_name_it`, etc.).
- `--generation-api-base` / `--generation-api-key` route the source model to your local vLLM server (overriding the defaults in `.env`).
- `ANALYSIS_API_BASE` keeps contrastive / behavioral analyzers on OpenAI with the key from `.env`.
- `EMBEDDING_API_BASE` pins embedding_clustering’s embedding lookups to OpenAI.
```

## Local Generation (HF vs vLLM)

You can run models locally in two ways when generating base outputs (via `scripts/generate_responses.py`):

- Hugging Face Transformers (prefix `hf:`)
  - Example: `--model hf:meta-llama/Llama-3.1-8B-Instruct`
  - Under the hood: `transformers.pipeline("text-generation", device_map="auto")` which uses `accelerate` to place weights on your local GPU(s).
  - Pros: Simple, great for quick experiments; supports most HF models.
  - Cons: Less throughput than vLLM for large batches; higher memory overhead in some cases.
  - Install: `pip install transformers accelerate`

- vLLM (prefix `vllm:`)
  - Example: `--model vllm:meta-llama/Llama-3.1-8B-Instruct`
  - Under the hood: `vllm.LLM` with `SamplingParams` (fast local inference engine).
  - Pros: High throughput and memory efficiency for longer runs; good for large datasets.
  - Cons: Requires `vllm` setup; fewer ready-made chat templates (we use a plain prompt string).
  - Install: `pip install vllm`

Provider APIs (default)
- If you pass a provider model (e.g., `openai/gpt-4o-mini`), we use LiteLLM and your API keys.
- No local model weights required.

Where this is used
- Local/provider selection is used by `scripts/generate_responses.py` when producing base outputs.
- The disguise step (`disguise.py`) uses LiteLLM by default. To run disguise against a local vLLM server, pass:
  - `--openai-api-base http://localhost:8000/v1` (or your vLLM endpoint)
  - `--openai-api-key EMPTY` (placeholder is ok for local)
  - and a model string like `openai/meta-llama/Llama-3.1-8B-Instruct` so LiteLLM routes to your local server.

### Using vLLM for Disguise (example)

1) Start a local vLLM server exposing an OpenAI‑compatible API
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 --dtype auto --max-model-len 4096
```

2) Generate base outputs via LiteLLM routed to vLLM (optional; you can also use the explicit `vllm:` backend)
```bash
python scripts/generate_responses.py \
  --model openai/meta-llama/Llama-3.1-8B-Instruct \
  --prompts_file data/datasets/chatbot_arena/chatbot_arena_prompts.txt \
  --output data/model-responses/chatbot_arena/full/llama31_8b.csv \
  --openai-api-base http://localhost:8000/v1 \
  --openai-api-key EMPTY
```

3) Run disguise against your local vLLM server using LiteLLM routing
```bash
python disguise.py \
  --model openai/meta-llama/Llama-3.1-8B-Instruct \
  --disguise_as gpt-4o \
  --method contrastive \
  --num_samples 200 \
  --openai-api-base http://localhost:8000/v1 \
  --openai-api-key EMPTY
```

4) Score and aggregate as usual
```bash
python -m scripts.scorer pairwise \
  --input data/results/chatbot_arena/disguised/my_run.csv \
  --output data/results/chatbot_arena/scores/my_run_scored/scored.csv
python scripts/aggregate_metrics.py --root data/results/chatbot_arena --output data/results/chatbot_arena/summary.csv --markdown data/results/chatbot_arena/summary.md
```

Helper script
- You can also use the provided helper to run this end‑to‑end locally:
  ```bash
  bash scripts/run_local_vllm.sh \
    --hf-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000 --dtype float16 --tp 4 \
    --prompts_file data/datasets/chatbot_arena/chatbot_arena_prompts.txt \
    --num_samples 200 --output_dir data/results/chatbot_arena/disguised \
    --source-model-openai openai/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --target-model gpt-4o
  ```

## Disguise Methods

### 1. Random Sampling (`--method random_sampling_system_prompting`)
**Best for**: Reliable baseline performance with good target model responses
```bash
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method random_sampling_system_prompting
```
- Randomly samples examples from target model responses
- Uses clean system prompting for instructions
- Simple, effective, and fast

### 2. Behavioral-Based (`--method behavioral_based_system_prompting`)
**Best for**: Capturing personality and deep communication essence
```bash
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method behavioral_based_system_prompting
```
- Analyzes target model's communication style, personality, and behavioral patterns
- Creates sophisticated "essence profile" capturing how the model thinks and responds
- Focuses on cognitive style, emotional resonance, and interaction patterns

### 3. Stylistic (`--method stylistic_system_prompting`)
**Best for**: Replicating measurable surface-level patterns
```bash
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method stylistic_system_prompting
```
- Analyzes and replicates measurable stylistic features (formatting, length, structure)
- Focuses on surface-level patterns that can be quantified
- Complements behavioral-based analysis with concrete metrics

### 4. Contrastive (`--method contrastive_system_prompting`)
**Best for**: Learning specific differences between models
```bash
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method contrastive_system_prompting
```
- Compares source and target models to identify distinguishing features
- Learns what makes the target model unique vs the source; produces explicit, actionable guidelines (system prompt)
- Requires both source and target model response data

Scoring
- Single-file scoring (base models): `python -m scripts.scorer single data/model-responses/<dataset>/full/<model>.csv --output data/results/<dataset>/scores/<model>/scored.csv`
- Pairwise scoring (disguised vs target):
  `python -m scripts.scorer pairwise --input data/results/<dataset>/<method>/<src>_as_<tgt>.csv --output data/results/<dataset>/<method>/scores/<pair>/scored.csv --judge-model openai/gpt-4.1-mini`
- Merge and score in one step: `python -m scripts.scorer compare --a data/model-responses/<dataset>/full/<src>.csv --b data/model-responses/<dataset>/full/<tgt>.csv --output data/results/<dataset>/comparisons/source_vs_target/<src>_vs_<tgt>.csv`

## File Structure

```
dementor/
├── scripts/
│   ├── run_pipeline.py              # Streamlined end-to-end pipeline (recommended)
│   ├── disguise.py                  # Core disguise script
│   ├── generate_responses.py        # Generate model responses
│   ├── scorer.py                    # Unified scoring CLI (single, pairwise, compare, merge)
│   ├── stylistic_analysis.py        # Heuristic analysis functions
│   └── methods/
│       ├── contrastive.py           # Contrastive method
│       ├── behavioral_based.py      # Behavioral-based method
│       ├── random_sampling.py       # Random sampling method
│       ├── stylistic.py             # Stylistic method
│       ├── get_method.py            # Method registry
│       ├── utils/                   # Shared utilities
│       └── extras/                  # Legacy/specialized methods
├── data/                            # Datasets and prompts
└── results/                         # Generated responses and scores

Additional:
- `examples/` — provider setup and copy‑paste commands
- `scripts/smoke_test.py` — offline smoke test (no API keys)
- `scripts/serve/` — optional helpers for vLLM/local serving experiments
- `workflows/README.md` — instructions for the SFT + DPO workflows, dataset setup, and adapter registry usage.
- `run_gsm8k_finetune_responses.sh` — helper to regenerate the 200-sample GSM8K evaluation responses (and judge scores) for all four finetune adapters (Tinker/OpenAI, SFT/DPO).

## Filename Convention

- We use official model identifiers in artifact filenames and tags, with only minimal sanitization for filesystem safety.
- Sanitization rule: replace `/` and `:` with `_`.
- Examples:
  - Source responses: `data/model-responses/gsm8k/full/meta-llama_Meta-Llama-3.1-8B-Instruct.csv`
  - Disguised vs target: `data/results/gsm8k/comparisons/disguised_vs_target/<method>/gpt-4.1_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv`
  - Baseline comparison: `data/results/gsm8k/comparisons/source_vs_target/openai_gpt-4o-mini_vs_meta-llama_Meta-Llama-3.1-8B-Instruct.csv`

Note: Older runs may contain short-name artifacts (e.g., `llama-3-8b`). These are deprecated; new runs and tools write official names.
```

## Evaluation

The framework provides two types of evaluation:

### LLM Judge Scoring (using Phi-4-mini via vLLM)
- **Semantic Score (1-4)**: How similar the meaning/content is
- **Stylistic Score (1-4)**: How similar the formatting/tone/style is

### Heuristic Scoring (fast, no LLM needed)
- Response length similarity  
- Formatting pattern matches (headers, lists, code blocks)
- Overall structural similarity

## 💡 Usage Examples

### Basic Usage
```bash
# Generate 200 disguised responses
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method behavioral_based --num_samples 200

# Use custom prompts
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method contrastive --prompts_file my_prompts.txt

# Skip evaluation (generation only)
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method random_sampling --skip_evaluation

# Heuristics only (faster evaluation)
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method behavioral_based --heuristics_only
```

### With Experiment Tracking
```bash
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method contrastive --use_wandb --run_name "contrastive_experiment_v1"
```

### Scoring and Summaries
```bash
# Score a CSV of response pairs with LLM judge + heuristics
python -m scripts.scorer pairwise \
  --input data/results/chatbot_arena/disguised/my_run.csv \
  --output data/results/chatbot_arena/scores/my_run_scored/scored.csv

# Heuristics only (no LLM judge)
python -m scripts.scorer pairwise \
  --input data/results/chatbot_arena/disguised/my_run.csv \
  --output data/results/chatbot_arena/scores/my_run_scored/scored.csv \
  --heuristics-only

# Programmatic usage: compute averages
python - << 'PY'
from scripts.scorer import score_pairwise
df = score_pairwise(
    'data/results/chatbot_arena/disguised/my_run.csv',
    'data/results/chatbot_arena/scores/my_run_scored/scored.csv',
    judge_model='openai/gpt-4.1-mini',
)
print(df[['semantic_score', 'stylistic_score', 'heuristic_match_score']].mean())
PY
```

## Method Comparison

| Method | Strengths | Best Use Case | Speed |
|--------|-----------|---------------|-------|
| **random_sampling** | Simple, reliable baseline | When you have good target examples | Fast |
| **behavioral_based** | Captures deep communication essence | Models with distinctive personality | Medium |
| **stylistic** | Measurable surface features | Consistent formatting/structure patterns | Fast |
| **contrastive** | Explicit change-list and rules | Understanding model distinctions | Slow |

## Recommended Combinations
- Contrastive + curated target examples: clear rules with targeted exemplars.
- Vibe-based + a few curated examples: capture personality with concrete anchors.
- Stylistic + clustering: enforce structure and pick representative examples automatically.

## Recommended Pipelines (Commands)
-- Contrastive (rules-only)
  - python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method contrastive --num_samples 200

-- Random-k examples baseline
  - python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method random_sampling --num_samples 200

-- Vibe-based (personality) with examples
  - python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as gpt-4o --method behavioral_based --num_samples 200

-- Scoring with summary metrics
  - python -m scripts.scorer pairwise --input data/results/chatbot_arena/disguised/my_run.csv --output data/results/chatbot_arena/scores/my_run_scored/scored.csv

## Pipelines (Step-by-step)

For larger datasets (e.g., GSM8K), prefer explicit steps over a single mega-command:

1) Generate base outputs for each model
```bash
python scripts/generate_responses.py \
  --model openai/gpt-4o-mini \
  --prompts_file data/datasets/chatbot_arena/chatbot_arena_prompts.txt \
  --output data/model-responses/chatbot_arena/full/openai_gpt-4o-mini.csv

python scripts/generate_responses.py \
  --model openai/gpt-4o \
  --prompts_file data/datasets/chatbot_arena/chatbot_arena_prompts.txt \
  --output data/model-responses/chatbot_arena/full/openai_gpt-4o.csv
```

Note: The canonical location for base model outputs is `data/model-responses/<dataset>/full/` (and optionally `/500/`). The `disguise.py` script will auto‑detect source/target CSVs there first and will still fall back to legacy `data/model-responses/base/` if present. You can always override with `--source_responses` and `--target_responses`.

Benchmark style archetypes
- Curated style/persona response CSVs are inputs and now live under `data/datasets/benchmarks/style_archetypes/`.
- These are not generated artifacts; they are used for analysis/visualization and reference.

2) Run disguise (contrastive rules)
```bash
python scripts/disguise.py \
  --model openai/gpt-4o-mini \
  --disguise_as gpt-4o \
  --method contrastive \
  --num_samples 200 \
  --source_responses data/model-responses/chatbot_arena/full/openai_gpt-4o-mini.csv \
  --target_responses data/model-responses/chatbot_arena/full/openai_gpt-4o.csv
```

3) Score and get metrics
```bash
python -m scripts.scorer pairwise \
  --input data/results/<dataset>/comparisons/disguised_vs_target/<method>/<src>_as_<tgt>.csv \
  --output data/results/<dataset>/comparisons/disguised_vs_target/<method>/<src>_as_<tgt>_scored.csv
```

Optional one-liner orchestrator
```bash
python scripts/run_pipeline.py \
  --prompts_file data/datasets/chatbot_arena/chatbot_arena_prompts.txt \
  --source-model openai/gpt-4o-mini \
  --target-model gpt-4o \
  --method contrastive \
  --num_samples 200
```

## Scripts Overview

- `scripts/generate_responses.py`
  - Generates base model outputs for a prompts file, with simple caching by prompt.
  - Backends: LiteLLM providers (default), local HuggingFace via `hf:` prefix, local vLLM via `vllm:` prefix.
  - Example: `python scripts/generate_responses.py --model openai/gpt-4o-mini --prompts_file data/datasets/chatbot_arena/chatbot_arena_prompts.txt --output data/model-responses/chatbot_arena/full/openai_gpt-4o-mini.csv`

- `scripts/scorer.py`
  - Unified CLI for single-file scoring, pairwise scoring, comparison + scoring, and legacy merges.
  - Examples:
    - `python -m scripts.scorer single data/model-responses/chatbot_arena/full/openai_gpt-4o-mini.csv --output data/results/chatbot_arena/scores/openai_gpt-4o-mini/scored.csv`
    - `python -m scripts.scorer compare --a base/modelA.csv --b base/modelB.csv --output results/compare_A_vs_B.csv --judge-model openai/gpt-4.1-mini`

- `scripts/run_pipeline.py`
  - Optional orchestrator that runs: generate base (both models) → baseline compare (source vs target) → disguise (disguised vs target) → score → optional aggregation + W&B table + three‑way summary. Uses cache to avoid recomputing.
  - Example shown above in Pipelines.

- `scripts/aggregate_metrics.py`
  - Aggregates `scored_metrics.csv` (and legacy `*_scores_metrics.json`) files to a CSV and optional Markdown; optionally logs a W&B summary table.
  - Example shown in Evaluation → Aggregating Metrics.

- `scripts/summarize_three_way.py`
  - Produces a compact table comparing baseline (source vs target) and disguised vs target metrics; can also log to W&B.
  - Example shown in Evaluation → Aggregating Metrics.

- `scripts/smoke_test.py`
  - Offline smoke test (no API keys): checks method.forward behavior and heuristics-only scoring pipeline.

## 🧠 How Vibe-Based Disguise Works

The behavioral-based method goes beyond surface-level mimicry to capture the **essence** of how a model communicates:

1. **Deep Analysis**: Examines cognitive style, personality traits, interaction patterns
2. **Essence Profiling**: Creates actionable personality guidelines 
3. **Behavioral Capture**: Identifies unique quirks, language patterns, response architecture
4. **Sophisticated Prompting**: Uses comprehensive personality profile for authentic disguise

This creates more authentic disguises that capture not just what a model says, but **how** it thinks and communicates.

## 📏 How Stylistic System Prompting Works

The stylistic method focuses on **measurable, surface-level features** that can be quantified and replicated:

1. **Quantitative Analysis**: Measures concrete features like response length, sentence structure, formatting patterns
2. **Pattern Detection**: Identifies consistent use of markdown, bullet points, code blocks, headers
3. **Structural Guidelines**: Creates specific rules for replicating measurable style elements
4. **Verification-Friendly**: Produces patterns that can be easily verified with heuristics

This complements behavioral-based analysis by providing concrete, measurable targets for style replication.

## 📊 Understanding Results

### Good Disguise Indicators:
- **Semantic scores 3-4**: Content meaning preserved
- **Stylistic scores 3-4**: Style successfully mimicked  
- **High heuristic match**: Surface patterns match well

### Troubleshooting Low Scores:
- Try different methods (behavioral-based often works better for distinctive models)
- Increase number of target examples
- Check that target model responses are representative
- Use contrastive method when models are very different

## 🗂️ File Structure Details

### Input Files Expected:
- Target model responses: `data/model-responses/<dataset>/full/{model_name}.csv`
- Source model responses: `data/model-responses/<dataset>/full/{source_model}.csv` (for contrastive)
- Prompts: Text file with one prompt per line (default: `data/datasets/chatbot_arena/chatbot_arena_prompts.txt`)

### Output Files Generated:
Results and scores are saved under `data/results/<dataset>/comparisons/...` with method-specific subfolders.

## 🚮 What Was Cleaned Up

**Removed useless scripts:**
- Legacy entrypoints (`prompt_llm.py`, `llm_scorer2.py`, `random_sample_disguise.py`, `compare_llm.py`) have been deleted; use the streamlined equivalents in `scripts/` noted above.

**Streamlined:**
- Clean method registry with only 3 core methods
- Unified scoring interface  
- Single command for complete experiments
- No backward compatibility bloat

## 🔧 Advanced Configuration

The core methods can be customized:

```python
# In your own scripts
from scripts.methods.get_method import get_method

# Vibe-based with custom settings
method = get_method('behavioral_based', 'source-model', 'target-model', disguise_df=target_data)

# Contrastive with both datasets
method = get_method('contrastive', 'source-model', 'target-model', disguise_df=target_data, source_df=source_data)

# Generate disguised prompt
disguised_messages = method.forward("Your prompt here")
```

## 🎯 Validation

Run a small end-to-end validation:
```bash
python validate_end_to_end.py \
  --model openai/gpt-4o-mini \
  --disguise_as gpt-4o \
  --method contrastive \
  --num_samples 20
```

This generates a small run, scores it, and writes metrics JSON/CSV.

### Offline Smoke Test
```bash
python scripts/smoke_test.py
```
Runs a tiny test without API keys: checks method.forward formatting and heuristic scoring pipeline.

### CI
This repository includes a minimal GitHub Actions workflow (`.github/workflows/smoke.yml`) that runs the offline smoke test on pushes and PRs to main/master using only lightweight dependencies (pandas, numpy, tqdm).

---

**Clean. Focused. Effective.** Four methods, one framework, reliable disguises.
### Model Setup (LiteLLM providers)
- For API providers, set the provider env var(s) before running `disguise.py`, e.g.:
  - OpenAI: `export OPENAI_API_KEY=...`
  - Anthropic: `export ANTHROPIC_API_KEY=...`
  - Together: `export TOGETHER_AI_API_KEY=...`
  - X.AI: `export XAI_API_KEY=...`
  - Google: `export GEMINI_API_KEY=...`
  - Then pass `--model` as a provider-qualified name (e.g., `openai/gpt-4o-mini`).
### W&B Integration
- Enable logging with `--use_wandb` on `disguise.py`.
### Aggregating Metrics and W&B Table
Aggregate all run metrics under a directory and optionally log a W&B table:
```bash
python scripts/aggregate_metrics.py --root data/results/chatbot_arena \
  --output results/summary.csv --markdown results/summary.md \
  --use-wandb --wandb-project streamlined-disguise --wandb-run-name metrics-aggregate
```
This writes a summary CSV/MD and logs a W&B Table and artifact.

- Defaults to project `streamlined-disguise`; override via `WANDB_PROJECT` if needed.
- Logs summary metrics (semantic/stylistic means, heuristic match) and, when possible, uploads the following as a W&B artifact:
  - results CSV, scored CSV, metrics JSON/CSV, and run summary JSON (composite method)
