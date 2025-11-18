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
  --prompts_file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --source-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --target-model meta-llama/Meta-Llama-3.1-8B-Instruct \
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

You can run models locally or via provider APIs using the `basic` backend of `scripts/generate_responses.py`:

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
- If you pass a provider model (e.g., `openai/gpt-4.1-mini`), we use LiteLLM and your API keys.
- No local model weights required.

Where this is used
- Local/provider selection is used by `scripts/generate_responses.py basic` when producing base outputs.
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
  --prompts-file data/datasets/chatbot_arena/chatbot_arena_prompts.csv \
  --output-csv data/model-responses/chatbot_arena/full/llama31_8b.csv \
  basic \
  --model openai/meta-llama/Llama-3.1-8B-Instruct \
  --openai-api-base http://localhost:8000/v1 \
  --openai-api-key EMPTY
```

3) Run disguise against your local vLLM server using LiteLLM routing
```bash
python disguise.py \
  --model openai/meta-llama/Llama-3.1-8B-Instruct \
  --disguise_as openai/gpt-4.1-mini \
  --method contrastive \
  --num_samples 200 \
  --openai-api-base http://localhost:8000/v1 \
  --openai-api-key EMPTY
```

4) Score and inspect as usual
```bash
python -m scripts.scorer pairwise \
  --input data/results/chatbot_arena/disguised/my_run.csv \
  --output data/results/chatbot_arena/scores/my_run_scored/scored.csv
```

Helper script
- You can also use the provided helper to run this end‑to‑end locally:
  ```bash
  bash scripts/run_local_vllm.sh \
    --hf-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000 --dtype float16 --tp 4 \
    --prompts_file data/datasets/chatbot_arena/chatbot_arena_prompts.csv \
    --num_samples 200 --output_dir data/results/chatbot_arena/disguised \
    --source-model-openai openai/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --target-model openai/gpt-4.1-mini
  ```

## Disguise Methods

| Method | Summary | Typical Use |
| --- | --- | --- |
| Random Sampling | Builds a few-shot prompt directly from target responses. | Quick baselines when target answers are strong and consistent. |
| Behavioral-Based | Learns persona/tone/structure from target outputs and turns them into a rich system prompt. | When you need the mimic to adopt the target model’s “voice.” |
| Stylistic | Enforces measurable formatting traits (length, markdown, lists, code). | When evaluation focuses on surface style or formatting fidelity. |
| Contrastive | Learns explicit corrections by comparing source vs target responses. | When the two models diverge sharply and you want targeted deltas. |

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
│   ├── generate_responses.py        # Finetune adapters + basic provider/HF/vLLM generation
│   ├── scorer.py                    # Unified scoring CLI (single, pairwise, compare, merge)
│   ├── stylistic_analysis.py        # Style heuristics used by scoring/methods
│   ├── gsm8k/                       # GSM8K-specific workflow runners
│   │   ├── run_all_workflows.sh
│   │   ├── run_openai_workflows.sh
│   │   └── run_gsm8k_finetune_responses.sh
│   │   ├── plot_gsm8k_eval200_finetune_scores.py
│   │   └── cleanup_gsm8k_results.sh
│   ├── tools/                       # One-off maintenance utilities
│   │   ├── compute_deltas.py
│   │   ├── convert_convergence_html.py
│   │   ├── summarize_three_way.py
│   │   └── save_tinker_adapter.py
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
- `examples/` — provider setup and copy‑paste commands.
- `scripts/smoke_test.py` — offline smoke test (no API keys).
- `scripts/serve/` — optional helpers for vLLM/local serving experiments.
- `scripts/gsm8k/` — GSM8K workflow + maintenance scripts (e.g., `scripts/gsm8k/run_all_workflows.sh`, `scripts/gsm8k/run_gsm8k_finetune_responses.sh`, `scripts/gsm8k/run_openai_workflows.sh`, `scripts/gsm8k/cleanup_gsm8k_results.sh`, `scripts/gsm8k/plot_gsm8k_eval200_finetune_scores.py`).
- `scripts/tools/` — low-frequency maintenance utilities (e.g., convergence HTML → PNG converter, score delta calculator, manual Tinker adapter saver).
- `workflows/README.md` — instructions for SFT + DPO workflows, dataset setup, and adapter registry usage that reference the `scripts/gsm8k/` launchers.

## Filename Convention

- We use official model identifiers in artifact filenames and tags, with only minimal sanitization for filesystem safety.
- Sanitization rule: replace `/` and `:` with `_`.
- Examples:
  - Source responses: `data/model-responses/gsm8k/full/meta-llama_Meta-Llama-3.1-8B-Instruct.csv`
  - Disguised vs target: `data/results/gsm8k/comparisons/disguised_vs_target/<method>/gpt-4.1_as_meta-llama_Meta-Llama-3.1-8B-Instruct.csv`
  - Baseline comparison: `data/results/gsm8k/comparisons/source_vs_target/openai_gpt-4.1-mini_vs_meta-llama_Meta-Llama-3.1-8B-Instruct.csv`

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
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as openai/gpt-4.1-mini --method behavioral_based --num_samples 200

# Use custom prompts
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as openai/gpt-4.1-mini --method contrastive --prompts_file my_prompts.csv

# Skip evaluation (generation only)
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as openai/gpt-4.1-mini --method random_sampling --skip_evaluation

# Heuristics only (faster evaluation)
```

### With Experiment Tracking
```bash
python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as openai/gpt-4.1-mini --method contrastive --use_wandb --run_name "contrastive_experiment_v1"
```

### Scoring and Summaries
```bash
# Score a CSV of response pairs with the LLM judge
python -m scripts.scorer pairwise \
  --input data/results/chatbot_arena/disguised/my_run.csv \
  --output data/results/chatbot_arena/scores/my_run_scored/scored.csv

# Heuristics only (no LLM judge)
python -m scripts.scorer pairwise \
  --input data/results/chatbot_arena/disguised/my_run.csv \
  --output data/results/chatbot_arena/scores/my_run_scored/scored.csv \

# Programmatic usage: compute averages
python - << 'PY'
from scripts.scorer import score_pairwise
df = score_pairwise(
    'data/results/chatbot_arena/disguised/my_run.csv',
    'data/results/chatbot_arena/scores/my_run_scored/scored.csv',
    judge_model='openai/gpt-4.1-mini',
)
print(df[['semantic_score', 'stylistic_score']].mean())
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
  - python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as openai/gpt-4.1-mini --method contrastive --num_samples 200

-- Random-k examples baseline
  - python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as openai/gpt-4.1-mini --method random_sampling --num_samples 200

-- Vibe-based (personality) with examples
  - python scripts/disguise.py --model google/gemma-3-1b-it --disguise_as openai/gpt-4.1-mini --method behavioral_based --num_samples 200

-- Scoring with summary metrics
  - python -m scripts.scorer pairwise --input data/results/chatbot_arena/disguised/my_run.csv --output data/results/chatbot_arena/scores/my_run_scored/scored.csv

## Pipelines (Step-by-step)

For larger datasets (e.g., GSM8K), prefer explicit steps over a single mega-command:

1) Generate base outputs for each model
```bash
python scripts/generate_responses.py \
  --prompts-file data/datasets/chatbot_arena/chatbot_arena_prompts.csv \
  --output-csv data/model-responses/chatbot_arena/full/openai_gpt-4.1-mini.csv \
  basic \
  --model openai/gpt-4.1-mini

python scripts/generate_responses.py \
  --prompts-file data/datasets/chatbot_arena/chatbot_arena_prompts.csv \
  --output-csv data/model-responses/chatbot_arena/full/meta-llama_Meta-Llama-3.1-8B-Instruct.csv \
  basic \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

Note: The canonical location for base model outputs is `data/model-responses/<dataset>/full/` (and optionally `/500/`). The `disguise.py` script will auto‑detect source/target CSVs there first and will still fall back to legacy `data/model-responses/base/` if present. You can always override with `--source_responses` and `--target_responses`.

Benchmark style archetypes
- Curated style/persona response CSVs are inputs and now live under `data/datasets/benchmarks/style_archetypes/`.
- These are not generated artifacts; they are used for analysis/visualization and reference.

2) Run disguise (contrastive rules)
```bash
python scripts/disguise.py \
  --model openai/gpt-4.1-mini \
  --disguise_as meta-llama/Meta-Llama-3.1-8B-Instruct \
  --method contrastive \
  --num_samples 200 \
  --source_responses data/model-responses/chatbot_arena/full/openai_gpt-4.1-mini.csv \
  --target_responses data/model-responses/chatbot_arena/full/meta-llama_Meta-Llama-3.1-8B-Instruct.csv
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
  --prompts_file data/datasets/chatbot_arena/chatbot_arena_prompts.csv \
  --source-model openai/gpt-4.1-mini \
  --target-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --method contrastive \
  --num_samples 200
```

## Scripts Overview

- `scripts/generate_responses.py`
  - Finetune-aware generation utility. Use the `basic` subcommand for raw provider/HF/vLLM sampling, or the `tinker`/`openai` subcommands to sample LoRA adapters.
  - Prompts file: CSV with a `prompt` column is the default; `.txt` (one prompt per line) still works for lightweight datasets.
  - Example: `python scripts/generate_responses.py --prompts-file data/datasets/chatbot_arena/chatbot_arena_prompts.csv --output-csv data/model-responses/chatbot_arena/full/openai_gpt-4.1-mini.csv basic --model openai/gpt-4.1-mini`

- `scripts/scorer.py`
  - Unified CLI for single-file scoring, pairwise scoring, comparison + scoring, and legacy merges.
  - Examples:
- `python -m scripts.scorer single data/model-responses/chatbot_arena/full/openai_gpt-4.1-mini.csv --output data/results/chatbot_arena/scores/openai_gpt-4.1-mini/scored.csv`
    - `python -m scripts.scorer compare --a base/modelA.csv --b base/modelB.csv --output results/compare_A_vs_B.csv --judge-model openai/gpt-4.1-mini`

- `scripts/run_pipeline.py`
  - Optional orchestrator that runs: generate base (both models) → baseline compare (source vs target) → disguise (disguised vs target) → score → optional aggregation + W&B table + three‑way summary. Uses cache to avoid recomputing.
  - Example shown above in Pipelines.

- `scripts/tools/summarize_three_way.py`
  - Produces a compact table comparing baseline (source vs target) and disguised vs target metrics; can also log to W&B.
  - Example shown in Evaluation → Aggregating Metrics.

- `scripts/tools/save_tinker_adapter.py`
  - Load a saved Tinker state URI and register it under a friendly adapter alias in `data/tinker_adapters.json`.

- `scripts/smoke_test.py`
- Offline smoke test (no API keys): checks method.forward behavior and basic scoring pipeline.

## 📊 Understanding Results

### Good Disguise Indicators:
- **Semantic scores 3-4**: Content meaning preserved
- **Stylistic scores 3-4**: Style successfully mimicked  
- **Stylistic scores 3-4**: Style successfully mimicked  

### Troubleshooting Low Scores:
- Try different methods (behavioral-based often works better for distinctive models)
- Increase number of target examples
- Check that target model responses are representative
- Use contrastive method when models are very different

## 🗂️ File Structure Details

### Input Files Expected:
- Target model responses: `data/model-responses/<dataset>/full/{model_name}.csv`
- Source model responses: `data/model-responses/<dataset>/full/{source_model}.csv` (for contrastive)
- Prompts: CSV with a `prompt` column (default: `data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv`; chatbot arena ships `data/datasets/chatbot_arena/chatbot_arena_prompts.csv`). Plain-text lists (e.g., `data/datasets/chatbot_arena/chatbot_arena_prompts.txt`) remain supported if you prefer one prompt per line.

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
  --model openai/gpt-4.1-mini \
  --disguise_as meta-llama/Meta-Llama-3.1-8B-Instruct \
  --method contrastive \
  --num_samples 20
```

This generates a small run, scores it, and writes metrics JSON/CSV.

### Offline Smoke Test
```bash
python scripts/smoke_test.py
```
Runs a tiny test without API keys: checks method.forward formatting and basic scoring pipeline.

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
  - Then pass `--model` as a provider-qualified name (e.g., `openai/gpt-4.1-mini`).
### W&B Integration
- Enable logging with `--use_wandb` on `disguise.py`.
