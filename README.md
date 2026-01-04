# Dementor – Streamlined LLM Disguise

Minimal toolkit for stealing the “voice” of one LLM and applying it to another. Pair concise prompt-based methods with optional SFT/DPO adapters; all evaluation artifacts land under `data/results/`.

All pairwise scores are reported on a 1–4 scale (decimals allowed) for both semantic and stylistic similarity.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up API keys (e.g., OpenAI API key):
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Quick Start
```bash
# Step 1: Build prompts CSV with a single 'prompt' column
python scripts/make_prompts.py \
  --input data/<dataset>/raw_prompts.csv \
  --column prompt \
  --output data/datasets/<dataset>/<dataset>_prompts_eval_200_seed42.csv

# Step 2: Generate target model responses
python scripts/generate_responses.py \
  --prompts-file data/datasets/<dataset>/<dataset>_prompts_eval_200_seed42.csv \
  --output-csv data/model-responses/<dataset>/full/<target_model>.csv \
  basic --model <provider>/<target_model>

# Step 3: Apply disguise (e.g. make Llama sound like GPT-4)
python scripts/disguise.py \
  --prompts_file data/datasets/<dataset>/<dataset>_prompts_eval_200_seed42.csv \
  --model <source_model> \
  --disguise_as <target_model> \
  --method <method_name> \
  --num_samples 50

# Step 4: Score disguised vs target responses (pairwise judge)
python -m scripts.scorer \
  --input data/results/<dataset>/eval200/<method>/<src>_as_<tgt>.csv \
  --output data/results/<dataset>/eval200/<method>/scores/<src>_as_<tgt>/scored.csv \
  --judge-model openai/gpt-4.1-mini

```
## Additional Scoring
```bash
# Pairwise (LLM judge only, default)
python -m scripts.scorer \
  --input data/results/<dataset>/<subset>/<method>/<src>_as_<tgt>.csv \
  --output data/results/<dataset>/<subset>/<method>/scores/<src>_as_<tgt>/scored.csv \
  --judge-model openai/gpt-4.1-mini

# Pairwise (LLM judge + heuristics)
python -m scripts.scorer \
  --input data/results/<dataset>/<subset>/<method>/<src>_as_<tgt>.csv \
  --output data/results/<dataset>/<subset>/<method>/scores/<src>_as_<tgt>/scored.csv \
  --judge-model openai/gpt-4.1-mini \
  --heuristics
```
*(Default judge model: `openai/gpt-4.1-mini`. Override with `--judge-model` if needed.)*
*(Here `<subset>` is the prompt split inferred from your prompts file, e.g., `eval200`, `train300`, `500`, or `full`.)*

## Reference Docs
- `docs/local_generation.md` – HF/vLLM/provider routing (includes the local vLLM walkthrough).
- `docs/stylometric_classifier.md` – stylometric attribution baseline (char/word/POS n-grams).
- `workflows/README.md` – GSM8K SFT/DPO orchestration + Tinker adapter registry.
- `examples/` – copy/paste provider configs.
- `AGENTS.md` – repo guidance + coding conventions for AI agents.

## Workflow
1. **Build prompts CSV/TXT** – `scripts/make_prompts.py` converts raw datasets into the `prompt` column expected by generators (e.g., `gsm8k_prompts_train_300_seed42.csv` for train, `gsm8k_prompts_eval_200_seed42.csv` for eval).
2. **Generate base responses** – `scripts/generate_responses.py` (provider/HF/vLLM/Tinker) → outputs a CSV with `prompt` + `model_response`. Run it separately for the 300 train prompts (used during SFT/DPO) and the 200 eval prompts (used for disguise/score).
3. **Disguise** – `scripts/disguise.py` consumes the prompt file + source/target response CSVs and writes `prompt, model_response, target_response`.
4. **Score** – `python -m scripts.scorer --input <comparison.csv> --output <scores.csv>` (LLM judge only; heuristics opt-in) expects `prompt, model_response, target_response`.
5. **Review outputs** – CSVs in `data/results/<dataset>/...`; cache lives in `cache/llm_cache/`.

### Column Contracts
- **Prompts (`scripts/make_prompts.py`)**: must emit a single column named `prompt`.
- **Base responses (`scripts/generate_responses.py`)**: expect `prompt` in the input file and produce `prompt`, `model_response`, plus metadata columns (e.g., `model`).
- **Disguised comparisons (`scripts/disguise.py`)**: consume the prompt file + source/target CSVs and write `prompt`, `model_response` (new disguised output), and `target_response`.
- **Scoring (`scripts/scorer.py`)**: requires the comparison CSV to include `prompt`, `model_response`, `target_response`; optional columns (method, source_model, etc.) are passed through untouched.

### Data & Results Layout
- `data/datasets/<dataset>/`: prompt CSV/TXT files produced by `make_prompts.py`. For example, for GSM8K we keep both `gsm8k_prompts_train_300_seed42.csv` and `gsm8k_prompts_eval_200_seed42.csv`.
- `data/model-responses/<dataset>/`:
  - `full/`, `500/`, etc. hold bulk generations (filenames follow `<provider_model>_responses.csv`).
  - `splits/seed42/train_300/` and `splits/seed42/eval_200/` contain the exact CSVs used during SFT/DPO (naming pattern `<model>_responses_<split>_seed42.csv`).
- `data/results/<dataset>/<subset>/<method>/`:
  - `<pair>.csv` is written by `scripts/disguise.py`, where `<pair>` is `<source>_as_<target>` with `/` replaced by `_`.
  - `scores/<pair>/` contains `scored.csv`, `scored_metrics.csv`, and `summary.json` emitted by `scripts.scorer.py`.
  - Legacy Chatbot Arena runs also mirror this under `data/results/chatbot_arena/comparisons/disguised_vs_target/<method>/`.
- `data/results/<dataset>/<workflow>/…` captures fine-tune outputs (e.g., `data/results/gsm8k/500/sft/...`), while `data/results/workflows/` stores logs from orchestrated runs.

## Prompt-based Disguise Options

| Prompt Method | What it does |
| --- | --- |
| `random_sampling` | Few-shot prompt of target answers. |
| `behavioral_based` | Builds persona / tone system prompt. |
| `stylistic` | Enforces formatting heuristics. |
| `contrastive` | Learns correction rules from src vs tgt pairs. |
| `stylistic_clustering` | Clusters target exemplars by formatting traits. |
| `embedding_clustering` | Embedding-based exemplar clusters per semantic regime. |

### Finetune Adapters
| Path | Summary | Entry point |
| --- | --- | --- |
| **SFT (LoRA)** | Runs Tinker + OpenAI supervised fine-tunes for the configured dataset/splits. | `workflows/pipeline.py` (instantiate `SFTWorkflowConfig` / `run_sft_workflow`) |
| **DPO** | Preference tuning stacked on the SFT adapters (OpenAI + Tinker variants). | `workflows/pipeline.py` (instantiate `DPOWorkflowConfig` / `run_dpo_workflow`) |

## Repo Layout (abridged)
```
scripts/
├── disguise.py           # Prompt-based disguises
├── generate_responses.py # Base generations (provider/HF/vLLM/Tinker)
├── scorer.py             # Pairwise + compare CLI
├── gsm8k/                # GSM8K-specific workflow runners, plotting, cleanup
├── tools/                # Maintenance utilities
└── methods/              # Method implementations + registry
data/
├── datasets/             # Prompts + style archetypes
└── model-responses/      # Generated baselines
└── results/              # Disguised outputs, scores, plots
workflows/                # SFT/DPO fine-tuning workflows
examples/                 # Example configurations
docs/                     # Additional documentation
```

## Inputs & Outputs
- **Prompts**: CSV with `prompt` column (e.g., `data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv`, `data/datasets/chatbot_arena/chatbot_arena_prompts.csv`). Plain-text lists still work for lightweight cases.
- **Base responses**: `data/model-responses/<dataset>/full/…`
- **Results / scores**: `data/results/<dataset>/comparisons/...`

Need more detail? Open `AGENTS.md`, `docs/local_generation.md`, or `workflows/README.md` depending on whether you’re coding, routing providers, or fine-tuning. Everything else lives in the scripts described above.
