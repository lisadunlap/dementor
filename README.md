# Dementor – Streamlined LLM Disguise

Minimal toolkit for stealing the “voice” of one LLM and applying it to another. Pair concise prompt-based methods with optional SFT/DPO adapters; all evaluation artifacts land under `data/results/`.

## Quick Start
```bash
pip install -r requirements.txt

# 1. Generate target-model responses
python scripts/generate_responses.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/model-responses/gsm8k/openai_gpt-4.1-mini.csv \
  basic --model openai/gpt-4.1-mini

# 2. Disguise the source model as the target
python scripts/disguise.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --disguise-as openai/gpt-4.1-mini \
  --method contrastive \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv
```

## Reference Docs
- `AGENTS.md` – repo guidance + coding conventions.
- `docs/conference_experiment_plan.md` – eight-step conference plan for behavioral-inertia experiments.
- `docs/local_generation.md` – HF/vLLM/provider routing (includes the local vLLM walkthrough).
- `workflows/README.md` – GSM8K SFT/DPO orchestration + Tinker adapter registry.
- `METHODS.md` – stable disguise method names and the intervention ladder.
- `examples/` – copy/paste provider configs.
- `scripts/gsm8k/` – automation for GSM8K workflows, cleanup, and plotting.

## Workflow in Four Steps
1. **Generate base responses** – `scripts/generate_responses.py` (provider/HF/vLLM/Tinker).
2. **Disguise** – `scripts/disguise.py`.
3. **Score** – `python -m scripts.scorer ...` for LLM judge + heuristics.
4. **Review outputs** – CSVs in `data/results/<dataset>/...`; cache lives in `cache/llm_cache/`.

## Disguise Options

| Prompt Method | What it does | When to use |
| --- | --- | --- |
| `random_sampling` | Few-shot prompt of target answers. | Fast baseline when target responses are clean. |
| `behavioral_based` | Builds persona / tone system prompt. | You need the target’s “voice.” |
| `stylistic` | Enforces formatting heuristics. | Rubric-heavy, surface-style benchmarks. |
| `contrastive` | Learns correction rules from src vs tgt pairs. | Models diverge sharply; need targeted edits. |
| `stylistic_clustering` | Clusters target exemplars by formatting traits. | Datasets with multiple style regimes. |
| `embedding_clustering` | Embedding-based exemplar clusters per semantic regime. | Mixed semantic tasks (math vs chit-chat). |

### Finetune Adapters
| Path | Summary | Entry point |
| --- | --- | --- |
| **SFT (LoRA)** | Tinker/OpenAI fine-tunes for GSM8K (300 train / 200 eval). | `python -m workflows.run_gsm8k_workflow --stage sft ...` |
| **DPO** | Preference tuning stacked on SFT adapters. | `python -m workflows.run_gsm8k_workflow --stage dpo ...` |

## Scoring Cheatsheet
```bash
# Single file (baseline quality)
python -m scripts.scorer single data/model-responses/<dataset>/full/<model>.csv \
  --output data/results/<dataset>/scores/<model>/scored.csv

# Pairwise (disguised vs target)
python -m scripts.scorer pairwise \
  --input data/results/<dataset>/comparisons/disguised_vs_target/<method>/<src>_as_<tgt>.csv \
  --output data/results/<dataset>/comparisons/disguised_vs_target/<method>/<src>_as_<tgt>_scored.csv \
  --judge-model openai/gpt-4.1-mini
```

## Repo Layout (abridged)
```
scripts/
├── disguise.py           # Prompt-based disguises
├── generate_responses.py # Base generations (provider/HF/vLLM/Tinker)
├── scorer.py             # Single, pairwise, compare CLIs
├── gsm8k/                # Workflow runners, plotting, cleanup
├── tools/                # Maintenance utilities
└── methods/              # Method implementations + registry
data/
├── datasets/             # Prompts + style archetypes
└── model-responses/      # Generated baselines
data/results/             # Disguised outputs, scores, plots
```

## Handy Commands
```bash
# Generate base responses (provider/HF/vLLM)
python scripts/generate_responses.py \
  --prompts-file data/datasets/chatbot_arena/chatbot_arena_prompts.csv \
  --output-csv data/model-responses/chatbot_arena/full/openai_gpt-4.1-mini.csv \
  basic --model openai/gpt-4.1-mini

# Contrastive disguise with custom prompts
python scripts/disguise.py \
  --model google/gemma-3-1b-it \
  --disguise-as openai/gpt-4.1-mini \
  --method contrastive \
  --prompts-file my_prompts.csv \
  --num-samples 200
```

## Inputs & Outputs
- **Prompts**: CSV with `prompt` column (`data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv`, `data/datasets/chatbot_arena/chatbot_arena_prompts.csv`). Plain-text lists still work for lightweight cases.
- **Base responses**: `data/model-responses/<dataset>/full/…`
- **Results / scores**: `data/results/<dataset>/comparisons/...`

Need more detail? Open `AGENTS.md`, `docs/local_generation.md`, or `workflows/README.md` depending on whether you’re coding, routing providers, or fine-tuning. Everything else lives in the scripts described above.
