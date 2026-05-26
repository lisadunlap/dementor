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

The paper analysis path uses Naz's adjective-matching latent analysis:
Big-Five + model-style descriptors are embedded with
`sentence-transformers/all-MiniLM-L6-v2`, responses are scored against those
descriptors, and joint SVD/probe metrics measure which behavioral axes move.
Activation steering is a local Transformers-hooks rung after Tinker adapter
export, not a Tinker remote-sampling feature.

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

## Directory Skeleton

The repository is split into reusable experiment code, workflow orchestration, paper planning docs, and data artifacts. Treat `scripts/`, `workflows/`, `docs/`, `AGENTS.md`, `METHODS.md`, and `examples/` as the source-of-truth code/docs layer. Treat most of `data/model-responses/` and `data/results/` as experiment artifacts.

```
dementor/
├── AGENTS.md                    # Agent-facing repo conventions and current project guidance
├── METHODS.md                   # Stable method names and the intervention ladder
├── README.md                    # Human-facing entry point
├── requirements.txt             # Core Python dependencies
├── main.py                      # Thin compatibility CLI; prefer direct scripts below
├── docs/
│   ├── conference_experiment_plan.md      # Framing and conference-level experiment plan
│   ├── experiment_implementation_plan.md  # Concrete matrix/model/run plan
│   └── local_generation.md                # Local HF/vLLM/provider generation notes
├── examples/
│   └── README.md                # Provider/model routing examples
├── scripts/
│   ├── generate_responses.py    # Base generations through provider, HF, vLLM, or Tinker
│   ├── disguise.py              # Prompt/intervention disguise runner
│   ├── scorer.py                # Single, pairwise, and compare scoring CLI
│   ├── make_matrix_splits.py    # Deterministic dataset split builder for matrix runs
│   ├── analysis/
│   │   ├── behavioral_inertia_metrics.py  # Behavioral axis movement / residual signature metrics
│   │   ├── latent_behavior_axes.py        # Naz-style adjective scoring + SVD axes
│   │   ├── activation_bridge.py           # Cross-model activation alignment utilities
│   │   ├── activation_steering.py         # Local Transformers steering hooks
│   │   └── run_intervention_ladder.py     # Aggregate prompting/SFT/DPO/steering outputs
│   ├── methods/
│   │   ├── base.py               # MethodBase contract
│   │   ├── get_method.py         # Method registry
│   │   ├── random_sampling.py    # Few-shot target exemplar prompting
│   │   ├── behavioral_based.py   # Persona/style prompt construction
│   │   ├── stylistic.py          # Surface-form style controls
│   │   └── contrastive.py        # Source-vs-target contrastive prompting
│   ├── gsm8k/                    # Legacy/convenience GSM8K shell runners and plots
│   ├── serve/                    # Shared local/server-side model utilities
│   ├── tests/                    # Lightweight regression tests and fixtures
│   ├── tools/                    # Adapter export, summaries, and artifact utilities
│   └── utils/                    # One-off CSV/data repair helpers
├── workflows/
│   ├── data.py                   # Shared train/eval artifact builders
│   ├── dpo.py                    # Preference-pair preparation
│   ├── openai.py                 # OpenAI fine-tuning backend
│   ├── tinker.py                 # Tinker LoRA/SFT/DPO backend and adapter registry
│   ├── pipeline.py               # Shared workflow orchestration helpers
│   ├── run_gsm8k_workflow.py     # Canonical GSM8K SFT/DPO entry point
│   ├── run_matrix.py             # Conference matrix runner
│   └── run_eval200_scoring.py    # Eval scoring orchestration
├── data/
│   ├── datasets/                 # Prompt splits and benchmark inputs
│   ├── model-responses/          # Base model responses; generated or imported
│   ├── recovered/                # Recovered Naz/BayLearn-era artifacts
│   ├── results/                  # Disguise outputs, scores, manifests, plots
│   └── tinker_adapters.json      # Local registry of Tinker adapter/sampler paths
├── figures/                      # Paper/report figures when materialized
└── cache/
    └── llm_cache/                # Persistent API cache; ignored by git
```

### What Belongs Where

- **New disguise method**: add implementation under `scripts/methods/`, register it in `scripts/methods/get_method.py`, and document the stable method name in `METHODS.md`.
- **New behavioral/latent metric**: add reusable code under `scripts/analysis/`; add a small fixture-backed test under `scripts/tests/` when the metric affects paper claims.
- **New full experiment run**: add orchestration to `workflows/` and keep the generated outputs under `data/results/`.
- **New model-response baseline**: write it under `data/model-responses/<dataset>/...`; commit only curated/provenance-critical baselines.
- **Generated matrix artifacts**: keep them local under `data/model-responses/matrix_baselines/`, `data/results/matrix/`, or `data/results/workflows/<run>/`. These are ignored by git while Tinker jobs are running.
- **External or recovered data**: keep provenance notes near the files, as in `data/results/call_center/MANIFEST.md`.

### Canonical Entry Points

- Base response generation: `python scripts/generate_responses.py ...`
- Prompt disguise: `python scripts/disguise.py ...`
- Scoring: `python -m scripts.scorer single|pairwise|compare ...`
- GSM8K SFT/DPO: `python -m workflows.run_gsm8k_workflow ...`
- Conference matrix: `python -m workflows.run_matrix ...`
- Behavioral inertia analysis: `python -m scripts.analysis.run_behavioral_inertia ...`
- Intervention ladder aggregation: `python -m scripts.analysis.run_intervention_ladder ...`

`main.py` remains for compatibility, but new automation should call the script/module entry points above directly.

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
