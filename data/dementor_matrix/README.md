---
license: mit
task_categories: [text-generation]
tags: [llm-imitation, behavioral-inertia, dementor]
---

# Dementor — matrix model responses

Generated model outputs for the Dementor LLM-imitation / behavioral-inertia study.
Companion to:
- **Code + prompt splits:** https://github.com/lisadunlap/dementor (branch `ethan`)
- **Trained adapters (228 LoRAs):** https://huggingface.co/dementor-research (SFT / DPO / self-SFT, grouped into per-dataset collections)

## Layout
- `baselines/<dataset>/<model>_train.csv` — each base model's responses on the
  train split (used as SFT completions and as prompting example pools).
- `cells/<dataset>/<source>_to_<target>/`
  - `gen/` — eval-split generations: source & target baselines (×2 seeds),
    prompting-rung disguises (just_name_it / random_sampling / stylistic),
    and SFT/DPO adapter outputs (one file per seed).
  - `cell_summary.csv` — behavioral-inertia metrics per rung
    (`persistence`, `anchored`, `trustworthy`, `over_assimilation`, …).
  - `cell.json` — the cell manifest.
- `matrix_ladder/<dataset>/matrix_ladder.{csv,png}` — aggregated cross-cell
  persistence ladder (all 12 source→target pairs) + headline figure per dataset.

## Scope
4 models (llama-3.1-8b, gpt-oss-20b, qwen3.6-27b, nemotron-nano-30b-a3b);
datasets gsm8k / chatbot_arena / writingprompts (train 500 prompts, disjoint
held-out eval). Reproduce or extend any cell with
`scripts/analysis/run_cell_pipeline.py` from the repo.
