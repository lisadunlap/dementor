"""Push matrix-generated model responses to a HuggingFace dataset repo.

Bundles the train-split baselines + per-cell eval generations + cell summaries
(not the heavy, regenerable per-method latent/figure artifacts) so collaborators
get the exact generations without re-running Tinker. Re-runnable: upload_folder
updates the repo as more cells are produced.

Usage: python -m scripts.tools.push_matrix_dataset [--private]
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

REPO = "ethantsliu/dementor-matrix-responses"
DATA = Path("data")
BASELINES = DATA / "model-responses" / "matrix_baselines"

README = """---
license: mit
task_categories: [text-generation]
tags: [llm-imitation, behavioral-inertia, dementor]
---

# Dementor — matrix model responses

Generated model outputs for the Dementor LLM-imitation / behavioral-inertia study.
Companion to:
- **Code + prompt splits:** https://github.com/lisadunlap/dementor (branch `ethan`)
- **Trained adapters (228 LoRAs):** https://huggingface.co/ethantsliu (filter `sft_`/`dpo_`/`self_sft_`)

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

## Scope
4 models (llama-3.1-8b, gpt-oss-20b, qwen3.6-27b, nemotron-nano-30b-a3b);
datasets gsm8k / chatbot_arena / writingprompts (train 500 prompts, disjoint
held-out eval). Reproduce or extend any cell with
`scripts/analysis/run_cell_pipeline.py` from the repo.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", action="store_true", help="Create the dataset repo private (default public).")
    args = ap.parse_args()
    load_dotenv()
    from huggingface_hub import HfApi

    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage / "README.md").write_text(README, encoding="utf-8")

        n_baselines = 0
        for f in sorted(BASELINES.rglob("*_train.csv")):
            dst = stage / "baselines" / f.relative_to(BASELINES)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst)
            n_baselines += 1

        n_cells = 0
        for cell in sorted(DATA.glob("results/*/analysis/cells/*")):
            if not cell.is_dir():
                continue
            parts = cell.parts
            dataset = parts[parts.index("results") + 1]
            base = stage / "cells" / dataset / cell.name
            gen = cell / "gen"
            if gen.is_dir():
                (base / "gen").mkdir(parents=True, exist_ok=True)
                for f in gen.glob("*.csv"):
                    shutil.copy(f, base / "gen" / f.name)
            for extra in ("cell_summary.csv", "cell.json", "cell_evaluation_summary.json"):
                if (cell / extra).exists():
                    base.mkdir(parents=True, exist_ok=True)
                    shutil.copy(cell / extra, base / extra)
            n_cells += 1

        print(f"staged {n_baselines} baselines + {n_cells} cells; creating repo...", flush=True)
        api.create_repo(REPO, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(stage),
            repo_id=REPO,
            repo_type="dataset",
            commit_message="Add matrix baselines + cell generations + summaries",
        )
        vis = "private" if args.private else "public"
        print(f"pushed ({vis}) -> https://huggingface.co/datasets/{REPO}", flush=True)


if __name__ == "__main__":
    main()
