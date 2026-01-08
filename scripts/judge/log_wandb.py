#!/usr/bin/env python3
"""Log stylometric ensemble outputs to W&B.

Usage:
  WANDB_ENTITY=clipinvariance WANDB_PROJECT=dementor-judge \
  python scripts/judge/log_wandb.py
"""

from __future__ import annotations

import glob
import os

import pandas as pd
import wandb


def extract_method(base: str) -> str:
    candidates = []
    for marker in ["_openai", "_meta-llama"]:
        idx = base.find(marker)
        if idx != -1:
            candidates.append(idx)
    if candidates:
        return base[: min(candidates)]
    return base.split("_")[0]


def main() -> None:
    entity = os.environ.get("WANDB_ENTITY", "clipinvariance")
    project = os.environ.get("WANDB_PROJECT", "dementor-judge")

    run = wandb.init(
        project=project,
        entity=entity,
        name="gsm8k_stylometric_ensemble",
        job_type="analysis",
    )

    pngs = [
        "data/results/gsm8k/eval200/plots/stylometric_ensemble_probs.png",
        "data/results/gsm8k/eval200/plots/stylometric_ensemble_unanimous.png",
    ]
    for path in pngs:
        if os.path.exists(path):
            wandb.log({os.path.basename(path): wandb.Image(path)})

    rows = []
    files = glob.glob("data/results/gsm8k/eval200/stylometric_probs_ensemble/*.csv")
    for path in files:
        base = os.path.basename(path)
        if base.startswith("self_"):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue

        method = extract_method(base)
        source = str(df["source_model"].iloc[0]) if "source_model" in df.columns else ""
        if "Meta-Llama-3.1-8B-Instruct" in source:
            direction = "Llama -> GPT"
            target_label = "gpt"
        elif "gpt-4.1-mini" in source:
            direction = "GPT -> Llama"
            target_label = "llama"
        else:
            continue

        gpt_col = "ensemble_prob_openai/gpt-4.1-mini"
        llama_col = "ensemble_prob_meta-llama/Meta-Llama-3.1-8B-Instruct"
        if gpt_col not in df.columns or llama_col not in df.columns:
            continue

        avg_gpt = float(df[gpt_col].mean())
        avg_llama = float(df[llama_col].mean())
        target_prob = avg_gpt if target_label == "gpt" else avg_llama
        unanimous_rate = float((df["ensemble_agreement_count"] == 3).mean())

        rows.append(
            {
                "method": method,
                "direction": direction,
                "target_probability": target_prob,
                "unanimous_rate": unanimous_rate,
            }
        )

    if rows:
        table = wandb.Table(dataframe=pd.DataFrame(rows))
        wandb.log({"stylometric_summary": table})

    artifact = wandb.Artifact("stylometric_probs_ensemble", type="dataset")
    for path in files:
        artifact.add_file(path)
    wandb.log_artifact(artifact)

    wandb.finish()
    print(f"Logged to W&B: {project} / {entity}")


if __name__ == "__main__":
    main()
