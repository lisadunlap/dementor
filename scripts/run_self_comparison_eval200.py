from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_PROMPTS_CSV = REPO_ROOT / "data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv"
OUTPUT_ROOT = REPO_ROOT / "data/results/gsm8k/eval200/self_comparisons"


def load_eval_prompts(path: Path) -> set[str]:
    df = pd.read_csv(path)
    if "prompt" not in df.columns:
        raise ValueError(f"{path} missing 'prompt' column")
    return set(df["prompt"].astype(str).tolist())


def filter_csv(source_csv: Path, prompts: set[str], dest_csv: Path) -> None:
    df = pd.read_csv(source_csv)
    if "prompt" not in df.columns:
        raise ValueError(f"{source_csv} missing 'prompt'")
    filtered = df[df["prompt"].astype(str).isin(prompts)].reset_index(drop=True)
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(dest_csv, index=False)
    print(f"Saved {len(filtered)} rows to {dest_csv}")


def score_with_judge(source_csv: Path, judge_model: str = "openai/gpt-4.1-mini") -> Path:
    score_dir = source_csv.parent / "scores" / source_csv.stem
    score_dir.mkdir(parents=True, exist_ok=True)
    scored_csv = score_dir / "scored.csv"
    subprocess.run(
        [
            "python",
            "scripts/scorer.py",
            "pairwise",
            "--input",
            str(source_csv),
            "--output",
            str(scored_csv),
            "--judge-model",
            judge_model,
        ],
        check=True,
    )
    return scored_csv


def compute_ci(values: Iterable[float]) -> Tuple[float, float]:
    vals = pd.Series(list(values)).dropna()
    if vals.empty:
        return 0.0, 0.0
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    ci = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, ci


def plot_results(results: List[Tuple[str, Path]]) -> None:
    metrics = ["semantic_score", "stylistic_score"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric in zip(axes, metrics):
        labels = []
        means = []
        cis = []
        for label, scored_path in results:
            df = pd.read_csv(scored_path)
            mean, ci = compute_ci(df[metric])
            labels.append(label)
            means.append(mean)
            cis.append(ci)
        ax.bar(labels, means, yerr=cis, capsize=6, color=["#4c72b0", "#dd8452"])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 4)
        ax.set_ylabel("Score (1-4)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("Self-Comparison (95% CI, eval-200 subset)")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / "self_comparison_ci.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote plot to {out_path}")


def main() -> None:
    prompts = load_eval_prompts(EVAL_PROMPTS_CSV)
    cases = [
        (
            REPO_ROOT / "data/results/gsm8k/500_archive_20251107-1941/gpt4.1_run1_vs_run2.csv",
            OUTPUT_ROOT / "gpt-4.1-mini_run1_vs_run2_eval200.csv",
            "gpt-4.1-mini run1 vs run2",
        ),
        (
            REPO_ROOT
            / "data/results/gsm8k/500_archive_20251107-1941/comparisons/self/meta-llama_Meta-Llama-3.1-8B-Instruct_run1_vs_run2/scored.csv",
            OUTPUT_ROOT / "llama-3.1-8b_run1_vs_run2_eval200.csv",
            "llama-3.1-8B run1 vs run2",
        ),
    ]
    scored_paths: List[Tuple[str, Path]] = []
    for source_csv, dest_csv, label in cases:
        filter_csv(source_csv, prompts, dest_csv)
        scored_path = score_with_judge(dest_csv)
        scored_paths.append((label, scored_path))
    plot_results(scored_paths)


if __name__ == "__main__":
    main()
