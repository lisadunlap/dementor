"""Plot stylometric ensemble summaries with confidence intervals."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


GPT_PROB_COL = "ensemble_prob_openai/gpt-4.1-mini"
LLAMA_PROB_COL = "ensemble_prob_meta-llama/Meta-Llama-3.1-8B-Instruct"
AGREEMENT_COL = "ensemble_agreement_count"

METHOD_ORDER = [
    "behavioral_based",
    "contrastive",
    "dpo",
    "embedding_clustering",
    "just_name_it",
    "random_sampling",
    "sft",
    "stylistic_clustering",
]

METHOD_LABELS = {
    "behavioral_based": "behavioral based",
    "contrastive": "contrastive",
    "dpo": "dpo",
    "embedding_clustering": "embedding clustering",
    "just_name_it": "just name it",
    "random_sampling": "random sampling",
    "sft": "sft",
    "stylistic_clustering": "stylistic clustering",
}


@dataclass(frozen=True)
class RunStats:
    label: str
    gpt_mean: float
    gpt_ci: float
    llama_mean: float
    llama_ci: float
    unanimous_mean: float
    unanimous_ci: float
    n: int


def _mean_ci(values: pd.Series, confidence: float) -> Tuple[float, float, int]:
    values = values.dropna()
    n = int(values.shape[0])
    if n == 0:
        return 0.0, 0.0, 0
    mean = float(values.mean())
    if n == 1:
        return mean, 0.0, n
    std = float(values.std(ddof=1))
    if std == 0.0:
        return mean, 0.0, n
    z_score = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    ci = z_score * (std / math.sqrt(n))
    return mean, ci, n


def _direction_from_source(source: str) -> Optional[str]:
    if "Meta-Llama-3.1-8B-Instruct" in source:
        return "Llama -> GPT"
    if "gpt-4.1-mini" in source:
        return "GPT -> Llama"
    return None


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def _load_run_stats(path: Path, confidence: float) -> Optional[RunStats]:
    df = pd.read_csv(path)
    if df.empty or GPT_PROB_COL not in df.columns or LLAMA_PROB_COL not in df.columns:
        return None
    if "source_model" not in df.columns or "method" not in df.columns:
        return None

    source = str(df["source_model"].iloc[0])
    direction = _direction_from_source(source)
    if direction is None:
        return None

    method = str(df["method"].iloc[0])
    label = f"{direction} ({_method_label(method)})"

    gpt_mean, gpt_ci, n = _mean_ci(df[GPT_PROB_COL], confidence)
    llama_mean, llama_ci, _ = _mean_ci(df[LLAMA_PROB_COL], confidence)
    unanimous = (df[AGREEMENT_COL] == 3).astype(float)
    unanimous_mean, unanimous_ci, _ = _mean_ci(unanimous, confidence)

    return RunStats(
        label=label,
        gpt_mean=gpt_mean,
        gpt_ci=gpt_ci,
        llama_mean=llama_mean,
        llama_ci=llama_ci,
        unanimous_mean=unanimous_mean,
        unanimous_ci=unanimous_ci,
        n=n,
    )


def _load_self_runs(input_dir: Path, confidence: float) -> List[RunStats]:
    runs = []
    for key, label in [("gpt", "GPT (self)"), ("llama", "Llama (self)")]:
        paths = sorted(input_dir.glob(f"self_{key}_run*.csv"))
        if not paths:
            continue
        frames = [pd.read_csv(path) for path in paths]
        df = pd.concat(frames, ignore_index=True)
        if df.empty or GPT_PROB_COL not in df.columns or LLAMA_PROB_COL not in df.columns:
            continue
        gpt_mean, gpt_ci, n = _mean_ci(df[GPT_PROB_COL], confidence)
        llama_mean, llama_ci, _ = _mean_ci(df[LLAMA_PROB_COL], confidence)
        unanimous = (df[AGREEMENT_COL] == 3).astype(float)
        unanimous_mean, unanimous_ci, _ = _mean_ci(unanimous, confidence)
        runs.append(
            RunStats(
                label=label,
                gpt_mean=gpt_mean,
                gpt_ci=gpt_ci,
                llama_mean=llama_mean,
                llama_ci=llama_ci,
                unanimous_mean=unanimous_mean,
                unanimous_ci=unanimous_ci,
                n=n,
            )
        )
    return runs


def _collect_runs(input_dir: Path, confidence: float) -> List[RunStats]:
    by_key: Dict[Tuple[str, str], RunStats] = {}
    for path in sorted(input_dir.glob("*.csv")):
        if path.name.startswith("self_"):
            continue
        stats = _load_run_stats(path, confidence)
        if stats is None:
            continue
        method = stats.label.split("(", 1)[-1].rstrip(")").replace(" ", "_")
        direction = stats.label.split(" (", 1)[0]
        by_key[(method, direction)] = stats

    ordered: List[RunStats] = []
    for method in METHOD_ORDER:
        for direction in ["GPT -> Llama", "Llama -> GPT"]:
            key = (method, direction)
            if key in by_key:
                ordered.append(by_key[key])
    ordered.extend(_load_self_runs(input_dir, confidence))
    return ordered


def _plot_probabilities(
    runs: Iterable[RunStats],
    output_path: Path,
    *,
    confidence: float,
) -> None:
    runs = list(runs)
    if not runs:
        raise ValueError("No runs available to plot.")

    labels = [run.label for run in runs]
    gpt_vals = [run.gpt_mean for run in runs]
    gpt_err_low = [min(run.gpt_ci, max(0.0, run.gpt_mean - 1e-6)) for run in runs]
    gpt_err_high = [min(run.gpt_ci, max(0.0, 1.0 - run.gpt_mean - 1e-6)) for run in runs]
    llama_vals = [run.llama_mean for run in runs]
    llama_err_low = [min(run.llama_ci, max(0.0, run.llama_mean - 1e-6)) for run in runs]
    llama_err_high = [min(run.llama_ci, max(0.0, 1.0 - run.llama_mean - 1e-6)) for run in runs]

    n_rows = len(runs)
    fig_height = max(4.5, 0.42 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    y_positions = list(range(n_rows))
    offset = 0.18

    ax.barh(
        [y - offset for y in y_positions],
        gpt_vals,
        height=0.32,
        color="#2f6ff0",
        label="GPT-4.1-mini",
    )
    ax.barh(
        [y + offset for y in y_positions],
        llama_vals,
        height=0.32,
        color="#e24a33",
        label="Llama-3.1-8B",
    )

    for y, mean, low, high in zip([y - offset for y in y_positions], gpt_vals, gpt_err_low, gpt_err_high):
        x_min = max(0.0, mean - low)
        x_max = min(0.97, mean + high)
        ax.hlines(y, x_min, x_max, colors="black", linewidth=1.0, clip_on=True)
    for y, mean, low, high in zip([y + offset for y in y_positions], llama_vals, llama_err_low, llama_err_high):
        x_min = max(0.0, mean - low)
        x_max = min(0.97, mean + high)
        ax.hlines(y, x_min, x_max, colors="black", linewidth=1.0, clip_on=True)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.margins(x=0)
    ax.set_xlim(0, 1.0)
    ax.margins(x=0)
    ax.set_xlabel(f"Average probability (mean ± {confidence * 100:.0f}% CI)")
    ax.set_title("Stylometric Ensemble Avg Probabilities")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        ncol=2,
    )

    for y, run in zip(y_positions, runs):
        ax.text(
            1.02,
            y - offset,
            f"GPT {run.gpt_mean:.3f} (n={run.n})",
            va="center",
            ha="left",
            fontsize=8,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )
        ax.text(
            1.02,
            y + offset,
            f"Llama {run.llama_mean:.3f} (n={run.n})",
            va="center",
            ha="left",
            fontsize=8,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.27, right=0.75, top=0.88)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_unanimous(
    runs: Iterable[RunStats],
    output_path: Path,
    *,
    confidence: float,
) -> None:
    runs = list(runs)
    if not runs:
        raise ValueError("No runs available to plot.")

    labels = [run.label for run in runs]
    values = [run.unanimous_mean for run in runs]
    err_low = [min(run.unanimous_ci, max(0.0, run.unanimous_mean - 1e-6)) for run in runs]
    err_high = [min(run.unanimous_ci, max(0.0, 1.0 - run.unanimous_mean - 1e-6)) for run in runs]

    n_rows = len(runs)
    fig_height = max(4.5, 0.35 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))

    y_positions = list(range(n_rows))

    ax.barh(
        y_positions,
        values,
        height=0.45,
        color="#2ecc71",
    )

    for y, mean, low, high in zip(y_positions, values, err_low, err_high):
        x_min = max(0.0, mean - low)
        x_max = min(0.97, mean + high)
        ax.hlines(y, x_min, x_max, colors="black", linewidth=1.0, clip_on=True)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.margins(x=0)
    ax.set_xlim(0, 1.0)
    ax.margins(x=0)
    ax.set_xlabel(f"Unanimous rate (mean ± {confidence * 100:.0f}% CI)")
    ax.set_title("Stylometric Ensemble Unanimous Agreement Rate")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for y, run in zip(y_positions, runs):
        ax.text(
            1.02,
            y,
            f"{run.unanimous_mean:.3f} (n={run.n})",
            va="center",
            ha="left",
            fontsize=8,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.27, right=0.75, top=0.88)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stylometric ensemble results with confidence intervals.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/results/gsm8k/eval200/stylometric_probs_ensemble"),
        help="Directory containing stylometric ensemble CSVs.",
    )
    parser.add_argument(
        "--output-probs",
        type=Path,
        default=Path("data/results/gsm8k/eval200/plots/stylometric_ensemble_probs.png"),
        help="Output path for probability plot.",
    )
    parser.add_argument(
        "--output-unanimous",
        type=Path,
        default=Path("data/results/gsm8k/eval200/plots/stylometric_ensemble_unanimous.png"),
        help="Output path for unanimous plot.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for error bars (default: 0.95).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if not 0 < args.confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")
    input_dir = args.input_dir.resolve()
    runs = _collect_runs(input_dir, args.confidence)
    _plot_probabilities(runs, args.output_probs, confidence=args.confidence)
    _plot_unanimous(runs, args.output_unanimous, confidence=args.confidence)


if __name__ == "__main__":
    main()
