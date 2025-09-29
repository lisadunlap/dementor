"""Utility to visualize disguise evaluation metrics for GSM8K 500-run summaries.

This script loads semantic and stylistic score statistics from summary.json files
and produces a grouped bar chart comparing the means with standard deviation error
bars across disguise methods. The resulting figure is saved as a PNG image.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go


@dataclass
class MetricSummary:
    """Holds aggregated semantic and stylistic score statistics.

    Attributes
    ----------
    semantic_mean: float
        Mean semantic score taken from the summary file.
    semantic_std: float
        Semantic score standard deviation from the summary file.
    stylistic_mean: float
        Mean stylistic score taken from the summary file.
    stylistic_std: float
        Stylistic score standard deviation from the summary file.
    """

    semantic_mean: float
    semantic_std: float
    stylistic_mean: float
    stylistic_std: float


SUMMARY_PATHS: Dict[str, Path] = {
    "base (gpt-4.1 vs llama 3 8b)": Path(
        "data/results/gsm8k/500/scores/meta-llama_vs_gpt4.1/summary.json"
    ),
    "vibe_based": Path(
        "data/results/gsm8k/500/vibe_based/scores/"
        "openai_gpt-4.1_as_meta-llama_Meta-Llama-3-8B-Instruct/summary.json"
    ),
    "random_sampling": Path(
        "data/results/gsm8k/500/random_sampling/scores/"
        "openai_gpt-4.1_as_meta-llama_Meta-Llama-3-8B-Instruct/summary.json"
    ),
    "contrastive_with_al_examples": Path(
        "data/results/gsm8k/500/contrastive_with_al_examples/scores/"
        "openai_gpt-4.1_as_meta-llama_Meta-Llama-3-8B-Instruct/summary.json"
    ),
}


def load_metric_summary(path: Path) -> MetricSummary:
    """Return the metric summary for a single disguise method.

    Parameters
    ----------
    path: Path
        Absolute or repository-relative path to a summary.json file that contains
        ``semantic_score_mean``, ``semantic_score_std``, ``stylistic_score_mean``,
        and ``stylistic_score_std`` under a top-level ``metrics`` key.
    """

    with path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)["metrics"]

    return MetricSummary(
        semantic_mean=float(metrics["semantic_score_mean"]),
        semantic_std=float(metrics["semantic_score_std"]),
        stylistic_mean=float(metrics["stylistic_score_mean"]),
        stylistic_std=float(metrics["stylistic_score_std"]),
    )


def build_figure(summaries: Dict[str, MetricSummary]) -> go.Figure:
    """Create a grouped bar chart comparing semantic and stylistic scores."""

    methods: List[str] = list(summaries.keys())
    semantic_means = [summaries[label].semantic_mean for label in methods]
    semantic_stds = [summaries[label].semantic_std for label in methods]
    stylistic_means = [summaries[label].stylistic_mean for label in methods]
    stylistic_stds = [summaries[label].stylistic_std for label in methods]

    fig = go.Figure()
    fig.add_bar(
        name="Semantic",
        x=methods,
        y=semantic_means,
        error_y=dict(type="data", array=semantic_stds, visible=True),
        offsetgroup="semantic",
    )
    fig.add_bar(
        name="Stylistic",
        x=methods,
        y=stylistic_means,
        error_y=dict(type="data", array=stylistic_stds, visible=True),
        offsetgroup="stylistic",
    )

    fig.update_layout(
        title="GSM8K-500 Disguise Evaluation Scores",
        barmode="group",
        xaxis_title="Method",
        yaxis_title="Score (mean ± std)",
        legend_title="Metric",
        template="plotly_white",
    )

    return fig


def main() -> None:
    """Generate the grouped bar chart and write it to figures/gsm8k_disguise_scores.png."""

    summaries = {label: load_metric_summary(path) for label, path in SUMMARY_PATHS.items()}
    figure = build_figure(summaries)

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gsm8k_disguise_scores.png"
    figure.write_image(str(output_path), scale=2, width=900, height=600)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
