"""Visualize GSM8K 300/200 finetuning evaluation scores.

This script aggregates the semantic and stylistic LLM-judge metrics stored in the
``scored_metrics.csv`` files that live under ``data/results/gsm8k/eval200`` and
renders a grouped bar chart comparing all 12 runs.  The visualization is written
as an interactive Plotly HTML document, and a tabular CSV summary is emitted for
quick reference.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import plotly.graph_objects as go

if __package__ is None or __package__ == "":
    # Ensure the repository root is on sys.path when executing as `python scripts/...`.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.visualize_scores import _format_pair as format_pair_label
from scripts.visualize_scores import _prettify_method as prettify_method


LOGGER = logging.getLogger(__name__)

# Keep the methods in a consistent order for easier visual scanning.
METHOD_ORDER: Tuple[str, ...] = (
    "behavioral_based",
    "random_sampling",
    "contrastive",
    "stylistic_clustering",
    "embedding_clustering",
)

EXCLUDED_METHODS = {"just_name_it"}


@dataclass
class RunMetrics:
    """Semantic and stylistic summary statistics for a single scoring run."""

    method: str
    method_label: str
    pair_id: str
    direction_label: str
    semantic_mean: float
    semantic_std: float
    semantic_count: int
    stylistic_mean: float
    stylistic_std: float
    stylistic_count: int
    path: Path


def _read_metrics_csv(path: Path) -> Dict[str, float]:
    """Return a mapping of metric -> value from a scored_metrics.csv file."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        payload = {row["metric"]: float(row["value"]) for row in reader}
    return payload


def _sort_key(method: str, pair_id: str) -> Tuple[int, str]:
    """Provide a stable ordering across methods and direction pairs."""

    try:
        method_rank = METHOD_ORDER.index(method)
    except ValueError:
        method_rank = len(METHOD_ORDER)
    return method_rank, pair_id


def _build_run_summary(path: Path, root: Path) -> RunMetrics:
    """Convert a single scored_metrics.csv path into a RunMetrics record."""

    rel_parts = path.relative_to(root).parts
    if len(rel_parts) < 3:
        raise ValueError(f"Unexpected scored_metrics location: {path}")
    method = rel_parts[0]
    pair_id = rel_parts[2]

    metrics = _read_metrics_csv(path)
    semantic_count = int(metrics["semantic_score_count"])
    stylistic_count = int(metrics["stylistic_score_count"])

    return RunMetrics(
        method=method,
        method_label=prettify_method(method),
        pair_id=pair_id,
        direction_label=format_pair_label(pair_id),
        semantic_mean=float(metrics["semantic_score_mean"]),
        semantic_std=float(metrics["semantic_score_std"]),
        semantic_count=semantic_count,
        stylistic_mean=float(metrics["stylistic_score_mean"]),
        stylistic_std=float(metrics["stylistic_score_std"]),
        stylistic_count=stylistic_count,
        path=path,
    )


def _collect_runs(root: Path, expected_count: int) -> List[RunMetrics]:
    """Load every scored_metrics.csv under root and sort them deterministically."""

    paths = sorted(root.rglob("scored_metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No scored_metrics.csv files found under {root}")

    runs = [_build_run_summary(path, root) for path in paths]
    runs = [run for run in runs if run.method not in EXCLUDED_METHODS]
    runs.sort(key=lambda item: _sort_key(item.method, item.pair_id))

    for run in runs:
        if expected_count and run.semantic_count != expected_count:
            LOGGER.warning(
                "Expected %d evals but %s reports %d (semantic)",
                expected_count,
                run.path,
                run.semantic_count,
            )
        if expected_count and run.stylistic_count != expected_count:
            LOGGER.warning(
                "Expected %d evals but %s reports %d (stylistic)",
                expected_count,
                run.path,
                run.stylistic_count,
            )
    return runs


def _runs_to_rows(runs: Iterable[RunMetrics]) -> List[Dict[str, object]]:
    """Convert RunMetrics structures into dictionaries for CSV export."""

    rows: List[Dict[str, object]] = []
    for run in runs:
        rows.append(
            {
                "method": run.method,
                "method_label": run.method_label,
                "direction": run.direction_label,
                "semantic_score_mean": run.semantic_mean,
                "semantic_score_std": run.semantic_std,
                "semantic_score_count": run.semantic_count,
                "stylistic_score_mean": run.stylistic_mean,
                "stylistic_score_std": run.stylistic_std,
                "stylistic_score_count": run.stylistic_count,
                "source_csv": str(run.path),
            }
        )
    return rows


def _write_summary_csv(rows: Sequence[Dict[str, object]], output_csv: Path) -> None:
    """Persist the aggregated metrics table to disk."""

    fieldnames = list(rows[0].keys())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_figure(runs: Sequence[RunMetrics]) -> go.Figure:
    """Return a grouped bar chart comparing semantic vs stylistic means."""

    x_labels = [
        f"{run.method_label}<br>{run.direction_label}"
        for run in runs
    ]
    semantic_means = [run.semantic_mean for run in runs]
    semantic_errors = [run.semantic_std for run in runs]
    stylistic_means = [run.stylistic_mean for run in runs]
    stylistic_errors = [run.stylistic_std for run in runs]

    fig = go.Figure()
    fig.add_bar(
        name="Semantic",
        x=x_labels,
        y=semantic_means,
        error_y=dict(type="data", array=semantic_errors, visible=True),
        marker_color="#1f77b4",
        offsetgroup="semantic",
    )
    fig.add_bar(
        name="Stylistic",
        x=x_labels,
        y=stylistic_means,
        error_y=dict(type="data", array=stylistic_errors, visible=True),
        marker_color="#ff7f0e",
        offsetgroup="stylistic",
    )

    fig.update_layout(
        title="GSM8K 300/200 Finetuning Scores (LLM Judge)",
        xaxis_title="Method • Direction",
        yaxis_title="Score (mean ± std)",
        barmode="group",
        template="plotly_white",
        legend_title="Metric",
        margin=dict(t=70, l=60, r=20, b=120),
    )
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GSM8K 300/200 finetuning scoring runs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/results/gsm8k/eval200"),
        help="Directory that contains the method/scores/.../scored_metrics.csv files.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("figures/gsm8k_eval200_finetune_scores.png"),
        help="Path for the static PNG visualization.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Optional path for an interactive Plotly HTML export.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("figures/gsm8k_eval200_finetune_scores.csv"),
        help="Where to write the aggregated summary CSV.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=200,
        help="Sanity-check that each run reports this many evals (warn if mismatched).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    runs = _collect_runs(args.root, args.expected_count)
    rows = _runs_to_rows(runs)
    _write_summary_csv(rows, args.output_csv)

    figure = _build_figure(runs)

    if args.output_image:
        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        figure.write_image(str(args.output_image), scale=2, width=1100, height=650)
        print(f"Saved PNG visualization to {args.output_image}")

    if args.output_html:
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(str(args.output_html), include_plotlyjs="cdn")
        print(f"Saved interactive visualization to {args.output_html}")

    print(f"Wrote summary table to {args.output_csv}")


if __name__ == "__main__":
    main()
