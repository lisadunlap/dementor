from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .behavioral_inertia_metrics import compute_behavioral_metrics
from .common import read_csv_robust, write_json
from .latent_behavior_axes import DEFAULT_DESCRIPTOR_ENCODER, plot_persistence_radar, run_latent_analysis


def run_behavioral_inertia(
    *,
    comparison_csv: str,
    output_dir: str,
    source_responses: str | None = None,
    source_model: str = "source",
    target_model: str = "target",
    method: str = "unknown",
    dataset: str = "unknown",
    source_col: str = "source_response",
    disguised_col: str = "model_response",
    target_col: str = "target_response",
    descriptor_mode: str = "big5_style",
    encoder_model: str = DEFAULT_DESCRIPTOR_ENCODER,
    k: int = 5,
    seed: int = 42,
    self_baseline_persistence: float | None = None,
) -> dict:
    out_dir = Path(output_dir)
    scores_df, _loads_df, config = run_latent_analysis(
        comparison_csv,
        output_dir=out_dir,
        source_responses=source_responses,
        source_model=source_model,
        target_model=target_model,
        method=method,
        dataset=dataset,
        source_col=source_col,
        disguised_col=disguised_col,
        target_col=target_col,
        descriptor_mode=descriptor_mode,
        encoder_model=encoder_model,
        k=k,
        seed=seed,
    )

    per_axis, metrics = compute_behavioral_metrics(
        scores_df,
        seed=seed,
        self_baseline_persistence=self_baseline_persistence,
    )
    per_axis.to_csv(out_dir / "per_axis_movement.csv", index=False)
    plot_persistence_radar(
        per_axis,
        out_dir / "figures" / "persistence_radar.png",
        title=f"Per-axis movement toward target\n{source_model} -> {target_model} | {method}",
    )

    summary = {
        **config,
        **metrics,
        "activation_bridge_mode": "none",
        "activation_source_probability_disguised": None,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run behavioral inertia analysis on one comparison CSV.")
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--source-responses")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-model", default="source")
    parser.add_argument("--target-model", default="target")
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--source-col", default="source_response")
    parser.add_argument("--disguised-col", default="model_response")
    parser.add_argument("--target-col", default="target_response")
    parser.add_argument("--descriptor-mode", choices=["big5_style", "style_only"], default="big5_style")
    parser.add_argument("--encoder-model", default=DEFAULT_DESCRIPTOR_ENCODER)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-baseline-persistence", type=float)
    args = parser.parse_args()

    summary = run_behavioral_inertia(
        comparison_csv=args.comparison_csv,
        source_responses=args.source_responses,
        output_dir=args.output_dir,
        source_model=args.source_model,
        target_model=args.target_model,
        method=args.method,
        dataset=args.dataset,
        source_col=args.source_col,
        disguised_col=args.disguised_col,
        target_col=args.target_col,
        descriptor_mode=args.descriptor_mode,
        encoder_model=args.encoder_model,
        k=args.k,
        seed=args.seed,
        self_baseline_persistence=args.self_baseline_persistence,
    )
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
