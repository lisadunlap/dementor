from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .behavioral_inertia_metrics import bootstrap_behavioral_metrics, compute_behavioral_metrics
from .common import read_csv_robust, write_json
from .latent_behavior_axes import DEFAULT_DESCRIPTOR_ENCODER, FEATURE_SETS, plot_persistence_radar, run_latent_analysis


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
    feature_set: str = "full",
    save_basis: str | None = None,
    load_basis: str | None = None,
    k: int = 5,
    seed: int = 42,
    min_axis_separation: float = 0.10,
    min_probe_accuracy: float = 0.70,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
    baseline: float | None = None,
    weighted_baseline: float | None = None,
    active_axes: list[str] | None = None,
    allow_duplicate_prompts: bool = False,
    basis_type: str = "variance",
    lda_shrinkage: float = 0.1,
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
        feature_set=feature_set,
        save_basis_path=save_basis,
        load_basis_path=load_basis,
        k=k,
        seed=seed,
        allow_duplicate_prompts=allow_duplicate_prompts,
        basis_type=basis_type,
        lda_shrinkage=lda_shrinkage,
    )

    per_axis, metrics = compute_behavioral_metrics(
        scores_df,
        seed=seed,
        min_axis_separation=min_axis_separation,
        min_probe_accuracy=min_probe_accuracy,
        active_axes=active_axes,
        axis_weights=config.get("variance_explained"),
        baseline=baseline,
        weighted_baseline=weighted_baseline,
        feature_sep_score=config.get("sep_norm"),
        global_sep_threshold=config.get("global_sep_threshold", 0.10),
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
        "activation_source_prob": None,
    }
    if bootstrap_samples > 0:
        active_axes = per_axis.loc[per_axis["active_axis"], "axis"].astype(str).tolist()
        boot_df, boot_summary = bootstrap_behavioral_metrics(
            scores_df,
            active_axes=active_axes,
            axis_weights=config.get("variance_explained"),
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        boot_df.to_csv(out_dir / "bootstrap_summary.csv", index=False)
        summary.update(boot_summary)
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
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="full")
    parser.add_argument("--save-basis")
    parser.add_argument("--load-basis")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-axis-separation", type=float, default=0.10)
    parser.add_argument("--min-probe-accuracy", type=float, default=0.70)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--self-baseline-persistence", dest="baseline", type=float)
    parser.add_argument("--self-baseline-weighted-persistence", dest="weighted_baseline", type=float)
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
        feature_set=args.feature_set,
        save_basis=args.save_basis,
        load_basis=args.load_basis,
        k=args.k,
        seed=args.seed,
        min_axis_separation=args.min_axis_separation,
        min_probe_accuracy=args.min_probe_accuracy,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        baseline=args.baseline,
        weighted_baseline=args.weighted_baseline,
    )
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
