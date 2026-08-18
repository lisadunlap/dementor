#!/usr/bin/env python
"""Imitation-erosion variance decomposition for the Dementor paper.

Computes the campaign headline statistics from the current, stage-separated
artifact. The metric is `mean_harm_erosion`: the per-adapter mean,
over the five harm-axis safety benchmarks, of (disguised harm - baseline harm).
Positive = the disguised (imitation) adapter has higher harmful compliance than
its own base model on the same prompts; negative = lower harmful compliance, not
necessarily greater overall safety.

We attribute the spread in erosion across adapters to three design factors via
a one-way eta-squared (SS_between / SS_total) per factor:

  * source  -- the base model that gets fine-tuned (the "disguising" model you
               start FROM); this equals `base_model` in the summary CSV
  * target  -- the model being imitated (whose behavior is copied)
  * dataset -- the imitation training corpus

The relative source/target shares are descriptive rather than a fixed expected
ordering: they are comparable after SFT and source is larger after DPO in the
completed campaign. `base_model` is constant per `source` and varies per `target`,
so the first slug in an adapter id is the fine-tuned base and the second is the
imitated target.

Usage
-----
    python experiments/imitation_safety/variance_decomp.py

Reads  data/results/safety/erosion_seed42_summary.csv
Writes data/results/safety/erosion_variance_stats.json (DPO default) or a stage-specific output.
Pure pandas/numpy; no plotting, no side effects beyond the JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root is two levels up from this file (experiments/imitation_safety/..).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
SUMMARY_CSV = REPO_ROOT / "data" / "results" / "safety" / "erosion_seed42_summary.csv"
LONG_CSV = REPO_ROOT / "data" / "results" / "safety" / "erosion_seed42_long.csv"
OUT_JSON = REPO_ROOT / "data" / "results" / "safety" / "erosion_variance_stats.json"

METRIC = "mean_harm_erosion"
FACTORS = ["source", "target", "dataset"]
ERODER_THRESHOLD = 0.01  # +1 percentage point; selected dynamically, never by model name
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPS = 10_000


def portable_path(path: Path) -> str:
    """Prefer a repository-relative artifact path over a machine-specific checkout path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def eta_squared(df: pd.DataFrame, factor: str, metric: str) -> float:
    """One-way eta^2 = SS_between / SS_total for `metric` grouped by `factor`."""
    y = df[metric].to_numpy(dtype=float)
    grand_mean = y.mean()
    ss_total = float(((y - grand_mean) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    # SS_between = sum over groups of n_g * (group_mean - grand_mean)^2.
    group_means = df.groupby(factor)[metric].mean()
    group_sizes = df.groupby(factor)[metric].size()
    ss_between = float((group_sizes * (group_means - grand_mean) ** 2).sum())
    return ss_between / ss_total


def source_cluster_mean_ci(df: pd.DataFrame, metric: str, *, seed=BOOTSTRAP_SEED) -> list[float]:
    """Percentile CI from resampling the 12 source-model means with replacement."""
    source_means = df.groupby("source")[metric].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(source_means, size=(BOOTSTRAP_REPS, len(source_means)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(draws, [2.5, 97.5])]


def eroder_absolute_harm(long_df: pd.DataFrame, source: str) -> dict:
    """Mean absolute harm (baseline vs disguised) over harm-axis benchmarks.

    The summary CSV only carries the erosion delta; the absolute levels live in
    the long CSV's `metric_baseline` / `metric_disguised` columns. We average
    over the harm-axis rows for the given source, matching how
    `mean_harm_erosion` is built.
    """
    sub = long_df[(long_df["axis"] == "harm") & (long_df["source"] == source)]
    baseline = float(sub["metric_baseline"].mean())
    disguised = float(sub["metric_disguised"].mean())
    return {
        "n_adapters": int(sub["adapter"].nunique()),
        "baseline_harm": baseline,
        "disguised_harm": disguised,
        "erosion": disguised - baseline,
    }


def ensure_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a validated SFT/DPO stage column.

    Legacy CSVs predate the explicit column, but their adapter ids still encode the
    rung. Keeping the backfill here makes old artifacts readable without allowing a
    new rebuild to pool both rungs silently.
    """
    out = df.copy()
    inferred = out["adapter"].astype(str).str.split("_", n=1).str[0]
    if "stage" not in out:
        out["stage"] = inferred
    elif not out["stage"].fillna("").eq(inferred).all():
        raise ValueError("stage column disagrees with adapter id")
    invalid = sorted(set(out["stage"]) - {"sft", "dpo"})
    if invalid:
        raise ValueError(f"invalid stages in erosion CSV: {invalid}")
    return out


def paired_stage_delta(df: pd.DataFrame) -> dict:
    """Exact-cell SFT→DPO change in mean harm erosion.

    Pairing keys include dataset, source, target, and seed. The evaluation pipeline
    fixes the prompt sampler for both stages, so these are the same campaign prompts.
    """
    df = df[df["source"] != df["target"]].copy()
    keys = ["dataset", "source", "target", "seed"]
    cols = keys + [METRIC]
    sft = df[df["stage"] == "sft"][cols].rename(columns={METRIC: "sft"})
    dpo = df[df["stage"] == "dpo"][cols].rename(columns={METRIC: "dpo"})
    paired = sft.merge(dpo, on=keys, how="inner", validate="one_to_one")
    delta = paired["dpo"] - paired["sft"]
    paired = paired.assign(delta=delta)
    return {
        "definition": "DPO erosion minus matching SFT erosion",
        "pair_keys": keys,
        "n_pairs": int(len(paired)),
        "mean_delta": float(delta.mean()) if len(delta) else None,
        "source_cluster_bootstrap_95ci": (
            source_cluster_mean_ci(paired, "delta") if len(delta) else None
        ),
        "median_delta": float(delta.median()) if len(delta) else None,
        "pct_dpo_more_erosive": float(100.0 * (delta > 0).mean()) if len(delta) else None,
    }


def _origin_slope(target_gap: np.ndarray, adapter_change: np.ndarray,
                  weights: np.ndarray | None = None) -> float:
    """Slope through the source origin: 0 stays at source; 1 reaches target."""
    if weights is None:
        weights = np.ones(len(target_gap), dtype=float)
    denominator = float(np.sum(weights * target_gap * target_gap))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(weights * target_gap * adapter_change) / denominator)


def _crossed_target_bootstrap(cells: pd.DataFrame, *, seed: int, reps: int) -> dict:
    """Pigeonhole bootstrap over the crossed source, target, and dataset axes.

    Matrix cells are not independent: every source, target, and training corpus is
    reused.  Independent multinomial weights on those three design axes preserve
    that crossed dependence while retaining all five benchmark means inside each
    resampled cell.
    """
    if reps <= 0:
        return {"reps": 0, "seed": seed, "intervals": {}}

    levels = {
        column: sorted(cells[column].unique())
        for column in ("source", "target", "dataset")
    }
    indices = {
        column: cells[column].map({value: i for i, value in enumerate(values)}).to_numpy()
        for column, values in levels.items()
    }
    target_gap = cells["target_gap"].to_numpy(dtype=float)
    adapter_change = cells["adapter_change"].to_numpy(dtype=float)
    distance_reduction = cells["target_distance_reduction"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = {
        "target_alignment_slope": [],
        "mean_target_distance_reduction": [],
        "mean_target_aligned_change": [],
        "safer_target_mean_adapter_change": [],
        "more_harmful_target_mean_adapter_change": [],
    }

    for _ in range(reps):
        weights = np.ones(len(cells), dtype=float)
        for column, values in levels.items():
            counts = np.bincount(
                rng.integers(len(values), size=len(values)), minlength=len(values)
            )
            weights *= counts[indices[column]]
        total = float(weights.sum())
        if total <= 0:
            continue
        slope = _origin_slope(target_gap, adapter_change, weights)
        if np.isfinite(slope):
            draws["target_alignment_slope"].append(slope)
        draws["mean_target_distance_reduction"].append(
            float(np.sum(weights * distance_reduction) / total)
        )
        draws["mean_target_aligned_change"].append(
            float(np.sum(weights * np.sign(target_gap) * adapter_change) / total)
        )
        for name, mask in (
            ("safer_target_mean_adapter_change", target_gap < 0),
            ("more_harmful_target_mean_adapter_change", target_gap > 0),
        ):
            stratum_weight = float(weights[mask].sum())
            if stratum_weight > 0:
                draws[name].append(
                    float(np.sum(weights[mask] * adapter_change[mask]) / stratum_weight)
                )

    intervals = {
        name: [float(value) for value in np.percentile(values, [2.5, 97.5])]
        for name, values in draws.items()
        if values
    }
    return {
        "method": "crossed source-target-dataset multinomial (pigeonhole) bootstrap",
        "reps": reps,
        "seed": seed,
        "intervals": intervals,
    }


def _joint_identity_target_bootstrap(cells: pd.DataFrame, *, seed: int, reps: int) -> dict:
    """Vertex bootstrap that resamples one model identity jointly in both dyad roles.

    The primary crossed bootstrap treats source-role and target-role effects as separate
    factors.  Here the same multinomial count is applied whenever a model appears as a
    source or target, so a directed cell receives ``count[source] * count[target]``.
    Dataset identities are still resampled independently.  This induced-dyad sensitivity
    check preserves correlations between a model's behavior in its two matrix roles.
    """
    method = "joint model-identity (directed-dyad vertex) and dataset multinomial bootstrap"
    if reps <= 0:
        return {"method": method, "reps": 0, "seed": seed, "intervals": {}}

    identities = sorted(set(cells["source"]) | set(cells["target"]))
    identity_index = {value: i for i, value in enumerate(identities)}
    source_index = cells["source"].map(identity_index).to_numpy()
    target_index = cells["target"].map(identity_index).to_numpy()
    datasets = sorted(cells["dataset"].unique())
    dataset_index = cells["dataset"].map(
        {value: i for i, value in enumerate(datasets)}
    ).to_numpy()
    target_gap = cells["target_gap"].to_numpy(dtype=float)
    adapter_change = cells["adapter_change"].to_numpy(dtype=float)
    distance_reduction = cells["target_distance_reduction"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = {
        "target_alignment_slope": [],
        "mean_target_distance_reduction": [],
        "mean_target_aligned_change": [],
        "safer_target_mean_adapter_change": [],
        "more_harmful_target_mean_adapter_change": [],
    }

    for _ in range(reps):
        identity_counts = np.bincount(
            rng.integers(len(identities), size=len(identities)),
            minlength=len(identities),
        )
        dataset_counts = np.bincount(
            rng.integers(len(datasets), size=len(datasets)),
            minlength=len(datasets),
        )
        weights = (
            identity_counts[source_index]
            * identity_counts[target_index]
            * dataset_counts[dataset_index]
        ).astype(float)
        total = float(weights.sum())
        if total <= 0:
            continue
        slope = _origin_slope(target_gap, adapter_change, weights)
        if np.isfinite(slope):
            draws["target_alignment_slope"].append(slope)
        draws["mean_target_distance_reduction"].append(
            float(np.sum(weights * distance_reduction) / total)
        )
        draws["mean_target_aligned_change"].append(
            float(np.sum(weights * np.sign(target_gap) * adapter_change) / total)
        )
        for name, mask in (
            ("safer_target_mean_adapter_change", target_gap < 0),
            ("more_harmful_target_mean_adapter_change", target_gap > 0),
        ):
            stratum_weight = float(weights[mask].sum())
            if stratum_weight > 0:
                draws[name].append(
                    float(np.sum(weights[mask] * adapter_change[mask]) / stratum_weight)
                )

    intervals = {
        name: [float(value) for value in np.percentile(values, [2.5, 97.5])]
        for name, values in draws.items()
        if values
    }
    return {
        "method": method,
        "reps": reps,
        "valid_draws": {name: len(values) for name, values in draws.items()},
        "seed": seed,
        "intervals": intervals,
    }


def target_relative_safety(long_df: pd.DataFrame, stage: str, *,
                           bootstrap_reps: int = BOOTSTRAP_REPS,
                           bootstrap_seed: int = BOOTSTRAP_SEED) -> dict:
    """Measure adapter movement along the source-to-target safety gap.

    For every harm benchmark, ``target_gap = H_target - H_source`` and
    ``adapter_change = H_adapter - H_source`` use the same configured prompt
    sample and RTL metric.  The primary unit averages the five harm benchmarks
    within each dataset/source/target/seed cell before analysis.
    """
    staged = ensure_stage(long_df)
    harm = staged[
        (staged["stage"] == stage)
        & (staged["axis"] == "harm")
        & (staged["baseline_available"] == True)  # noqa: E712
        & (staged["source"] != staged["target"])
    ].copy()
    baseline_unique = harm.groupby(["source", "benchmark"])["metric_baseline"].nunique()
    if len(baseline_unique) == 0 or not baseline_unique.eq(1).all():
        raise ValueError("source baseline is not unique for every model and harm benchmark")
    target_baselines = (
        harm.groupby(["source", "benchmark"], as_index=False)["metric_baseline"].first()
        .rename(columns={"source": "target", "metric_baseline": "metric_target"})
    )
    rows = harm.merge(
        target_baselines, on=["target", "benchmark"], how="left", validate="many_to_one"
    )
    if rows["metric_target"].isna().any():
        raise ValueError("target baseline coverage is incomplete")
    rows["target_gap"] = rows["metric_target"] - rows["metric_baseline"]
    rows["adapter_change"] = rows["metric_disguised"] - rows["metric_baseline"]
    rows["target_distance_reduction"] = (
        rows["target_gap"].abs()
        - (rows["target_gap"] - rows["adapter_change"]).abs()
    )
    rows["target_aligned_change"] = np.sign(rows["target_gap"]) * rows["adapter_change"]

    keys = ["dataset", "source", "target", "seed"]
    benchmark_counts = rows.groupby(keys)["benchmark"].nunique()
    expected_benchmarks = int(rows["benchmark"].nunique())
    if expected_benchmarks == 0 or not benchmark_counts.eq(expected_benchmarks).all():
        raise ValueError("target-relative cells do not have uniform harm-benchmark coverage")
    cells = rows.groupby(keys, as_index=False)[["target_gap", "adapter_change"]].mean()
    # The paper's adapter-level safety metric first averages the five harm
    # benchmarks.  Recompute nonlinear distance after that average rather than
    # averaging five benchmark-specific absolute distances.
    cells["target_distance_reduction"] = (
        cells["target_gap"].abs()
        - (cells["target_gap"] - cells["adapter_change"]).abs()
    )
    cells["target_aligned_change"] = np.sign(cells["target_gap"]) * cells["adapter_change"]
    nonzero = cells[cells["target_gap"] != 0]

    def stratum(mask: pd.Series) -> dict:
        part = cells[mask]
        return {
            "n_cells": int(len(part)),
            "mean_target_gap": float(part["target_gap"].mean()) if len(part) else None,
            "mean_adapter_change": float(part["adapter_change"].mean()) if len(part) else None,
            "mean_target_distance_reduction": (
                float(part["target_distance_reduction"].mean()) if len(part) else None
            ),
        }

    per_benchmark = {}
    for benchmark, part in rows.groupby("benchmark"):
        per_benchmark[benchmark] = {
            "n_rows": int(len(part)),
            "target_alignment_slope": _origin_slope(
                part["target_gap"].to_numpy(dtype=float),
                part["adapter_change"].to_numpy(dtype=float),
            ),
            "mean_target_distance_reduction": float(part["target_distance_reduction"].mean()),
            "pct_rows_closer_to_target": float(
                100.0 * (part.loc[part["target_gap"] != 0, "target_distance_reduction"] > 0).mean()
            ),
        }

    return {
        "definition": {
            "target_gap": "target base harm minus source base harm",
            "adapter_change": "adapter harm minus source base harm",
            "target_alignment_slope": (
                "origin-constrained slope of adapter_change on target_gap; "
                "0 stays at source and 1 reaches target"
            ),
            "target_distance_reduction": (
                "absolute source-target gap minus absolute adapter-target gap; positive is closer"
            ),
        },
        "stage": stage,
        "n_benchmark_rows": int(len(rows)),
        "n_cells": int(len(cells)),
        "harm_benchmarks": expected_benchmarks,
        "target_alignment_slope": _origin_slope(
            cells["target_gap"].to_numpy(dtype=float),
            cells["adapter_change"].to_numpy(dtype=float),
        ),
        "mean_absolute_target_gap": float(cells["target_gap"].abs().mean()),
        "mean_target_aligned_change": float(cells["target_aligned_change"].mean()),
        "mean_target_distance_reduction": float(cells["target_distance_reduction"].mean()),
        "pct_nonzero_gap_cells_closer_to_target": float(
            100.0 * (nonzero["target_distance_reduction"] > 0).mean()
        ),
        "pct_nonzero_gap_cells_moving_in_target_direction": float(
            100.0
            * (np.sign(nonzero["adapter_change"]) == np.sign(nonzero["target_gap"])).mean()
        ),
        "strata": {
            "safer_target": stratum(cells["target_gap"] < 0),
            "equal_mean_harm_target": stratum(cells["target_gap"] == 0),
            "more_harmful_target": stratum(cells["target_gap"] > 0),
        },
        "per_benchmark": per_benchmark,
        "bootstrap": _crossed_target_bootstrap(
            cells, seed=bootstrap_seed, reps=bootstrap_reps
        ),
        "joint_model_identity_bootstrap": _joint_identity_target_bootstrap(
            cells, seed=bootstrap_seed, reps=bootstrap_reps
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sft", "dpo"), default="dpo")
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--long-csv", type=Path, default=LONG_CSV)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="compute exploratory statistics before the configured campaign is complete",
    )
    args = parser.parse_args()

    all_df = ensure_stage(pd.read_csv(args.summary_csv))
    df = all_df[all_df["stage"] == args.stage].copy()
    df = df[df["baseline_available"] == True].copy()  # noqa: E712
    # Self-imitation is a matched-compute control, not an off-diagonal disguise cell.
    # The configured campaign enumerates source != target, but keep this filter so old
    # aggregate artifacts cannot leak self-controls into a headline rebuild.
    self_pairs = int((df["source"] == df["target"]).sum())
    if self_pairs:
        print(f"[filter] dropping {self_pairs} self-imitation cells (source == target)")
        df = df[df["source"] != df["target"]].copy()

    y = df[METRIC].to_numpy(dtype=float)
    n = int(len(df))

    # --- Variance decomposition ------------------------------------------
    variance_decomp = {f: eta_squared(df, f, METRIC) for f in FACTORS}

    # --- Headline distribution stats -------------------------------------
    overall_mean = float(np.mean(y))
    median = float(np.median(y))
    pct_high_eroders = float(100.0 * np.mean(y > 0.10))   # > +0.10
    pct_lower_harm = float(100.0 * np.mean(y < 0.0))
    pct_below_minus_1pp = float(100.0 * np.mean(y < -0.01))
    pct_at_least_5pp = float(100.0 * np.mean(y >= 0.05))
    max_erosion = float(np.max(y))
    min_erosion = float(np.min(y))

    # --- Per-source mean erosion (sorted ascending) ----------------------
    per_source = df.groupby("source")[METRIC].mean().sort_values()
    per_source_mean = {src: float(val) for src, val in per_source.items()}

    # --- Eroder absolute harm (from long CSV) ----------------------------
    long_df = ensure_stage(pd.read_csv(args.long_csv))
    long_df = long_df[long_df["stage"] == args.stage].copy()
    long_df = long_df[long_df["baseline_available"] == True].copy()  # noqa: E712
    eroder_sources = [src for src, value in per_source.items() if value > ERODER_THRESHOLD]
    eroders = {src: eroder_absolute_harm(long_df, src) for src in eroder_sources}

    from dementor import config
    n_models = len(config.campaign_roster())
    expected = (
        n_models
        * (n_models - 1)
        * len(config.campaign_dataset_names())
        * len(config.campaign_seeds())
    )
    if n != expected and not args.allow_partial:
        raise SystemExit(
            f"refusing partial {args.stage.upper()} analysis: {n}/{expected} campaign cells; "
            "finish the coverage audit or pass --allow-partial for exploration"
        )

    stats = {
        "metric": METRIC,
        "stage": args.stage,
        "input_csv": portable_path(args.summary_csv),
        "n_adapters": n,
        "expected_campaign_adapters": expected,
        "coverage_complete": n == expected,
        "variance_decomposition_eta2": variance_decomp,
        "variance_decomposition_pct": {
            f: 100.0 * v for f, v in variance_decomp.items()
        },
        "overall_mean_erosion": overall_mean,
        "source_cluster_bootstrap_95ci": source_cluster_mean_ci(df, METRIC),
        "median_erosion": median,
        "max_erosion": max_erosion,
        "min_erosion": min_erosion,
        "pct_adapters_gt_0.10": pct_high_eroders,
        "pct_adapters_lower_harm": pct_lower_harm,
        "pct_adapters_below_minus_1pp": pct_below_minus_1pp,
        "pct_adapters_at_least_5pp": pct_at_least_5pp,
        "mean_over_refusal_delta": float(df["mean_over_refusal_delta"].mean()),
        "per_source_mean_erosion_sorted": per_source_mean,
        "eroders": eroders,
        "eroder_threshold": ERODER_THRESHOLD,
        "paired_sft_to_dpo": paired_stage_delta(
            all_df[all_df["baseline_available"] == True].copy()  # noqa: E712
        ),
        "target_relative_safety": target_relative_safety(
            pd.read_csv(args.long_csv), args.stage
        ),
    }

    output = args.output or (OUT_JSON if args.stage == "dpo" else OUT_JSON.with_name(
        "erosion_variance_stats_sft.json"
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        json.dump(stats, fh, indent=2)

    # --- Clean console summary -------------------------------------------
    print("=" * 66)
    print(f"Imitation-erosion variance decomposition ({args.stage.upper()}, mean_harm_erosion)")
    print("=" * 66)
    print(f"n adapters (baseline_available): {n}")
    print()
    print("Variance decomposition  eta^2 = SS_between / SS_total")
    for f in FACTORS:
        print(f"  {f:<8s} {variance_decomp[f]:.4f}  ({100.0*variance_decomp[f]:5.1f}%)")
    print()
    print("Headline distribution stats")
    print(f"  overall mean erosion : {overall_mean:+.4f}")
    print(f"  median erosion       : {median:+.4f}")
    print(f"  max / min erosion    : {max_erosion:+.4f} / {min_erosion:+.4f}")
    print(f"  % adapters > +0.10   : {pct_high_eroders:.1f}%")
    print(f"  % adapters lower harm: {pct_lower_harm:.1f}%")
    print()
    print("Per-source mean erosion (sorted)")
    for src, val in per_source_mean.items():
        print(f"  {src:<20s} {val:+.4f}")
    print()
    print("Eroders: absolute harm (baseline -> disguised, harm-axis mean)")
    for src, e in eroders.items():
        print(
            f"  {src:<20s} {e['baseline_harm']:.4f} -> {e['disguised_harm']:.4f}"
            f"  (erosion {e['erosion']:+.4f}, n={e['n_adapters']})"
        )
    print()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
