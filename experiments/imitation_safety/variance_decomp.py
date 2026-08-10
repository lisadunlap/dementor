#!/usr/bin/env python
"""Imitation-erosion variance decomposition for the Dementor paper.

Computes the campaign headline statistics from the current, stage-separated
artifact. The metric is `mean_harm_erosion`: the per-adapter mean,
over the five harm-axis safety benchmarks, of (disguised harm - baseline harm).
Positive = the disguised (imitation) adapter is MORE harmful than its own base
model on the same prompts; negative = it got safer.

We attribute the spread in erosion across adapters to three design factors via
a one-way eta-squared (SS_between / SS_total) per factor:

  * source  -- the base model that gets fine-tuned (the "disguising" model you
               start FROM); this equals `base_model` in the summary CSV
  * target  -- the model being imitated (whose behavior is copied)
  * dataset -- the imitation training corpus

The dominance of `source` is the paper's point: safety erosion tracks WHICH MODEL
YOU START FROM (the disguising base), NOT which model you imitate. Verified against
the CSV: `base_model` is constant per `source` and varies per `target`, so the
first slug in an adapter id is the fine-tuned base and the second is the imitated
target.

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
    return {
        "definition": "DPO erosion minus matching SFT erosion",
        "pair_keys": keys,
        "n_pairs": int(len(paired)),
        "mean_delta": float(delta.mean()) if len(delta) else None,
        "median_delta": float(delta.median()) if len(delta) else None,
        "pct_dpo_more_erosive": float(100.0 * (delta > 0).mean()) if len(delta) else None,
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
    pct_got_safer = float(100.0 * np.mean(y < 0.0))       # < 0
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
        "input_csv": str(args.summary_csv),
        "n_adapters": n,
        "expected_campaign_adapters": expected,
        "coverage_complete": n == expected,
        "variance_decomposition_eta2": variance_decomp,
        "variance_decomposition_pct": {
            f: 100.0 * v for f, v in variance_decomp.items()
        },
        "overall_mean_erosion": overall_mean,
        "median_erosion": median,
        "max_erosion": max_erosion,
        "min_erosion": min_erosion,
        "pct_adapters_gt_0.10": pct_high_eroders,
        "pct_adapters_got_safer": pct_got_safer,
        "per_source_mean_erosion_sorted": per_source_mean,
        "eroders": eroders,
        "eroder_threshold": ERODER_THRESHOLD,
        "paired_sft_to_dpo": paired_stage_delta(
            all_df[all_df["baseline_available"] == True].copy()  # noqa: E712
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
    print(f"  % adapters got safer : {pct_got_safer:.1f}%")
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
