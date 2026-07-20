#!/usr/bin/env python
"""Imitation-erosion variance decomposition for the Dementor paper.

Reproduces the headline "source ~79% / target ~3% / dataset ~0.2%" claim as a
one-command artifact. The metric is `mean_harm_erosion`: the per-adapter mean,
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
Writes data/results/safety/erosion_variance_stats.json
Pure pandas/numpy; no plotting, no side effects beyond the JSON artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root is two levels up from this file (experiments/imitation_safety/..).
REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = REPO_ROOT / "data" / "results" / "safety" / "erosion_seed42_summary.csv"
LONG_CSV = REPO_ROOT / "data" / "results" / "safety" / "erosion_seed42_long.csv"
OUT_JSON = REPO_ROOT / "data" / "results" / "safety" / "erosion_variance_stats.json"

METRIC = "mean_harm_erosion"
FACTORS = ["source", "target", "dataset"]
# The two source models that actually erode safety on average (the "eroders").
ERODER_SOURCES = ["ministral-8b", "granite-4-h-small"]


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


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["baseline_available"] == True].copy()  # noqa: E712

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
    long_df = pd.read_csv(LONG_CSV)
    long_df = long_df[long_df["baseline_available"] == True].copy()  # noqa: E712
    eroders = {src: eroder_absolute_harm(long_df, src) for src in ERODER_SOURCES}

    stats = {
        "metric": METRIC,
        "input_csv": str(SUMMARY_CSV.relative_to(REPO_ROOT)),
        "n_adapters": n,
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
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(stats, fh, indent=2)

    # --- Clean console summary -------------------------------------------
    print("=" * 66)
    print("Imitation-erosion variance decomposition (mean_harm_erosion)")
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
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
