"""Survivor-enrichment significance for DPO behavioral persistence.

The README/paper cite "Fisher exact p ~ 0.008" for the claim that gpt-oss/nemotron
*source* models retain a DPO-resistant behavioral residue while qwen/llama launder to
the floor. No `fisher_exact` call existed anywhere in the repo, so that number was
unreproducible from committed code. This script reproduces it from the committed
``data/results/multiseed_ci_s3.csv`` and prints BOTH the optimistic and the honest test:

  (1) CELL-LEVEL 2x2 Fisher exact (survivor cell vs source-tier) -> p ~ 0.008
      Treats all 36 DPO cells as independent. PSEUDO-REPLICATED.
  (2) SOURCE-LEVEL exact permutation test over the 4 unique sources (effective n=4)
      -> p ~ 0.33. This is the honest inferential unit and the number to cite.

The cell-level test's "significance" is the n=4 two-tier source split mechanically
tripled across 3 datasets x 3 targets (9 non-independent cells per source), so its small
p-value overstates the evidence. The repo's own docs (docs/argument.md, docs/strategy.md)
already concede the honest model-level p is ~0.33.

Usage:
    python -m experiments.analysis.survivor_enrichment
    python -m experiments.analysis.survivor_enrichment --threshold 0.3
    python -m experiments.analysis.survivor_enrichment --help
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_CSV = Path("data/results/multiseed_ci_s3.csv")
# The two source models whose fingerprint survives DPO (vs qwen/llama -> ~0).
SURVIVOR_TIER = ("gpt-oss-20b", "nemotron-nano-30b-a3b")


def cell_level_fisher(dpo: pd.DataFrame, tier, threshold: float):
    """2x2 Fisher exact: survivor-tier source (rows) x survivor cell, mean>threshold (cols).

    Returns (table, two_sided_p, n_survivor_cells). `table` is
    [[tier&survivor, tier&non], [other&survivor, other&non]].
    """
    is_tier = dpo["source"].isin(tier)
    is_surv = dpo["mean"] > threshold
    table = [
        [int((is_tier & is_surv).sum()), int((is_tier & ~is_surv).sum())],
        [int((~is_tier & is_surv).sum()), int((~is_tier & ~is_surv).sum())],
    ]
    _odds, p = stats.fisher_exact(table, alternative="two-sided")
    n_surv = table[0][0] + table[1][0]
    return table, float(p), n_surv


def source_level_permutation(dpo: pd.DataFrame, tier):
    """Exact two-sided permutation test over the per-source mean DPO persistences.

    Statistic = mean(tier sources) - mean(other sources). Enumerate every C(n_src, k) way
    to label k of the sources as the "survivor tier" (k = len(tier)); the two-sided p is
    the fraction of labelings whose |statistic| >= |observed|. With 4 sources and a 2-2
    split there are C(4,2)=6 labelings, so the smallest attainable two-sided p is 2/6=0.33.

    Returns (per_source_mean_sorted, observed_stat, p, n_labelings).
    """
    src_mean = dpo.groupby("source")["mean"].mean()
    sources = list(src_mean.index)

    def stat(labeled) -> float:
        labeled = list(labeled)
        hi = src_mean[src_mean.index.isin(labeled)].mean()
        lo = src_mean[~src_mean.index.isin(labeled)].mean()
        return float(hi - lo)

    obs = stat(tier)
    labelings = list(itertools.combinations(sources, len(tier)))
    diffs = np.array([stat(c) for c in labelings])
    p = float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))
    return src_mean.sort_values(ascending=False), obs, p, len(labelings)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Survivor-enrichment significance (cell-level Fisher vs honest "
                    "source-level permutation) for DPO behavioral persistence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help="Multiseed CI table with a 'base_rung' column.")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="Point-estimate survivor cutoff on DPO persistence mean.")
    ap.add_argument("--tier", nargs="+", default=list(SURVIVOR_TIER),
                    help="Source model(s) forming the survivor tier.")
    a = ap.parse_args(argv)

    if not a.csv.exists():
        raise SystemExit(
            f"Missing {a.csv} — expected the committed multiseed CI table. "
            f"Run `python -m experiments.analysis.decontaminate --adapter-seeds 3` to (re)generate it."
        )

    df = pd.read_csv(a.csv)
    if "base_rung" not in df.columns:
        raise SystemExit(f"{a.csv} has no 'base_rung' column; got {list(df.columns)}.")
    dpo = df[df["base_rung"] == "dpo"].copy()
    if dpo.empty:
        raise SystemExit(f"No base_rung=='dpo' rows in {a.csv}.")
    tier = tuple(a.tier)
    n_src = dpo["source"].nunique()

    print(f"Loaded {len(dpo)} DPO cells from {a.csv} "
          f"({n_src} unique sources, {len(dpo) // max(n_src, 1)} cells each).")
    print(f"Survivor definition: DPO persistence mean > {a.threshold:g}.  "
          f"Survivor tier = {', '.join(tier)}.\n")

    # (1) Cell-level Fisher exact -------------------------------------------------------
    table, p_cell, n_surv = cell_level_fisher(dpo, tier, a.threshold)
    tier_label = "/".join(s.split("-")[0] for s in tier)
    print("(1) CELL-LEVEL 2x2 Fisher exact  [optimistic / PSEUDO-REPLICATED]")
    print(f"                                  survivor   non-survivor")
    print(f"      tier  ({tier_label:<20}) {table[0][0]:>6}   {table[0][1]:>11}")
    print(f"      other (qwen/llama)           {table[1][0]:>6}   {table[1][1]:>11}")
    print(f"      -> {n_surv} survivor cells, all in the tier row.  Fisher exact p = {p_cell:.4f}\n")

    # (2) Source-level exact permutation ------------------------------------------------
    src_mean, obs, p_src, n_lab = source_level_permutation(dpo, tier)
    print(f"(2) SOURCE-LEVEL exact permutation test  [HONEST backbone, effective n={n_src}]")
    for s, v in src_mean.items():
        marker = "  <- tier" if s in tier else ""
        print(f"      {s:24} mean DPO persistence = {v:.3f}{marker}")
    print(f"      statistic = mean(tier) - mean(other) = {obs:.3f};  "
          f"two-sided over C({n_src},{len(tier)})={n_lab} label splits")
    print(f"      -> permutation p = {p_src:.4f}\n")

    print(f"SUMMARY:  cell-level Fisher p = {p_cell:.4f}   |   source-level permutation p = {p_src:.4f}")
    print("CAVEAT: the cell-level test is PSEUDO-REPLICATED — 36 non-independent cells but only "
          f"{n_src} unique source values\n        ({len(dpo) // max(n_src, 1)} cells/source); cite the "
          "source-level p (effective n=4), not the 0.008.")


if __name__ == "__main__":
    main()
