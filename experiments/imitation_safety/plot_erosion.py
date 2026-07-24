#!/usr/bin/env python
"""Consolidated figures for the imitation safety-erosion matrix.

Reads the aggregated CSVs written by build_erosion_csv.py:
  data/results/safety/erosion_<seed>_summary.csv   (one row per disguise adapter)
  data/results/safety/erosion_<seed>_long.csv       (one row per adapter x benchmark)

Writes publication-style figures to data/results/safety/figures/:
  erosion_heatmap.png        source x target mean harm erosion (the imitation matrix)
  erosion_by_source.png      per-source erosion + absolute baseline vs disguised harm
  erosion_by_dataset.png     erosion by fine-tuning dataset
  erosion_distribution.png   distribution of per-adapter erosion (null-centred check)
and prints a text results table.

Erosion = harm(disguised A_as_B) - harm(baseline A), mean over the 5 harm benchmarks.
Positive = safety ERODED; negative = model got SAFER under disguise.

Usage: plot_erosion.py [--seed seed42]
Pure matplotlib (no seaborn). Re-run after the eval completes for final figures.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erosion_common as EC

HARM_BENCHES = ["advbench", "harmbench", "strongreject", "sorrybench", "sgbench"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="seed42")
    args = ap.parse_args()
    base = EC.RESULTS_SAFETY
    figdir = os.path.join(base, "figures")
    os.makedirs(figdir, exist_ok=True)

    summ = pd.read_csv(os.path.join(base, f"erosion_{args.seed}_summary.csv"))
    summ = summ[summ.baseline_available == True].copy()
    lng = pd.read_csv(os.path.join(base, f"erosion_{args.seed}_long.csv"))
    lng = lng[lng.baseline_available == True].copy()
    he = "mean_harm_erosion"
    n = len(summ)
    print(f"[plot] {n} disguise adapters with baselines ({args.seed})")

    # ---- Fig 1: source x target erosion heatmap (mean over datasets) ------------------
    # Report in PERCENTAGE POINTS.  Two display fixes so genuinely-distinct cells don't read as
    # "repeated constants": (1) annotate at 0.1pp precision (2-decimal *fractions* collapsed a whole
    # safe-source row like aya's -2.5..-3.6pp to a single "-0.03"); (2) clip the colour scale to the
    # 95th percentile so one outlier cell (the gpt-oss-20b RTL false-positive, +15.6pp) doesn't wash
    # every other row into an indistinguishable shade.
    piv = summ.pivot_table(index="source", columns="target", values=he, aggfunc="mean") * 100.0  # -> pp
    row_order = piv.mean(axis=1).sort_values(ascending=False).index
    piv = piv.reindex(row_order)
    finite = np.abs(piv.values[np.isfinite(piv.values)])
    vmax = max(float(np.nanpercentile(finite, 95)) if finite.size else 5.0, 1.0)
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * piv.shape[1] + 4), 0.5 * piv.shape[0] + 3))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=9)
    ax.set_xlabel("target (imitated model)"); ax.set_ylabel("source (disguised model)")
    ax.set_title(f"Imitation safety-erosion matrix (mean over datasets, {args.seed})\n"
                 "red = safety eroded, blue = safer; erosion = harm(disguised) − harm(baseline), in pp", fontsize=10)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 * vmax else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="mean harm erosion (pp; colour clipped at 95th pct)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "erosion_heatmap.png"), dpi=150); plt.close(fig)

    # ---- Fig 2: per-source erosion + absolute baseline vs disguised harm --------------
    h = lng[lng.benchmark.isin(HARM_BENCHES)]
    g = h.groupby("source").agg(base=("metric_baseline", "mean"),
                                disg=("metric_disguised", "mean")).reset_index()
    src_err = summ.groupby("source")[he].agg(["mean", "std"]).reset_index()
    g = g.merge(src_err, on="source").sort_values("mean", ascending=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    y = np.arange(len(g))
    a1.barh(y, g["mean"], xerr=g["std"], color=["#c0392b" if v > 0.02 else "#7f8c8d" if v > -0.01 else "#2471a3" for v in g["mean"]])
    a1.axvline(0, color="k", lw=0.8); a1.set_yticks(y); a1.set_yticklabels(g["source"], fontsize=9)
    a1.invert_yaxis(); a1.set_xlabel("mean harm erosion (± SD over targets×datasets)")
    a1.set_title("Erosion is a SOURCE property")
    a2.barh(y - 0.2, g["base"], 0.4, label="baseline (unadapted)", color="#95a5a6")
    a2.barh(y + 0.2, g["disg"], 0.4, label="disguised (mean)", color="#e67e22")
    a2.set_yticks(y); a2.set_yticklabels(g["source"], fontsize=9); a2.invert_yaxis()
    a2.set_xlabel("absolute harm rate"); a2.set_title("Absolute harm: baseline vs disguised"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "erosion_by_source.png"), dpi=150); plt.close(fig)

    # ---- Fig 3: erosion by dataset ---------------------------------------------------
    ds = summ.groupby("dataset")[he].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ds["dataset"], ds["mean"], yerr=ds["std"], color="#5499c7")
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("mean harm erosion (± SD)")
    ax.set_title(f"Erosion by fine-tuning dataset ({args.seed})")
    for i, r in ds.iterrows():
        ax.text(i, r["mean"], f'n={int(r["count"])}', ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "erosion_by_dataset.png"), dpi=150); plt.close(fig)

    # ---- Fig 4: distribution of per-adapter erosion (null-centred check) --------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(summ[he], bins=40, color="#7f8c8d", edgecolor="white")
    ax.axvline(0, color="k", lw=1); ax.axvline(summ[he].mean(), color="#c0392b", lw=1.5, ls="--",
                                               label=f"mean {summ[he].mean():+.4f}")
    ax.set_xlabel("per-adapter mean harm erosion"); ax.set_ylabel("# disguise adapters")
    ax.set_title(f"Erosion is null-centred ({n} adapters; {100*(summ[he]>0.05).mean():.0f}% exceed +0.05, "
                 f"{100*(summ[he]<0).mean():.0f}% got safer)")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(figdir, "erosion_distribution.png"), dpi=150); plt.close(fig)

    # ---- text summary ----------------------------------------------------------------
    print(f"\n=== EROSION SUMMARY ({args.seed}, n={n}) ===")
    print(f"overall mean harm erosion: {summ[he].mean():+.4f} (median {summ[he].median():+.4f})")
    print(f">+0.05: {(summ[he]>0.05).sum()}  >+0.10: {(summ[he]>0.10).sum()}  got safer: {(summ[he]<0).sum()}")
    print("\nby source:\n", g[["source", "base", "disg", "mean", "std"]].round(4).to_string(index=False))
    print("\nby dataset:\n", ds.round(4).to_string(index=False))
    print(f"\n[plot] wrote 4 figures -> {figdir}")


if __name__ == "__main__":
    main()
