#!/usr/bin/env python
"""Publication (AAAI two-column) steering figures, rebuilt from the recorded metrics.

Replaces four figures whose artwork asserted numbers that do not appear anywhere in the
repository (per-model "match rates" that no pipeline computes; gpt-oss labelled coupled at
-4.2 pp where we record it CLEAN; theta annotations claimed not to exist).  Every value
plotted here is read from a file:

  fig01_benchmark_dissociation  per-benchmark cone / fingerprint / random dHarm
                                <- rebuild_steering_tables.py per_benchmark_clean.csv
  fig02_model_orthogonality     per-model cos(fingerprint, refusal), 4 layers each
                                <- repl80_rdo/roster_geometry.csv
  fig03_steering_effects        per-model cone / fingerprint / random dHarm
                                <- rebuild_steering_tables.py per_model.csv
  fig04_positive_control        per-model refusal collapse under cone ablation
                                <- rebuild_steering_tables.py per_model.csv

AXIS DISCIPLINE follows rebuild_steering_tables.py: the harm axis reports dHarm as an
INCREASE in pp over baseline_harm; the refusal axis reports dRefusal as a DECREASE in pp
from baseline_refrate.  The two are never mixed in one panel.

Usage:
  rebuild_steering_tables.py --outdir TABLES
  plot_steering_paper.py --tables TABLES --outdir FIGDIR [--square-only]
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RDO = os.environ.get("DEMENTOR_RDO_DIR", "/data/ethantsliu/exp_steer_safety/repl80_rdo")

# The 13 models that are both a source and a target in the imitation square, in steering slugs.
SQUARE = ["ministral-8b", "nemotron-nano", "gemma-4-31b", "qwen3.6-35b", "gpt-oss-120b",
          "gemma-4-e4b", "qwen3.6-27b", "phi-4", "gpt-oss-20b", "olmo-3-7b", "qwen3.5-4b",
          "llama-3.1-8b", "aya-expanse-8b"]

HARM_BENCH = ["advbench", "harmbench", "strongreject", "sorrybench", "sgbench"]
OVERREF_BENCH = ["xstest", "orbench_toxic"]

# Categorical slots 1-3 of the validated palette.  One arm keeps one colour in EVERY figure --
# colour follows the entity, never its rank or its position in a sort.
# Checked with the palette validator (light surface, all-pairs): worst CVD dE 9.2, worst
# normal-vision dE 24.0, all gates PASS.  Aqua warns on contrast vs surface, which is why every
# aqua mark below also carries a direct value label.
C_CONE = "#2a78d6"        # slot 1 blue   -- refusal cone (the positive control)
C_FINGERPRINT = "#eb6834"  # slot 2 orange -- fingerprint direction (the arm under test)
C_RANDOM = "#1baf7a"       # slot 3 aqua   -- random direction (the null floor)
INK = "#1a1a19"
INK_MUTED = "#6b6a63"
GRID = "#d9d8d2"


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.6,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "text.color": INK, "axes.labelcolor": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "figure.dpi": 150,
    })


def _grid(ax, axis="x"):
    ax.grid(True, axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, path):
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


# ------------------------------------------------------------------ fig 01
def fig_benchmark(pb, path):
    """Per-benchmark dHarm for the three arms.

    One linear axis, no dual scale: the cone bar runs to ~60 pp while fingerprint and random
    sit under 2 pp, and that gap IS the finding.  The small arms are direct-labelled so they
    stay readable at the scale the cone forces.
    """
    d = pb[pb.benchmark.isin(HARM_BENCH + OVERREF_BENCH)].copy()
    d["order"] = d.benchmark.map({b: i for i, b in enumerate(HARM_BENCH + OVERREF_BENCH)})
    d = d.sort_values("order")
    y = np.arange(len(d))
    h = 0.26
    ARMS = [(h, C_CONE, "refusal cone (positive control)", "cone"),
            (0.0, C_FINGERPRINT, "fingerprint (under test)", "fingerprint"),
            (-h, C_RANDOM, "random direction (null)", "random")]
    # Two panels rather than one: at the scale the cone forces (~60 pp) the fingerprint and
    # random bars are sub-pixel, so the right panel replots just those two on their own scale.
    # Small multiples, NOT a second x-axis on the same plot.
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(3.4, 2.5), sharey=True,
                                  gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.08})
    for a in (ax, axz):
        _grid(a)
    for off, col, lab, arm in ARMS:
        v = d[f"dharm_pooled_{arm}_mean"].values
        e = d[f"dharm_pooled_{arm}_sd"].values
        ax.barh(y + off, v, h * 0.92, color=col, label=lab, zorder=3,
                xerr=e if arm == "cone" else None,
                error_kw=dict(ecolor=INK_MUTED, lw=0.5, capsize=1.5))
        if arm != "cone":
            axz.barh(y + off, v, h * 0.92, color=col, zorder=3)
            for yy, vv in zip(y + off, v):
                axz.text(vv + 0.06, yy, f"{vv:+.1f}", va="center", ha="left",
                         fontsize=5.0, color=INK_MUTED)
    for a in (ax, axz):
        a.axvline(0, color=INK_MUTED, lw=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([b.replace("orbench_toxic", "orbench") for b in d.benchmark], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel("$\\Delta$harm (pp)")
    axz.set_xlabel("zoom: $\\pm$2 pp")
    axz.set_xlim(-2.0, 2.9)
    axz.set_xticks([-2, 0, 2])
    ax.set_title("Ablating identity does not move harm; ablating refusal does", loc="left",
                 fontsize=7.5)
    n = int(d.dharm_pooled_cone_n.max())
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", frameon=False,
               handlelength=1.1, ncol=3, columnspacing=1.0,
               bbox_to_anchor=(0.5, -0.10))
    fig.text(0.5, -0.17, f"cone error bars are SD across models; n$\\leq${n} "
                         f"model$\\times$benchmark evals",
             ha="center", fontsize=5.2, color=INK_MUTED)
    save(fig, path)


# ------------------------------------------------------------------ fig 02
def fig_orthogonality(geo, path, square_only):
    """cos(fingerprint, refusal) per model.

    A dot per layer plus the model mean.  Diverging is the right job here (sign matters, zero
    is the reference), but with every value inside +-0.13 a single ink colour plus a zero line
    reads better than a two-hue ramp that would map noise to hue.
    """
    g = geo.copy()
    if square_only:
        g = g[g.slug.isin(SQUARE)]
    m = g.groupby("slug").fp_refusal.agg(["mean", "min", "max", "count"]).reset_index()
    m = m.sort_values("mean")
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(3.4, 0.16 * len(m) + 1.0))
    _grid(ax)
    ax.axvspan(-0.1, 0.1, color=GRID, alpha=0.45, zorder=1,
               label="$|\\cos|<0.1$ (near-orthogonal)")
    ax.hlines(y, m["min"], m["max"], color=INK_MUTED, lw=0.8, zorder=3)
    ax.scatter(m["mean"], y, s=14, color=C_FINGERPRINT, zorder=4, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=INK, lw=0.7, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(m.slug, fontsize=5.8)
    ax.set_xlabel("$\\cos$(fingerprint direction, refusal direction)")
    # State the claim at the strength the data supports: every LAYER MEAN is inside +-0.13, but
    # individual layers range wider, so "near-orthogonal in every model" would overclaim.
    ax.set_title(f"Layer-mean identity/refusal angle is near-orthogonal\n"
                 f"in all {len(m)} models (means {m['mean'].min():+.2f} to {m['mean'].max():+.2f})",
                 loc="left", fontsize=7.5)
    ax.legend(loc="upper left", frameon=False, handlelength=1.1)
    lo, hi = m["min"].min(), m["max"].max()
    fig.text(0.5, -0.01, f"dot = mean over {int(m['count'].iloc[0])} layers, "
                         f"line = per-layer range (widest {lo:+.2f} to {hi:+.2f})",
             ha="center", fontsize=5.2, color=INK_MUTED)
    save(fig, path)


# ------------------------------------------------------------------ fig 03
def fig_steering(pm, path, square_only):
    """Per-model dHarm for the three arms, sorted by cone effect."""
    d = pm.copy()
    if square_only:
        d = d[d.model.isin(SQUARE)]
    d = d.dropna(subset=["dharm_pooled_cone"]).sort_values("dharm_pooled_cone")
    y = np.arange(len(d))
    h = 0.26
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(3.4, 0.17 * len(d) + 1.3), sharey=True,
                                  gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.08})
    for a in (ax, axz):
        _grid(a)
    for off, col, lab, arm in [(h, C_CONE, "refusal cone", "cone"),
                               (0.0, C_FINGERPRINT, "fingerprint", "fingerprint"),
                               (-h, C_RANDOM, "random", "random")]:
        v = d[f"dharm_pooled_{arm}"].values
        ax.barh(y + off, v, h * 0.92, color=col, label=lab, zorder=3)
        if arm != "cone":
            axz.barh(y + off, v, h * 0.92, color=col, zorder=3)
    for a in (ax, axz):
        a.axvline(0, color=INK_MUTED, lw=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(d.model, fontsize=5.8)
    ax.set_xlabel("$\\Delta$harm (pp)")
    axz.set_xlabel("zoom: $\\pm$5 pp")
    axz.set_xlim(-5, 5)
    axz.set_xticks([-5, 0, 5])
    ax.set_title("Per model: the cone moves harm, the fingerprint does not", loc="left",
                 fontsize=7.5)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", frameon=False,
               handlelength=1.1, ncol=3, columnspacing=1.2, bbox_to_anchor=(0.5, -0.012))
    save(fig, path)


# ------------------------------------------------------------------ fig 04
def fig_positive_control(records, path, square_only):
    """The positive control: cone ablation collapses refusal from its baseline to near zero.

    REFUSAL axis only -- no harm number appears in this panel.

    Both endpoints are averaged over the SAME evals.  per_model.csv cannot be used here: its
    baseline_refrate_mean is averaged over every benchmark a model has, while its
    drefusal_pooled_cone is averaged over only the benchmarks that produced a cone arm (for
    deepseek-distill-8b, 12 vs 7).  Subtracting those two means put its post-ablation refusal
    at -11%, which is not a rate.  Pairing per eval keeps the difference meaningful.
    """
    rows = []
    for r in records:
        cone = r.get("arms", {}).get("cone")
        if not cone or r.get("baseline_refrate") is None or cone.get("refrate_pooled") is None:
            continue
        # A cell whose cone ablation collapsed coherence has no valid operating point: its
        # refusal rate falls because the model emits repetition, not because it complied.
        # Plotting that as a positive control would show the intervention "working" on a model
        # it merely destroyed.  refrate_pooled is NaN for these, which `is None` does not catch.
        if r.get("verdict") == "PC_INVALID" or cone["refrate_pooled"] != cone["refrate_pooled"]:
            continue
        rows.append({"model": r["model"],
                     "before": 100.0 * r["baseline_refrate"],
                     "after": 100.0 * cone["refrate_pooled"]})
    d = pd.DataFrame(rows)
    if square_only:
        d = d[d.model.isin(SQUARE)]
    d = d.groupby("model").agg(before=("before", "mean"), after=("after", "mean"),
                               n=("before", "size")).reset_index()
    d = d.sort_values("before")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(3.4, 0.19 * len(d) + 1.3))
    _grid(ax)
    ax.hlines(y, d["after"], d["before"], color=GRID, lw=2.2, zorder=2)
    ax.scatter(d["before"], y, s=13, color=INK_MUTED, zorder=4,
               label="unsteered baseline", edgecolor="white", linewidth=0.4)
    ax.scatter(d["after"], y, s=13, color=C_CONE, zorder=4,
               label="after refusal-cone ablation", edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(d.model, fontsize=5.8)
    ax.set_xlim(-2, 102)
    ax.set_xlabel("refusal rate (%)")
    med = (d["before"] - d["after"]).median()
    ax.set_title(f"Positive control: ablating the refusal cone collapses\n"
                 f"refusal (median {med:.0f} pp drop, {len(d)} models)", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", frameon=False,
               handlelength=1.1, ncol=2, columnspacing=1.5, bbox_to_anchor=(0.5, -0.012))
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True, help="outdir of rebuild_steering_tables.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--square-only", action="store_true",
                    help="restrict per-model panels to the 13-model imitation square")
    args = ap.parse_args()
    style()
    os.makedirs(args.outdir, exist_ok=True)
    pb = pd.read_csv(os.path.join(args.tables, "per_benchmark_clean.csv"))
    pm = pd.read_csv(os.path.join(args.tables, "per_model.csv"))
    geo = pd.read_csv(os.path.join(RDO, "roster_geometry.csv"))
    with open(os.path.join(args.tables, "records.json")) as fh:
        records = json.load(fh)

    fig_benchmark(pb, os.path.join(args.outdir, "fig01_benchmark_dissociation"))
    fig_orthogonality(geo, os.path.join(args.outdir, "fig02_model_orthogonality"), args.square_only)
    fig_steering(pm, os.path.join(args.outdir, "fig03_steering_effects"), args.square_only)
    fig_positive_control(records, os.path.join(args.outdir, "fig04_positive_control"),
                         args.square_only)

    stats = {
        "n_models_per_model_panels": int(pm.dropna(subset=["dharm_pooled_cone"]).shape[0]),
        "n_models_geometry": int(geo.slug.nunique()),
        "geometry_cos_range": [float(geo.fp_refusal.min()), float(geo.fp_refusal.max())],
        "benchmarks": pb.benchmark.tolist(),
    }
    with open(os.path.join(args.outdir, "steering_fig_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"[fig] stats -> {args.outdir}/steering_fig_stats.json")


if __name__ == "__main__":
    main()
