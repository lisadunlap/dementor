#!/usr/bin/env python
"""Publication (AAAI two-column) figures for the imitation safety-erosion matrix.

Sibling of plot_erosion.py.  plot_erosion.py emits 150-dpi PNGs at exploratory sizes; this
script emits vector PDFs sized for an AAAI two-column layout and adds the two panels the
paper's Figure 05 needs but plot_erosion.py never drew (target-conditioned erosion; the
categorical breakdown).  It deliberately PRESERVES the two display decisions documented in
plot_erosion.py's Fig-1 comment:

  * heatmap cells are annotated at 0.1 pp precision;
  * the diverging colour scale is clipped at the 95th percentile of |cell| so an outlier
    cannot wash every other row out.

Nothing else about the styling is invented: same RdBu_r map, same accent colours as
plot_erosion.py (#c0392b / #7f8c8d / #2471a3 / #5499c7 / #e67e22 / #95a5a6).

POPULATION: the configured core-12 campaign has 528 off-diagonal adapters per stage. SFT and DPO
are selected explicitly with ``--stage`` and are never pooled.

Outputs (to <outdir>, default $DEMENTOR_RESULTS_SAFETY/figures_paper):
  fig05a_erosion_matrix_wide.pdf/.png   configured-roster heatmap at \\textwidth (7.0in)
  fig05a_erosion_matrix_col.pdf         same at \\columnwidth (3.32in)
  fig05b_erosion_summary_wide.pdf/.png  4-panel summary at \\textwidth
  fig05b_erosion_summary_col.pdf        4-panel summary stacked 2x2 at \\columnwidth
  erosion_matrix_cells.csv              machine-readable heatmap cells (pp, 1 dp, n datasets)
  erosion_paper_stats.json              every number quoted in the captions

Usage:  plot_erosion_paper.py [--seed seed42] [--stage dpo|sft] [--outdir DIR]
Pure matplotlib, no GPU, no network.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erosion_common as EC

HE = "mean_harm_erosion"
COL_W, TEXT_W = 3.32, 7.0          # AAAI \columnwidth / \textwidth in inches
RED, GREY, BLUE = "#c0392b", "#7f8c8d", "#2471a3"
STEEL, ORANGE, SILVER = "#5499c7", "#e67e22", "#95a5a6"

# Categorical bins over per-adapter mean harm erosion, in PERCENTAGE POINTS.
# Half-open [lo, hi): every adapter lands in exactly one bin, the four counts sum to n.
BINS = [(-np.inf, -1.0, "safer", "$<\\!-1.0$", BLUE),
        (-1.0,     1.0, "flat", "$[-1.0,+1.0)$", GREY),
        (1.0,      5.0, "mild", "$[+1.0,+5.0)$", ORANGE),
        (5.0,   np.inf, "strong", "$\\geq\\!+5.0$", RED)]


def rcparams():
    plt.rcParams.update({
        # NB: savefig.bbox is left at "standard" on purpose -- "tight" re-expands the canvas past
        # the requested figsize, and these PDFs must drop into AAAI \columnwidth / \textwidth at
        # exactly 1.0 scale.  tight_layout() below does the label packing instead.
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "standard",
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


def complete_square(off):
    """Largest model subset that is BOTH a source and a target and whose off-diagonal
    sub-square is fully populated.  Greedy hole-peeling; deterministic on this data."""
    cnt = off.pivot_table(index="source", columns="target", values=HE, aggfunc="size")
    cur = sorted(set(off.source) & set(off.target))
    while True:
        sub = cnt.reindex(cur).reindex(columns=cur)
        holes = sub.isna().values.copy()
        np.fill_diagonal(holes, False)
        if not holes.any():
            return cur
        sc = {m: 0 for m in cur}
        for i, s in enumerate(cur):
            for j, t in enumerate(cur):
                if holes[i, j]:
                    sc[s] += 1
                    sc[t] += 1
        cur.remove(max(sc, key=lambda m: sc[m]))


def breakdown(pp):
    rows = []
    for lo, hi, lab, edge, c in BINS:
        m = (pp >= lo) & (pp < hi)
        rows.append({"bin": lab, "edge_label": edge, "lo_pp": None if np.isinf(lo) else lo,
                     "hi_pp": None if np.isinf(hi) else hi, "colour": c,
                     "n": int(m.sum()), "pct": round(float(100 * m.mean()), 1)})
    assert sum(r["n"] for r in rows) == len(pp)
    return rows


# --------------------------------------------------------------------------- Figure A
def fig_matrix(piv, cnt, path, width, cell_fs, tick_fs, annotate=True):
    finite = np.abs(piv.values[np.isfinite(piv.values)])
    vmax = max(float(np.nanpercentile(finite, 95)), 1.0)       # 95th-pct colour clip (preserved)
    h = width * (piv.shape[0] + 2.2) / (piv.shape[1] + 3.0)
    fig, ax = plt.subplots(figsize=(width, h))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=tick_fs)
    ax.set_xlabel("target (imitated model)")
    ax.set_ylabel("source (disguised, fine-tuned model)")
    if annotate:
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isnan(v):
                    ax.text(j, i, "--", ha="center", va="center", fontsize=cell_fs, color=GREY)
                    continue
                # 0.1 pp annotation precision (preserved); asterisk = single-dataset cell
                mark = "*" if cnt.values[i, j] == 1 else ""
                ax.text(j, i, f"{v:+.1f}{mark}", ha="center", va="center", fontsize=cell_fs,
                        color="white" if abs(v) > 0.6 * vmax else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("mean harm erosion (pp; colour clipped at 95th pct)", fontsize=tick_fs)
    cb.ax.tick_params(labelsize=tick_fs)
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)
    return vmax


# --------------------------------------------------------------------------- Figure B
def fig_summary(pop, path, width, layout):
    src = pop.groupby("source")["pp"].agg(["mean", "std", "count"]).sort_values("mean")
    tgt = pop.groupby("target")["pp"].agg(["mean", "std", "count"]).sort_values("mean")
    bd = breakdown(pop.pp)
    mu = float(pop.pp.mean())

    if layout == "wide":
        fig, axs = plt.subplots(2, 2, figsize=(width, 5.0))
        axs = axs.ravel()
    else:
        fig, axs = plt.subplots(4, 1, figsize=(width, 8.4))
    a1, a2, a3, a4 = axs
    tfs = 7.5 if layout == "wide" else 5.8   # panel-title size: \columnwidth needs smaller

    def barh(ax, t, title, xlab):
        y = np.arange(len(t))
        col = [RED if v > 0.5 else GREY if v > -0.5 else BLUE for v in t["mean"]]
        ax.barh(y, t["mean"], xerr=t["std"].fillna(0.0), color=col,
                error_kw=dict(lw=0.5, ecolor="#444444"))
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{m} ({int(n)})" for m, n in zip(t.index, t["count"])], fontsize=5.5)
        ax.set_ylim(-0.7, len(t) - 0.3)
        ax.set_xlabel(xlab)
        ax.set_title(title, loc="left", fontsize=tfs)

    barh(a1, src, "(a) erosion by SOURCE (disguised model)", "mean erosion (pp) $\\pm$ SD")
    barh(a2, tgt, "(b) erosion by TARGET (imitated model)", "mean erosion (pp) $\\pm$ SD")

    a3.hist(pop.pp, bins=40, color=GREY, edgecolor="white", linewidth=0.3)
    a3.axvline(0, color="k", lw=0.8)
    a3.axvline(mu, color=RED, lw=1.0, ls="--")
    a3.set_yscale("symlog", linthresh=10)
    a3.set_xlabel("per-adapter mean harm erosion (pp)")
    a3.set_ylabel("# adapters (symlog)")
    a3.set_title(f"(c) distribution of per-adapter erosion (n={len(pop)})", loc="left", fontsize=tfs)
    a3.text(0.97, 0.93, f"mean {mu:+.3f} pp (dashed)\nmedian {float(pop.pp.median()):+.2f} pp\n"
                        f"{100 * (pop.pp < 0).mean():.1f}% below zero",
            transform=a3.transAxes, ha="right", va="top", color=RED, fontsize=6)

    x = np.arange(len(bd))
    a4.bar(x, [b["pct"] for b in bd], color=[b["colour"] for b in bd], width=0.68)
    a4.set_xticks(x)
    a4.set_xticklabels([f'{b["bin"]}\n{b["edge_label"]}' for b in bd], fontsize=5.5)
    a4.set_ylabel("% of adapters")
    a4.set_title("(d) categorical breakdown (half-open pp bins)", loc="left", fontsize=min(tfs, 6.8))
    for xi, b in zip(x, bd):
        a4.text(xi, b["pct"] + 1.5, f'{b["pct"]:.1f}%  (n={b["n"]})', ha="center", fontsize=6)
    a4.set_ylim(0, max(b["pct"] for b in bd) * 1.22)

    fig.tight_layout(pad=0.6, w_pad=1.4, h_pad=1.2)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)
    return src, tgt, bd, mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="seed42")
    ap.add_argument("--stage", choices=("sft", "dpo"), default="dpo")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    base = EC.RESULTS_SAFETY
    out = args.outdir or os.path.join(
        base, "figures_paper" if args.stage == "dpo" else "figures_paper_sft"
    )
    os.makedirs(out, exist_ok=True)
    rcparams()

    df = pd.read_csv(os.path.join(base, f"erosion_{args.seed}_summary.csv"))
    if "stage" not in df:
        df["stage"] = df["adapter"].astype(str).str.split("_", n=1).str[0]
    df = df[df.stage == args.stage].copy()
    df = df[df.baseline_available == True].copy()
    df["pp"] = df[HE] * 100.0
    off = df[df.source != df.target].copy()
    square = complete_square(off)
    sq = off[off.source.isin(square) & off.target.isin(square)].copy()

    # ---- Figure A: the configured complete imitation matrix ------------------------
    piv = sq.pivot_table(index="source", columns="target", values="pp", aggfunc="mean")
    cnt = sq.pivot_table(index="source", columns="target", values="pp", aggfunc="size")
    order = piv.mean(axis=1).sort_values(ascending=False).index          # rows by mean erosion
    piv = piv.reindex(order).reindex(columns=order)                     # same order on both axes
    cnt = cnt.reindex(order).reindex(columns=order)
    vmax = fig_matrix(piv, cnt, os.path.join(out, "fig05a_erosion_matrix_wide"),
                      TEXT_W, cell_fs=9.0, tick_fs=9.0)
    fig_matrix(piv, cnt, os.path.join(out, "fig05a_erosion_matrix_col"),
               COL_W, cell_fs=4.2, tick_fs=5.0)

    cells = []
    for s in piv.index:
        for t in piv.columns:
            if s == t:
                cells.append(dict(source=s, target=t, erosion_pp="", n_datasets=0,
                                  cell_type="diagonal (self-imitation, omitted)"))
            else:
                c = int(cnt.loc[s, t])
                cells.append(dict(source=s, target=t, erosion_pp=round(float(piv.loc[s, t]), 1),
                                  n_datasets=c,
                                  cell_type={1: "single-dataset", 3: "3-dataset",
                                             4: "full 4-dataset"}.get(c, f"{c}-dataset")))
    pd.DataFrame(cells).to_csv(os.path.join(out, "erosion_matrix_cells.csv"), index=False)
    piv.round(1).to_csv(os.path.join(out, "erosion_matrix_values_pp.csv"))
    cnt.fillna(0).astype(int).to_csv(os.path.join(out, "erosion_matrix_n_datasets.csv"))

    # ---- Figure B: 4-panel summary over the off-diagonal roster --------------------
    src, tgt, bd, mu = fig_summary(off, os.path.join(out, "fig05b_erosion_summary_wide"),
                                   TEXT_W, "wide")
    fig_summary(off, os.path.join(out, "fig05b_erosion_summary_col"), COL_W, "col")

    cell_n = sq.groupby(["source", "target"]).size()
    stats = {
        "seed": args.seed,
        "stage": args.stage,
        "n_full_roster": len(df), "n_sources": int(df.source.nunique()),
        "n_targets": int(df.target.nunique()),
        "n_offdiagonal": len(off), "n_self_imitation_diagonal": len(df) - len(off),
        "mean_pp_full_roster": round(float(df.pp.mean()), 3),
        "mean_pp_offdiagonal": round(float(off.pp.mean()), 3),
        "median_pp_offdiagonal": round(float(off.pp.median()), 3),
        "sd_pp_offdiagonal": round(float(off.pp.std()), 3),
        "pct_below_zero_full_roster": round(float(100 * (df.pp < 0).mean()), 1),
        "pct_below_zero_offdiagonal": round(float(100 * (off.pp < 0).mean()), 1),
        "min_pp": float(df.pp.min()), "max_pp": float(df.pp.max()),
        "square_models": square, "n_square": len(square),
        "square_adapters": len(sq),
        "square_mean_pp": round(float(sq.pp.mean()), 3),
        "square_cells_filled": int(len(cell_n)),
        "square_cells_possible": len(square) * (len(square) - 1),
        "square_cells_single_dataset": int((cell_n == 1).sum()),
        "square_cells_three_dataset": int((cell_n == 3).sum()),
        "square_cells_four_dataset": int((cell_n == 4).sum()),
        "square_cells_below_four": int((cell_n < 4).sum()),
        "colour_clip_vmax_pp": round(vmax, 2),
        "outliers_gt_10pp": df[df.pp > 10].sort_values("pp", ascending=False)[
            ["source", "target", "dataset", "pp"]].round(1).to_dict("records"),
        "per_source_offdiagonal_pp": src.sort_values("mean", ascending=False).round(3)
            .reset_index().to_dict("records"),
        "per_target_offdiagonal_pp": tgt.sort_values("mean", ascending=False).round(3)
            .reset_index().to_dict("records"),
        "categorical_breakdown_offdiagonal": [{k: v for k, v in b.items() if k != "colour"}
                                              for b in bd],
        "aya_row_pp": {t: round(float(piv.loc["aya-expanse-8b", t]), 1)
                       for t in piv.columns if t != "aya-expanse-8b"},
    }
    stats["aya_row_min_pp"] = min(stats["aya_row_pp"].values())
    stats["aya_row_max_pp"] = max(stats["aya_row_pp"].values())
    with open(os.path.join(out, "erosion_paper_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=float)

    print(f"[paper] {len(df)} adapters | off-diagonal {len(off)} | "
          f"{len(square)}x{len(square)} square {len(sq)} adapters")
    print(f"[paper] mean erosion: full roster {stats['mean_pp_full_roster']:+.3f} pp, "
          f"off-diagonal {stats['mean_pp_offdiagonal']:+.3f} pp, "
          f"square {stats['square_mean_pp']:+.3f} pp")
    print(f"[paper] square cells {stats['square_cells_filled']}/"
          f"{stats['square_cells_possible']} filled; "
          f"{stats['square_cells_single_dataset']} single-dataset, "
          f"{stats['square_cells_three_dataset']} 3-dataset, "
          f"{stats['square_cells_four_dataset']} full 4-dataset")
    print("[paper] categorical breakdown (off-diagonal, n=%d):" % len(off))
    for b in bd:
        print(f"          {b['bin']:<22} n={b['n']:>3}  {b['pct']:>5.1f}%")
    print(f"[paper] colour clip |v| <= {vmax:.2f} pp (95th pct)")
    print(f"[paper] wrote -> {out}")


if __name__ == "__main__":
    main()
