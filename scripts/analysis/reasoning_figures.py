"""D2 figures — H1 ladder, H2 capability scatter, H3 per-source survivors.

Reads only the CSVs produced by reasoning_transfer.py and gsm8k_capability.py.
Writes PNGs under figures/reasoning/. CPU only.

Usage:
  PYTHONPATH=. ./.venv/bin/python -m scripts.analysis.reasoning_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path("data/results/reasoning")
FIG = Path("figures/reasoning")
RUNGS = ["just_name_it", "random_sampling", "stylistic", "sft", "dpo"]
RUNG_LABEL = ["just_name_it", "random_samp", "stylistic", "sft", "dpo"]


def _boot_ci(vals, n=2000, seed=0):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return vals.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def fig_ladder():
    joined = pd.read_csv(RES / "reasoning_vs_style_percell.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [
        ("persist_reasoning", "reasoning structure", "#c0392b", "-"),
        ("persist_style_recomp", "surface style", "#2980b9", "-"),
        ("persist_reasoning_lenres", "reasoning (length-residualized)", "#c0392b", "--"),
    ]
    x = np.arange(len(RUNGS))
    for col, label, color, ls in series:
        means, los, his = [], [], []
        for i, r in enumerate(RUNGS):
            g = joined[joined["rung"] == r][col].dropna()
            m, lo, hi = _boot_ci(g, seed=i)
            means.append(m); los.append(lo); his.append(hi)
        means = np.array(means)
        ax.plot(x, means, ls, color=color, marker="o", label=label, lw=2)
        if ls == "-":
            ax.fill_between(x, los, his, color=color, alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels(RUNG_LABEL, rotation=20)
    ax.set_ylabel("source persistence (1 = residue survives)")
    ax.set_xlabel("disguise rung (increasing imitation pressure →)")
    ax.set_title("H1: reasoning structure persists differently from surface style\n"
                 "(Fisher-LDA projection persistence, n=36 cells, 95% bootstrap CI)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_rs_vs_style_ladder.png", dpi=140)
    plt.close(fig)


def fig_cap_scatter():
    df = pd.read_csv(RES / "gsm8k_capability_joined.csv")
    sub = df[df["rung"].isin(["sft", "dpo"]) & (~df["low_power"])].dropna(
        subset=["cap_transfer", "transfer_reasoning", "transfer_style"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, xcol, name, color in [
        (axes[0], "transfer_reasoning", "reasoning transfer (1-persist)", "#c0392b"),
        (axes[1], "transfer_style", "style transfer (1-persist)", "#2980b9"),
    ]:
        ax.scatter(sub[xcol], sub["cap_transfer"], c=color, alpha=0.7,
                   edgecolor="k", linewidth=0.3)
        if len(sub) >= 2 and sub[xcol].std() > 1e-9:
            b, a = np.polyfit(sub[xcol], sub["cap_transfer"], 1)
            xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 20)
            ax.plot(xs, a + b * xs, color=color, lw=1.5)
            r = np.corrcoef(sub[xcol], sub["cap_transfer"])[0, 1]
            ax.set_title(f"{name}\nPearson r = {r:+.3f}  (n={len(sub)})")
        ax.set_xlabel(xcol)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("gsm8k capability transfer")
    fig.suptitle("H2: capability transfer vs reasoning/style transfer (gsm8k SFT+DPO, low-power excluded)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_cap_vs_transfer_scatter.png", dpi=140)
    plt.close(fig)


def fig_persource():
    tbl = pd.read_csv(RES / "h3_persource_reasoning.csv")
    order = ["nemotron", "gpt-oss", "qwen", "llama"]
    tbl = tbl.set_index("source").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(order))
    w = 0.38
    colors = ["#c0392b" if s else "#7f8c8d" for s in tbl["is_survivor"]]
    ax.bar(x - w / 2, tbl["dpo_reasoning_persist"], w, color=colors,
           label="DPO reasoning persistence")
    ax.bar(x + w / 2, tbl["survivor_residue_cached"], w, color="#bdc3c7",
           label="cached DPO style residue (survivor split)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{'SURVIVOR' if v else 'launderer'}"
                        for s, v in zip(order, tbl["is_survivor"])])
    ax.set_ylabel("DPO source persistence")
    from scipy.stats import spearmanr
    rho, _ = spearmanr(tbl["dpo_reasoning_persist"], tbl["survivor_residue_cached"])
    ax.set_title("H3: per-source DPO reasoning persistence tracks the 7/3 survivor split\n"
                 f"Spearman(reasoning, survivor) = {rho:+.2f}   "
                 f"vs known-inverting distinctiveness baseline r = -0.31")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_persource_reasoning_survivors.png", dpi=140)
    plt.close(fig)


def main():
    fig_ladder()
    fig_cap_scatter()
    fig_persource()
    print(f"wrote 3 figures -> {FIG}/")
    for p in sorted(FIG.glob("*.png")):
        print("  ", p, f"({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
