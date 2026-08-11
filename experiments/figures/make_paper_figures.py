"""Generate four figures for the historical four-model persistence study.

Config- and data-driven: each figure's numbers are loaded from the results CSV
where a clean source exists, with the originally published values kept as
``VERIFIED_*`` constants that double as an offline fallback and the ``--validate``
reference. Styling is unchanged, so the PDFs are visually identical to the
hand-coded version for those legacy results. This is not the core-12 safety-paper figure builder;
use ``experiments/imitation_safety/regenerate_campaign.py`` for the current campaign.

  python -m experiments.figures.make_paper_figures               # write to overleaf_img
  python -m experiments.figures.make_paper_figures --out-dir DIR  # write elsewhere
  python -m experiments.figures.make_paper_figures --validate     # check data vs literals
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dementor import config

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 10,
})

# Source-model order shared by fig_source_split and fig_inversion.
SOURCE_SLUGS = ["llama-3.1-8b", "qwen3.6-27b", "gpt-oss-20b", "nemotron-nano-30b-a3b"]

# Results CSV locations (anchored on the repo root, never the cwd).
MULTISEED_CSV = config.path("results") / "multiseed_ci_s3.csv"            # data/results/
DISTINCT_CSV = config.project_root() / "results" / "source_distinctiveness.csv"
B2C_CSV = config.project_root() / "results" / "durability" / "b2c_cross_dataset.csv"


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --- data loaders ------------------------------------------------------------
# Each loader returns the figure's numbers from a results CSV, rounded to the
# 3-decimal precision the figures were authored at (so the bars/labels render
# identically), falling back to the VERIFIED_* literals if the CSV is absent.

# fig_source_split: per-source mean DPO persistence over the 3x3 transfer matrix.
VERIFIED_SOURCE_SPLIT = [0.012, 0.077, 0.190, 0.211]


def load_source_split() -> list[float]:
    if not MULTISEED_CSV.exists():
        return list(VERIFIED_SOURCE_SPLIT)
    df = pd.read_csv(MULTISEED_CSV)
    m = df[df["base_rung"] == "dpo"].groupby("source")["mean"].mean()
    return [round(float(m[s]), 3) for s in SOURCE_SLUGS]


# fig_inversion: per-source DPO persistence vs. distinctiveness in two spaces.
VERIFIED_INVERSION = {
    "persistence": [0.012, 0.077, 0.190, 0.211],
    "minilm": [0.130, 0.145, 0.440, 0.447],
    "structural": [0.577, 0.163, 0.353, 0.368],
}


def load_inversion() -> dict[str, list[float]]:
    if not DISTINCT_CSV.exists():
        return {k: list(v) for k, v in VERIFIED_INVERSION.items()}
    df = pd.read_csv(DISTINCT_CSV).set_index("model")
    col = lambda c: [round(float(df.loc[s, c]), 3) for s in SOURCE_SLUGS]
    return {"persistence": col("dpo_persistence"),
            "minilm": col("distinct_minilm"),
            "structural": col("distinct_structural")}


# fig_capability_null: ->nemotron DPO persistence per dataset, meaned over the
# three Phase-B sources.
CAPNULL_DATASETS = ["gsm8k", "writingprompts", "chatbot_arena", "oasst1"]
VERIFIED_CAPNULL = [0.613, 0.000, 0.000, 0.009]


def load_capability_null() -> list[float]:
    if not B2C_CSV.exists():
        return list(VERIFIED_CAPNULL)
    df = pd.read_csv(B2C_CSV)
    m = df[df["is_nemotron_target"]].groupby("dataset")["dpo_persistence"].mean()
    return [round(float(m[d]), 3) for d in CAPNULL_DATASETS]


# fig_safety_placebo: refusal-erosion attack vs. self-imitation placebo. The
# placebo (self-target) arm is not present in any single results CSV and the
# attack aggregation is not cleanly reconstructable, so these stay as VERIFIED
# literals (do not re-derive).
VERIFIED_SAFETY_ATTACK = [0.654, 0.036, 0.021, 0.018]
VERIFIED_SAFETY_PLACEBO = [0.05, 0.0, 0.0, 0.013]


# === FIGURE 1: source split ===
def fig_source_split(img):
    labels = ["llama-3.1-8b", "qwen3.6-27b", "gpt-oss-20b", "nemotron-nano"]
    vals = load_source_split()
    # launder = llama, qwen ; retain = gpt-oss, nemotron
    launder_color = "#c44e52"
    retain_color = "#4c72b0"
    colors = [launder_color, launder_color, retain_color, retain_color]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.65)
    ax.axhline(0.15, ls="--", color="grey", lw=1)
    ax.text(0.05, 0.158, "retain threshold", ha="left", va="bottom",
            fontsize=7, color="grey")
    ax.set_ylim(0, 0.3)
    ax.set_ylabel("DPO persistence")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7)
    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=launder_color, label="launder"),
                       Patch(color=retain_color, label="retain")],
              frameon=False, fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 1.02), ncol=2, handlelength=1.0,
              columnspacing=1.0)
    despine(ax)
    fig.tight_layout()
    fig.savefig(img / "fig_source_split.pdf", bbox_inches="tight")
    plt.close(fig)


# === FIGURE 2: safety placebo ===
def fig_safety_placebo(img):
    sources = ["llama", "qwen", "gpt-oss", "nemotron"]
    attack = VERIFIED_SAFETY_ATTACK
    placebo = VERIFIED_SAFETY_PLACEBO

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(len(sources))
    w = 0.38
    ax.bar(x - w / 2, attack, w, label="cross-model imitation", color="#c44e52")
    ax.bar(x + w / 2, placebo, w, label="self-imitation (placebo)", color="#bbbbbb")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("fraction of refusals eroded")
    ax.set_xticks(x)
    ax.set_xticklabels(sources, fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    despine(ax)
    fig.tight_layout()
    fig.savefig(img / "fig_safety_placebo.pdf", bbox_inches="tight")
    plt.close(fig)


# === FIGURE 3: distinctiveness inversion ===
def fig_inversion(img):
    names = ["llama", "qwen", "gpt-oss", "nemotron"]
    data = load_inversion()
    persistence = data["persistence"]
    minilm = data["minilm"]
    structural = data["structural"]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)

    def panel(ax, xs, title):
        ax.scatter(xs, persistence, color="#4c72b0", zorder=3, s=30)
        for xi, yi, nm in zip(xs, persistence, names):
            ax.annotate(nm, (xi, yi), textcoords="offset points",
                        xytext=(4, 3), fontsize=6.5)
        # trend line
        m, b = np.polyfit(xs, persistence, 1)
        xline = np.linspace(min(xs), max(xs), 50)
        ax.plot(xline, m * xline + b, color="grey", lw=1, ls="-", zorder=2)
        ax.set_xlabel("distinctiveness")
        ax.set_title(title, fontsize=8)
        despine(ax)

    panel(axes[0], minilm, "MiniLM space (r=+0.97)")
    panel(axes[1], structural, "Structural space (r=$-$0.31)")
    axes[0].set_ylabel("DPO persistence")
    fig.tight_layout()
    fig.savefig(img / "fig_distinctiveness_inversion.pdf", bbox_inches="tight")
    plt.close(fig)


# === FIGURE 4: capability null ===
def fig_capability_null(img):
    labels = ["GSM8K", "WritingPrompts", "Chatbot Arena", "OpenAssistant"]
    vals = load_capability_null()
    colors = ["#c44e52"] + ["#4c72b0"] * 3  # highlight GSM8K outlier

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.65)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel(r"$\rightarrow$Nemotron DPO persistence")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7)
    despine(ax)
    fig.tight_layout()
    fig.savefig(img / "fig_capability_null.pdf", bbox_inches="tight")
    plt.close(fig)


def validate(tol: float = 0.02) -> None:
    """Assert each data-loaded series matches its VERIFIED literal within ``tol``."""
    inv = load_inversion()
    checks = [
        ("source_split", MULTISEED_CSV, load_source_split(), VERIFIED_SOURCE_SPLIT),
        ("capability_null", B2C_CSV, load_capability_null(), VERIFIED_CAPNULL),
        ("inversion.persistence", DISTINCT_CSV, inv["persistence"], VERIFIED_INVERSION["persistence"]),
        ("inversion.minilm", DISTINCT_CSV, inv["minilm"], VERIFIED_INVERSION["minilm"]),
        ("inversion.structural", DISTINCT_CSV, inv["structural"], VERIFIED_INVERSION["structural"]),
    ]
    ok = True
    for name, src, loaded, verified in checks:
        maxdiff = max((abs(a - b) for a, b in zip(loaded, verified)), default=0.0)
        bad = maxdiff > tol
        ok = ok and not bad
        flag = "" if src.exists() else "  [CSV MISSING -> using fallback]"
        print(f"[{'FAIL' if bad else 'ok'}] {name}: loaded={loaded} verified={verified} "
              f"maxdiff={maxdiff:.4f}{flag}")
    print("[--] safety_placebo: literal (no clean data source) — not validated")
    if not ok:
        raise SystemExit("validation FAILED: loaded numbers diverge from VERIFIED literals")
    print(f"validation OK: all data-loaded numbers match VERIFIED literals within tol={tol}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the 4 Dementor paper figures.")
    ap.add_argument("--out-dir", default=str(config.path("overleaf_img")),
                    help="directory to write the PDFs (default: config path 'overleaf_img')")
    ap.add_argument("--validate", action="store_true",
                    help="check data-loaded numbers against VERIFIED literals, then exit")
    args = ap.parse_args()

    if args.validate:
        validate()
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_source_split(out)
    fig_safety_placebo(out)
    fig_inversion(out)
    fig_capability_null(out)
    print("All 4 figures written to", out)


if __name__ == "__main__":
    main()
