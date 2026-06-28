"""D3 FULL-matrix safety analysis (3 datasets x 12 pairs x 2 rungs x 3 seeds = 216 cells).

Extends the pilot analysis with the things the full matrix unlocks:
  - per-DATASET replication: does the gsm8k refusal-erosion hold on writingprompts /
    chatbot_arena too?
  - SEED CIs: 3 adapter seeds per cell -> bootstrap CI on the headline erosion.
  - dataset-correct H5: |drift| vs style persistence joined to EACH dataset's own
    matrix ladder (the pilot script wrongly used the gsm8k ladder for all rows).

Comparator: adapter base_model == SOURCE, so drift = r_imitated - r_native(SOURCE).
Inputs: results/safety/{native_refusal,adapter_refusal}/*.csv (verdicts only).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/safety"
FIG = ROOT / "figures"
NATIVE_DIR = OUT / "native_refusal"
ADAPTER_DIR = OUT / "adapter_refusal"
DATASETS = ["gsm8k", "writingprompts", "chatbot_arena"]


def _harm(df):
    return df["category"].astype(str) == "harmful"


def load_native():
    rows = []
    for p in sorted(NATIVE_DIR.glob("*.csv")):
        m = re.match(r"(.+)_seed(\d+)\.csv$", p.name)
        if not m or p.name.startswith("_raw"):
            continue
        df = pd.read_csv(p)
        rows.append({"model": m.group(1), "seed": int(m.group(2)),
                     "r_native_harmful": float(df[_harm(df)]["refused"].mean()),
                     "r_native_over": float(df[~_harm(df)]["refused"].mean())})
    return pd.DataFrame(rows)


def load_adapters():
    pat = re.compile(r"(sft|dpo)_(gsm8k|writingprompts|chatbot_arena)_(.+)_as_(.+)_seed(\d+)\.csv$")
    rows = []
    for p in sorted(ADAPTER_DIR.glob("*.csv")):
        m = pat.match(p.name)
        if not m or p.name.startswith("_raw"):
            continue
        df = pd.read_csv(p)
        rows.append({"rung": m.group(1), "dataset": m.group(2), "source": m.group(3),
                     "target": m.group(4), "seed": int(m.group(5)),
                     "r_imitated_harmful": float(df[_harm(df)]["refused"].mean()),
                     "r_imitated_over": float(df[~_harm(df)]["refused"].mean())})
    return pd.DataFrame(rows)


def boot_ci(vals, n=4000, seed=0):
    vals = np.asarray([v for v in vals if not np.isnan(v)])
    if len(vals) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    native = load_native()
    r_nat = native.groupby("model")["r_native_harmful"].mean().to_dict()
    r_nat_over = native.groupby("model")["r_native_over"].mean().to_dict()

    a = load_adapters()
    assert len(a) == 216, f"expected 216 adapter cells, got {len(a)}"
    a["r_native_source"] = a["source"].map(r_nat)
    a["drift"] = a["r_imitated_harmful"] - a["r_native_source"]
    a["abs_drift"] = a["drift"].abs()
    a["frac_erased"] = ((a["r_native_source"] - a["r_imitated_harmful"]).clip(lower=0)
                        / a["r_native_source"].replace(0, np.nan))
    a.to_csv(OUT / "safety_full_refusal_ladder.csv", index=False)

    # dataset-correct style join
    lad = pd.concat([pd.read_csv(ROOT / f"results/matrix_ladder/{d}_matrix_ladder.csv")
                     .assign(dataset=d)[["dataset", "source", "target", "rung",
                                         "persistence", "anchored"]] for d in DATASETS])
    m = a.merge(lad, on=["dataset", "source", "target", "rung"], how="left")
    m.to_csv(OUT / "safety_full_drift_vs_style.csv", index=False)

    print("=" * 74)
    print("NATIVE harmful refusal per source (mean/2 seeds):",
          {k: round(v, 3) for k, v in sorted(r_nat.items())})

    print("\n--- DPO refusal EROSION: per-dataset replication (headline) ---")
    summary = []
    for ds in DATASETS:
        for rung in ["sft", "dpo"]:
            sub = a[(a["dataset"] == ds) & (a["rung"] == rung)]
            fe = sub["frac_erased"].dropna()
            lo, hi = boot_ci(fe.values)
            # seed stability: erosion per seed
            per_seed = sub.groupby("seed")["frac_erased"].mean().round(3).to_dict()
            summary.append({"dataset": ds, "rung": rung, "n_cells": len(sub),
                            "mean_r_imitated": round(sub["r_imitated_harmful"].mean(), 3),
                            "mean_drift": round(sub["drift"].mean(), 3),
                            "frac_erased": round(fe.mean(), 3),
                            "erased_ci_lo": round(lo, 3), "erased_ci_hi": round(hi, 3),
                            "per_seed_erased": per_seed})
            if rung == "dpo":
                print(f"  {ds:15s} DPO: erased {fe.mean()*100:4.1f}% of native refusals "
                      f"[95%CI {lo*100:.1f}-{hi*100:.1f}]  (drift {sub['drift'].mean():+.3f}; "
                      f"per-seed {per_seed})")
    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "safety_full_by_dataset.csv", index=False)

    print("\n--- POOLED (all 3 datasets) ---")
    for rung in ["sft", "dpo"]:
        sub = a[a["rung"] == rung]
        fe = sub["frac_erased"].dropna()
        lo, hi = boot_ci(fe.values)
        print(f"  {rung}: erased {fe.mean()*100:.1f}% [95%CI {lo*100:.1f}-{hi*100:.1f}] "
              f"(n={len(fe)} cells; mean drift {sub['drift'].mean():+.3f})")

    print("\n--- H5: corr(|drift|, style persistence), dataset-correct ---")
    for rung in ["sft", "dpo", "both"]:
        s = (m if rung == "both" else m[m["rung"] == rung]).dropna(subset=["abs_drift", "persistence"])
        pr = s["abs_drift"].corr(s["persistence"])
        pa = s["abs_drift"].corr(s["anchored"])
        print(f"  {rung:5s} n={len(s):3d}  pearson(|drift|,persistence)={pr:+.3f}  "
              f"pearson(|drift|,anchored)={pa:+.3f}")

    print("\n--- H6 over-refusal (safe prompts) ---")
    print(f"  native over-refusal mean: {np.nanmean(list(r_nat_over.values())):.3f}")
    for rung in ["sft", "dpo"]:
        sub = a[a["rung"] == rung]
        print(f"  {rung}: imitated over-refusal {sub['r_imitated_over'].mean():.3f}")

    _figures(a, m)
    print("\nWrote results/safety/safety_full_*.csv + figures/safety_full_*.png")


def _figures(a, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = ["native", "sft", "dpo"]
    for ds in DATASETS:
        nat = a[a["dataset"] == ds]["r_native_source"].mean()
        ys = [nat,
              a[(a["dataset"] == ds) & (a["rung"] == "sft")]["r_imitated_harmful"].mean(),
              a[(a["dataset"] == ds) & (a["rung"] == "dpo")]["r_imitated_harmful"].mean()]
        ax.plot(x, ys, "o-", lw=2, label=ds)
    ax.set_ylabel("mean harmful refusal rate")
    ax.set_title("D3 full matrix: refusal erosion by training dataset")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(FIG / "safety_full_ladder.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    s = m.dropna(subset=["abs_drift", "persistence"])
    for ds in DATASETS:
        ss = s[s["dataset"] == ds]
        ax.scatter(ss["persistence"], ss["abs_drift"], alpha=0.6, s=30, label=ds)
    r = s["abs_drift"].corr(s["persistence"])
    ax.set_title(f"H5: |refusal drift| vs style persistence (r={r:+.2f}, n={len(s)})")
    ax.set_xlabel("style persistence"); ax.set_ylabel("|refusal drift|")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "safety_full_drift_vs_style.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
