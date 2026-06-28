"""D3 safety pilot analysis — rates only, no completions.

Consumes the verdict CSVs written by run_safety_ladder.py under results/safety/
(native_refusal/*.csv, adapter_refusal/*.csv) and produces:

  results/safety/safety_native_refusal.csv     per SOURCE base model native rates
  results/safety/safety_refusal_ladder.csv      per (dataset,source,target,rung,seed)
  results/safety/safety_drift_vs_style.csv      merged with style persistence (H5)
  results/safety/safety_xstest_overrefusal.csv  over-refusal guardrail (H6)
  figures/safety_ladder.png
  figures/safety_drift_vs_style.png

COMPARATOR CONVENTION: adapter base_model == SOURCE, so the native comparator
for an A->B (source->target) adapter rung is r_native(SOURCE), NOT r_native(target).
drift = r_imitated - r_native(SOURCE).

No model spend. No completions persisted (input is already verdicts).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/safety")
FIG = Path("figures")
NATIVE_DIR = OUT / "native_refusal"
ADAPTER_DIR = OUT / "adapter_refusal"
STYLE_CSV = Path("results/matrix_ladder/gsm8k_matrix_ladder.csv")

HARMFUL = "harmful"  # category label for the 120-harmful subset


def _harmful_mask(df: pd.DataFrame) -> pd.Series:
    return df["category"].astype(str) == HARMFUL


def load_native() -> pd.DataFrame:
    """Per (source, seed) native rate on the harmful subset + over-refusal subset."""
    rows = []
    for p in sorted(NATIVE_DIR.glob("*.csv")):
        if p.name.startswith("_raw"):
            continue
        m = re.match(r"(.+)_seed(\d+)\.csv$", p.name)
        if not m:
            continue
        model, seed = m.group(1), int(m.group(2))
        df = pd.read_csv(p)
        h = df[_harmful_mask(df)]
        o = df[~_harmful_mask(df)]
        rows.append({
            "model": model, "seed": seed,
            "n_harmful": len(h),
            "r_native_harmful": float(h["refused"].mean()) if len(h) else np.nan,
            "n_over": len(o),
            "r_native_over": float(o["refused"].mean()) if len(o) else np.nan,
        })
    return pd.DataFrame(rows)


def native_rate_map(native_df: pd.DataFrame) -> dict[str, float]:
    """Mean over seeds of the per-source native harmful refusal rate."""
    g = native_df.groupby("model")["r_native_harmful"].mean()
    return g.to_dict()


def native_over_map(native_df: pd.DataFrame) -> dict[str, float]:
    g = native_df.groupby("model")["r_native_over"].mean()
    return g.to_dict()


def load_adapters() -> pd.DataFrame:
    """Parse adapter verdict CSVs -> per-cell rates on harmful + over subsets."""
    pat = re.compile(r"(sft|dpo)_(gsm8k|writingprompts|chatbot_arena)_(.+)_as_(.+)_seed(\d+)\.csv$")
    rows = []
    for p in sorted(ADAPTER_DIR.glob("*.csv")):
        if p.name.startswith("_raw"):
            continue
        m = pat.match(p.name)
        if not m:
            continue
        rung, ds, src, tgt, seed = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
        df = pd.read_csv(p)
        h = df[_harmful_mask(df)]
        o = df[~_harmful_mask(df)]
        rows.append({
            "dataset": ds, "source": src, "target": tgt, "rung": rung, "seed": seed,
            "n_harmful": len(h),
            "r_imitated_harmful": float(h["refused"].mean()) if len(h) else np.nan,
            "n_over": len(o),
            "r_imitated_over": float(o["refused"].mean()) if len(o) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    native = load_native()
    native.to_csv(OUT / "safety_native_refusal.csv", index=False)

    r_native = native_rate_map(native)          # per SOURCE
    r_native_over = native_over_map(native)

    adapters = load_adapters()
    # drift / pull against SOURCE (= adapter base).
    adapters["r_native_source"] = adapters["source"].map(r_native)
    adapters["r_native_target"] = adapters["target"].map(r_native)
    adapters["drift"] = adapters["r_imitated_harmful"] - adapters["r_native_source"]
    adapters["abs_drift"] = adapters["drift"].abs()
    # erased-fraction: of the source's native refusals, what fraction did imitation drop.
    def _erased(row):
        rn = row["r_native_source"]
        if rn is None or np.isnan(rn) or rn <= 0:
            return np.nan
        return max(0.0, (rn - row["r_imitated_harmful"])) / rn
    adapters["frac_native_erased"] = adapters.apply(_erased, axis=1)
    adapters.to_csv(OUT / "safety_refusal_ladder.csv", index=False)

    # ---- H5: |drift| vs style persistence (anchored) ----
    style = pd.read_csv(STYLE_CSV)[["source", "target", "rung", "anchored", "persistence"]]
    merged = adapters.merge(style, on=["source", "target", "rung"], how="left")
    merged.to_csv(OUT / "safety_drift_vs_style.csv", index=False)

    # ---- XSTest over-refusal guardrail (H6) ----
    over = adapters[["dataset", "source", "target", "rung", "seed",
                     "r_imitated_over", "r_native_source"]].copy()
    over["r_native_source_over"] = over["source"].map(r_native_over)
    over["over_drift"] = over["r_imitated_over"] - over["r_native_source_over"]
    over.to_csv(OUT / "safety_xstest_overrefusal.csv", index=False)

    # ---------- HEADLINE ----------
    print("=" * 72)
    print("NATIVE refusal rate per SOURCE (harmful subset, mean over seeds):")
    for mdl, r in sorted(r_native.items()):
        print(f"  {mdl:26s} native_harmful={r:.3f}   native_over={r_native_over.get(mdl, float('nan')):.3f}")

    # mean imitated by rung
    print("\nIMITATED harmful refusal rate by rung (mean over all 12 pairs, seed1):")
    for rung in ["sft", "dpo"]:
        sub = adapters[adapters["rung"] == rung]
        print(f"  {rung:4s}  mean r_imitated={sub['r_imitated_harmful'].mean():.3f}  "
              f"mean drift(vs source)={sub['drift'].mean():+.3f}  "
              f"mean |drift|={sub['abs_drift'].mean():.3f}")

    # headline: erased %
    for rung in ["sft", "dpo"]:
        sub = adapters[adapters["rung"] == rung]
        # pooled erased: pooled native minus pooled imitated over the cells with native>0.
        valid = sub.dropna(subset=["frac_native_erased"])
        pooled = valid["frac_native_erased"].mean() if len(valid) else float("nan")
        print(f"\nHEADLINE [{rung}]: benign-output imitation erased "
              f"{pooled*100:.1f}% of SOURCE native refusals (mean over "
              f"{len(valid)}/{len(sub)} cells with native>0).")

    # H5 correlation
    print("\nH5 — corr(|drift|, style anchored persistence):")
    for rung in ["sft", "dpo", "both"]:
        sub = merged if rung == "both" else merged[merged["rung"] == rung]
        s = sub.dropna(subset=["abs_drift", "anchored"])
        if len(s) >= 3 and s["abs_drift"].std() > 0 and s["anchored"].std() > 0:
            pr = s["abs_drift"].corr(s["anchored"], method="pearson")
            sr = s["abs_drift"].corr(s["anchored"], method="spearman")
            print(f"  {rung:5s} n={len(s)}  pearson={pr:+.3f}  spearman={sr:+.3f}")
        else:
            print(f"  {rung:5s} n={len(s)}  (insufficient variance)")

    # H6 guardrail
    print("\nH6 — over-refusal (XSTest-style safe prompts):")
    print(f"  native over-refusal (mean over sources): {np.nanmean(list(r_native_over.values())):.3f}")
    for rung in ["sft", "dpo"]:
        sub = over[over["rung"] == rung]
        print(f"  {rung:4s} imitated over-refusal mean={sub['r_imitated_over'].mean():.3f}  "
              f"over_drift mean={sub['over_drift'].mean():+.3f}")

    _figures(adapters, merged)
    print("\nWrote results/safety/*.csv + figures/safety_*.png")


def _figures(adapters: pd.DataFrame, merged: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: native vs sft vs dpo mean refusal (the ladder).
    fig, ax = plt.subplots(figsize=(6, 4))
    rungs = ["native(source)", "sft", "dpo"]
    native_mean = adapters["r_native_source"].mean()
    vals = [native_mean,
            adapters[adapters["rung"] == "sft"]["r_imitated_harmful"].mean(),
            adapters[adapters["rung"] == "dpo"]["r_imitated_harmful"].mean()]
    ax.plot(rungs, vals, "o-", color="crimson", lw=2)
    for x, v in zip(rungs, vals):
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 8))
    ax.set_ylabel("mean harmful refusal rate")
    ax.set_title("D3 pilot (gsm8k seed1): refusal laundering ladder")
    ax.set_ylim(0, max(0.05, max(vals) * 1.3))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "safety_ladder.png", dpi=130)
    plt.close(fig)

    # Fig 2: |drift| vs style persistence scatter (H5).
    fig, ax = plt.subplots(figsize=(6, 4))
    s = merged.dropna(subset=["abs_drift", "anchored"])
    colors = {"sft": "steelblue", "dpo": "crimson"}
    for rung in ["sft", "dpo"]:
        ss = s[s["rung"] == rung]
        ax.scatter(ss["anchored"], ss["abs_drift"], label=rung,
                   color=colors[rung], alpha=0.7, s=35)
    if len(s) >= 3 and s["anchored"].std() > 0:
        r = s["abs_drift"].corr(s["anchored"])
        ax.set_title(f"H5: |refusal drift| vs style persistence (pearson r={r:+.2f})")
    ax.set_xlabel("style anchored persistence")
    ax.set_ylabel("|refusal drift| (harmful)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "safety_drift_vs_style.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
