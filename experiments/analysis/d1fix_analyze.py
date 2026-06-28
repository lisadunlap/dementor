"""D1-FIX analysis: capability transfer vs style persistence on MATH-500.

Reads the graded MATH-500 generations under results/d1fix_capgen/ (base models +
SFT/DPO adapters), joins per-cell STYLE persistence from the gsm8k matrix ladder,
and tests the "capability is the payload while style launders" thesis on a benchmark
that actually has a capability gap (MATH-500 spread 0.35, unlike ceiling'd gsm8k).

Definitions (cell = source->target, adapter base_model = SOURCE):
  acc_source   = base[source]                       (capability origin)
  acc_target   = base[target]                       (capability destination)
  acc_imitator = adapter accuracy at the rung
  gap          = acc_target - acc_source ; gap_ok = |gap| >= 0.05
  cap_xfer     = (acc_imitator - acc_source) / gap  (on gap_ok cells)
  delta_acc    = acc_imitator - acc_source

The audit-relevant "payload" case is the POSITIVE-gap cells (weak source imitating a
stronger target): does training on a stronger model's outputs lift competence while the
style audit reads ~0? Dissociation = corr(cap_xfer, style persistence) at DPO.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "results/d1fix_capgen"
LADDER = ROOT / "results/matrix_ladder/gsm8k_matrix_ladder.csv"
OUT_PERCELL = ROOT / "results/d1fix_capability_transfer_percell.csv"
OUT_STATS = ROOT / "results/d1fix_dissociation_stats.csv"
OUT_FIG = ROOT / "results/d1fix_fig_dissociation.png"

SLUGS = ["llama-3.1-8b", "gpt-oss-20b", "qwen3.6-27b", "nemotron-nano-30b-a3b"]


def boot_ci(x, y, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(x[b])) < 2 or len(set(y[b])) < 2:
            continue
        vals.append(fn(x[b], y[b]))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    base = {s: pd.read_csv(GEN / f"base_{s}.csv")["correct"].mean() for s in SLUGS}
    print("MATH-500 base accuracy:")
    for s in sorted(base, key=lambda k: -base[k]):
        print(f"  {s:26s} {base[s]:.3f}")

    rows = []
    for f in sorted(GEN.glob("*_as_*.csv")):
        rung, rest = f.stem.split("_", 1)
        src, tgt = rest.split("_as_")
        acc_imit = float(pd.read_csv(f)["correct"].mean())
        gap = base[tgt] - base[src]
        rows.append(dict(
            source=src, target=tgt, rung=rung,
            acc_source=round(base[src], 4), acc_target=round(base[tgt], 4),
            acc_imitator=round(acc_imit, 4), gap=round(gap, 4),
            delta_acc=round(acc_imit - base[src], 4),
            cap_xfer=round((acc_imit - base[src]) / gap, 4) if abs(gap) >= 0.05 else np.nan,
            gap_ok=abs(gap) >= 0.05, payload_cell=gap >= 0.05))
    cap = pd.DataFrame(rows)

    lad = pd.read_csv(LADDER)[["source", "target", "rung", "persistence", "anchored"]]
    m = cap.merge(lad, on=["source", "target", "rung"], how="left")
    m.to_csv(OUT_PERCELL, index=False)
    print(f"\nwrote {OUT_PERCELL} ({len(m)} rows)")

    stat_rows = []
    for rung in ["dpo", "sft"]:
        sub = m[(m["rung"] == rung) & m["gap_ok"]].dropna(subset=["cap_xfer", "persistence"])
        pay = sub[sub["payload_cell"]]
        x = sub["cap_xfer"].to_numpy(); y = sub["persistence"].to_numpy()
        pr, pp = (stats.pearsonr(x, y) if len(x) > 2 else (np.nan, np.nan))
        sr, sp = (stats.spearmanr(x, y) if len(x) > 2 else (np.nan, np.nan))
        lo, hi = boot_ci(x, y, lambda a, b: stats.pearsonr(a, b)[0]) if len(x) > 2 else (np.nan, np.nan)
        # false-clean: style reads ~clean but capability moved toward target
        fc = sub[(sub["persistence"] <= 0.10) & (sub["cap_xfer"] >= 0.25) & (sub["delta_acc"] > 0)]
        print(f"\n=== {rung.upper()} ===")
        print(f"  gap_ok cells: {len(sub)} (payload/positive-gap: {len(pay)})")
        print(f"  mean cap_xfer        = {sub['cap_xfer'].mean():.3f}  "
              f"(payload cells: {pay['cap_xfer'].mean():.3f})")
        print(f"  mean delta_acc       = {sub['delta_acc'].mean():+.3f}  "
              f"(payload cells: {pay['delta_acc'].mean():+.3f})")
        print(f"  dissociation r(cap_xfer, persistence) = {pr:.3f} (p={pp:.3f}, 95%CI[{lo:.2f},{hi:.2f}])"
              f"  spearman={sr:.3f}")
        print(f"  false-clean cells (persist<=.1 & cap_xfer>=.25 & dacc>0): {len(fc)}")
        if len(fc):
            print(fc[["source", "target", "persistence", "cap_xfer", "delta_acc"]].to_string(index=False))
        stat_rows.append(dict(rung=rung, n_gap_ok=len(sub), n_payload=len(pay),
                              mean_cap_xfer=round(float(sub["cap_xfer"].mean()), 4),
                              mean_cap_xfer_payload=round(float(pay["cap_xfer"].mean()), 4) if len(pay) else np.nan,
                              mean_delta_acc=round(float(sub["delta_acc"].mean()), 4),
                              mean_delta_acc_payload=round(float(pay["delta_acc"].mean()), 4) if len(pay) else np.nan,
                              pearson_r=round(float(pr), 4), pearson_p=round(float(pp), 4),
                              pearson_ci_lo=round(lo, 4), pearson_ci_hi=round(hi, 4),
                              spearman_rho=round(float(sr), 4), n_false_clean=len(fc)))
    pd.DataFrame(stat_rows).to_csv(OUT_STATS, index=False)
    print(f"\nwrote {OUT_STATS}")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, rung in zip(axes, ["dpo", "sft"]):
            sub = m[(m["rung"] == rung) & m["gap_ok"]].dropna(subset=["cap_xfer", "persistence"])
            ax.scatter(sub["persistence"], sub["cap_xfer"],
                       c=sub["payload_cell"].map({True: "C3", False: "C0"}), s=60)
            for _, r in sub.iterrows():
                ax.annotate(f"{r['source'][:4]}>{r['target'][:4]}",
                            (r["persistence"], r["cap_xfer"]), fontsize=6, alpha=0.7)
            ax.axhline(0, color="k", lw=0.5); ax.axvline(0.1, color="grey", ls=":", lw=0.8)
            ax.set_xlabel("style persistence"); ax.set_ylabel("capability transfer")
            ax.set_title(f"{rung.upper()} (red=payload: weak imitating strong)")
        fig.suptitle("D1-fix (MATH-500): capability transfer vs style persistence")
        fig.tight_layout()
        fig.savefig(OUT_FIG, dpi=120)
        print(f"wrote {OUT_FIG}")
    except Exception as e:
        print(f"[fig skipped] {e}")


if __name__ == "__main__":
    main()
