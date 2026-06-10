"""B2b verdict — does CAPABILITY or SIZE drive DPO style durability? (n=7, 2x2 design)

B2a left capability and size confounded (its 2 retainers were both LARGE and STRONG).
B2b adds Qwen3-4B: SMALL (4B) but STRONG (MATH 0.64) — the missing small+strong corner.

The capability x size 2x2 (durability = DPO style persistence; retain >= 0.15, launder <= 0.12):
                weak (MATH<0.60)          strong (MATH>=0.60)
  small (<=8B)  llama-3.1-8b              Qwen3-4B   <- THE decisive new cell
  large(>=20B)  qwen3.6-27b              {gpt-oss,nemotron,llama-3.3-70b,qwen3-32b}

Decisive read on Qwen3-4B:
  RETAINS  -> capability drives durability; SIZE ruled out (a 4B retains; weak models
              launder at any size). The strong claim, now confound-controlled.
  LAUNDERS -> B2a retention was SIZE, not capability. Durability tracks scale.

Reads qwen3-4b's 4 DPO cells from the eval (cell_summary.csv), stacks onto the n=6 table
(results/durability/cap_vs_durability_n6.csv), and writes the n=7 table + 2x2 +
Spearman(capability) vs Spearman(size) to results/durability/cap_vs_durability_n7.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
CELLS = ROOT / "data/results/gsm8k/analysis/cells"
DUR = ROOT / "results/durability"

NEW_SRC = "qwen3-4b"
NEW_CAP = 0.6375  # B1 census MATH-500
TARGETS = ["llama-3.1-8b", "qwen3.6-27b", "nemotron-nano-30b-a3b", "gpt-oss-20b"]

# Approx parameter counts (B, total) for the size axis.
SIZE_B = {
    "llama-3.1-8b": 8, "gpt-oss-20b": 20, "qwen3.6-27b": 27,
    "nemotron-nano-30b-a3b": 30, "llama-3.3-70b": 70, "qwen3-32b": 32, "qwen3-4b": 4,
}
LAUNDER_MAX, RETAIN_MIN = 0.12, 0.15


def dpo_persist(src: str, tgt: str) -> float | None:
    f = CELLS / f"{src}_to_{tgt}" / "cell_summary.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    row = df[df["method"] == "dpo"]
    return None if row.empty else float(row["persistence"].iloc[0])


def main() -> None:
    # --- qwen3-4b per-cell DPO persistence ---
    vals, cell_rows = [], []
    for tgt in TARGETS:
        p = dpo_persist(NEW_SRC, tgt)
        cell_rows.append({"source": NEW_SRC, "target": tgt,
                          "dpo_persistence": None if p is None else round(p, 4)})
        if p is not None:
            vals.append(p)
    cells_df = pd.DataFrame(cell_rows)
    print("qwen3-4b per-cell DPO persistence:")
    print(cells_df.to_string(index=False))
    if not vals:
        raise SystemExit("\nNo qwen3-4b cells scored yet — run the eval first.")
    mean_persist = float(np.mean(vals))
    print(f"\n  qwen3-4b: {len(vals)}/4 cells -> mean DPO persistence = {mean_persist:.4f}")
    if len(vals) < 4:
        print(f"  [warn] only {len(vals)}/4 cells — verdict provisional.")

    # --- n=6 table -> n=7 ---
    t = pd.read_csv(DUR / "cap_vs_durability_n6.csv")
    t = pd.concat([t, pd.DataFrame([{
        "source": NEW_SRC, "tier": "new(small+strong)",
        "dpo_style_persist": round(mean_persist, 4), "math_capability": NEW_CAP,
    }])], ignore_index=True)
    t["size_b"] = t["source"].map(SIZE_B)
    t.to_csv(DUR / "cap_vs_durability_n7.csv", index=False)

    # --- which predicts durability: capability or size? ---
    rho_cap, p_cap = spearmanr(t["math_capability"], t["dpo_style_persist"])
    rho_sz, p_sz = spearmanr(t["size_b"], t["dpo_style_persist"])

    print("\n=== n=7 table ===")
    print(t.sort_values("dpo_style_persist", ascending=False).to_string(index=False))
    print(f"\n  Spearman(capability, durability) = {rho_cap:+.3f} (p={p_cap:.3f})")
    print(f"  Spearman(size,       durability) = {rho_sz:+.3f} (p={p_sz:.3f})")

    # --- 2x2 quadrant means ---
    def quad(r):
        s = "small" if r.size_b <= 8 else "large"
        c = "strong" if r.math_capability >= 0.60 else "weak"
        return f"{s}+{c}"
    t["quad"] = t.apply(quad, axis=1)
    print("\n=== capability x size 2x2 (mean DPO persistence; retain>=0.15) ===")
    q = t.groupby("quad").agg(persist=("dpo_style_persist", "mean"),
                              members=("source", lambda s: ",".join(s))).reset_index()
    print(q.to_string(index=False))

    # --- decisive call on qwen3-4b ---
    band = ("launders" if mean_persist <= LAUNDER_MAX
            else "retains" if mean_persist >= RETAIN_MIN else "ambiguous")
    print(f"\n  Qwen3-4B (small 4B, strong MATH {NEW_CAP}): persist={mean_persist:.4f} -> {band}")
    if band == "retains":
        verdict = (f"CAPABILITY drives durability, SIZE ruled out: a 4B model RETAINS "
                   f"(persist {mean_persist:.3f}) like the large strong models, while weak models "
                   f"launder at 8B AND 27B. Capability ranks durability ({rho_cap:+.2f}) better "
                   f"than size ({rho_sz:+.2f}).")
    elif band == "launders":
        verdict = (f"SIZE drives durability: the small-but-strong 4B LAUNDERS (persist "
                   f"{mean_persist:.3f}) despite high capability, so B2a's retention was scale. "
                   f"Size ranks durability ({rho_sz:+.2f}) vs capability ({rho_cap:+.2f}).")
    else:
        verdict = (f"AMBIGUOUS: Qwen3-4B lands in the {LAUNDER_MAX}-{RETAIN_MIN} band "
                   f"(persist {mean_persist:.3f}). Capability rho={rho_cap:+.2f}, size rho={rho_sz:+.2f}.")
    print(f"\n  >>> VERDICT: {verdict}")
    print(f"\n  wrote {DUR/'cap_vs_durability_n7.csv'}")


if __name__ == "__main__":
    main()
