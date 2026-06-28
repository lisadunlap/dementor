"""B2a verdict — recompute the capability<->durability relationship at n=6.

The original 4 study models confound durability with capability: the 2 retainers
(nemotron, gpt-oss) are also the 2 most capable; the 2 launderers (qwen, llama) the 2
least. Every durability metric co-ranks rho=1.0 with capability over n=4, so we cannot
tell whether durability is a real axis or just capability.

B2a adds 2 HIGH-capability sources from LAUNDERER lineages (Llama-3.3-70B, Qwen3-32B).
They sit OFF the original capability=durability diagonal. Pre-registered test:
  - if they LAUNDER (low DPO style persistence, like their lineage cousins) despite high
    capability  -> capability and durability DISSOCIATE -> durability is a SEPARATE AXIS.
  - if they RETAIN (high persistence, snapping onto the diagonal) -> durability IS
    capability (honest null).
  - in between / mixed -> AMBIGUOUS: hard-stop and report before spending on B2b.

Reads the 8 trained cells' calibrated DPO persistence from
data/results/gsm8k/analysis/cells/<src>_to_<tgt>/cell_summary.csv, averages per new
source, joins capability from results/durability/extra_census.csv, stacks onto the
original results/durability/per_source_durability.csv, and writes the n=6 table +
Spearman to results/durability/cap_vs_durability_n6.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "data/results/gsm8k/analysis/cells"
DUR = ROOT / "results/durability"

NEW = {
    "llama-3.3-70b": "llama(launderer)",
    "qwen3-32b": "qwen(launderer)",
}
TARGETS = ["llama-3.1-8b", "qwen3.6-27b", "nemotron-nano-30b-a3b", "gpt-oss-20b"]

# Empirical per-source bands from the original 4 (dpo_style_persist):
#   retainers nemotron 0.211 / gpt-oss 0.190 ; launderers qwen 0.077 / llama 0.012.
LAUNDER_MAX = 0.12   # <= this => behaves like a launderer
RETAIN_MIN = 0.15    # >= this => behaves like a retainer (gap is the ambiguous band)


def dpo_persist(src: str, tgt: str) -> float | None:
    f = CELLS / f"{src}_to_{tgt}" / "cell_summary.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    row = df[df["method"] == "dpo"]
    if row.empty:
        return None
    return float(row["persistence"].iloc[0])


def main() -> None:
    # --- per-cell DPO persistence for the 8 new cells ---
    cell_rows, per_src = [], {}
    for src in NEW:
        vals = []
        for tgt in TARGETS:
            p = dpo_persist(src, tgt)
            cell_rows.append({"source": src, "target": tgt,
                              "dpo_persistence": None if p is None else round(p, 4)})
            if p is not None:
                vals.append(p)
        per_src[src] = float(np.mean(vals)) if vals else np.nan
        print(f"  {src}: {len(vals)}/4 cells -> mean DPO persistence = "
              f"{per_src[src]:.4f}" if vals else f"  {src}: NO cells found")
    cells_df = pd.DataFrame(cell_rows)
    cells_df.to_csv(DUR / "cap_vs_durability_n6_cells.csv", index=False)
    print("\nper-cell DPO persistence (8 new cells):")
    print(cells_df.to_string(index=False))

    missing = cells_df["dpo_persistence"].isna().sum()
    if missing:
        print(f"\n[warn] {missing} cell(s) not yet scored — verdict is provisional.")

    # --- capability for the new sources ---
    cen = pd.read_csv(DUR / "extra_census.csv").set_index("slug")["math500_acc"].to_dict()

    # --- original 4-row table ---
    orig = pd.read_csv(DUR / "per_source_durability.csv", comment="#")

    new_rows = [{
        "source": src, "tier": "new(launderer-lineage)",
        "dpo_style_persist": round(per_src[src], 4),
        "math_capability": round(float(cen.get(src, np.nan)), 4),
        "lineage": lin,
    } for src, lin in NEW.items()]
    combined = pd.concat(
        [orig[["source", "tier", "dpo_style_persist", "math_capability"]],
         pd.DataFrame(new_rows)[["source", "tier", "dpo_style_persist", "math_capability"]]],
        ignore_index=True)
    combined.to_csv(DUR / "cap_vs_durability_n6.csv", index=False)

    # --- Spearman cap<->durability at n=4 (orig) and n=6 (with new sources) ---
    rho4, p4 = spearmanr(orig["math_capability"], orig["dpo_style_persist"])
    rho6, p6 = spearmanr(combined["math_capability"], combined["dpo_style_persist"])

    print("\n=== n=6 capability vs DPO style durability ===")
    print(combined.to_string(index=False))
    print(f"\n  Spearman(capability, dpo_style_persist):  n=4 rho={rho4:+.3f} (p={p4:.3f})"
          f"  ->  n=6 rho={rho6:+.3f} (p={p6:.3f})")

    # --- pre-registered verdict ---
    def band(p):
        if np.isnan(p):
            return "unknown"
        if p <= LAUNDER_MAX:
            return "launders"
        if p >= RETAIN_MIN:
            return "retains"
        return "ambiguous"

    bands = {s: band(per_src[s]) for s in NEW}
    print("\n  new high-capability sources (predicted: LAUNDER despite high cap):")
    for s in NEW:
        print(f"    {s:14s} cap={cen.get(s, float('nan')):.3f}  "
              f"persist={per_src[s]:.4f}  -> {bands[s]}")

    vals = list(bands.values())
    if all(b == "launders" for b in vals):
        verdict = ("SEPARATE AXIS: both high-capability launderer-lineage sources LAUNDER "
                   f"(persistence <= {LAUNDER_MAX}) despite high MATH capability. Capability and "
                   f"durability DISSOCIATE — rho collapses {rho4:+.2f} -> {rho6:+.2f}. Durability "
                   "is a source/lineage property, not capability.")
    elif all(b == "retains" for b in vals):
        verdict = ("DURABILITY == CAPABILITY (honest null): both new capable sources RETAIN, "
                   f"snapping onto the diagonal (rho stays {rho6:+.2f}). The cross-axis story is "
                   "capability wearing a costume.")
    else:
        verdict = ("AMBIGUOUS / MIXED: new sources split or land in the ambiguous band "
                   f"({LAUNDER_MAX}-{RETAIN_MIN}). Hard-stop per the plan — report and decide on "
                   "B2b (3 datasets / K=3) before more spend.")
    print(f"\n  >>> PRE-REGISTERED VERDICT: {verdict}")
    print(f"\n  wrote {DUR/'cap_vs_durability_n6.csv'}")


if __name__ == "__main__":
    main()
