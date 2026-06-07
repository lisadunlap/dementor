"""A1 — Per-source durability table (n=4) + Spearman matrix.

Joins existing per-source results into one 4-row table to make the "durability axis" pattern
explicit AND honestly bounded. STRICTLY SUBORDINATE to A2 (variance_decomp.py): A2 showed the
per-source trait is real+stable for STYLE and SAFETY but NOT for reasoning. This table is a
descriptive summary, not inference.

Columns: dpo_style_persist, dpo_reasoning_persist, refusal_retention (=1-mean frac_erased@DPO),
math_capability (MATH-500 native acc), distinct_structural (the inverter), distinct_minilm,
native_xdataset_stability (A4, imitation-independent).

CAVEAT (load-bearing, written into the CSV + printed): n=4 sources, 2-2 survivor split. Every
rho here is a rank-direction descriptor, NOT inference: a perfect rho=+/-1 over 4 points has
permutation p~0.33 for the 2-vs-2 partition (~1 bit). All columns are CONFOUNDED with capability
(they all co-rank with MATH acc). Only Phase B/B2 (new sources dissociating capability from
predicted durability) can break that confound.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/durability"

CAVEAT = ("n=4 sources, 2-2 split. rho = rank-direction only (perm p~0.33, ~1 bit); "
          "ALL columns confounded with capability; only Phase B breaks it. "
          "A2 verdict: style+safety durability are stable per-source traits; reasoning is NOT.")


def canon(name: str) -> str:
    n = str(name).lower()
    if "gpt" in n:
        return "gpt-oss-20b"
    if "llama" in n:
        return "llama-3.1-8b"
    if "qwen" in n:
        return "qwen3.6-27b"
    if "nemotron" in n:
        return "nemotron-nano-30b-a3b"
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    dist = pd.read_csv(ROOT / "results/source_distinctiveness.csv")
    dist["source"] = dist["model"].map(canon)
    tbl = dist[["source", "dpo_persistence", "distinct_structural", "distinct_minilm", "tier"]].copy()
    tbl = tbl.rename(columns={"dpo_persistence": "dpo_style_persist"})

    reas = pd.read_csv(ROOT / "data/results/reasoning/h3_persource_reasoning.csv")
    reas["source"] = reas["source"].map(canon)
    tbl = tbl.merge(reas[["source", "dpo_reasoning_persist"]], on="source", how="left")

    saf = pd.read_csv(ROOT / "results/safety/safety_full_refusal_ladder.csv")
    saf["source"] = saf["source"].map(canon)
    ret = saf[saf["rung"] == "dpo"].groupby("source")["frac_erased"].mean()
    tbl["refusal_retention"] = tbl["source"].map(1 - ret)

    cap = pd.read_csv(ROOT / "results/d1fix_capability_census.csv")
    cap = cap[cap["benchmark"] == "math500"].copy()
    cap["source"] = cap["model"].map(canon)
    tbl = tbl.merge(cap[["source", "acc"]].rename(columns={"acc": "math_capability"}), on="source", how="left")

    s7p = ROOT / "results/findings/native_xdataset_stability.csv"
    if s7p.exists():
        s7 = pd.read_csv(s7p)
        s7["source"] = s7["model"].map(canon)
        tbl = tbl.merge(s7[["source", "native_xdataset_stability"]], on="source", how="left")

    metric_cols = ["dpo_style_persist", "dpo_reasoning_persist", "refusal_retention",
                   "math_capability", "distinct_structural", "distinct_minilm",
                   "native_xdataset_stability"]
    metric_cols = [c for c in metric_cols if c in tbl.columns]
    tbl = tbl.sort_values("dpo_style_persist", ascending=False).reset_index(drop=True)
    tbl = tbl[["source", "tier"] + metric_cols].round(4)

    out_csv = OUT / "per_source_durability.csv"
    with open(out_csv, "w") as f:
        f.write(f"# CAVEAT: {CAVEAT}\n")
    tbl.to_csv(out_csv, mode="a", index=False)

    corr = tbl[metric_cols].corr(method="spearman").round(3)
    corr.to_csv(OUT / "per_source_durability_corr.csv")

    print("=== A1 per-source durability table (n=4) ===")
    print(tbl.to_string(index=False))
    print(f"\n=== Spearman correlation matrix (n=4; {CAVEAT}) ===")
    print(corr.to_string())
    cap_corr = corr["math_capability"].drop("math_capability")
    print("\n  Every durability column's rank-corr with MATH capability:")
    print(cap_corr.to_string())
    print(f"\n  >>> Confound is visible: mean |rho(metric, capability)| = {cap_corr.abs().mean():.2f}. "
          "Phase B is required to separate durability from capability.")


if __name__ == "__main__":
    main()
