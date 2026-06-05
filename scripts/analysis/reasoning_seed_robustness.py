"""D2 R5 — seed-robustness of reasoning-structure persistence (cache-only, $0).

The headline D2 numbers (reasoning_transfer.py) fit the Fisher-LDA source/target
axis on the seed1 endpoints and project the (single-seed) disguised rungs onto it.
R5 in the plan flags this as a single-seed estimate. This module re-runs the
IDENTICAL estimator with the cached *_seed2 endpoints as the axis-defining basis
(available for all 36 cells), and a POOLED variant (seed1+seed2 endpoints), then
reports:

  (1) per-cell, per-rung reasoning persistence on seed2 endpoints + pooled,
  (2) seed1-vs-seed2 agreement (Pearson + Spearman, per-cell, overall and by rung),
  (3) whether H3 (per-source DPO reasoning persistence ranks the 7/3 survivor
      split, Spearman +1.00) HOLDS on seed2 and pooled,
  (4) whether the H1 DPO gap Δ(reasoning-style) survives on seed2/pooled.

Disguised rungs are single-seed for 35/36 cells (only one cell has rung seed2/3),
so the available cross-seed signal is the source/target *endpoints* that define
the measurement axis — exactly the basis the persistence metric is sensitive to.
This is the honest, fully-cached robustness test.

Pure numpy/pandas/scipy, CPU only. No API, no Tinker, no GPU.

Usage:
  PYTHONPATH=. ./.venv/bin/python -m scripts.analysis.reasoning_seed_robustness
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

from scripts.analysis.reasoning_structure import _read_responses, reasoning_matrix
from scripts.analysis.reasoning_transfer import (
    OUT_DIR,
    RUNGS,
    RUNG_ORDER,
    SURVIVORS,
    _bootstrap_ci,
    _cell_persistence,
    _iter_cells,
)
from scripts.analysis.structural_decomp import structural_matrix

# cached multiseed DPO style persistence per source (the 7/3 survivor residue);
# identical constant used in reasoning_transfer.h3_stats.
SURVIVOR_RESIDUE = {
    "nemotron": 0.210668, "gpt-oss": 0.189985,
    "qwen": 0.076945, "llama": 0.012033,
}
SURVIVOR_ORDER = "nemotron>gpt-oss>qwen>llama"


def compute_percell_for_basis(basis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cell, per-rung reasoning + identical-pipeline style persistence, where
    the source/target axis-defining endpoints come from `basis`.

    basis in {"seed1", "seed2", "pooled"}. Disguised rungs are always the single
    cached rung_<r>.csv (seed1 sampling), so this isolates basis-seed sensitivity.
    """
    reasoning_rows, style_rows = [], []
    for dataset, _pair, src, tgt, gen in _iter_cells():
        if basis == "pooled":
            src_texts = _read_responses(gen / "source_seed1.csv") + _read_responses(gen / "source_seed2.csv")
            tgt_texts = _read_responses(gen / "target_seed1.csv") + _read_responses(gen / "target_seed2.csv")
        else:
            suffix = "seed1" if basis == "seed1" else "seed2"
            src_texts = _read_responses(gen / f"source_{suffix}.csv")
            tgt_texts = _read_responses(gen / f"target_{suffix}.csv")

        rs_src = reasoning_matrix(src_texts)[0]
        rs_tgt = reasoning_matrix(tgt_texts)[0]
        st_src = structural_matrix(src_texts)[0]
        st_tgt = structural_matrix(tgt_texts)[0]

        for rung in RUNGS:
            dis_texts = _read_responses(gen / f"rung_{rung}.csv")
            rs_dis = reasoning_matrix(dis_texts)[0]
            p, m, sep, na = _cell_persistence(rs_src, rs_tgt, rs_dis)
            reasoning_rows.append({
                "dataset": dataset, "source": src, "target": tgt, "rung": rung,
                "basis": basis, "persist_reasoning": p, "movement_reasoning": m,
                "axis_separation": sep, "n_active_features": na,
            })
            st_dis = structural_matrix(dis_texts)[0]
            ps, ms, seps, nas = _cell_persistence(st_src, st_tgt, st_dis)
            style_rows.append({
                "dataset": dataset, "source": src, "target": tgt, "rung": rung,
                "basis": basis, "persist_style_recomp": ps, "movement_style_recomp": ms,
                "axis_separation_style": seps, "n_active_features_style": nas,
            })
    return pd.DataFrame(reasoning_rows), pd.DataFrame(style_rows)


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan"), float("nan"), int(len(a))
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0]), int(len(a))


def seed_agreement(rs1: pd.DataFrame, rs2: pd.DataFrame) -> pd.DataFrame:
    """seed1-vs-seed2 per-cell persistence agreement, overall and per rung."""
    key = ["dataset", "source", "target", "rung"]
    j = rs1.merge(rs2, on=key, suffixes=("_s1", "_s2"))
    rows = []
    for rung in ["ALL"] + RUNGS:
        g = j if rung == "ALL" else j[j["rung"] == rung]
        a = g["persist_reasoning_s1"].to_numpy()
        b = g["persist_reasoning_s2"].to_numpy()
        pear, spear, n = _corr(a, b)
        mad = float(np.nanmean(np.abs(a - b))) if n else float("nan")
        rows.append({
            "rung": rung, "n_cells": n,
            "pearson_s1_s2": pear, "spearman_s1_s2": spear,
            "mean_abs_diff": mad,
            "mean_persist_s1": float(np.nanmean(a)) if n else float("nan"),
            "mean_persist_s2": float(np.nanmean(b)) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def h3_for_basis(reasoning: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-source DPO reasoning persistence ranking vs the 7/3 survivor residue."""
    sources = ["nemotron", "gpt-oss", "qwen", "llama"]
    dpo = reasoning[reasoning["rung"] == "dpo"]
    persource = dpo.groupby("source")["persist_reasoning"].mean()
    tbl = pd.DataFrame({
        "source": sources,
        "dpo_reasoning_persist": [float(persource.get(s, np.nan)) for s in sources],
        "survivor_residue_cached": [SURVIVOR_RESIDUE[s] for s in sources],
        "is_survivor": [s in SURVIVORS for s in sources],
    })
    rho, _ = spearmanr(tbl["dpo_reasoning_persist"], tbl["survivor_residue_cached"])
    # observed source ordering by DPO reasoning persistence (descending)
    order = ">".join(tbl.sort_values("dpo_reasoning_persist", ascending=False)["source"])
    # clean 7/3 split = both survivors rank above both launderers
    surv_vals = tbl.loc[tbl["is_survivor"], "dpo_reasoning_persist"]
    laun_vals = tbl.loc[~tbl["is_survivor"], "dpo_reasoning_persist"]
    split_clean = bool(surv_vals.min() > laun_vals.max())
    meta = {
        "spearman_reasoning_vs_survivor": float(rho),
        "observed_order": order,
        "survivor_order": SURVIVOR_ORDER,
        "survivors_above_launderers": split_clean,
    }
    return tbl, meta


def h1_dpo_gap(reasoning: pd.DataFrame, style: pd.DataFrame) -> dict:
    """H1 headline: DPO Δ(persist_reasoning - persist_style_recomp), bootstrap CI."""
    key = ["dataset", "source", "target", "rung"]
    j = reasoning.merge(style, on=key, how="left")
    g = j[j["rung"] == "dpo"][["persist_reasoning", "persist_style_recomp"]].dropna()
    delta = (g["persist_reasoning"] - g["persist_style_recomp"]).to_numpy()
    mean, lo, hi = _bootstrap_ci(delta, seed=RUNG_ORDER["dpo"])
    try:
        _, wp = wilcoxon(g["persist_reasoning"], g["persist_style_recomp"])
    except ValueError:
        wp = float("nan")
    return {
        "n": int(len(g)),
        "mean_persist_reasoning": float(g["persist_reasoning"].mean()),
        "mean_persist_style": float(g["persist_style_recomp"].mean()),
        "mean_delta": mean, "delta_ci_low": lo, "delta_ci_high": hi,
        "wilcoxon_p": float(wp),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rs1, st1 = compute_percell_for_basis("seed1")
    rs2, st2 = compute_percell_for_basis("seed2")
    rsp, stp = compute_percell_for_basis("pooled")

    rs2.to_csv(OUT_DIR / "reasoning_persistence_percell_seed2.csv", index=False)
    rsp.to_csv(OUT_DIR / "reasoning_persistence_percell_pooled.csv", index=False)

    agree = seed_agreement(rs1, rs2)
    agree.to_csv(OUT_DIR / "reasoning_seed_agreement.csv", index=False)

    h3_s1, m1 = h3_for_basis(rs1)
    h3_s2, m2 = h3_for_basis(rs2)
    h3_sp, mp = h3_for_basis(rsp)
    h3_s2.assign(basis="seed2").to_csv(OUT_DIR / "h3_persource_reasoning_seed2.csv", index=False)
    h3_sp.assign(basis="pooled").to_csv(OUT_DIR / "h3_persource_reasoning_pooled.csv", index=False)

    g1 = h1_dpo_gap(rs1, st1)
    g2 = h1_dpo_gap(rs2, st2)
    gp = h1_dpo_gap(rsp, stp)

    # ---- consolidated seed-robustness summary --------------------------------
    rob_rows = []
    for basis, h3m, h1g, rsdf in [
        ("seed1", m1, g1, rs1), ("seed2", m2, g2, rs2), ("pooled", mp, gp, rsp),
    ]:
        rob_rows.append({
            "basis": basis,
            "h3_spearman_reasoning_vs_survivor": h3m["spearman_reasoning_vs_survivor"],
            "h3_observed_source_order": h3m["observed_order"],
            "h3_survivors_above_launderers": h3m["survivors_above_launderers"],
            "h1_dpo_mean_delta": h1g["mean_delta"],
            "h1_dpo_delta_ci": f"[{h1g['delta_ci_low']:.3f},{h1g['delta_ci_high']:.3f}]",
            "h1_dpo_wilcoxon_p": h1g["wilcoxon_p"],
            "h1_dpo_gap_positive_sig": bool(
                np.isfinite(h1g["delta_ci_low"]) and h1g["delta_ci_low"] > 0
            ),
        })
    rob = pd.DataFrame(rob_rows)
    rob.to_csv(OUT_DIR / "reasoning_seed_robustness.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("=" * 84)
    print("D2 R5 — SEED-ROBUSTNESS of reasoning-structure persistence (cache-only, $0)")
    print("=" * 84)
    print("\nNOTE: disguised rungs are single-seed for 35/36 cells; the cross-seed")
    print("signal is the source/target ENDPOINTS that DEFINE the Fisher-LDA axis.")
    print("seed2 = axis fit on *_seed2 endpoints; pooled = seed1+seed2 endpoints.\n")

    print("--- (2) seed1-vs-seed2 per-cell persistence agreement ---")
    print(agree.round(4).to_string(index=False))

    print("\n--- (3) H3: per-source DPO reasoning persistence vs 7/3 survivor split ---")
    for basis, tbl, m in [("seed1", h3_s1, m1), ("seed2", h3_s2, m2), ("pooled", h3_sp, mp)]:
        print(f"\n[{basis}] order={m['observed_order']}  "
              f"Spearman(vs survivor)={m['spearman_reasoning_vs_survivor']:+.3f}  "
              f"survivors>launderers={m['survivors_above_launderers']}")
        print(tbl.round(4).to_string(index=False))

    print("\n--- (4) H1: DPO Δ(reasoning - style) across bases ---")
    for basis, g in [("seed1", g1), ("seed2", g2), ("pooled", gp)]:
        print(f"[{basis}] Δ={g['mean_delta']:+.3f} "
              f"CI[{g['delta_ci_low']:.3f},{g['delta_ci_high']:.3f}] "
              f"wilcoxon_p={g['wilcoxon_p']:.4f}")

    print("\n--- consolidated robustness table ---")
    print(rob.to_string(index=False))
    print(f"\nwrote seed-robustness CSVs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
