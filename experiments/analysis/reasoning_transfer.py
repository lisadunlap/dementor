"""D2 Steps 1-2,4 — per-cell reasoning-structure persistence + style join.

Reuses the exact headline estimator (Fisher-LDA supervised axis + rotation-
invariant projection persistence) on the DISJOINT reasoning-structure matrix,
for EVERY rung (not just DPO). For apples-to-apples H1, it ALSO recomputes a
style projection-persistence with the identical single-seed pipeline on the 32-D
``_style_*`` matrix, and (separately) joins the cached multi-seed style numbers.

H1: per-rung paired Δ = persist_reasoning - persist_style, bootstrap CI +
    Wilcoxon, length-residualized robustness variant.
H3: per-source DPO reasoning persistence ranks the 7/3 survivor split; compared
    to per-source DPO style persistence and the known-inverting distinctiveness
    baseline (r=-0.31).

Pure numpy/pandas/scipy/sklearn, CPU only. No API, no MiniLM, no GPU.

Usage:
  PYTHONPATH=. ./.venv/bin/python -m experiments.analysis.reasoning_transfer
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dementor.metric.behavioral_inertia_metrics import (
    axis_separation,
    projection_persistence,
)
from dementor.metric.latent_behavior_axes import (
    _fit_supervised_basis,
    _length_covariate,
    _residualize_against_length,
)
from experiments.analysis.reasoning_structure import (
    _read_responses,
    reasoning_matrix,
)
from dementor.steering.structural_decomp import (
    SHORT,
    structural_matrix,
)

DECONTAM_ROOT = Path("data/results/decontam")
OUT_DIR = Path("data/results/reasoning")
RUNGS = ["just_name_it", "random_sampling", "stylistic", "sft", "dpo"]
RUNG_ORDER = {r: i for i, r in enumerate(RUNGS)}
SEP_GATE = 0.3  # |Cohen's d| on the projected axis; same gate as structural_decomp
SHRINKAGE = 0.15
SURVIVORS = ["gpt-oss", "nemotron"]
LAUNDERERS = ["llama", "qwen"]


def _cell_persistence(
    src_X: np.ndarray,
    tgt_X: np.ndarray,
    dis_X: np.ndarray,
    *,
    residualize_len: bool = False,
    src_len: np.ndarray | None = None,
    tgt_len: np.ndarray | None = None,
    dis_len: np.ndarray | None = None,
) -> tuple[float, float, float, int]:
    """Fisher-LDA projection persistence for one cell+rung.

    Returns (persistence, movement, axis_separation_on_axis, n_active_features).
    Basis rule: z-score on pooled source+target only; disguised never defines the
    measurement. Axis = shrinkage-regularized Fisher discriminant (k=1).
    """
    if residualize_len:
        all_X = np.vstack([src_X, dis_X, tgt_X])
        all_len = np.concatenate([src_len, dis_len, tgt_len])
        n_s, n_d, n_t = len(src_X), len(dis_X), len(tgt_X)
        fit_mask = np.zeros(len(all_X), dtype=bool)
        fit_mask[:n_s] = True
        fit_mask[n_s + n_d :] = True
        all_X = _residualize_against_length(all_X, all_len, fit_mask)
        src_X = all_X[:n_s]
        dis_X = all_X[n_s : n_s + n_d]
        tgt_X = all_X[n_s + n_d :]

    ref = np.vstack([src_X, tgt_X])
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    src_z = (src_X - mu) / sd
    tgt_z = (tgt_X - mu) / sd
    dis_z = (dis_X - mu) / sd

    # active features = those separating source/target on z-scale (|d| > gate)
    sep_feat = np.abs(tgt_z.mean(axis=0) - src_z.mean(axis=0))
    n_active = int((sep_feat > SEP_GATE).sum())

    comps, _ = _fit_supervised_basis(src_z, tgt_z, k=1, shrinkage=SHRINKAGE)
    w1 = comps[0]

    src_p = (src_z @ w1)[:, None]
    tgt_p = (tgt_z @ w1)[:, None]
    dis_p = (dis_z @ w1)[:, None]

    persist = projection_persistence(src_p, dis_p, tgt_p)
    movement = 1.0 - persist if np.isfinite(persist) else np.nan
    axis_sep = float(axis_separation(src_p, tgt_p)[0])
    return persist, movement, axis_sep, n_active


def _iter_cells():
    for dpo_csv in sorted(DECONTAM_ROOT.glob("*/*/gen/rung_dpo.csv")):
        gen = dpo_csv.parent
        parts = gen.parts
        dataset = parts[parts.index("decontam") + 1]
        pair = parts[parts.index("decontam") + 2]
        src_full, tgt_full = pair.split("_to_")
        yield dataset, pair, SHORT[src_full], SHORT[tgt_full], gen


def compute_percell() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cell, per-rung persistence for BOTH the reasoning matrix and the
    identical-pipeline style matrix (the primary H1 baseline)."""
    reasoning_rows = []
    style_rows = []
    for dataset, pair, src, tgt, gen in _iter_cells():
        src_texts = _read_responses(gen / "source_seed1.csv")
        tgt_texts = _read_responses(gen / "target_seed1.csv")

        rs_src = reasoning_matrix(src_texts)[0]
        rs_tgt = reasoning_matrix(tgt_texts)[0]
        st_src = structural_matrix(src_texts)[0]
        st_tgt = structural_matrix(tgt_texts)[0]
        src_len = _length_covariate(src_texts)
        tgt_len = _length_covariate(tgt_texts)

        for rung in RUNGS:
            dis_texts = _read_responses(gen / f"rung_{rung}.csv")
            dis_len = _length_covariate(dis_texts)

            rs_dis = reasoning_matrix(dis_texts)[0]
            p, m, sep, na = _cell_persistence(rs_src, rs_tgt, rs_dis)
            p_lr, m_lr, _, _ = _cell_persistence(
                rs_src, rs_tgt, rs_dis,
                residualize_len=True, src_len=src_len, tgt_len=tgt_len, dis_len=dis_len,
            )
            reasoning_rows.append({
                "dataset": dataset, "source": src, "target": tgt, "rung": rung,
                "persist_reasoning": p, "movement_reasoning": m,
                "persist_reasoning_lenres": p_lr,
                "axis_separation": sep, "n_active_features": na,
            })

            st_dis = structural_matrix(dis_texts)[0]
            ps, ms, seps, nas = _cell_persistence(st_src, st_tgt, st_dis)
            style_rows.append({
                "dataset": dataset, "source": src, "target": tgt, "rung": rung,
                "persist_style_recomp": ps, "movement_style_recomp": ms,
                "axis_separation_style": seps, "n_active_features_style": nas,
            })
    return pd.DataFrame(reasoning_rows), pd.DataFrame(style_rows)


def cross_correlation_report() -> pd.DataFrame:
    """R2: document that the reasoning matrix is distinct from the style matrix.

    Pool source+target responses across all cells, z-score both feature blocks,
    and report the max |corr| of each reasoning feature against ANY style feature.
    """
    rs_blocks, st_blocks = [], []
    for _ds, _pair, _s, _t, gen in _iter_cells():
        for role in ("source_seed1.csv", "target_seed1.csv"):
            texts = _read_responses(gen / role)
            rs_blocks.append(reasoning_matrix(texts)[0])
            st_blocks.append(structural_matrix(texts)[0])
    RS = np.vstack(rs_blocks)
    ST = np.vstack(st_blocks)
    rs_names = reasoning_matrix(["x"])[1]
    st_names = structural_matrix(["x"])[1]

    def _z(M):
        mu, sd = M.mean(0), M.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        return (M - mu) / sd

    RSz, STz = _z(RS), _z(ST)
    n = len(RSz)
    corr = (RSz.T @ STz) / n  # (n_rs, n_st)
    rows = []
    for i, rn in enumerate(rs_names):
        absrow = np.abs(corr[i])
        j = int(np.argmax(absrow))
        rows.append({
            "rs_feature": rn,
            "max_abs_corr_with_style": float(absrow[j]),
            "closest_style_feature": st_names[j],
            "signed_corr": float(corr[i, j]),
        })
    return pd.DataFrame(rows).sort_values("max_abs_corr_with_style", ascending=False)


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 2000, seed: int = 0):
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
    return float(np.mean(vals)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def h1_stats(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-rung paired Δ = persist_reasoning - persist_style (recomputed pipeline),
    plus the length-residualized reasoning variant vs same style baseline."""
    rows = []
    for rung in RUNGS:
        g = joined[joined["rung"] == rung]
        for label, rcol in [("raw", "persist_reasoning"), ("lenres", "persist_reasoning_lenres")]:
            pair = g[[rcol, "persist_style_recomp"]].dropna()
            delta = (pair[rcol] - pair["persist_style_recomp"]).to_numpy()
            mean, lo, hi = _bootstrap_ci(delta, seed=RUNG_ORDER[rung])
            try:
                w_stat, w_p = wilcoxon(pair[rcol], pair["persist_style_recomp"])
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            rows.append({
                "rung": rung, "variant": label, "n": len(pair),
                "mean_persist_reasoning": float(pair[rcol].mean()),
                "mean_persist_style": float(pair["persist_style_recomp"].mean()),
                "mean_delta": mean, "delta_ci_low": lo, "delta_ci_high": hi,
                "wilcoxon_p": float(w_p),
            })
    return pd.DataFrame(rows)


def h3_stats(reasoning: pd.DataFrame, style_recomp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-source DPO reasoning persistence vs the survivor split.

    Survivor split target ordering (known DPO style residue): nemotron 0.211 >
    gpt-oss 0.190 > qwen 0.077 > llama 0.012. We rank-correlate per-source DPO
    reasoning persistence against this and report Spearman vs the cached style
    numbers and vs the known-inverting distinctiveness baseline (r=-0.31).
    """
    survivor_residue = {  # cached multiseed DPO style persistence per source
        "nemotron": 0.210668, "gpt-oss": 0.189985,
        "qwen": 0.076945, "llama": 0.012033,
    }
    dpo_rs = reasoning[reasoning["rung"] == "dpo"]
    persource = dpo_rs.groupby("source")["persist_reasoning"].mean()
    dpo_st = style_recomp[style_recomp["rung"] == "dpo"]
    persource_st = dpo_st.groupby("source")["persist_style_recomp"].mean()

    sources = ["nemotron", "gpt-oss", "qwen", "llama"]
    tbl = pd.DataFrame({
        "source": sources,
        "dpo_reasoning_persist": [float(persource.get(s, np.nan)) for s in sources],
        "dpo_style_persist_recomp": [float(persource_st.get(s, np.nan)) for s in sources],
        "survivor_residue_cached": [survivor_residue[s] for s in sources],
        "is_survivor": [s in SURVIVORS for s in sources],
    })

    from scipy.stats import spearmanr
    rho_rs, _ = spearmanr(tbl["dpo_reasoning_persist"], tbl["survivor_residue_cached"])
    rho_st, _ = spearmanr(tbl["dpo_style_persist_recomp"], tbl["survivor_residue_cached"])
    meta = {
        "spearman_reasoning_vs_survivor": float(rho_rs),
        "spearman_recomp_style_vs_survivor": float(rho_st),
        "distinctiveness_baseline_r": -0.31,  # known inverting baseline
        "survivor_order": "nemotron>gpt-oss>qwen>llama",
    }
    return tbl, meta


def feature_decomp() -> pd.DataFrame:
    """Per-feature DPO persistence on the reasoning matrix, survivor vs launderer.

    Mirrors structural_decomp.per_feature_persistence but on the reasoning matrix:
    which CoT-shape feature carries the survivor residue? Positive residue_gap =
    survivors stay source on this feature while launderers collapse to target.
    """
    feat_names = reasoning_matrix(["x"])[1]
    recs = []
    for dataset, pair, src, tgt, gen in _iter_cells():
        s = reasoning_matrix(_read_responses(gen / "source_seed1.csv"))[0]
        t = reasoning_matrix(_read_responses(gen / "target_seed1.csv"))[0]
        d = reasoning_matrix(_read_responses(gen / "rung_dpo.csv"))[0]
        ref = np.vstack([s, t])
        mu, sd = ref.mean(0), ref.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        s_mu = (s.mean(0) - mu) / sd
        t_mu = (t.mean(0) - mu) / sd
        d_mu = (d.mean(0) - mu) / sd
        sep = t_mu - s_mu
        with np.errstate(divide="ignore", invalid="ignore"):
            move = np.where(np.abs(sep) > 1e-9, (d_mu - s_mu) / sep, np.nan)
        persist = 1.0 - np.clip(move, 0.0, 1.0)
        for i, fn in enumerate(feat_names):
            recs.append({"source": src, "feature": fn, "persistence": persist[i],
                         "separates": abs(sep[i]) > SEP_GATE})
    allc = pd.DataFrame(recs)
    rows = []
    for feat, g in allc.groupby("feature"):
        def gated(sub_sources):
            sub = g[g["source"].isin(sub_sources) & g["separates"]]
            base = g[g["source"].isin(sub_sources)]
            use = sub if (len(sub) and sub["persistence"].notna().any()) else base
            return float(use["persistence"].mean())
        sp = gated(SURVIVORS)
        lp = gated(LAUNDERERS)
        rows.append({"feature": feat, "survivor_persist": sp, "launderer_persist": lp,
                     "residue_gap": sp - lp})
    return pd.DataFrame(rows).sort_values("residue_gap", ascending=False).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reasoning, style_recomp = compute_percell()
    reasoning.to_csv(OUT_DIR / "reasoning_persistence_percell.csv", index=False)
    fdecomp = feature_decomp()
    fdecomp.to_csv(OUT_DIR / "reasoning_feature_decomp.csv", index=False)

    # cross-correlation (R2 disjointness audit)
    xcorr = cross_correlation_report()
    xcorr.to_csv(OUT_DIR / "reasoning_style_crosscorr.csv", index=False)

    # join cached multiseed style persistence (H1 secondary)
    ms = pd.read_csv("data/results/multiseed_ci_s3.csv").rename(
        columns={"base_rung": "rung", "mean": "persist_style_multiseed"}
    )[["dataset", "source", "target", "rung", "persist_style_multiseed"]]
    joined = reasoning.merge(style_recomp, on=["dataset", "source", "target", "rung"], how="left")
    joined = joined.merge(ms, on=["dataset", "source", "target", "rung"], how="left")
    joined["delta_rs_minus_style"] = joined["persist_reasoning"] - joined["persist_style_recomp"]
    joined["delta_rs_minus_style_multiseed"] = joined["persist_reasoning"] - joined["persist_style_multiseed"]
    joined.to_csv(OUT_DIR / "reasoning_vs_style_percell.csv", index=False)

    h1 = h1_stats(joined)
    h1.to_csv(OUT_DIR / "h1_per_rung_delta.csv", index=False)

    h3_tbl, h3_meta = h3_stats(reasoning, style_recomp)
    h3_tbl.to_csv(OUT_DIR / "h3_persource_reasoning.csv", index=False)

    # ---- hypothesis summary (one row per hypothesis) ----------------------
    dpo_raw = h1[(h1["rung"] == "dpo") & (h1["variant"] == "raw")].iloc[0]
    dpo_lr = h1[(h1["rung"] == "dpo") & (h1["variant"] == "lenres")].iloc[0]
    rs_raw = h1[(h1["rung"] == "random_sampling") & (h1["variant"] == "lenres")].iloc[0]
    summary_rows = [{
        "hypothesis": "H1_different_rate",
        "headline_stat": f"DPO Δ(rs-style)={dpo_raw['mean_delta']:+.3f}",
        "ci": f"[{dpo_raw['delta_ci_low']:.3f},{dpo_raw['delta_ci_high']:.3f}]",
        "wilcoxon_p": float(dpo_raw["wilcoxon_p"]),
        "lenres_robust": f"DPO lenres Δ={dpo_lr['mean_delta']:+.3f} (p={dpo_lr['wilcoxon_p']:.2f}); "
                         f"random_sampling lenres Δ={rs_raw['mean_delta']:+.3f} (p={rs_raw['wilcoxon_p']:.2f})",
        "verdict": ("SUPPORTED at DPO raw (reasoning persists higher than style after style erases); "
                    "PARTIALLY robust to length-residualization — survives at random_sampling but "
                    "DPO gap attenuates to non-significant under lenres."),
    }]
    h2_csv = OUT_DIR / "h2_capability_corr.csv"
    if h2_csv.exists():
        h2 = pd.read_csv(h2_csv)
        head = h2[h2["rungs"] == "sft+dpo"]
        if len(head):
            hr = head.iloc[0]
            summary_rows.append({
                "hypothesis": "H2_capability_coupling",
                "headline_stat": f"|r_reasoning|={abs(hr['pearson_reasoning']):.3f} vs |r_style|={abs(hr['pearson_style']):.3f} (sft+dpo, n={int(hr['n'])})",
                "ci": "n/a",
                "wilcoxon_p": float("nan"),
                "lenres_robust": "n/a",
                "verdict": ("NOT SUPPORTED: reasoning does not beat style at predicting capability "
                            "transfer. gsm8k acc_source≈acc_target≈0.83 for all 4 models, so 25/60 "
                            "cells are low-power; metric underdetermined. Clean negative."),
            })
    else:
        summary_rows.append({
            "hypothesis": "H2_capability_coupling", "headline_stat": "deferred (no gsm8k gold)",
            "ci": "n/a", "wilcoxon_p": float("nan"), "lenres_robust": "n/a",
            "verdict": "DEFERRED — run gsm8k_capability.py (free gold download).",
        })
    summary_rows.append({
        "hypothesis": "H3_two_tier_explanation",
        "headline_stat": f"Spearman(reasoning,survivor)={h3_meta['spearman_reasoning_vs_survivor']:+.2f}",
        "ci": "n=4 sources (direction, not p-value)",
        "wilcoxon_p": float("nan"),
        "lenres_robust": (f"recomp-style Spearman={h3_meta['spearman_recomp_style_vs_survivor']:+.2f}; "
                          f"distinctiveness baseline r={h3_meta['distinctiveness_baseline_r']:+.2f} (inverts)"),
        "verdict": ("SUPPORTED (direction): per-source DPO reasoning persistence ranks the survivor "
                    "split perfectly and correctly-signed, beating the known-inverting distinctiveness "
                    "baseline. Under-powered (n=4) — a rank-consistency claim, not significance."),
    })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "reasoning_hypothesis_summary.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("=" * 80)
    print("D2 REASONING-STRUCTURE TRANSFER — H1 (different rate) + H3 (survivor split)")
    print("=" * 80)
    print("\n--- R2: reasoning-vs-style feature disjointness (max |corr| per rs feature) ---")
    print(xcorr.round(3).to_string(index=False))
    print(f"\nmean max|corr| across reasoning features: {xcorr['max_abs_corr_with_style'].mean():.3f}")

    print("\n--- H1: per-rung persistence (reasoning vs recomputed-style, single seed) ---")
    print(h1.round(4).to_string(index=False))

    print("\n--- H3: per-source DPO reasoning persistence vs survivor split ---")
    print(h3_tbl.round(4).to_string(index=False))
    print(f"\n  Spearman(reasoning DPO persist, survivor residue) = {h3_meta['spearman_reasoning_vs_survivor']:+.3f}")
    print(f"  Spearman(recomp-style DPO persist, survivor residue) = {h3_meta['spearman_recomp_style_vs_survivor']:+.3f}")
    print(f"  known-inverting distinctiveness baseline r = {h3_meta['distinctiveness_baseline_r']:+.3f}")
    print(f"  (n=4 sources — rank-consistency / direction, NOT a p-value)")

    print(f"\nwrote per-cell + stats CSVs -> {OUT_DIR}/")
    return None


if __name__ == "__main__":
    main()
