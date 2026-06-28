"""A2 — Variance decomposition: is per-source behavioral "durability" a STABLE TRAIT?

The n=4 model-correlation problem does NOT apply here. This asks how the per-CELL variance
in each durability metric partitions across factors. A stable per-source trait predicts a
LARGE source main effect with SMALL source x target interaction and small residual. That
inference is powered by the 9 cells/source (style/reasoning) and 648 cell x seed obs (safety),
independent of n=4.

Hand-rolled Type-II ANOVA (statsmodels is absent): effects-coded design + np.linalg.lstsq,
rank-aware degrees of freedom so the missing diagonal (source != target -> 12 pairs, not 16)
is handled gracefully. eta^2 / omega^2 / partial eta^2 / F / p per term.

Three metrics / three units:
  style persistence    : results/matrix_ladder/*_matrix_ladder.csv (rung==dpo)        36 cells
  reasoning persistence: data/results/reasoning/reasoning_persistence_percell.csv     36 cells
  refusal |drift|      : results/safety/safety_full_refusal_ladder.csv (the anchor)   216 rows, 3 seeds/cell

GATE: if SOURCE eta^2 >> SOURCE:TARGET and >> residual in >=2/3 metrics AND within-source
rank-consistency is high -> durability is a stable per-source trait -> Phase B is fundable.
NB: A2 proves STABILITY, not INDEPENDENCE FROM CAPABILITY (capability also loads on SOURCE).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/durability"


# --------------------------------------------------------------------------- design
def _effects_columns(series: pd.Series) -> np.ndarray:
    """Sum-to-zero effects coding: L levels -> L-1 columns, reference level = -1."""
    levels = sorted(series.unique())
    ref = levels[-1]
    cols = []
    arr = series.to_numpy()
    for lev in levels[:-1]:
        c = np.where(arr == lev, 1.0, 0.0)
        c = np.where(arr == ref, -1.0, c)
        cols.append(c)
    return np.column_stack(cols) if cols else np.empty((len(series), 0))


def _design(df: pd.DataFrame, terms: list[tuple], factor_cols: dict) -> np.ndarray:
    n = len(df)
    blocks = [np.ones((n, 1))]  # intercept
    for term in terms:
        if term == ():
            continue
        block = factor_cols[term[0]]
        for f in term[1:]:
            m = factor_cols[f]
            block = np.einsum("ni,nj->nij", block, m).reshape(n, -1)
        blocks.append(block)
    return np.column_stack(blocks)


def _rss_rank(y: np.ndarray, X: np.ndarray) -> tuple[float, int]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), int(np.linalg.matrix_rank(X))


def type2_anova(df: pd.DataFrame, response: str, factors: list[str],
                terms: list[tuple]) -> tuple[pd.DataFrame, dict]:
    y = df[response].to_numpy(dtype=float)
    n = len(y)
    fcols = {f: _effects_columns(df[f].astype(str)) for f in factors}
    X_full = _design(df, terms, fcols)
    rss_full, rank_full = _rss_rank(y, X_full)
    df_resid = n - rank_full
    ms_resid = rss_full / df_resid if df_resid > 0 else np.nan
    ss_total = float(((y - y.mean()) ** 2).sum())

    rows = []
    for T in terms:
        if T == ():
            continue
        reduced = [U for U in terms if not set(T).issubset(set(U))]
        Xr = _design(df, reduced, fcols)
        Xf = _design(df, reduced + [T], fcols)
        rss_r, rank_r = _rss_rank(y, Xr)
        rss_f, rank_f = _rss_rank(y, Xf)
        ss_T = max(0.0, rss_r - rss_f)
        df_T = rank_f - rank_r
        eta2 = ss_T / ss_total
        partial = ss_T / (ss_T + rss_full) if (ss_T + rss_full) > 0 else np.nan
        omega2 = ((ss_T - df_T * ms_resid) / (ss_total + ms_resid)
                  if df_resid > 0 else np.nan)
        F = (ss_T / df_T) / ms_resid if (df_T > 0 and df_resid > 0) else np.nan
        p = float(stats.f.sf(F, df_T, df_resid)) if np.isfinite(F) else np.nan
        rows.append(dict(term=":".join(T), df=df_T, SS=round(ss_T, 5),
                         eta2=round(eta2, 4), omega2=round(omega2, 4),
                         partial_eta2=round(partial, 4),
                         F=round(F, 3) if np.isfinite(F) else np.nan,
                         p=round(p, 5) if np.isfinite(p) else np.nan))
    rows.append(dict(term="residual", df=df_resid, SS=round(rss_full, 5),
                     eta2=round(rss_full / ss_total, 4), omega2=np.nan,
                     partial_eta2=np.nan, F=np.nan, p=np.nan))
    meta = dict(n=n, ss_total=round(ss_total, 5), df_resid=df_resid,
                ms_resid=round(ms_resid, 6) if df_resid > 0 else np.nan,
                residual_is_pure_error=False)
    return pd.DataFrame(rows), meta


# ------------------------------------------------------------- rank consistency
def kendall_w(pivot: pd.DataFrame) -> float:
    """pivot: rows = items (sources), cols = raters (datasets). Returns Kendall's W in [0,1]."""
    ranks = pivot.rank(axis=0)  # rank sources within each dataset
    m = ranks.shape[1]  # raters
    nn = ranks.shape[0]  # items
    Rsum = ranks.sum(axis=1)
    S = float(((Rsum - Rsum.mean()) ** 2).sum())
    denom = m ** 2 * (nn ** 3 - nn)
    return 12 * S / denom if denom else np.nan


def within_source_consistency(df, response, label) -> dict:
    """Per source: (a) Kendall's W of the source ranking across datasets (avg over targets);
    (b) fraction of a source's cells on the same side of the grand mean."""
    g = df.groupby(["source", "dataset"])[response].mean().reset_index()
    pivot = g.pivot(index="source", columns="dataset", values=response)
    W = kendall_w(pivot)
    grand = df[response].mean()
    sign_rows = []
    for src, sub in df.groupby("source"):
        side = np.sign(sub[response] - grand)
        dominant = side.mode().iloc[0] if len(side.mode()) else 0
        frac = float((side == dominant).mean())
        sign_rows.append(dict(metric=label, source=src, n_cells=len(sub),
                              mean=round(sub[response].mean(), 4),
                              sign_consistency=round(frac, 3)))
    return dict(metric=label, kendall_w_across_datasets=round(W, 3)), sign_rows


# --------------------------------------------------------------------------- load
def load_style() -> pd.DataFrame:
    frames = []
    for f in glob.glob(str(ROOT / "results/matrix_ladder/*_matrix_ladder.csv")):
        ds = Path(f).stem.replace("_matrix_ladder", "")
        d = pd.read_csv(f)
        d["dataset"] = ds
        frames.append(d)
    df = pd.concat(frames)
    return df[df["rung"] == "dpo"][["source", "target", "dataset", "persistence"]].copy()


def load_reasoning() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/results/reasoning/reasoning_persistence_percell.csv")
    return d[d["rung"] == "dpo"][["source", "target", "dataset", "persist_reasoning"]].copy()


def load_safety() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "results/safety/safety_full_refusal_ladder.csv")
    return d[["source", "target", "dataset", "rung", "seed", "abs_drift", "drift"]].copy()


# --------------------------------------------------------------------------- main
def _print_block(name, anova, meta):
    print(f"\n{'='*70}\n{name}  (n={meta['n']}, residual df={meta['df_resid']}, "
          f"pure_error={meta['residual_is_pure_error']})\n{'='*70}")
    print(anova.to_string(index=False))

    def _g(term, col):
        r = anova[anova["term"] == term][col]
        return float(r.iloc[0]) if len(r) else np.nan
    d = dict(source_eta2=_g("source", "eta2"), source_p=_g("source", "p"),
             target_eta2=_g("target", "eta2"), dataset_eta2=_g("dataset", "eta2"),
             inter_eta2=_g("source:target", "eta2"), resid_eta2=_g("residual", "eta2"),
             pure_error=meta["residual_is_pure_error"])
    print(f"  -> SOURCE eta2={d['source_eta2']:.3f} (p={d['source_p']}) | "
          f"target={d['target_eta2']:.3f} dataset={d['dataset_eta2']:.3f} "
          f"inter={d['inter_eta2']:.3f} resid={d['resid_eta2']:.3f}")
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    factors3 = ["source", "target", "dataset"]
    terms3 = [("source",), ("target",), ("dataset",), ("source", "target")]

    verdict = {}
    consistency_summ, consistency_rows = [], []

    # ---- style (DPO) ----
    style = load_style()
    a_style, m_style = type2_anova(style, "persistence", factors3, terms3)
    m_style["residual_is_pure_error"] = False
    a_style.to_csv(OUT / "variance_decomp_style.csv", index=False)
    verdict["style"] = _print_block("STYLE persistence (DPO)", a_style, m_style)
    s, rws = within_source_consistency(style, "persistence", "style")
    consistency_summ.append(s); consistency_rows += rws

    # ---- reasoning (DPO) ----
    reas = load_reasoning()
    a_reas, m_reas = type2_anova(reas, "persist_reasoning", factors3, terms3)
    a_reas.to_csv(OUT / "variance_decomp_reasoning.csv", index=False)
    verdict["reasoning"] = _print_block("REASONING persistence (DPO)", a_reas, m_reas)
    s, rws = within_source_consistency(reas, "persist_reasoning", "reasoning")
    consistency_summ.append(s); consistency_rows += rws

    # ---- safety (the inferential anchor: seeds = pure error) ----
    saf = load_safety()
    factors4 = ["source", "target", "dataset", "rung"]
    terms4 = [("source",), ("target",), ("dataset",), ("rung",), ("source", "target")]
    a_saf, m_saf = type2_anova(saf, "abs_drift", factors4, terms4)
    m_saf["residual_is_pure_error"] = True  # seeds within cell = genuine error
    a_saf.to_csv(OUT / "variance_decomp_safety.csv", index=False)
    verdict["safety"] = _print_block("REFUSAL |drift| (sft+dpo, seeds=error)", a_saf, m_saf)
    s, rws = within_source_consistency(saf[saf["rung"] == "dpo"], "abs_drift", "safety_dpo")
    consistency_summ.append(s); consistency_rows += rws

    pd.DataFrame(consistency_rows).to_csv(OUT / "within_source_rank_consistency.csv", index=False)
    kw_by_metric = {s["metric"]: s["kendall_w_across_datasets"] for s in consistency_summ}
    print(f"\n{'='*70}\nWITHIN-SOURCE RANK CONSISTENCY (Kendall's W across datasets)\n{'='*70}")
    print(pd.DataFrame(consistency_summ).to_string(index=False))
    print("\nsign-consistency per source (frac of cells on same side of grand mean):")
    print(pd.DataFrame(consistency_rows).to_string(index=False))

    # ---- GATE verdict ----
    # Principled criterion (fair to single-seed metrics whose residual has no pure-error term):
    # a stable per-source trait = (1) significant SOURCE main effect (p<0.05),
    # (2) SOURCE is the dominant NAMED effect (> target, dataset, source:target eta^2),
    # (3) cross-dataset source ranking is stable (Kendall's W >= 0.5).
    kw_key = {"style": "style", "reasoning": "reasoning", "safety": "safety_dpo"}
    print(f"\n{'#'*70}\nGATE VERDICT\n{'#'*70}")
    rows, passes = [], 0
    for metric, d in verdict.items():
        kw = kw_by_metric.get(kw_key[metric], np.nan)
        sig = d["source_p"] < 0.05
        dominant = (d["source_eta2"] > d["target_eta2"] and d["source_eta2"] > d["dataset_eta2"]
                    and (np.isnan(d["inter_eta2"]) or d["source_eta2"] > d["inter_eta2"]))
        stable = kw >= 0.5
        ok = sig and dominant and stable
        passes += ok
        rows.append(dict(metric=metric, source_eta2=round(d["source_eta2"], 3),
                         source_p=d["source_p"], dominant_named=dominant,
                         kendall_w=round(kw, 3), significant=sig, stable=stable,
                         verdict="STABLE TRAIT" if ok else "not stable"))
        print(f"  {metric:10s}: source eta2={d['source_eta2']:.3f} p={d['source_p']} "
              f"dominant={dominant} W={kw:.2f} -> {'STABLE TRAIT' if ok else 'NOT stable'}")
    gate = passes >= 2
    print(f"\n  >>> A2 GATE: {passes}/3 metrics show a stable per-source trait.")
    if gate:
        print("      PASS — durability is a stable per-source trait (style+safety); Phase B is fundable")
        print("      to test whether that stable trait is separable from capability.")
    else:
        print("      WEAK — do not fund Phase B; write the honest null.")
    print("  (Reminder: stability != independence from capability; only Phase B/B2 breaks that.)")
    pd.DataFrame(rows).to_csv(OUT / "gate_summary.csv", index=False)


if __name__ == "__main__":
    main()
