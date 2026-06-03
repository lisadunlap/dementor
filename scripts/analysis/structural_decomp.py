"""D4 / Move 2b — feature-level decomposition of the surviving DPO residue.

The headline persistence metric (latent_behavior_axes.st_axis) projects every
response onto the *scaled* source->target difference-of-means direction: source
maps to 0, target to 1, and the disguised coordinate is the movement fraction.
"Persistence" is 1 - movement. That metric collapses 32 structural features
(+ a MiniLM block) into one opaque number, so "the surviving residue is
structural" is currently an inference, not a named claim.

This script runs the *same* projection idea but along each individual hand-coded
structural feature (the 32-D ``_style_scalar_features`` + ``_style_binary_features``
basis, z-scored on source+target only — exactly the basis rule the real metric
uses, where disguised/DPO rows never define the measurement). For each of the 36
de-confounded cells and each feature it asks: after one epoch of LoRA-DPO toward
the target, how far does the source *stay* from the target on THIS feature?

It then identifies which named features (length, markdown, headers, bullets,
LaTeX/math, ...) carry the survivor residue: the features on which the survivor
sources (gpt-oss, nemotron) stay far from target while the launderers (qwen,
llama) collapse to the floor. That turns "structural" into a falsifiable,
feature-level claim and quantifies how concentrated the residue is.

Pure pandas/numpy, CPU only. No API, no MiniLM, no GPU.

Usage:
  PYTHONPATH=. ./.venv/bin/python -m scripts.analysis.structural_decomp
  PYTHONPATH=. ./.venv/bin/python -m scripts.analysis.structural_decomp --csv-out data/results/structural_decomp_percell.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.latent_behavior_axes import _style_binary_features, _style_scalar_features

# Survivor sources retain a DPO residue; launderers go to the floor (strategy.md
# §1/§4: per-source DPO persistence nemotron 0.211, gpt-oss 0.190 | qwen 0.077,
# llama 0.012). The "structural residue" claim is about the SURVIVOR minus
# LAUNDERER gap, feature by feature.
SHORT = {
    "llama-3.1-8b": "llama",
    "qwen3.6-27b": "qwen",
    "gpt-oss-20b": "gpt-oss",
    "nemotron-nano-30b-a3b": "nemotron",
}
SURVIVORS = ["gpt-oss", "nemotron"]
LAUNDERERS = ["llama", "qwen"]

DECONTAM_ROOT = Path("data/results/decontam")
# A feature whose source and target barely differ carries no source->target
# signal, so its "movement"/"persistence" is noise. We gate on |z-scored
# separation| the same way big5_directions gates movement on |sep_d|>0.3
# (Cohen's d). On the z-scored (pooled-std) scale sep IS a Cohen's d.
SEP_GATE = 0.3
RESPONSE_COL = "model_response"


def structural_matrix(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """The 32-D hand-coded structural feature matrix (raw, un-scaled)."""
    scalars, scalar_names = _style_scalar_features(texts)
    binaries, binary_names = _style_binary_features(texts)
    X = np.hstack([scalars, binaries]).astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, scalar_names + binary_names


def _read_responses(path: Path) -> list[str]:
    return pd.read_csv(path)[RESPONSE_COL].astype(str).tolist()


def per_feature_persistence(cell_gen: Path, feat_names: list[str]) -> pd.DataFrame:
    """One row per structural feature for a single cell.

    Mirrors the headline st_axis construction, but one feature at a time:
      * z-score each feature on the pooled source+target rows (disguised/DPO
        never defines the measurement — the basis rule from
        fit_behavioral_axis_basis);
      * sep   = z(target).mean - z(source).mean   (a Cohen's d on this feature);
      * move  = (z(dpo).mean - z(source).mean) / sep   (0 = stayed source,
                1 = reached target);
      * persistence = 1 - clip(move, 0, 1)   (1 = DPO output still looks like the
                source on this feature, i.e. residue survived).
    """
    src = structural_matrix(_read_responses(cell_gen / "source_seed1.csv"))[0]
    tgt = structural_matrix(_read_responses(cell_gen / "target_seed1.csv"))[0]
    dpo = structural_matrix(_read_responses(cell_gen / "rung_dpo.csv"))[0]

    # z-score on source+target only (the reference), per feature.
    ref = np.vstack([src, tgt])
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)  # near-constant feature -> contributes ~0

    src_mu = (src.mean(axis=0) - mu) / sd
    tgt_mu = (tgt.mean(axis=0) - mu) / sd
    dpo_mu = (dpo.mean(axis=0) - mu) / sd

    sep = tgt_mu - src_mu
    with np.errstate(divide="ignore", invalid="ignore"):
        move = np.where(np.abs(sep) > 1e-9, (dpo_mu - src_mu) / sep, np.nan)
    persistence = 1.0 - np.clip(move, 0.0, 1.0)

    return pd.DataFrame(
        {
            "feature": feat_names,
            "sep_d": sep,            # signed Cohen's d, source->target on this feature
            "abs_sep_d": np.abs(sep),
            "movement": move,        # raw (un-clipped) for diagnostics
            "persistence": persistence,
            "separates": np.abs(sep) > SEP_GATE,
        }
    )


def load_all_cells() -> pd.DataFrame:
    feat_names = structural_matrix(["probe"])[1]
    rows = []
    for dpo_csv in sorted(DECONTAM_ROOT.glob("*/*/gen/rung_dpo.csv")):
        gen = dpo_csv.parent
        dataset = gen.parts[gen.parts.index("decontam") + 1]
        pair = gen.parts[gen.parts.index("decontam") + 2]
        src_full, tgt_full = pair.split("_to_")
        df = per_feature_persistence(gen, feat_names)
        df["dataset"] = dataset
        df["source"] = SHORT[src_full]
        df["target"] = SHORT[tgt_full]
        df["cell"] = f"{dataset}/{pair}"
        rows.append(df)
    if not rows:
        raise SystemExit(f"No DPO cells found under {DECONTAM_ROOT}/*/*/gen/rung_dpo.csv")
    return pd.concat(rows, ignore_index=True)


def _gated_mean(g: pd.DataFrame) -> float:
    """Mean persistence over cells where the feature actually separates s<->t.

    Falls back to the ungated mean only if a feature separates in no cell of the
    group (so the column is never empty), but flags coverage via n_sep.
    """
    sub = g[g["separates"]]
    if len(sub) and sub["persistence"].notna().any():
        return float(sub["persistence"].mean())
    return float(g["persistence"].mean())


def summarize(allc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    feat_names = list(dict.fromkeys(allc["feature"]))

    # ---- per (source, feature): gated mean DPO persistence ----------------
    recs = []
    for (src, feat), g in allc.groupby(["source", "feature"]):
        sub = g[g["separates"]]
        recs.append(
            {
                "source": src,
                "feature": feat,
                "persistence": _gated_mean(g),
                "n_sep": int(len(sub)),          # cells where this feature separates
                "n_cells": int(len(g)),
                "mean_abs_sep_d": float(g["abs_sep_d"].mean()),
            }
        )
    by_src = pd.DataFrame(recs)

    persist_wide = by_src.pivot(index="feature", columns="source", values="persistence")
    nsep_wide = by_src.pivot(index="feature", columns="source", values="n_sep")
    for col in [*SURVIVORS, *LAUNDERERS]:
        if col not in persist_wide.columns:
            persist_wide[col] = np.nan
            nsep_wide[col] = 0

    feat = pd.DataFrame(index=persist_wide.index)
    feat["survivor_persist"] = persist_wide[SURVIVORS].mean(axis=1)
    feat["launderer_persist"] = persist_wide[LAUNDERERS].mean(axis=1)
    # The residue carried by a feature = how much MORE the survivors stay source
    # on it than the launderers do. Positive => survivor-specific structural residue.
    feat["residue_gap"] = feat["survivor_persist"] - feat["launderer_persist"]
    for s in [*SURVIVORS, *LAUNDERERS]:
        feat[f"persist_{s}"] = persist_wide[s]
    feat["survivor_nsep"] = nsep_wide[SURVIVORS].sum(axis=1).astype(int)
    feat["mean_abs_sep_d"] = (
        by_src.groupby("feature")["mean_abs_sep_d"].mean().reindex(feat.index)
    )
    feat = feat.reset_index().sort_values("residue_gap", ascending=False).reset_index(drop=True)

    # ---- concentration of the residue -------------------------------------
    # How much of the total positive survivor-vs-launderer residue lives in the
    # top-k features? If a few named features carry most of it, "structural" is a
    # concentrated, falsifiable claim rather than a diffuse hand-wave.
    pos = feat[feat["residue_gap"] > 0].copy()
    total_pos = float(pos["residue_gap"].sum())
    conc = {}
    cum = 0.0
    for k in (1, 3, 5, 10):
        cum_k = float(pos["residue_gap"].head(k).sum())
        conc[k] = (cum_k / total_pos) if total_pos > 1e-12 else float("nan")
    # Herfindahl on the share distribution (1 = all in one feature, ->0 = diffuse).
    shares = (pos["residue_gap"] / total_pos).to_numpy() if total_pos > 1e-12 else np.array([])
    hhi = float((shares ** 2).sum()) if shares.size else float("nan")

    meta = {
        "n_cells": int(allc["cell"].nunique()),
        "n_features": len(feat_names),
        "survivor_sources": SURVIVORS,
        "launderer_sources": LAUNDERERS,
        "sep_gate": SEP_GATE,
        "survivor_mean_persist_over_features": float(feat["survivor_persist"].mean()),
        "launderer_mean_persist_over_features": float(feat["launderer_persist"].mean()),
        "n_features_positive_residue": int((feat["residue_gap"] > 0).sum()),
        "total_positive_residue_gap": total_pos,
        "concentration_topk_share": conc,
        "residue_hhi": hhi,
    }
    return feat, by_src, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-out", default=None, help="Write per-cell per-feature rows here.")
    ap.add_argument("--feat-csv-out", default=None, help="Write per-feature survivor/launderer summary here.")
    a = ap.parse_args()

    allc = load_all_cells()
    feat, by_src, meta = summarize(allc)

    if a.csv_out:
        Path(a.csv_out).parent.mkdir(parents=True, exist_ok=True)
        allc.to_csv(a.csv_out, index=False)
    if a.feat_csv_out:
        Path(a.feat_csv_out).parent.mkdir(parents=True, exist_ok=True)
        feat.to_csv(a.feat_csv_out, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)

    print("=" * 78)
    print("D4 STRUCTURAL DECOMPOSITION — which named features carry the DPO residue")
    print("=" * 78)
    print(
        f"cells={meta['n_cells']}  features={meta['n_features']}  "
        f"sep-gate |d|>{meta['sep_gate']}  (per-feature: source->0, target->1, "
        f"persistence=1-movement)"
    )
    print(f"survivors={SURVIVORS}   launderers={LAUNDERERS}")
    print(
        f"\nmean DPO persistence over all features:  "
        f"survivors={meta['survivor_mean_persist_over_features']:.3f}   "
        f"launderers={meta['launderer_mean_persist_over_features']:.3f}"
    )

    show = feat[
        [
            "feature",
            "survivor_persist",
            "launderer_persist",
            "residue_gap",
            "persist_gpt-oss",
            "persist_nemotron",
            "persist_qwen",
            "persist_llama",
            "survivor_nsep",
            "mean_abs_sep_d",
        ]
    ].copy()

    print("\n--- TOP features carrying the survivor residue (sorted by survivor-launderer gap) ---")
    print("    (survivor stays source on this feature; launderer collapses to target)")
    print(show.head(12).round(3).to_string(index=False))

    print("\n--- BOTTOM features (launderers stay MORE source than survivors; negative gap) ---")
    print(show.tail(5).round(3).to_string(index=False))

    print("\n--- features survivors keep MOST source-like (absolute survivor persistence) ---")
    print(
        show.sort_values("survivor_persist", ascending=False)
        .head(8)[["feature", "survivor_persist", "launderer_persist", "residue_gap", "survivor_nsep"]]
        .round(3)
        .to_string(index=False)
    )

    c = meta["concentration_topk_share"]
    print("\n--- CONCENTRATION of the residue across the 32 features ---")
    print(f"  features with positive survivor residue: {meta['n_features_positive_residue']}/{meta['n_features']}")
    print(f"  total positive residue-gap mass: {meta['total_positive_residue_gap']:.3f}")
    print(
        f"  top-1 feature carries {c[1]*100:.0f}%   "
        f"top-3 {c[3]*100:.0f}%   top-5 {c[5]*100:.0f}%   top-10 {c[10]*100:.0f}%  of it"
    )
    print(f"  Herfindahl index of the residue shares: {meta['residue_hhi']:.3f}  (1=all in one feature, 0=diffuse)")

    if a.csv_out:
        print(f"\nwrote per-cell rows -> {a.csv_out}")
    if a.feat_csv_out:
        print(f"wrote per-feature summary -> {a.feat_csv_out}")


if __name__ == "__main__":
    main()
