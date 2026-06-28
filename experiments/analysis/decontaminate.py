"""D1 de-confound: re-evaluate the 36-cell matrix on chat-template-CLEANED text.

Applies the FIXED `clean_response` (dementor.training.matrix) to each cell's cached
gen/ outputs and re-runs the behavioural evaluator — no regeneration, no retraining
— then reports a before/after persistence table with each cell's gpt-oss CoT-leak
rate as a covariate. Resumable (per-cell cache).

Usage: python -m experiments.analysis.decontaminate
"""
from __future__ import annotations

import argparse
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path("data")
CLEAN_ROOT = DATA / "results" / "decontam"
RUNGS = ["just_name_it", "random_sampling", "stylistic", "sft", "dpo"]


def parse_cell(cell: Path):
    src, tgt = cell.name.split("_to_")
    dataset = cell.parts[cell.parts.index("results") + 1]
    return dataset, src, tgt


def model_for(fname: str, src_id: str, tgt_id: str) -> str:
    # target_seed*.csv are the target model; source baselines + every rung are the source.
    return tgt_id if fname.startswith("target_") else src_id


def leak_rate(cell: Path) -> float:
    f = cell / "staged" / "dpo.csv"
    if not f.exists():
        return float("nan")
    d = pd.read_csv(f)
    return float(d["model_response"].astype(str).str.contains("<|channel|>", regex=False).mean())


def _summary(path: Path) -> pd.DataFrame:
    """cell_summary.csv stores the rung name under `method` (feature_set=='full')."""
    cs = pd.read_csv(path)
    if "feature_set" in cs.columns:
        cs = cs[cs["feature_set"] == "full"]
    return cs[["method", "persistence"]].rename(columns={"method": "rung"})


def reeval(cell: Path, root: Path, *, adapter_seeds: int = 1, bootstrap: int = 300) -> pd.DataFrame:
    # Imported lazily (torch/transformers-heavy) so that importing this module just for
    # the small statistical helpers below (e.g. _ci95_halfwidth in tests) stays CPU-only.
    from dementor.metric.run_cell_pipeline import assemble_and_run, SLUG_TO_MODEL
    from dementor.training.matrix import clean_response
    dataset, src, tgt = parse_cell(cell)
    src_id, tgt_id = SLUG_TO_MODEL[src], SLUG_TO_MODEL[tgt]
    out_cell = root / dataset / cell.name
    cgen = out_cell / "gen"
    cgen.mkdir(parents=True, exist_ok=True)
    for f in (cell / "gen").glob("*.csv"):
        d = pd.read_csv(f)
        if "model_response" in d.columns:
            mid = model_for(f.name, src_id, tgt_id)
            d["model_response"] = [clean_response(mid, str(t)) if pd.notna(t) else t
                                   for t in d["model_response"]]
        d.to_csv(cgen / f.name, index=False)
    args = types.SimpleNamespace(dataset=dataset, source=src, target=tgt, bootstrap=bootstrap,
                                 adapter_seeds=adapter_seeds, calibration_judge=None, calibration_n=0)
    assemble_and_run(args, out_cell, cgen)
    cs = _summary(out_cell / "cell_summary.csv")
    cs["dataset"], cs["source"], cs["target"], cs["leak_rate"] = dataset, src, tgt, leak_rate(cell)
    return cs


def _pair(df):
    s = df["source"].str.replace("-3.1-8b", "").str.replace("3.6-27b", "").str.replace("-nano-30b-a3b", "")
    t = df["target"].str.replace("-3.1-8b", "").str.replace("3.6-27b", "").str.replace("-nano-30b-a3b", "")
    return s + "->" + t


def _report_before_after(frames) -> None:
    old_parts = []
    for p in DATA.glob("results/*/analysis/matrix_ladder/matrix_ladder.csv"):
        o = pd.read_csv(p)
        o["dataset"] = p.parts[p.parts.index("results") + 1]  # matrix_ladder.csv has no dataset col
        old_parts.append(o)
    old = pd.concat(old_parts, ignore_index=True)[["dataset", "source", "target", "rung", "persistence"]] \
        .rename(columns={"persistence": "persistence_before"})
    new = pd.concat(frames, ignore_index=True).rename(columns={"persistence": "persistence_after"})
    merged = new.merge(old, on=["dataset", "source", "target", "rung"], how="left")
    merged["delta"] = merged["persistence_after"] - merged["persistence_before"]
    out = DATA / "results" / "decontam_before_after.csv"
    merged.to_csv(out, index=False)
    dpo = merged[merged["rung"] == "dpo"].copy()
    dpo["pair"] = _pair(dpo)
    cols = ["dataset", "pair", "leak_rate", "persistence_before", "persistence_after", "delta"]
    print("\n=== DPO persistence: before vs after de-confound (highest-before first) ===")
    print(dpo.sort_values("persistence_before", ascending=False)[cols].round(3).head(12).to_string(index=False))
    print(f"corr(leak_rate, delta) across 36 DPO cells = {dpo['leak_rate'].corr(dpo['delta']):.2f}")
    print(f"wrote {out}")


def _ci95_halfwidth(sd, n):
    """Per-cell 95% CI half-width using the Student-t multiplier, not the normal 1.96.

        half_width = t(0.975, df=n-1) * sd / sqrt(n)

    Two correctness fixes over the old ``1.96 * sd.fillna(0) / sqrt(n)``:
      * For the n=3 adapter seeds the right 95% multiplier is t(0.975, df=2)=4.303, not
        1.96 — the normal approximation understates the half-width ~2.2x at n=3.
      * n<2 has no within-cell variance estimate, so the CI is UNDEFINED (-> NaN) and the
        cell is excluded from CI-based survivor tests, instead of being handed a spurious
        zero-width interval by ``sd.fillna(0)`` (which let n=1 cells falsely "survive").

    On the current data this changes the DPO survivor count (``mean - ci95 > 0.3``) from
    6 (buggy 1.96 + fillna(0)) to 3 (correct Student-t, n>=2 only) — verified.

    Accepts scalars or arrays/Series; returns a NumPy value/array (NaN where n<2).
    """
    sd = np.asarray(sd, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        hw = stats.t.ppf(0.975, df=np.maximum(n - 1.0, 1.0)) * sd / np.sqrt(n)
    return np.where(n >= 2, hw, np.nan)


def _report_multiseed(frames, seeds) -> None:
    d = pd.concat(frames, ignore_index=True)
    d["base_rung"] = d["rung"].str.replace(r"_seed\d+$", "", regex=True)
    g = (d.groupby(["dataset", "source", "target", "base_rung"])["persistence"]
         .agg(mean="mean", sd="std", n="count").reset_index())
    # Student-t 95% half-width (t(.975,df=n-1)), NOT the normal 1.96; n<2 -> NaN (excluded).
    # This changes the DPO survivor count (mean-ci95>0.3) from 6 (buggy 1.96+fillna0) to 3.
    g["ci95"] = _ci95_halfwidth(g["sd"], g["n"])
    out = DATA / "results" / f"multiseed_ci_s{seeds}.csv"
    g.to_csv(out, index=False)
    for rung in ("dpo", "sft"):
        r = g[g["base_rung"] == rung].copy()
        r["pair"] = _pair(r)
        print(f"\n=== {rung.upper()}: persistence mean ± 95% CI across {seeds} seeds (de-confounded) ===")
        print(r.sort_values("mean", ascending=False)[["dataset", "pair", "mean", "ci95", "n"]]
              .round(3).head(10).to_string(index=False))
    dpo = g[g["base_rung"] == "dpo"]
    # n<2 cells carry an UNDEFINED CI (ci95=NaN) and are EXCLUDED — not handed a spurious
    # zero-width interval that lets them falsely clear the survivor bar.
    surv = dpo[(dpo["n"] >= 2) & (dpo["mean"] - dpo["ci95"] > 0.3)]
    print(f"\nDPO survivors with 95%-CI lower bound > 0.3 (Student-t, n>=2): {len(surv)} cells "
          f"(median seed-sd across cells: {dpo['sd'].median():.3f})")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-seeds", type=int, default=1, help="Evaluate sft/dpo over N seeds (3 = CIs).")
    ap.add_argument("--bootstrap", type=int, default=300)
    a = ap.parse_args()
    root = DATA / "results" / ("decontam" if a.adapter_seeds == 1 else f"decontam_s{a.adapter_seeds}")

    cells = [c for c in sorted(DATA.glob("results/*/analysis/cells/*")) if (c / "staged").is_dir()]
    frames = []
    for i, cell in enumerate(cells, 1):
        dataset, src, tgt = parse_cell(cell)
        cache = root / dataset / cell.name / "cell_summary.csv"
        print(f"  [{i}/{len(cells)}] {dataset}/{cell.name}" + (" (cached)" if cache.exists() else ""), flush=True)
        try:
            if cache.exists():
                cs = _summary(cache)
                cs["dataset"], cs["source"], cs["target"], cs["leak_rate"] = dataset, src, tgt, leak_rate(cell)
                frames.append(cs)
            else:
                frames.append(reeval(cell, root, adapter_seeds=a.adapter_seeds, bootstrap=a.bootstrap))
        except Exception as exc:
            print(f"    [skip] {type(exc).__name__}: {str(exc)[:120]}", flush=True)

    if a.adapter_seeds > 1:
        _report_multiseed(frames, a.adapter_seeds)
    else:
        _report_before_after(frames)


if __name__ == "__main__":
    main()
