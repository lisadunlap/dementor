"""D1 de-confound: re-evaluate the 36-cell matrix on chat-template-CLEANED text.

Applies the FIXED `clean_response` (workflows.run_matrix) to each cell's cached
gen/ outputs and re-runs the behavioural evaluator — no regeneration, no retraining
— then reports a before/after persistence table with each cell's gpt-oss CoT-leak
rate as a covariate. Resumable (per-cell cache).

Usage: python -m scripts.analysis.decontaminate
"""
from __future__ import annotations

import types
from pathlib import Path

import pandas as pd

from scripts.analysis.run_cell_pipeline import assemble_and_run, SLUG_TO_MODEL
from workflows.run_matrix import clean_response

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


def reeval(cell: Path, bootstrap: int = 300) -> pd.DataFrame:
    dataset, src, tgt = parse_cell(cell)
    src_id, tgt_id = SLUG_TO_MODEL[src], SLUG_TO_MODEL[tgt]
    out_cell = CLEAN_ROOT / dataset / cell.name
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
                                 adapter_seeds=1, calibration_judge=None, calibration_n=0)
    assemble_and_run(args, out_cell, cgen)
    cs = _summary(out_cell / "cell_summary.csv")
    cs["dataset"], cs["source"], cs["target"], cs["leak_rate"] = dataset, src, tgt, leak_rate(cell)
    return cs


def main() -> None:
    cells = [c for c in sorted(DATA.glob("results/*/analysis/cells/*")) if (c / "staged").is_dir()]
    old_parts = []
    for p in DATA.glob("results/*/analysis/matrix_ladder/matrix_ladder.csv"):
        o = pd.read_csv(p)
        o["dataset"] = p.parts[p.parts.index("results") + 1]  # matrix_ladder.csv has no dataset col
        old_parts.append(o)
    old = pd.concat(old_parts, ignore_index=True)[["dataset", "source", "target", "rung", "persistence"]] \
        .rename(columns={"persistence": "persistence_before"})

    frames = []
    for i, cell in enumerate(cells, 1):
        dataset, src, tgt = parse_cell(cell)
        cache = CLEAN_ROOT / dataset / cell.name / "cell_summary.csv"
        print(f"  [{i}/{len(cells)}] {dataset}/{cell.name}" + (" (cached)" if cache.exists() else ""), flush=True)
        try:
            if cache.exists():
                cs = _summary(cache)
                cs["dataset"], cs["source"], cs["target"], cs["leak_rate"] = dataset, src, tgt, leak_rate(cell)
                frames.append(cs)
            else:
                frames.append(reeval(cell))
        except Exception as exc:
            print(f"    [skip] {type(exc).__name__}: {str(exc)[:120]}", flush=True)

    new = pd.concat(frames, ignore_index=True).rename(columns={"persistence": "persistence_after"})
    merged = new.merge(old, on=["dataset", "source", "target", "rung"], how="left")
    merged["delta"] = merged["persistence_after"] - merged["persistence_before"]
    out = DATA / "results" / "decontam_before_after.csv"
    merged.to_csv(out, index=False)

    dpo = merged[merged["rung"] == "dpo"].copy()
    dpo["pair"] = dpo["source"].str.replace("-3.1-8b", "").str.replace("3.6-27b", "").str.replace("-nano-30b-a3b", "") \
        + "->" + dpo["target"].str.replace("-3.1-8b", "").str.replace("3.6-27b", "").str.replace("-nano-30b-a3b", "")
    cols = ["dataset", "pair", "leak_rate", "persistence_before", "persistence_after", "delta"]
    print("\n=== DPO persistence: before vs after de-confound (highest-before first) ===")
    print(dpo.sort_values("persistence_before", ascending=False)[cols].round(3).head(12).to_string(index=False))
    leaky = dpo[dpo["leak_rate"] > 0.1]
    print(f"\nleaky cells (leak>0.1): n={len(leaky)}, mean delta={leaky['delta'].mean():.3f} "
          f"| clean cells: mean |delta|={dpo[dpo['leak_rate'] <= 0.1]['delta'].abs().mean():.3f}")
    print(f"corr(leak_rate, delta) across 36 DPO cells = {dpo['leak_rate'].corr(dpo['delta']):.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
