"""D3 — encoder-swap robustness. Re-score the DE-CONFOUNDED cells (decontam cleaned
gen) with a structurally different text encoder and check the rung ordering and the
DPO survivor set replicate — i.e. persistence is not a MiniLM artifact. No
regeneration; reuses the cleaned gen from `decontaminate.py`. Resumable (per-cell).

Usage: python -m scripts.analysis.encoder_swap [--encoder sentence-transformers/all-mpnet-base-v2]
"""
from __future__ import annotations

import argparse
import types
from pathlib import Path

import pandas as pd

from scripts.analysis.decontaminate import _summary
from scripts.analysis.run_cell_pipeline import assemble_and_run

DATA = Path("data")
DECONTAM = DATA / "results" / "decontam"
RUNGS = ["just_name_it", "random_sampling", "stylistic", "sft", "dpo"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--bootstrap", type=int, default=300)
    a = ap.parse_args()
    tag = a.encoder.split("/")[-1]
    out_root = DATA / "results" / f"encoder_swap_{tag}"
    pe = "persistence_" + tag

    cells = sorted(c for c in DECONTAM.glob("*/*") if (c / "gen").is_dir())
    rows = []
    for i, cell in enumerate(cells, 1):
        dataset = cell.parts[cell.parts.index("decontam") + 1]  # cells live under results/decontam/<dataset>/
        src, tgt = cell.name.split("_to_")
        out_cell = out_root / dataset / cell.name
        cache = out_cell / "cell_summary.csv"
        print(f"  [{i}/{len(cells)}] {dataset}/{cell.name}" + (" (cached)" if cache.exists() else ""), flush=True)
        try:
            if not cache.exists():
                args = types.SimpleNamespace(dataset=dataset, source=src, target=tgt,
                                             bootstrap=a.bootstrap, adapter_seeds=1,
                                             calibration_judge=None, calibration_n=0, encoder_model=a.encoder)
                assemble_and_run(args, out_cell, cell / "gen")
            new = _summary(cache).rename(columns={"persistence": pe})
            old = _summary(cell / "cell_summary.csv").rename(columns={"persistence": "persistence_minilm"})
            m = new.merge(old, on="rung")
            m["dataset"], m["source"], m["target"] = dataset, src, tgt
            rows.append(m)
        except Exception as exc:
            print(f"    [skip] {type(exc).__name__}: {str(exc)[:100]}", flush=True)

    allc = pd.concat(rows, ignore_index=True)
    out = DATA / "results" / f"encoder_swap_{tag}.csv"
    allc.to_csv(out, index=False)
    pm = "persistence_minilm"
    print(f"\n=== encoder-swap: MiniLM vs {tag}  (n={len(allc)} cell-rungs) ===")
    print(f"  Spearman = {allc[pm].corr(allc[pe], method='spearman'):.3f} | Pearson = {allc[pm].corr(allc[pe]):.3f}")
    g = allc.groupby("rung").agg(minilm=(pm, "mean"), other=(pe, "mean")).reindex(RUNGS)
    print("\nrung-mean ladder under each encoder (does the ordering hold?):")
    print(g.round(3).to_string())
    d = allc[allc["rung"] == "dpo"]
    sm = set(zip(d.loc[d[pm] > 0.3, "source"], d.loc[d[pm] > 0.3, "target"], d.loc[d[pm] > 0.3, "dataset"]))
    se = set(zip(d.loc[d[pe] > 0.3, "source"], d.loc[d[pe] > 0.3, "target"], d.loc[d[pe] > 0.3, "dataset"]))
    jac = len(sm & se) / max(1, len(sm | se))
    print(f"\nDPO survivor set (>0.3): MiniLM={len(sm)}, {tag}={len(se)}, Jaccard={jac:.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
