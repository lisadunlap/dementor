"""Batch driver over the open-source matrix: run a behavioral-inertia cell for each
(source, target) pair on a dataset, then aggregate every cell into one cross-cell
ladder (seeds averaged per rung) + figure — the headline deliverable.

Reuses run_cell_pipeline per cell (idempotent: cached generations are skipped), so
reruns are cheap. Use --aggregate-only to (re)build the figure from cells already
on disk without touching Tinker.

Usage:
  python -m scripts.analysis.run_matrix_cells --dataset gsm8k \
      [--sources llama-3.1-8b ...] [--targets ...] [--adapter-seeds 1] [--aggregate-only]
"""
from __future__ import annotations

import argparse
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.analysis.run_cell_pipeline import DATA, SLUG_TO_MODEL, assemble_and_run, generate
from scripts.analysis.run_intervention_ladder import INTERVENTION_ORDER

MODELS = list(SLUG_TO_MODEL)


def _rung_base(method: str) -> str:
    return str(method).split("_seed")[0]  # sft_seed2 -> sft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sources", nargs="*", default=MODELS, choices=MODELS)
    ap.add_argument("--targets", nargs="*", default=MODELS, choices=MODELS)
    ap.add_argument("--eval-size", type=int, default=200)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--adapter-seeds", type=int, default=1)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--aggregate-only", action="store_true", help="Skip generation; rebuild figure from cells on disk.")
    a = ap.parse_args()

    out = DATA / f"results/{a.dataset}/analysis/matrix_ladder"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for src in a.sources:
        for tgt in a.targets:
            if src == tgt:
                continue
            cell = DATA / f"results/{a.dataset}/analysis/cells/{src}_to_{tgt}"
            gen = cell / "gen"
            cargs = types.SimpleNamespace(
                dataset=a.dataset, source=src, target=tgt, eval_size=a.eval_size,
                parallel=a.parallel, temperature=0.7, bootstrap=a.bootstrap,
                adapter_seeds=a.adapter_seeds, calibration_judge=None, calibration_n=0,
            )
            if not a.aggregate_only:
                print(f"\n### cell {src} -> {tgt}", flush=True)
                try:
                    generate(cargs, gen)
                    assemble_and_run(cargs, cell, gen)
                except Exception as exc:
                    print(f"  [skip cell] {src}->{tgt}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                    continue
            summary_csv = cell / "cell_summary.csv"
            if summary_csv.exists():
                df = pd.read_csv(summary_csv)
                df["source"], df["target"] = src, tgt
                frames.append(df)

    if not frames:
        print("No cells found to aggregate.")
        return

    allc = pd.concat(frames, ignore_index=True)
    allc["rung"] = allc["method"].map(_rung_base)
    agg = (
        allc.groupby(["source", "target", "rung"])
        .agg(
            persistence=("persistence", "mean"),
            persistence_sd=("persistence", "std"),
            anchored=("anchored", "mean") if "anchored" in allc.columns else ("persistence", "mean"),
            n_seeds=("method", "nunique"),
            trustworthy=("trustworthy", "all") if "trustworthy" in allc.columns else ("persistence", "count"),
        )
        .reset_index()
    )
    agg["order"] = agg["rung"].map(
        lambda m: INTERVENTION_ORDER.index(m) if m in INTERVENTION_ORDER else len(INTERVENTION_ORDER)
    )
    agg = agg.sort_values(["source", "target", "order"]).reset_index(drop=True)
    agg.to_csv(out / "matrix_ladder.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    for (src, tgt), grp in agg.groupby(["source", "target"]):
        grp = grp.sort_values("order")
        ax.plot(grp["rung"], grp["persistence"], marker="o", label=f"{src} → {tgt}")
    ax.set_ylabel("persistence  (1 = stays source, 0 = reaches target)")
    ax.set_xlabel("intervention rung")
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.axhline(1.0, color="gray", lw=0.6)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.set_title(f"Fingerprint persistence across intervention rungs — {a.dataset}")
    fig.tight_layout()
    fig.savefig(out / "matrix_ladder.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    n_pairs = agg[["source", "target"]].drop_duplicates().shape[0]
    print(f"\nwrote {out}/matrix_ladder.csv + matrix_ladder.png  ({n_pairs} pairs)")
    print(agg[["source", "target", "rung", "persistence", "persistence_sd", "anchored", "n_seeds", "trustworthy"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
