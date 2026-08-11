#!/usr/bin/env python3
"""Write an exact coverage manifest for canonical base-steering artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.figures import rebuild_steering_figures as figures
from dementor import config


def build_manifest() -> dict:
    base = figures.load_cells()
    fpall = figures.load_cells("fpall")
    models = sorted(base)
    gated = figures.roster(base)
    full_base = [model for model in models
                 if all(benchmark in base[model] for benchmark in figures.HARM_BENCHMARKS)]
    full_fpall = [model for model in models
                  if all(benchmark in fpall.get(model, {})
                         for benchmark in figures.HARM_BENCHMARKS)]
    gated_full_fpall = sorted(set(gated) & set(full_fpall))

    def cell_record(cell: dict) -> dict:
        return {
            "verdict": cell.get("verdict"),
            "n_prompts": cell.get("n_prompts"),
            "sampling": cell.get("sampling"),
            "subsample_seed": cell.get("subsample_seed"),
            "metrics_source": cell.get("metrics_source"),
        }

    def sampling_summary(cells: dict) -> dict:
        counts = Counter(
            (cell.get("sampling") or "unknown", str(cell.get("n_prompts") or "unknown"))
            for benchmarks in cells.values()
            for benchmark, cell in benchmarks.items()
            if benchmark in figures.HARM_BENCHMARKS
        )
        return {
            f"{sampling}:n={n_prompts}": count
            for (sampling, n_prompts), count in sorted(counts.items())
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "work_root": figures.RDO,
        "harm_benchmarks": figures.HARM_BENCHMARKS,
        "evaluation_standard": {
            "max_prompts": config.campaign_evaluation()["max_prompts"],
            "subsample_seed": config.campaign_evaluation()["subsample_seed"],
            "scope": "new and exactly harmonized cells; legacy cells retain native denominators",
        },
        "sampling_summary": {
            "base": sampling_summary(base),
            "fpall": sampling_summary(fpall),
        },
        "counts": {
            "models_with_any_canonical_base_cell": len(models),
            "models_with_complete_base_harm": len(full_base),
            "positive_control_gated_models": len(gated),
            "models_with_complete_fpall_harm": len(full_fpall),
            "gated_models_with_complete_fpall_harm": len(gated_full_fpall),
        },
        "complete_base_harm_models": full_base,
        "positive_control_gated_models": gated,
        "complete_fpall_harm_models": full_fpall,
        "gated_complete_fpall_harm_models": gated_full_fpall,
        "models": {
            model: {
                "base_harm_benchmarks": sorted(set(base[model]) & set(figures.HARM_BENCHMARKS)),
                "fpall_harm_benchmarks": sorted(
                    set(fpall.get(model, {})) & set(figures.HARM_BENCHMARKS)
                ),
                "base_verdicts": {
                    benchmark: base[model][benchmark]["verdict"]
                    for benchmark in figures.HARM_BENCHMARKS
                    if benchmark in base[model]
                },
                "base_cells": {
                    benchmark: cell_record(base[model][benchmark])
                    for benchmark in figures.HARM_BENCHMARKS
                    if benchmark in base[model]
                },
                "fpall_cells": {
                    benchmark: cell_record(fpall[model][benchmark])
                    for benchmark in figures.HARM_BENCHMARKS
                    if benchmark in fpall.get(model, {})
                },
            }
            for model in models
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/results/safety/base_steering_coverage.json")
    args = parser.parse_args()
    manifest = build_manifest()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
