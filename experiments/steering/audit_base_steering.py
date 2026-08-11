#!/usr/bin/env python3
"""Write an exact coverage manifest for canonical base-steering artifacts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.figures import rebuild_steering_figures as figures


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
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "work_root": figures.RDO,
        "harm_benchmarks": figures.HARM_BENCHMARKS,
        "evaluation": {"max_prompts": 200, "subsample_seed": 42},
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
