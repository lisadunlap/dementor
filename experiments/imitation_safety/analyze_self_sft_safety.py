#!/usr/bin/env python3
"""Aggregate the strict 48-cell self-SFT safety control with the publication RTL metric."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dementor import config

HARM = ("advbench", "harmbench", "strongreject", "sorrybench", "sgbench")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True,
                        help="completion self_sft/eval/work directory")
    parser.add_argument("--output", type=Path,
                        default=Path("data/results/safety/self_sft_headlines.json"))
    args = parser.parse_args()
    roster = [model["slug"] for model in config.campaign_roster("imitation_safety")]
    datasets = config.campaign_dataset_names("imitation_safety")
    expected = {
        f"self_sft_{dataset}_{slug}_as_{slug}_seed42"
        for dataset in datasets for slug in roster
    }
    rows = []
    problems = {}
    for item_id in sorted(expected):
        path = args.work_root / item_id / "metrics.json"
        try:
            item = json.loads(path.read_text())
            baseline = json.loads(
                (args.work_root / f"baseline_{item['source']}" / "metrics.json").read_text()
            )
            effects = []
            for benchmark in HARM:
                adapted = item["per_benchmark"][benchmark]
                base = baseline["per_benchmark"][benchmark]
                if adapted["n"] != 200 or base["n"] != 200:
                    raise ValueError(f"{benchmark}: denominator mismatch")
                effects.append(float(adapted["rtl_genuine_harm"])
                               - float(base["rtl_genuine_harm"]))
            rows.append({"id": item_id, "source": item["source"], "dataset": item["dataset"],
                         "mean_harm_erosion": float(np.mean(effects))})
        except Exception as error:
            problems[item_id] = str(error)
    if problems or len(rows) != 48:
        raise RuntimeError(f"self-SFT coverage failed: rows={len(rows)} problems={problems}")
    values = np.asarray([row["mean_harm_erosion"] for row in rows])
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {"adapters": 48, "harm_benchmarks_per_adapter": 5,
                     "all_safety_scores": 336, "prompts_per_harm_benchmark": 200},
        "mean_harm_erosion": float(values.mean()),
        "median_harm_erosion": float(np.median(values)),
        "min_harm_erosion": float(values.min()),
        "max_harm_erosion": float(values.max()),
        "by_source": {source: float(np.mean([row["mean_harm_erosion"] for row in rows
                                               if row["source"] == source]))
                      for source in roster},
        "by_dataset": {dataset: float(np.mean([row["mean_harm_erosion"] for row in rows
                                                 if row["dataset"] == dataset]))
                       for dataset in datasets},
        "cells": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"adapters": 48, "mean_harm_erosion_pp": 100 * values.mean(),
                      "median_harm_erosion_pp": 100 * np.median(values)}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
