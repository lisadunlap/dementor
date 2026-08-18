#!/usr/bin/env python3
"""Freeze the remaining seven-benchmark steering and self-SFT campaign work.

This is intentionally stricter than the legacy figure loader: a steering cell is
complete only when its preferred metrics encode the campaign's exact seed-42,
200-prompt sample (XSTest contains 200 total prompts, of which 111 are benign).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dementor import config
from experiments.figures import rebuild_steering_figures as figures


HARM = ("advbench", "harmbench", "strongreject", "sorrybench", "sgbench")
OVERREFUSAL = ("xstest", "orbench_hard")
BENCHMARKS = HARM + OVERREFUSAL


def _physical_roster(live_root: Path) -> tuple[list[str], dict[str, dict]]:
    worklist = json.loads((live_root / "rdo_worklist.json").read_text())
    specs = {item["slug"]: item for item in worklist["models"]}
    canonical_with_results = set(figures.load_cells())
    physical = [
        slug for slug in specs
        if config.canonical_steering_slug(slug) in canonical_with_results
    ]
    if len(physical) != 29 or len({config.canonical_steering_slug(s) for s in physical}) != 29:
        raise RuntimeError(f"expected 29 distinct steering models, found {len(physical)}")
    return physical, specs


def _preferred_metrics(eval_dir: Path) -> dict | None:
    # The shared loader rejects incomplete early "metrics_n200" intersections.
    return figures._load_preferred_metrics(str(eval_dir))


def _exact(eval_dir: Path, benchmark: str) -> bool:
    metrics = _preferred_metrics(eval_dir)
    if not metrics or metrics.get("benchmark", benchmark) != benchmark:
        return False
    provenance = metrics.get("_evaluation_provenance", {})
    return provenance.get("n_prompts") == 200


def _legacy_eval_dir(live_root: Path, slug: str, benchmark: str, suffix: str) -> Path:
    candidate = live_root / slug / f"eval_{benchmark}{suffix}"
    # One late-added model retains its canonical AdvBench cell in the older bare
    # ``eval/`` location. The paper loader intentionally uses that as fallback.
    if not suffix and benchmark == "advbench" and _preferred_metrics(candidate) is None:
        return live_root / slug / "eval"
    return candidate


def _gpu_count(slug: str, spec: dict) -> int:
    if slug == "llama-3.3-70b":
        return 3
    return 2 if spec.get("needs_mp") else 1


def build_steering(live_root: Path, campaign_root: Path) -> dict:
    roster, specs = _physical_roster(live_root)
    tasks = []
    completed = []
    for operator in ("original", "fpall"):
        suffix = "" if operator == "original" else "_fpall"
        for slug in roster:
            for benchmark in BENCHMARKS:
                legacy = _legacy_eval_dir(live_root, slug, benchmark, suffix)
                record = {
                    "id": f"{operator}:{slug}:{benchmark}",
                    "operator": operator,
                    "slug": slug,
                    "canonical_slug": config.canonical_steering_slug(slug),
                    "benchmark": benchmark,
                    "max_prompts": 200,
                    "subsample_seed": 42,
                    "gpu_count": _gpu_count(slug, specs[slug]),
                    "params_b": specs[slug].get("params_b"),
                    "legacy_eval_dir": str(legacy),
                    "campaign_model_dir": str(campaign_root / "steering" / slug),
                }
                if _exact(legacy, benchmark):
                    record["status"] = "complete_existing"
                    completed.append(record)
                else:
                    record["status"] = "pending"
                    tasks.append(record)

    expected = Counter((x["operator"], "harm" if x["benchmark"] in HARM else "overrefusal") for x in tasks)
    expected_counts = {
        "original:harm": 86,
        "original:overrefusal": 56,
        "fpall:harm": 90,
        "fpall:overrefusal": 58,
    }
    actual_counts = {f"{op}:{axis}": n for (op, axis), n in sorted(expected.items())}
    if actual_counts != expected_counts:
        raise RuntimeError(f"live coverage changed: expected {expected_counts}, found {actual_counts}")
    if len(tasks) != 290 or len(completed) != 116:
        raise RuntimeError(f"expected 290 pending + 116 complete, got {len(tasks)} + {len(completed)}")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live_root": str(live_root),
        "campaign_root": str(campaign_root),
        "design": {"models": 29, "benchmarks": list(BENCHMARKS), "operators": ["original", "fpall"]},
        "counts": {"target": 406, "complete_existing": len(completed), "pending": len(tasks), **actual_counts},
        "tasks": tasks,
        "complete_existing": completed,
    }


def build_self_sft() -> dict:
    models = config.campaign_roster()
    datasets = config.campaign_dataset_names()
    existing = {"aya-expanse-8b", "llama-3.1-8b", "olmo-3-7b", "ministral-8b"}
    cells = []
    for model in models:
        for dataset in datasets:
            cells.append({
                "id": f"self_sft_{dataset}_{model['slug']}_as_{model['slug']}_seed42",
                "source": model["id"],
                "slug": model["slug"],
                "dataset": dataset,
                "seed": 42,
                "backend": model["backend"],
                "status": "complete_existing" if model["slug"] in existing else "pending",
            })
    counts = Counter((x["status"], x["backend"]) for x in cells)
    if len(cells) != 48 or counts[("complete_existing", "local")] != 16:
        raise RuntimeError(f"unexpected self-SFT coverage: {dict(counts)}")
    if counts[("pending", "tinker")] != 12 or counts[("pending", "local")] != 20:
        raise RuntimeError(f"unexpected self-SFT pending split: {dict(counts)}")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "target": 48,
            "complete_existing": 16,
            "pending": 32,
            "pending_tinker": 12,
            "pending_local": 20,
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live-root",
        type=Path,
        required=True,
        help="source steering evaluation root containing rdo_worklist.json",
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.campaign_root / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    steering = build_steering(args.live_root, args.campaign_root)
    self_sft = build_self_sft()
    (out / "steering_completion.json").write_text(json.dumps(steering, indent=2) + "\n")
    (out / "self_sft_completion.json").write_text(json.dumps(self_sft, indent=2) + "\n")
    print(json.dumps({"steering": steering["counts"], "self_sft": self_sft["counts"]}, indent=2))


if __name__ == "__main__":
    main()
