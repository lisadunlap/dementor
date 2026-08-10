#!/usr/bin/env python
"""Audit exact SFT/DPO safety coverage for a named config campaign.

The audit is read-only. Multiple ``--work-root`` arguments logically synchronize
boxes: identical checkpoints are deduplicated by id and conflicting checkpoints
abort. The JSON manifest is safe to use as the missing-cell worklist source.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import build_erosion_csv as BUILD  # noqa: E402
import erosion_common as EC  # noqa: E402
from dementor import config  # noqa: E402


def expected_ids(campaign: str, stage: str) -> list[str]:
    slugs = [m["slug"] for m in config.campaign_roster(campaign)]
    return [
        f"{stage}_{dataset}_{source}_as_{target}_seed{seed}"
        for dataset in config.campaign_dataset_names(campaign)
        for source in slugs
        for target in slugs
        if source != target
        for seed in config.campaign_seeds(campaign)
    ]


def item_problems(
    item: dict,
    baseline: dict | None,
    evaluation: dict | None = None,
) -> list[str]:
    problems = []
    evaluation = evaluation or {}
    expected_max = evaluation.get("max_prompts")
    expected_seed = evaluation.get("subsample_seed")
    for label, payload in (("adapter", item), ("baseline", baseline)):
        if payload is None:
            continue
        if expected_max is not None and payload.get("subsample_max_prompts") != expected_max:
            problems.append(
                f"{label}:max_prompts={payload.get('subsample_max_prompts')}!={expected_max}"
            )
        if expected_seed is not None and payload.get("subsample_seed") != expected_seed:
            problems.append(
                f"{label}:subsample_seed={payload.get('subsample_seed')}!={expected_seed}"
            )
    if baseline is None:
        problems.append("baseline_missing")
    per_benchmark = item.get("per_benchmark", {})
    for benchmark in EC.DEFAULT_BENCHMARKS:
        metric = per_benchmark.get(benchmark)
        if metric is None:
            problems.append(f"{benchmark}:missing")
            continue
        if baseline is not None:
            base_metric = baseline.get("per_benchmark", {}).get(benchmark)
            if base_metric is None:
                problems.append(f"{benchmark}:baseline_missing")
            elif metric.get("n") != base_metric.get("n"):
                problems.append(
                    f"{benchmark}:n={metric.get('n')}!=baseline_n={base_metric.get('n')}"
                )
    return problems


def generation_state(item_id: str, roots: list[str]) -> dict:
    """Classify an item without top-level metrics by its resumable generation checkpoints.

    ``missing`` historically meant only "no metrics.json", which made a fully generated cell look
    safe to regenerate.  Report benchmark CSVs across all synchronized roots so dispatchers can
    send complete generations directly to batched judging and generate only genuinely absent data.
    """
    present = [
        benchmark
        for benchmark in EC.DEFAULT_BENCHMARKS
        if any((Path(root) / item_id / benchmark / "all_gens.csv").exists() for root in roots)
    ]
    missing = [benchmark for benchmark in EC.DEFAULT_BENCHMARKS if benchmark not in present]
    if not missing:
        status = "generated_only"
    elif present:
        status = "partially_generated"
    else:
        status = "missing_generation"
    return {"status": status, "present": present, "missing": missing}


def audit(campaign: str, roots: list[str]) -> dict:
    expected_by_stage = {stage: expected_ids(campaign, stage) for stage in ("sft", "dpo")}
    expected_set = set(expected_by_stage["sft"]) | set(expected_by_stage["dpo"])
    evaluation = config.campaign_evaluation(campaign)
    items = BUILD.load_items(
        roots,
        accept=lambda item: item.get("kind") == "baseline" or item.get("id") in expected_set,
    )
    baselines = {
        item["base_model"]: item for item in items.values() if item.get("kind") == "baseline"
    }
    report = {
        "campaign": campaign,
        "work_roots": roots,
        "evaluation": evaluation,
        "models": [m["slug"] for m in config.campaign_roster(campaign)],
        "datasets": config.campaign_dataset_names(campaign),
        "seeds": config.campaign_seeds(campaign),
        "stages": {},
    }
    for stage in ("sft", "dpo"):
        expected = expected_by_stage[stage]
        missing = []
        generated_only = []
        partially_generated = {}
        missing_generation = []
        incomplete = {}
        complete = []
        for item_id in expected:
            item = items.get(item_id)
            if item is None:
                missing.append(item_id)
                generation = generation_state(item_id, roots)
                if generation["status"] == "generated_only":
                    generated_only.append(item_id)
                elif generation["status"] == "partially_generated":
                    partially_generated[item_id] = generation
                else:
                    missing_generation.append(item_id)
                continue
            problems = item_problems(
                item,
                baselines.get(item.get("base_model")),
                evaluation,
            )
            if problems:
                incomplete[item_id] = problems
            else:
                complete.append(item_id)
        report["stages"][stage] = {
            "expected": len(expected),
            "complete": len(complete),
            "missing": missing,
            "generated_only": generated_only,
            "partially_generated": partially_generated,
            "missing_generation": missing_generation,
            "incomplete": incomplete,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="imitation_safety")
    parser.add_argument("--work-root", action="append", dest="work_roots")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="exit nonzero unless both stages are complete")
    args = parser.parse_args()
    roots = args.work_roots or [EC.WORK]
    report = audit(args.campaign, roots)
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"[coverage] wrote {args.output}")
    for stage, stage_report in report["stages"].items():
        print(
            f"[coverage] {stage}: {stage_report['complete']}/{stage_report['expected']} complete; "
            f"{len(stage_report['missing'])} missing metrics "
            f"({len(stage_report['generated_only'])} generated-only, "
            f"{len(stage_report['partially_generated'])} partial, "
            f"{len(stage_report['missing_generation'])} need generation); "
            f"{len(stage_report['incomplete'])} incomplete"
        )
    if args.strict and any(
        result["complete"] != result["expected"] for result in report["stages"].values()
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
