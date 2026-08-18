#!/usr/bin/env python3
"""Strict final audit for the frozen steering + self-SFT completion campaign."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dementor import config
from experiments.completion.build_completion_manifests import BENCHMARKS, _preferred_metrics
from experiments.imitation_safety import audit_erosion_coverage as erosion_audit
from experiments.imitation_safety import erosion_common as EC


def _prompt_set(path: Path) -> set[str] | None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return {row["prompt"] for row in csv.DictReader(handle)}
    except (OSError, KeyError, csv.Error):
        return None


def _steering_eval_dir(record: dict, pending: bool) -> Path:
    if not pending:
        return Path(record["legacy_eval_dir"])
    suffix = "_fpall" if record["operator"] == "fpall" else ""
    return Path(record["campaign_model_dir"]) / f"eval_{record['benchmark']}{suffix}"


def audit_steering(manifest: dict) -> dict:
    records = [(record, False) for record in manifest["complete_existing"]]
    records += [(record, True) for record in manifest["tasks"]]
    problems: dict[str, list[str]] = {}
    complete = []
    prompt_cache: dict[str, set[str]] = {}
    for record, pending in records:
        task_id = record["id"]
        benchmark = record["benchmark"]
        eval_dir = _steering_eval_dir(record, pending)
        issues = []
        metrics = _preferred_metrics(eval_dir)
        if metrics is None:
            issues.append("metrics_missing")
        else:
            provenance = metrics.get("_evaluation_provenance", {})
            if metrics.get("benchmark", benchmark) != benchmark:
                issues.append(f"benchmark={metrics.get('benchmark')}!={benchmark}")
            if provenance.get("n_prompts") != 200:
                issues.append(f"n_prompts={provenance.get('n_prompts')}!=200")
            if benchmark not in prompt_cache:
                prompt_cache[benchmark] = _prompt_set(Path(EC.get_subsample(benchmark, 200, 42))) or set()
            selected = prompt_cache[benchmark]
            judged = _prompt_set(eval_dir / "all_judged.csv")
            if not selected or judged is None:
                issues.append("prompt_evidence_missing")
            elif not selected.issubset(judged):
                issues.append(f"seed42_prompt_coverage={len(selected & judged)}/{len(selected)}")
            elif provenance.get("metrics_source") != "metrics_n200.json" and judged != selected:
                issues.append(f"native_prompt_set={len(judged)}!=seed42_set={len(selected)}")
        if issues:
            problems[task_id] = issues
        else:
            complete.append(task_id)

    by_operator = Counter(item.split(":", 1)[0] for item in complete)
    model_benchmark = Counter()
    for task_id in complete:
        operator, slug, benchmark = task_id.split(":", 2)
        model_benchmark[(operator, config.canonical_steering_slug(slug), benchmark)] += 1
    full_models = {
        operator: sorted({
            canonical for op, canonical, _ in model_benchmark
            if op == operator and all(model_benchmark[(operator, canonical, bench)] == 1
                                      for bench in BENCHMARKS)
        })
        for operator in ("original", "fpall")
    }
    return {
        "expected": 406,
        "complete": len(complete),
        "by_operator": dict(by_operator),
        "models_with_all_seven": {key: len(value) for key, value in full_models.items()},
        "full_models": full_models,
        "problems": problems,
    }


def _adapter_problem(
    root: Path,
    cell: dict,
    entry: dict | None,
    tinker_jobs: dict[str, dict],
) -> str | None:
    expected_id = (
        f"self_sft_{cell['dataset']}_{cell['slug']}_as_{cell['slug']}_seed42"
    )
    if cell.get("seed") != 42 or cell.get("id") != expected_id:
        return "manifest_not_exact_seed42_self_sft"
    if entry is None:
        return "registry_missing"
    backend = entry.get("backend") or cell["backend"]
    reused_huggingface = (
        cell.get("status") == "complete_existing"
        and entry.get("backend") == "huggingface"
    )
    if entry.get("backend") and entry["backend"] != cell["backend"] and not reused_huggingface:
        return f"backend={entry['backend']}!={cell['backend']}"
    if entry.get("base_model") != cell["source"]:
        return f"base_model={entry.get('base_model')}!={cell['source']}"
    if backend == "local":
        adapter = Path(entry.get("path", ""))
        expected = (
            root / "self_sft" / "runs" / cell["dataset"]
            / f"{cell['slug']}_as_{cell['slug']}_seed42"
        )
        if adapter.resolve() != expected.resolve():
            return f"local_adapter_path={adapter}!={expected}"
        config_path = adapter / "adapter_config.json"
        weights_path = adapter / "adapter_model.safetensors"
        try:
            adapter_config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            return "local_adapter_config_invalid"
        if not weights_path.is_file() or weights_path.stat().st_size <= 0:
            return "local_adapter_weights_invalid"
        configured_base = adapter_config.get("base_model_name_or_path")
        if configured_base and configured_base != cell["source"]:
            return f"adapter_base_model={configured_base}!={cell['source']}"
        return None
    if backend == "tinker":
        sampler = entry.get("sampler_path") or entry.get("path") or ""
        if not str(sampler).startswith("tinker://"):
            return "tinker_sampler_invalid"
        job = tinker_jobs.get(cell["id"])
        if not job:
            return "tinker_training_provenance_missing"
        if job.get("error"):
            return "tinker_training_failed"
        if (
            job.get("source") != cell["source"]
            or job.get("target") != cell["source"]
            or job.get("dataset") != cell["dataset"]
            or job.get("seed") != 42
            or job.get("weights_name") != cell["id"]
        ):
            return "tinker_training_provenance_mismatch"
        job_sampler = job.get("sampler_path") or ""
        if job_sampler != sampler:
            return "tinker_sampler_provenance_mismatch"
        return None
    if backend == "huggingface":
        repo = entry.get("hf_repo") or ""
        if not repo or repo.rsplit("/", 1)[-1] != cell["id"]:
            return "huggingface_repo_not_exact_self_sft"
        if not entry.get("self_control") or entry.get("reconciled_from") != "existing_huggingface_adapter":
            return "huggingface_reuse_provenance_missing"
        return None
    return f"unsupported_backend={backend}"


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _expected_metric_ns() -> dict[str, int]:
    result = {}
    for benchmark in BENCHMARKS:
        path = Path(EC.get_subsample(benchmark, 200, 42))
        expected = "refuse" if EC.BENCH_AXIS[benchmark] == "harm" else "comply"
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                result[benchmark] = sum(
                    1 for row in csv.DictReader(handle)
                    if (row.get("expected") or "refuse") == expected
                )
        except (OSError, csv.Error):
            result[benchmark] = -1
    return result


def audit_self_sft(root: Path, manifest: dict) -> dict:
    registry = json.loads((root / "registry" / "tinker_adapters.json").read_text())
    tinker_run = _load_json(root / "manifests" / "self_sft_tinker_run.json", {})
    tinker_jobs = {
        job.get("weights_name"): job
        for job in tinker_run.get("jobs", [])
        if isinstance(job, dict) and job.get("weights_name")
    }
    adapter_problems = {}
    metric_problems = {}
    adapters_complete = []
    metrics_complete = []
    evaluation = {"max_prompts": 200, "subsample_seed": 42}
    expected_ns = _expected_metric_ns()
    work = root / "self_sft" / "eval" / "work"
    for cell in manifest["cells"]:
        item_id = cell["id"]
        adapter_issue = _adapter_problem(
            root, cell, registry.get(item_id), tinker_jobs
        )
        if adapter_issue:
            adapter_problems[item_id] = adapter_issue
        else:
            adapters_complete.append(item_id)

        metrics_path = work / item_id / "metrics.json"
        try:
            item = json.loads(metrics_path.read_text())
            item["_checkpoint_path"] = str(metrics_path)
        except (OSError, json.JSONDecodeError):
            metric_problems[item_id] = ["metrics_missing"]
            continue
        slug = cell["slug"]
        baseline_path = work / f"baseline_{slug}" / "metrics.json"
        try:
            baseline = json.loads(baseline_path.read_text())
            baseline["_checkpoint_path"] = str(baseline_path)
        except (OSError, json.JSONDecodeError):
            baseline = None
        issues = erosion_audit.item_problems(item, baseline, evaluation)
        for benchmark, expected_n in expected_ns.items():
            metric = item.get("per_benchmark", {}).get(benchmark)
            if metric is not None and metric.get("n") != expected_n:
                issues.append(
                    f"{benchmark}:n={metric.get('n')}!=exact_seed42_n={expected_n}"
                )
        if issues:
            metric_problems[item_id] = issues
        else:
            metrics_complete.append(item_id)
    return {
        "adapters_expected": 48,
        "adapters_complete": len(adapters_complete),
        "adapter_problems": adapter_problems,
        "scores_expected": 336,
        "scores_complete": 7 * len(metrics_complete),
        "metric_cells_complete": len(metrics_complete),
        "metric_problems": metric_problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    steering_manifest = json.loads((root / "manifests" / "steering_completion.json").read_text())
    self_manifest = json.loads((root / "manifests" / "self_sft_completion.json").read_text())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(root),
        "steering": audit_steering(steering_manifest),
        "self_sft": audit_self_sft(root, self_manifest),
    }
    report["complete"] = (
        report["steering"]["complete"] == 406
        and report["steering"]["by_operator"] == {"original": 203, "fpall": 203}
        and report["steering"]["models_with_all_seven"] == {"original": 29, "fpall": 29}
        and report["self_sft"]["adapters_complete"] == 48
        and report["self_sft"]["scores_complete"] == 336
    )
    output = args.output or root / "manifests" / "completion_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        "complete": report["complete"],
        "steering": f"{report['steering']['complete']}/406",
        "steering_models_all_seven": report["steering"]["models_with_all_seven"],
        "self_sft_adapters": f"{report['self_sft']['adapters_complete']}/48",
        "self_sft_scores": f"{report['self_sft']['scores_complete']}/336",
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")
    return 1 if args.strict and not report["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
