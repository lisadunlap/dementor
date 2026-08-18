#!/usr/bin/env python3
"""Train and evaluate the 12 missing Tinker self-SFT controls.

The launcher may wait for a credential to appear in the repository ``.env``.
It never prints or copies the credential.  Training, remote sampling, and local
batched judging are independently resumable from the registry/work checkpoints.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODELS = (
    "Qwen/Qwen3.6-27B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "openai/gpt-oss-20b",
)
DATASETS = ("chatbot_arena", "gsm8k", "oasst1", "writingprompts")
BENCHMARKS = "advbench,harmbench,strongreject,sorrybench,sgbench,xstest,orbench_hard"


def _dotenv_has_credential(path: Path) -> bool:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return False
    values = {
        line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip().strip('"').strip("'")
        for line in lines
        if "=" in line and not line.lstrip().startswith("#")
    }
    return any(values.get(name) for name in ("TINKER_API_KEY", "TINKER_CREDENTIAL_CMD"))


def _credential_present(repo: Path) -> bool:
    return bool(
        os.environ.get("TINKER_API_KEY")
        or os.environ.get("TINKER_CREDENTIAL_CMD")
        or _dotenv_has_credential(repo / ".env")
    )


def _wait_pid(pid: int, poll_seconds: int) -> None:
    while Path(f"/proc/{pid}").exists():
        time.sleep(poll_seconds)


def _local_evaluation_running() -> bool:
    result = subprocess.run(
        ["ps", "-eo", "cmd="], stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, check=False,
    )
    return any(
        "run_local_self_sft_evaluation.py" in line
        for line in result.stdout.splitlines()
    )


def _reconcile_registry_provenance(registry_path: Path, manifest_path: Path) -> None:
    """Enrich minimal Tinker registry rows from the exact launch manifest.

    ``tinker_backend`` durably records sampler/state URIs as soon as each remote
    job finishes, but those low-level rows intentionally know nothing about the
    matrix cell.  The completion audit also requires the base model and backend,
    while the Tinker run manifest is the source of truth for source/target,
    dataset, and seed.  Merge those records under the registry's existing
    cross-process lock so a concurrently finishing worker cannot be clobbered.
    """
    run = json.loads(manifest_path.read_text())
    jobs = [job for job in run.get("jobs", []) if isinstance(job, dict)]
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry = json.loads(registry_path.read_text())
        for job in jobs:
            weights_name = job.get("weights_name")
            if not weights_name or job.get("error") or weights_name not in registry:
                continue
            entry = dict(registry[weights_name])
            sampler = job.get("sampler_path") or entry.get("sampler_path") or entry.get("path")
            if sampler:
                entry["path"] = sampler
                entry["sampler_path"] = sampler
            checkpoint = job.get("checkpoint_path") or entry.get("checkpoint_path")
            if checkpoint:
                entry["checkpoint_path"] = checkpoint
            entry.update({
                "backend": "tinker",
                "base_model": job.get("base_model") or job.get("source"),
                "source": job.get("source"),
                "target": job.get("target"),
                "dataset": job.get("dataset"),
                "seed": job.get("seed"),
                "stage": "self_sft",
                "self_control": True,
            })
            registry[weights_name] = entry
        tmp = registry_path.with_suffix(registry_path.suffix + ".tmp")
        tmp.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, registry_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait-for-credentials", action="store_true")
    parser.add_argument("--wait-before-judge-pid", type=int)
    args = parser.parse_args()

    root = args.campaign_root.resolve()
    repo = Path(__file__).resolve().parents[2]
    while not _credential_present(repo):
        if not args.wait_for_credentials:
            print("Tinker credential unavailable; no remote job was launched.", flush=True)
            return 2
        print("Waiting for a Tinker credential in the environment or repo .env.", flush=True)
        time.sleep(args.poll_seconds)

    completion = json.loads((root / "manifests" / "self_sft_completion.json").read_text())
    cells = [
        cell for cell in completion["cells"]
        if cell["status"] == "pending" and cell["backend"] == "tinker"
    ]
    ids = [cell["id"] for cell in cells]
    if len(ids) != 12:
        raise RuntimeError(f"expected exactly 12 Tinker self-SFT cells, found {len(ids)}")

    registry = root / "registry" / "tinker_adapters.json"
    output_root = root / "self_sft" / "runs"
    manifest_out = root / "manifests" / "self_sft_tinker_run.json"
    train = [
        args.python, "-m", "dementor.training.matrix", "launch-self-sft",
        "--parallel", str(args.parallel),
        "--models", *MODELS,
        "--datasets", *DATASETS,
        "--output-root", str(output_root),
        "--registry-path", str(registry),
        "--manifest-out", str(manifest_out),
    ]
    rc = subprocess.run(train, cwd=repo).returncode
    if rc:
        return rc

    _reconcile_registry_provenance(registry, manifest_out)

    entries = json.loads(registry.read_text())
    missing = [item_id for item_id in ids if item_id not in entries]
    invalid = []
    for item_id in ids:
        entry = entries.get(item_id, {})
        sampler = entry.get("sampler_path") or entry.get("path") or ""
        if item_id in entries and not str(sampler).startswith("tinker://"):
            invalid.append(item_id)
    if missing or invalid:
        print(json.dumps({"missing_registry": missing, "invalid_sampler": invalid}, indent=2))
        return 1

    env = dict(os.environ)
    env.update({
        "DEMENTOR_REGISTRY": str(registry),
        "DEMENTOR_IMITATION_ROOT": str(root / "self_sft" / "eval"),
        "DEMENTOR_RESULTS_SAFETY": str(root / "self_sft" / "eval" / "results"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TMPDIR": str(root / "tmp"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    item_csv = ",".join(ids)
    common = [
        "--source-backend", "tinker", "--stage", "self_sft", "--seed", "seed42",
        "--items", item_csv, "--benchmarks", BENCHMARKS,
        "--max-prompts", "200", "--subsample-seed", "42",
    ]
    sample = [
        args.python, "experiments/imitation_safety/tinker_erosion.py", "sample",
        *common, "--sample-workers", "64",
    ]
    rc = subprocess.run(sample, cwd=repo, env=env).returncode
    if rc:
        return rc

    if args.wait_before_judge_pid:
        _wait_pid(args.wait_before_judge_pid, args.poll_seconds)
    priority = root / "self_sft" / "eval" / ".priority_active"
    while _local_evaluation_running() or priority.exists():
        time.sleep(args.poll_seconds)
    priority.parent.mkdir(parents=True, exist_ok=True)
    priority.write_text("Tinker self-SFT seven-benchmark judging\n")
    judge = [
        args.python, "experiments/imitation_safety/tinker_erosion.py", "judge",
        *common, "--gpus", "0,1,2,3,5,7", "--batch-size", "5",
        "--parallel", "4",
    ]
    try:
        return subprocess.run(judge, cwd=repo, env=env).returncode
    finally:
        priority.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
