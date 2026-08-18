#!/usr/bin/env python3
"""Resumable six-GPU scheduler for the frozen 290-cell steering completion."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.completion.build_completion_manifests import _exact


GPU_POOL = (0, 1, 2, 3, 5, 7)
LOCAL_LANES = {
    "llama70": ({0, 1, 2}, 4),
    "gemma31": ({3, 7}, 4),
    "small": ({5}, 12),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def _reserved_by_local(root: Path) -> set[int]:
    status_path = root / "manifests" / "self_sft_local_status.json"
    if not status_path.exists():
        return set()
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        return set().union(*(gpus for gpus, _ in LOCAL_LANES.values()))
    reserved = set()
    for lane, (gpus, total) in LOCAL_LANES.items():
        records = [v for k, v in status.items() if k.startswith(lane + ":")]
        terminal = sum(v.get("status") in {"complete", "failed"} for v in records)
        if terminal < total:
            reserved.update(gpus)
    return reserved


def _gpu_free() -> set[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
    )
    free = set()
    for line in result.stdout.splitlines():
        try:
            index, memory = (int(x.strip()) for x in line.split(",", 1))
        except ValueError:
            continue
        if index in GPU_POOL and memory < 1000:
            free.add(index)
    return free


def _reserved_by_external_workers() -> set[int]:
    """Honor GPU declarations from training/evaluation workers outside this scheduler.

    Memory-only polling has a race between sequential adapter jobs, when a lane has
    released its model but its supervisor still owns the same cards. Reading the
    worker environment makes that reservation explicit for its whole process life.
    """
    markers = (
        "launch-local-self-sft-cell",
        "run_granite_self_sft_retry.py",
        "run_erosion_item.py",
        "__judge_worker",
        "run_benchmark_eval.py",
    )
    reserved: set[int] = set()
    result = subprocess.run(
        ["ps", "-eo", "pid=,cmd="], stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, check=False,
    )
    for line in result.stdout.splitlines():
        if not any(marker in line for marker in markers):
            continue
        # Sequential lane supervisors can briefly sit between child processes.
        # Their command line retains the CUDA assignment even when their own
        # environment does not, so honor it to avoid launching into that gap.
        for assignment in re.findall(r"CUDA_VISIBLE_DEVICES=([0-9,]+)", line):
            reserved.update(
                gpu for gpu in map(int, assignment.split(",")) if gpu in GPU_POOL
            )
        try:
            pid = int(line.strip().split(None, 1)[0])
            environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except (OSError, ValueError, IndexError):
            continue
        for entry in environment:
            if not entry.startswith(b"CUDA_VISIBLE_DEVICES="):
                continue
            for value in entry.split(b"=", 1)[1].decode().split(","):
                try:
                    gpu = int(value)
                except ValueError:
                    continue
                if gpu in GPU_POOL:
                    reserved.add(gpu)
    return reserved


def _link_reusable_parts(source: Path, destination: Path) -> None:
    parts = destination / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for name in ("baseline_b0.csv", "cone_b0.6.csv", "cone_b1.0.csv", "cone_b1.4.csv"):
        src = source / "parts" / name
        dst = parts / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    manifest_path = root / "manifests" / "steering_completion.json"
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest["tasks"]
    if len(tasks) != 290:
        raise RuntimeError(f"refusing noncanonical manifest with {len(tasks)} tasks")
    # Large/multi-GPU models start first so they cannot become a serial tail.
    tasks.sort(key=lambda x: (0 if x["operator"] == "original" else 1,
                              -x["gpu_count"], -(x.get("params_b") or 0), x["id"]))
    status_path = root / "manifests" / "steering_status.json"
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        status = {}
    if args.dry_run:
        print(json.dumps({"tasks": len(tasks), "by_gpu_count": {
            str(n): sum(t["gpu_count"] == n for t in tasks) for n in (1, 2, 3)
        }}, indent=2))
        return 0

    repo = Path(__file__).resolve().parents[2]
    logs = root / "logs" / "steering"
    logs.mkdir(parents=True, exist_ok=True)
    running: dict[str, dict] = {}
    task_by_id = {task["id"]: task for task in tasks}

    def output_dir(task: dict) -> Path:
        suffix = "_fpall" if task["operator"] == "fpall" else ""
        return Path(task["campaign_model_dir"]) / f"eval_{task['benchmark']}{suffix}"

    def is_complete(task: dict) -> bool:
        return _exact(output_dir(task), task["benchmark"])

    def harmonize_if_possible(task: dict) -> bool:
        """Rescore mixed-denominator fpall output on the exact seed-42 set.

        Some legacy originals are exact only through ``metrics_n200.json`` while
        their reusable baseline/cone CSVs still contain 300 prompts.  Linking
        those controls into a newly generated 200-prompt fpall cell produces a
        valid 300-prompt union, but not an exact campaign artifact.  The existing
        harmonizer can recover the exact intersection without another GPU run.
        It refuses partial prompt coverage, so a successful strict check remains
        the authority after this best-effort pass.
        """
        eval_dir = output_dir(task)
        if task["operator"] != "fpall" or is_complete(task):
            return is_complete(task)
        if not (eval_dir / "all_judged.csv").is_file():
            return False
        log = logs / f"{task['id'].replace(':', '__')}.harmonize.log"
        result = subprocess.run(
            [
                args.python,
                "experiments/steering/harmonize_to_200.py",
                "--eval-dir",
                str(eval_dir),
            ],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write_text(result.stdout or "")
        return result.returncode == 0 and is_complete(task)

    def original_source(task: dict) -> Path | None:
        original_id = f"original:{task['slug']}:{task['benchmark']}"
        pending_original = task_by_id.get(original_id)
        if pending_original is not None:
            path = output_dir(pending_original)
            return path if is_complete(pending_original) else None
        completed = next((x for x in manifest["complete_existing"] if x["id"] == original_id), None)
        return Path(completed["legacy_eval_dir"]) if completed else None

    while True:
        # Reap children and validate actual artifacts, never just return codes.
        for task_id, run in list(running.items()):
            rc = run["process"].poll()
            if rc is None:
                continue
            task = task_by_id[task_id]
            complete = rc == 0 and (
                is_complete(task) or harmonize_if_possible(task)
            )
            record = status.setdefault(task_id, {})
            record.update({
                "status": "complete" if complete else "failed",
                "returncode": rc,
                "finished_at": _now(),
                "elapsed_seconds": round(time.time() - run["started"], 1),
            })
            run["handle"].close()
            del running[task_id]
            _atomic_json(status_path, status)

        # Reconcile an exact cached intersection even when an older supervisor
        # already exhausted its retry budget before this recovery path existed.
        for task in tasks:
            if (
                task["id"] not in running
                and int(status.get(task["id"], {}).get("attempts", 0)) >= args.max_attempts
                and not is_complete(task)
            ):
                harmonize_if_possible(task)

        done = 0
        for task in tasks:
            if is_complete(task):
                done += 1
                status.setdefault(task["id"], {}).update({"status": "complete"})
        if done == len(tasks):
            _atomic_json(status_path, status)
            print(f"all {done} steering completion tasks are exact", flush=True)
            return 0

        allocated = set().union(*(run["gpus"] for run in running.values())) if running else set()
        evaluation_priority = root / "self_sft" / "eval" / ".priority_active"
        granite_retry = root / "manifests" / "granite_retry_active"
        campaign_reserved = {5} if granite_retry.exists() else set()
        available = [] if evaluation_priority.exists() else sorted(
            _gpu_free() - _reserved_by_local(root) - _reserved_by_external_workers()
            - campaign_reserved - allocated
        )
        launched = False
        for task in tasks:
            task_id = task["id"]
            if task_id in running or is_complete(task):
                continue
            record = status.get(task_id, {})
            attempts = int(record.get("attempts", 0))
            if attempts >= args.max_attempts:
                continue
            if task["operator"] == "fpall":
                source = original_source(task)
                if source is None:
                    continue
            need = int(task["gpu_count"])
            if len(available) < need:
                continue
            gpus = set(available[:need])
            del available[:need]
            out = Path(task["campaign_model_dir"])
            eval_out = output_dir(task)
            if task["operator"] == "fpall":
                _link_reusable_parts(source, eval_out)
            cmd = [
                args.python, "experiments/steering/run_benchmark_eval.py", task["slug"],
                "--benchmarks", task["benchmark"], "--max-prompts", "200", "--outdir", str(out),
            ]
            # Qwen3.6-35B-A3B reaches the H100 memory ceiling at batch 16 on
            # long SG-Bench prompts.  cuDNN SDPA then fails its graph execution
            # on the final partial batch even though earlier batches succeed.
            # Batch size is throughput-only and does not change the greedy
            # generations, so use the verified lower-memory setting for every
            # Qwen3.6 benchmark and preserve exact campaign semantics.
            if task["slug"] == "qwen3.6-35b":
                cmd.extend(["--gen-batch", "8"])
            if task["operator"] == "fpall":
                cmd.append("--single-dir-all-layers")
            env = dict(os.environ)
            env.update({
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, sorted(gpus))),
                "DEMENTOR_STEER_WORK": manifest["live_root"],
                "DEMENTOR_SEED": "42",
                "TMPDIR": str(root / "tmp"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            # Qwen3.6's 262k-token vocabulary makes the fp32 logits used by
            # the post-generation perplexity pass much larger than its
            # generation activations.  The historical PPL batch of 16 needs
            # another ~4.7 GiB after a 200-prompt SorryBench arm and OOMs just
            # before the durable CSV write.  Perplexity is computed row-wise,
            # so reducing this batch changes only peak memory and throughput,
            # not the metric or selected prompts.
            if task["slug"] == "qwen3.6-35b":
                env["DEMENTOR_PPL_BS"] = "4"
            # ``cone_eval`` rebuilds grader subprocess environments through
            # ``steer_config.hf_env()``, whose canonical input is
            # DEMENTOR_HF_HOME.  Passing only HF_HOME/HF_HUB_CACHE here is not
            # sufficient: the grader would fall back to ~/.cache/huggingface,
            # miss an otherwise complete shared cache while offline, and fail
            # after generation had already finished.  Bridge the standard HF
            # setting into the canonical campaign setting for every worker.
            hf_home = os.environ.get("DEMENTOR_HF_HOME") or os.environ.get("HF_HOME")
            if hf_home:
                env["DEMENTOR_HF_HOME"] = hf_home
                env["HF_HOME"] = hf_home
                env["HF_HUB_CACHE"] = os.environ.get(
                    "HF_HUB_CACHE", os.path.join(hf_home, "hub")
                )
            if need > 1:
                env["DEMENTOR_MP"] = "1"
            else:
                env.pop("DEMENTOR_MP", None)
            log = logs / f"{task_id.replace(':', '__')}.attempt{attempts + 1}.log"
            handle = log.open("a")
            handle.write(f"START {_now()} GPUs={sorted(gpus)} {' '.join(cmd)}\n")
            handle.flush()
            process = subprocess.Popen(cmd, cwd=repo, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            status[task_id] = {
                **record, "status": "running", "attempts": attempts + 1,
                "pid": process.pid, "gpus": sorted(gpus), "started_at": _now(), "log": str(log),
            }
            running[task_id] = {"process": process, "gpus": gpus, "started": time.time(), "handle": handle}
            _atomic_json(status_path, status)
            launched = True
        if not running and not launched:
            exhausted = [t["id"] for t in tasks if not is_complete(t)
                         and int(status.get(t["id"], {}).get("attempts", 0)) >= args.max_attempts]
            blocked = [t["id"] for t in tasks if t["operator"] == "fpall" and not is_complete(t)
                       and original_source(t) is None]
            if exhausted and len(exhausted) + len(blocked) >= len(tasks) - done:
                print(json.dumps({"exhausted": exhausted, "blocked": blocked}, indent=2), flush=True)
                return 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
