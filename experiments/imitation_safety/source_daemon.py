#!/usr/bin/env python3
"""GPU-lease scheduler for persistent per-source local DPO generation."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import daemon_common as DC  # noqa: E402
import erosion_common as EC  # noqa: E402
import fidelity_common as FC  # noqa: E402
import gpu_lease  # noqa: E402

RUNNER = HERE / "run_erosion_source.py"
LOG_DIR = HERE / "logs"
LOG_DIR.mkdir(exist_ok=True)
LEASE_HOLDER = "persistent_source_daemon"
dlog = DC.make_dlog(str(LOG_DIR / "source_daemon.log"))

GEN_BATCH_BY_BASE = {
    "adamo1139/aya-expanse-8b-ungated": 128,
    "google/gemma-4-E4B-it": 256,
    "meta-llama/Llama-3.1-8B-Instruct": 128,
    "mistralai/Ministral-8B-Instruct-2410": 128,
    "allenai/OLMo-3-7B-Instruct": 128,
}


def item_done(item: dict, benchmarks: list[str], also_fidelity: bool) -> bool:
    root = Path(EC.WORK) / item["id"]
    return all((root / b / "all_gens.csv").is_file() for b in benchmarks) and (
        not also_fidelity or FC.gens_done(item["id"])
    )


def build_groups(sources: set[str] | None, items_filter: set[str] | None,
                 benchmarks: list[str], also_fidelity: bool) -> list[dict]:
    adapters, _ = EC.build_worklist(seed="seed42", local_only=True)
    adapters = [item for item in adapters if item["id"].startswith("dpo_")]
    if sources:
        adapters = [item for item in adapters
                    if item["source"] in sources or item["base_model"] in sources]
    if items_filter:
        adapters = [item for item in adapters if item["id"] in items_filter]
    grouped = {}
    for item in adapters:
        if item_done(item, benchmarks, also_fidelity):
            continue
        group = grouped.setdefault(item["source"], {
            "id": "source_" + item["source"], "source": item["source"],
            "base_model": item["base_model"], "needs_mp": item["needs_mp"], "items": [],
        })
        group["items"].append(item)
    return list(grouped.values())


def required_gpus(group: dict) -> int:
    return 3 if group["base_model"] == "meta-llama/Llama-3.3-70B-Instruct" else 1


def external_running_sources() -> set[str]:
    """Sources owned by persistent workers launched by another supervisor."""
    try:
        output = subprocess.run(["ps", "-eo", "cmd="], stdout=subprocess.PIPE,
                                text=True, check=True).stdout
    except Exception:
        return set()
    found = set()
    for line in output.splitlines():
        marker = "run_erosion_source.py --source "
        if marker in line:
            found.add(line.split(marker, 1)[1].split()[0])
    return found


def external_reserved_gpus() -> set[int]:
    """CUDA declarations of persistent workers owned by another supervisor."""
    reserved = set()
    try:
        output = subprocess.run(["ps", "-eo", "pid=,cmd="], stdout=subprocess.PIPE,
                                text=True, check=True).stdout
    except Exception:
        return reserved
    for line in output.splitlines():
        if "run_erosion_source.py" not in line:
            continue
        try:
            pid = int(line.strip().split(None, 1)[0])
            environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except (OSError, ValueError, IndexError):
            continue
        for entry in environment:
            if entry.startswith(b"CUDA_VISIBLE_DEVICES="):
                reserved.update(int(value) for value in entry.split(b"=", 1)[1].split(b",")
                                if value.strip())
    return reserved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", help="comma-separated source slugs/base IDs")
    parser.add_argument("--items", help="optional comma-separated cell IDs")
    parser.add_argument("--benchmarks", default=",".join(EC.DEFAULT_BENCHMARKS))
    parser.add_argument("--max-prompts", type=int, default=EC.DEFAULT_MAX_PROMPTS)
    parser.add_argument("--subsample-seed", type=int, default=EC.DEFAULT_SUBSAMPLE_SEED)
    parser.add_argument("--also-fidelity", action="store_true")
    parser.add_argument("--util-max", type=int, default=5)
    parser.add_argument("--mem-max", type=int, default=5000)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    benchmarks = [value for value in args.benchmarks.split(",") if value]
    sources = {value for value in (args.sources or "").split(",") if value} or None
    items_filter = {value for value in (args.items or "").split(",") if value} or None
    groups = build_groups(sources, items_filter, benchmarks, args.also_fidelity)
    dlog(f"persistent scheduler groups={[(g['source'], len(g['items'])) for g in groups]} "
         f"gpus={EC.GPUS}")
    if args.dry_run:
        return

    gpu_lease.reap()
    running = {}
    failed = set()
    while True:
        for source, record in list(running.items()):
            process = record["process"]
            if process.poll() is None:
                continue
            for gpu in record["gpus"]:
                gpu_lease.release(gpu, holder=LEASE_HOLDER)
            dlog(f"done source={source} rc={process.returncode}")
            if process.returncode:
                failed.add(source)
            del running[source]

        groups = build_groups(sources, items_filter, benchmarks, args.also_fidelity)
        inflight = set(running) | external_running_sources()
        pending = [group for group in groups
                   if group["source"] not in inflight and group["source"] not in failed]
        if not pending and not running:
            dlog("ALL DONE" if not groups else f"STOP failed_sources={sorted(failed)}")
            return

        used = ({gpu for record in running.values() for gpu in record["gpus"]}
                | external_reserved_gpus())
        available = []
        for gpu in EC.GPUS:
            if gpu in used:
                continue
            util, memory = DC.gpu_stat(gpu)
            if util <= args.util_max and memory < args.mem_max:
                available.append(gpu)

        # Largest source first so the 70B lane starts as soon as three cards are available.
        for group in sorted(pending, key=required_gpus, reverse=True):
            count = required_gpus(group)
            if len(available) < count:
                continue
            chosen, claimed = available[:count], []
            for gpu in chosen:
                if gpu_lease.try_claim(gpu, holder=LEASE_HOLDER):
                    claimed.append(gpu)
                else:
                    break
            if len(claimed) != count:
                for gpu in claimed:
                    gpu_lease.release(gpu, holder=LEASE_HOLDER)
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, chosen)))
            if count > 1:
                env["DEMENTOR_MP"] = "1"
            command = [EC.PY, str(RUNNER), "--source", group["source"],
                       "--benchmarks", ",".join(benchmarks),
                       "--max-prompts", str(args.max_prompts),
                       "--subsample-seed", str(args.subsample_seed),
                       "--gen-batch", str(GEN_BATCH_BY_BASE.get(group["base_model"], 32))]
            if args.also_fidelity:
                command.append("--also-fidelity")
            if items_filter:
                command += ["--items", ",".join(item["id"] for item in group["items"])]
            log = open(LOG_DIR / f"persistent_{group['source']}.log", "a")
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            running[group["source"]] = {"process": process, "gpus": chosen, "log": log}
            available = [gpu for gpu in available if gpu not in chosen]
            dlog(f"LAUNCH source={group['source']} cells={len(group['items'])} GPUs={chosen}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
