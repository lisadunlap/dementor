#!/usr/bin/env python3
"""Run the 20 missing local self-SFT controls in three disjoint GPU lanes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("chatbot_arena", "gsm8k", "oasst1", "writingprompts")
LANES = (
    ("llama70", "0,1,2", "meta-llama/Llama-3.3-70B-Instruct", True),
    ("gemma31", "3,7", "google/gemma-4-31B-it", True),
    ("small", "5", None, False),
)
SMALL = (
    "google/gemma-4-E4B-it",
    "microsoft/phi-4",
    "ibm-granite/granite-4.0-h-small",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    output_root = root / "self_sft" / "runs"
    registry = root / "registry" / "tinker_adapters.json"
    log_root = root / "logs" / "self_sft_local"
    status_path = root / "manifests" / "self_sft_local_status.json"
    log_root.mkdir(parents=True, exist_ok=True)
    lock = threading.RLock()
    status: dict[str, dict] = {}

    def publish() -> None:
        with lock:
            tmp = status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(status, indent=2) + "\n")
            os.replace(tmp, status_path)

    def run_one(lane: str, gpus: str, model: str, dataset: str, mp: bool) -> bool:
        slug = model.split("/")[-1].lower().replace("-instruct", "").replace("-it", "")
        job_id = f"{lane}:{dataset}:{slug}"
        log_path = log_root / f"{lane}_{dataset}_{slug}.log"
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": gpus,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TMPDIR": str(root / "tmp"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        })
        if mp:
            env["DEMENTOR_MP"] = "1"
        else:
            env.pop("DEMENTOR_MP", None)
        cmd = [
            args.python, "-m", "dementor.training.matrix", "launch-local-self-sft-cell",
            "--source", model, "--dataset", dataset, "--seed", "42",
            "--output-root", str(output_root), "--registry-path", str(registry),
        ]
        start = time.time()
        with lock:
            status[job_id] = {
                "status": "running", "lane": lane, "gpus": gpus, "model": model,
                "dataset": dataset, "started_at": datetime.now(timezone.utc).isoformat(),
                "log": str(log_path),
            }
        publish()
        with log_path.open("a") as handle:
            handle.write(f"START {' '.join(cmd)}\n")
            handle.flush()
            result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], env=env,
                                    stdout=handle, stderr=subprocess.STDOUT, text=True)
        with lock:
            status[job_id].update({
                "status": "complete" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "elapsed_seconds": round(time.time() - start, 1),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
        publish()
        return result.returncode == 0

    def run_lane(lane: str, gpus: str, model: str | None, mp: bool) -> bool:
        jobs = [(model, d) for d in DATASETS] if model else [(m, d) for m in SMALL for d in DATASETS]
        ok = True
        for model_id, dataset in jobs:
            ok = run_one(lane, gpus, model_id, dataset, mp) and ok
        return ok

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda lane: run_lane(*lane), LANES))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
