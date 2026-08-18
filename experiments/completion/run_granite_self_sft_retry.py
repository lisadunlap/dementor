#!/usr/bin/env python3
"""Retry the four Granite self-SFT cells with its lazy kernels available."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("chatbot_arena", "gsm8k", "oasst1", "writingprompts")
MODEL = "ibm-granite/granite-4.0-h-small"
SLUG = "granite-4-h-small"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gpu5_free() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
    )
    for line in result.stdout.splitlines():
        try:
            index, memory = (int(value.strip()) for value in line.split(",", 1))
        except ValueError:
            continue
        if index == 5:
            return memory < 1000
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    repo = Path(__file__).resolve().parents[2]
    output_root = root / "self_sft" / "runs"
    registry = root / "registry" / "tinker_adapters.json"
    logs = root / "logs" / "self_sft_local"
    status_path = root / "manifests" / "granite_retry_status.json"
    reservation = root / "manifests" / "granite_retry_active"
    reservation.write_text("GPU 5 reserved for Granite self-SFT retry\n")
    status = {}
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        pass

    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "5",
        # GraniteMoeHybrid resolves causal-conv1d from kernels-community at
        # model construction time.  The model weights themselves are cached.
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "TORCH_EXTENSIONS_DIR": str(root / "tmp" / "torch_extensions"),
        "XDG_CACHE_HOME": str(root / "tmp" / "xdg_cache"),
        "TMPDIR": str(root / "tmp"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
    })
    try:
        while not _gpu5_free():
            time.sleep(15)
        for dataset in DATASETS:
            alias = f"self_sft_{dataset}_{SLUG}_as_{SLUG}_seed42"
            adapter = output_root / dataset / f"{SLUG}_as_{SLUG}_seed42" / "adapter_config.json"
            if adapter.exists():
                status[alias] = {"status": "complete", "validated_at": _now()}
                continue
            log = logs / f"granite_retry_{dataset}.log"
            cmd = [
                args.python, "-m", "dementor.training.matrix", "launch-local-self-sft-cell",
                "--source", MODEL, "--dataset", dataset, "--seed", "42",
                "--output-root", str(output_root), "--registry-path", str(registry),
            ]
            status[alias] = {"status": "running", "started_at": _now(), "log": str(log)}
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            started = time.time()
            with log.open("a") as handle:
                handle.write(f"START {_now()} {' '.join(cmd)}\n")
                handle.flush()
                rc = subprocess.run(cmd, cwd=repo, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, text=True).returncode
            valid = adapter.exists()
            status[alias].update({
                "status": "complete" if rc == 0 and valid else "failed",
                "returncode": rc,
                "adapter_valid": valid,
                "elapsed_seconds": round(time.time() - started, 1),
                "finished_at": _now(),
            })
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            if rc or not valid:
                return 1
        return 0
    finally:
        reservation.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
