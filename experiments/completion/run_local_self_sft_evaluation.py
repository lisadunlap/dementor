#!/usr/bin/env python3
"""Wait for local self-SFT adapters, then evaluate all 20 on seven benchmarks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


BENCHMARKS = "advbench,harmbench,strongreject,sorrybench,sgbench,xstest,orbench_hard"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--wait-pids", default="")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    priority = root / "self_sft" / "eval" / ".priority_active"
    priority.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the evaluation pool before waiting for training supervisors.  If a
    # training lane exits slightly before another one, the steering scheduler
    # must not backfill the newly free cards and collide with evaluation startup.
    priority.write_text("local self-SFT seven-benchmark evaluation pending\n")
    try:
        for raw in args.wait_pids.split(","):
            if not raw.strip():
                continue
            pid = int(raw)
            while Path(f"/proc/{pid}").exists():
                time.sleep(30)

        manifest = json.loads((root / "manifests" / "self_sft_completion.json").read_text())
        cells = [
            c for c in manifest["cells"]
            if c["status"] == "pending" and c["backend"] == "local"
        ]
        ids = [c["id"] for c in cells]
        registry = json.loads((root / "registry" / "tinker_adapters.json").read_text())
        missing = [item_id for item_id in ids if item_id not in registry]
        invalid = [
            item_id for item_id in ids
            if item_id in registry
            and not (Path(registry[item_id].get("path", "")) / "adapter_config.json").exists()
        ]
        if missing or invalid:
            print(json.dumps({"missing_registry": missing, "invalid_adapter": invalid}, indent=2))
            return 1

        priority.write_text("local self-SFT seven-benchmark evaluation running\n")
        env = dict(os.environ)
        env.update({
            "DEMENTOR_REGISTRY": str(root / "registry" / "tinker_adapters.json"),
            "DEMENTOR_IMITATION_ROOT": str(root / "self_sft" / "eval"),
            "DEMENTOR_RESULTS_SAFETY": str(root / "self_sft" / "eval" / "results"),
            "DEMENTOR_GPUS": "0,1,2,3,5,7",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TMPDIR": str(root / "tmp"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        repo = Path(__file__).resolve().parents[2]
        item_csv = ",".join(ids)
        generate = [
            args.python, "experiments/imitation_safety/erosion_daemon.py",
            "--items", item_csv, "--benchmarks", BENCHMARKS, "--max-prompts", "200",
            "--subsample-seed", "42", "--generate-only", "--retry-errors",
        ]
        rc = subprocess.run(generate, cwd=repo, env=env).returncode
        if rc:
            return rc
        judge = [
            args.python, "experiments/imitation_safety/tinker_erosion.py", "judge",
            "--source-backend", "local", "--stage", "self_sft", "--seed", "seed42",
            "--items", item_csv, "--benchmarks", BENCHMARKS, "--max-prompts", "200",
            "--subsample-seed", "42", "--gpus", "0,1,2,3,5,7", "--batch-size", "5",
            "--parallel", "4",
        ]
        return subprocess.run(judge, cwd=repo, env=env).returncode
    finally:
        priority.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
