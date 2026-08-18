#!/usr/bin/env python3
"""Recover exhausted mixed-denominator fingerprint cells without using a GPU.

The live completion supervisor predates the scoped harmonization recovery in
``run_steering_completion.py`` and cannot be restarted safely while Tinker is
waiting on its exact PID.  This companion watches only terminal failed fpall
tasks.  When their judged prompt union contains the exact seed-42 200-prompt
set, it writes ``metrics_n200.json`` through the canonical harmonizer.  The
main supervisor then observes the same strict ``_exact`` gate and reconciles
the task as complete on its next poll.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.completion.build_completion_manifests import _exact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()

    root = args.campaign_root.resolve()
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "manifests" / "steering_completion.json").read_text()
    )
    tasks = manifest["tasks"]
    status_path = root / "manifests" / "steering_status.json"

    while True:
        status = json.loads(status_path.read_text())
        recovered = 0
        for task in tasks:
            if task["operator"] != "fpall":
                continue
            eval_dir = (
                Path(task["campaign_model_dir"])
                / f"eval_{task['benchmark']}_fpall"
            )
            if _exact(eval_dir, task["benchmark"]):
                continue
            record = status.get(task["id"], {})
            if record.get("status") != "failed" or not (
                eval_dir / "all_judged.csv"
            ).is_file():
                continue
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
            print(result.stdout or "", end="", flush=True)
            if result.returncode == 0 and _exact(eval_dir, task["benchmark"]):
                recovered += 1
                print(f"recovered exact task {task['id']}", flush=True)

        exact = sum(
            _exact(
                Path(task["campaign_model_dir"])
                / f"eval_{task['benchmark']}{'_fpall' if task['operator'] == 'fpall' else ''}",
                task["benchmark"],
            )
            for task in tasks
        )
        if exact == len(tasks):
            print(f"all {exact} steering completion tasks are exact", flush=True)
            return 0
        if recovered:
            print(f"recovered {recovered} exhausted task(s)", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
