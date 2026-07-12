#!/usr/bin/env python
"""Lightweight Tinker judging watchdog.

Checks status on an interval, logs GPU/process state, and restarts the Tinker
resume loop if judging stops while sampled work remains.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "experiments" / "steering" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOG_DIR / "tinker_watchdog.log"
RESUME_LOG = LOG_DIR / "tinker_watchdog_resume.log"

BENCHMARKS = "advbench,harmbench,strongreject,xstest,sorrybench,orbench_hard,sgbench"
GPUS = "0,1,2,3"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def env() -> dict[str, str]:
    e = dict(os.environ)
    e.setdefault("DEMENTOR_REPO", str(REPO))
    e.setdefault("DEMENTOR_DATA", str(REPO / "data"))
    e.setdefault("DEMENTOR_DATA_ROOT", str(REPO / "data"))
    e.setdefault("DEMENTOR_REGISTRY", str(REPO / "data" / "local_adapters" / "registry_partner.json"))
    e.setdefault("DEMENTOR_GPUS", GPUS)
    e.setdefault("DEMENTOR_HF_HOME", "/home/ubuntu/.cache/huggingface")
    e.setdefault("DEMENTOR_MODELS_DIR", str(REPO / "data" / "models"))
    e.setdefault("DEMENTOR_STEER_WORK", str(REPO / "data"))
    e.setdefault("HF_HOME", e["DEMENTOR_HF_HOME"])
    e.setdefault("HF_HUB_CACHE", str(Path(e["DEMENTOR_HF_HOME"]) / "hub"))
    e.setdefault("HF_HUB_DISABLE_XET", "1")
    e.setdefault("HF_HUB_OFFLINE", "1")
    e.setdefault("RTL_JUDGE_BATCH_SIZE", "96")
    e.setdefault("RTL_JUDGE_MAX_RESP_CHARS", "1800")
    e["PYTHONPATH"] = (
        str(REPO)
        + os.pathsep
        + str(REPO / "experiments" / "steering")
        + os.pathsep
        + str(REPO / "experiments" / "steering" / "port")
        + os.pathsep
        + e.get("PYTHONPATH", "")
    )
    return e


ENV = env()
sys.path.insert(0, str(REPO / "experiments" / "imitation_safety"))
import tinker_erosion as T  # noqa: E402


def ps_text() -> str:
    return subprocess.run(
        ["ps", "-eo", "pid,ppid,pgid,stat,etime,%cpu,%mem,cmd"],
        cwd=REPO,
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    ).stdout


def active_judge_lines() -> list[str]:
    lines = []
    for line in ps_text().splitlines():
        if "__judge_worker" in line or "judge_all.py" in line:
            if "tinker_watchdog.py" not in line:
                lines.append(line)
    return lines


def active_resume_lines() -> list[str]:
    lines = []
    for line in ps_text().splitlines():
        if "tinker_watchdog.py --resume" in line:
            lines.append(line)
    return lines


def counts() -> tuple[int, int, int]:
    adapters, baselines = T.tinker_worklist("all")
    worklist = baselines + adapters
    benches = BENCHMARKS.split(",")
    sampled = sum(T._sampled(it["id"], benches) for it in worklist)
    judged = sum(T._done(it["id"]) for it in worklist)
    return sampled, judged, len(worklist)


def gpu_snapshot() -> str:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return " | ".join(x.strip() for x in (r.stdout or "").splitlines() if x.strip())


def launch_resume() -> None:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--resume"]
    with RESUME_LOG.open("a") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=ENV,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(f"launched resume pid={proc.pid}")


def resume_main() -> int:
    sys.path.insert(0, str(REPO / "experiments" / "steering"))
    import run_nemotron_then_tinker_judge as q

    q.resume_tinker()
    return 0


def watch(interval: int) -> int:
    log(f"watchdog start interval={interval}s repo={REPO}")
    while True:
        sampled, judged, total = counts()
        remaining = total - judged
        judges = active_judge_lines()
        resumes = active_resume_lines()
        log(
            f"status sampled={sampled}/{total} judged={judged}/{total} "
            f"remaining={remaining} active_judges={len(judges)} active_resumes={len(resumes)} "
            f"gpu={gpu_snapshot()}"
        )
        if remaining <= 0:
            log("all Tinker judging complete; watchdog exiting")
            return 0
        if not judges and not resumes:
            log("no active Tinker judge/resume process while work remains; restarting resume loop")
            launch_resume()
        time.sleep(interval)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.resume:
        return resume_main()
    return watch(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
