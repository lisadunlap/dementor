#!/usr/bin/env python3
"""Stateful, low-noise watcher for the imitation-square sequencer daemon.

Polls every POLL_SECONDS and prints ONE stdout line per MEANINGFUL, not-yet-reported
transition (each printed line becomes a single coordinator notification). All routine
"still nominal" heartbeats are suppressed. Dedup state persists to watch_state.json so a
restart of this watcher never re-reports an already-announced milestone.

Meaningful events (coordinator-defined):
  1. Any new [error]/[fatal] log line, or the daemon PID vanishing -> report immediately.
  2. All 5 local baselines finished, and the first real local training cell starting.
  3. Every +15 completed cells; and when the local queue is fully drained (square done).
  4. Tinker running-submit-total crossing 25 / 50 / 75 / final.
  5. GPU-collision risk: one of OUR processes co-resident on a card with a foreign user.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Env-overridable (canonical DEMENTOR_REPO / DEMENTOR_DATA); match sequencer.py's resolution so the
# watcher reads the SAME state dir the daemon writes. Set DEMENTOR_TRAIN_STATE to watch a campaign
# whose state lives elsewhere (e.g. our /data exp_imitation_square).
def _env(name, default):
    v = os.environ.get(name)
    return v if v else default

_HERE = Path(__file__).resolve().parent                          # experiments/imitation_train
REPO = Path(_env("DEMENTOR_REPO", str(_HERE.parent.parent)))
DATA_ROOT = Path(_env("DEMENTOR_DATA", str(REPO / "data")))
STATE_DIR = Path(_env("DEMENTOR_TRAIN_STATE", str(DATA_ROOT / "imitation_train")))
QUEUE_PATH = STATE_DIR / "queue_state.json"
REGISTRY_PATH = DATA_ROOT / "tinker_adapters.json"
SEQ_LOG = STATE_DIR / "sequencer.log"
WATCH_STATE = STATE_DIR / "watch_state.json"
DAEMON_PATTERN = "sequencer.py"
POLL_SECONDS = 900  # 15 min
COMPLETED_STEP = 15
SUBMIT_MILESTONES = [25, 50, 75]
N_LOCAL_BASELINES = 5
DATASET = "chatbot_arena"
SEED = 42
ME = os.environ.get("USER", "ethantsliu")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def emit(msg: str) -> None:
    print(f"[watch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_watch_state() -> dict:
    if WATCH_STATE.exists():
        try:
            return json.loads(WATCH_STATE.read_text())
        except Exception:
            pass
    return {
        "reported_error_count": 0,
        "daemon_dead_reported": False,
        "last_completed_milestone": 0,
        "submit_milestones_reported": [],
        "final_submit_reported": False,
        "baselines_done_reported": False,
        "first_local_cell_reported": False,
        "local_drained_reported": False,
        "square_done_reported": False,
        "reported_collisions": [],
    }


def save_watch_state(s: dict) -> None:
    tmp = WATCH_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2))
    os.replace(tmp, WATCH_STATE)


def daemon_alive() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", DAEMON_PATTERN], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return bool(out)
    except Exception:
        return True  # fail-open: don't false-alarm on a flaky pgrep


def read_queue() -> dict:
    """Registry-derived square progress (ground truth, restart-stable). The daemon drops
    fully-covered cells from queue_state.json, so completion MUST be counted from the
    registry -- an ordered core pair is 'done' iff both its sft_ and dpo_ adapters exist.
    The 12x12 core square has 132 ordered pairs; 25 were pre-covered before the daemon
    started, so the daemon's own workload is the remaining 107 (46 tinker + 61 local).
    running/failed are pulled from queue_state.json (live, best-effort, informational)."""
    from dementor import config

    try:
        reg = set(json.loads(REGISTRY_PATH.read_text()).keys())
    except Exception:
        reg = set()
    core = [m for m in config.roster() if m.get("imitation") == "core"]
    ids = [m["id"] for m in core]
    slug = {m["id"]: m["slug"] for m in core}
    backend = {m["id"]: m["backend"] for m in core}

    covered = local_covered = tinker_covered = 0
    local_pairs = tinker_pairs = 0
    for s in ids:
        for t in ids:
            if s == t:
                continue
            is_local = backend[s] == "local"
            if is_local:
                local_pairs += 1
            else:
                tinker_pairs += 1
            sl = f"{DATASET}_{slug[s]}_as_{slug[t]}_seed{SEED}"
            if f"sft_{sl}" in reg and f"dpo_{sl}" in reg:
                covered += 1
                if is_local:
                    local_covered += 1
                else:
                    tinker_covered += 1

    # Daemon workload baselines (pairs already fully covered at daemon inception, from its
    # own init log: 107 missing = 46 tinker-source + 61 local-source, out of 66+66 pairs).
    # So pre-covered = 20 tinker-source + 5 local-source = 25 total; the daemon must fill 107.
    PRE = {"all": 25, "local": 5, "tinker": 20}
    running = failed = 0
    try:
        s = json.loads(QUEUE_PATH.read_text())
        from collections import Counter

        by_status = Counter(r.get("status") for r in s.values())
        running = by_status.get("running", 0)
        failed = by_status.get("failed", 0)
    except Exception:
        pass

    return {
        "total": (tinker_pairs + local_pairs) - PRE["all"],        # 132 - 25 = 107
        "done": covered - PRE["all"],                               # cells the daemon has filled
        "running": running,
        "failed": failed,
        "local_total": local_pairs - PRE["local"],                 # 61
        "local_done": local_covered - PRE["local"],
        "tinker_total": tinker_pairs - PRE["tinker"],              # 46 - 25 = 21 the daemon must fill
        "tinker_done": tinker_covered - PRE["tinker"],
    }


def read_log_signals() -> dict:
    """Scan sequencer.log for error count, submit total, baseline/first-cell markers."""
    sig = {
        "error_count": 0,
        "error_tail": [],
        "submit_total": 0,
        "missing_baselines": None,
        "first_local_cell": False,
    }
    if not SEQ_LOG.exists():
        return sig
    try:
        text = SEQ_LOG.read_text(errors="replace")
    except Exception:
        return sig
    # Errors: CURRENT run log only (archived logs hold old, already-resolved errors).
    err_lines = [l for l in text.splitlines() if "] [error]" in l or "] [fatal]" in l]
    sig["error_count"] = len(err_lines)
    sig["error_tail"] = err_lines[-3:]
    # Tinker submissions: CUMULATIVE across every run log (each run's counter restarts at 1
    # on a daemon relaunch, so true paid exposure = sum of each run-log's final counter).
    total = 0
    for logf in sorted(STATE_DIR.glob("sequencer.log*")):
        try:
            t = logf.read_text(errors="replace")
        except Exception:
            continue
        vals = [int(m) for m in re.findall(r"running submit total=(\d+)", t)]
        hb = [int(m) for m in re.findall(r"submitted_tinker_total=(\d+)", t)]
        total += max(vals + hb + [0])
    sig["submit_total"] = total
    mb = re.findall(r"missing_baselines=(\d+)", text)
    if mb:
        sig["missing_baselines"] = int(mb[-1])
    sig["first_local_cell"] = "[local] launched cell" in text
    return sig


def gpu_collisions() -> list[str]:
    """Return ['GPU5 ours=1446155 foreign=js_park:2737394', ...] for any card carrying
    BOTH a process owned by ME and one owned by another user."""
    try:
        uuid_lines = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()
        app_lines = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()
    except Exception:
        return []
    uuid_to_idx = {}
    for line in uuid_lines:
        if "," not in line:
            continue
        idx_s, uuid = [x.strip() for x in line.split(",", 1)]
        uuid_to_idx[uuid] = int(idx_s)
    per_gpu: dict[int, dict[str, list]] = {}
    for line in app_lines:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        idx = uuid_to_idx.get(parts[1])
        if idx is None:
            continue
        try:
            owner = subprocess.run(
                ["ps", "-o", "user:32=", "-p", str(pid)], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            owner = ""
        if not owner:
            continue
        per_gpu.setdefault(idx, {"mine": [], "foreign": []})
        (per_gpu[idx]["mine"] if owner == ME else per_gpu[idx]["foreign"]).append(f"{owner}:{pid}")
    out = []
    for idx, procs in sorted(per_gpu.items()):
        if procs["mine"] and procs["foreign"]:
            out.append(f"GPU{idx} ours=[{','.join(procs['mine'])}] foreign=[{','.join(procs['foreign'])}]")
    return out


def check(state: dict) -> dict:
    q = read_queue()
    sig = read_log_signals()

    # 1a. Daemon death.
    if not daemon_alive():
        if not state["daemon_dead_reported"]:
            emit(f"ALERT: sequencer daemon ({DAEMON_PATTERN}) is NOT running -- process vanished. "
                 f"Last queue snapshot: done={q.get('done')} running={q.get('running')} pending={q.get('pending')}.")
            state["daemon_dead_reported"] = True
    else:
        state["daemon_dead_reported"] = False  # rearm if it comes back (self-restart)

    # 1b. New errors.
    if sig["error_count"] > state["reported_error_count"]:
        new_n = sig["error_count"] - state["reported_error_count"]
        tail = " || ".join(sig["error_tail"])
        emit(f"ALERT: {new_n} new [error]/[fatal] line(s) in sequencer.log (total {sig['error_count']}). Recent: {tail}")
        state["reported_error_count"] = sig["error_count"]

    # 2a. All local baselines done.
    if not state["baselines_done_reported"] and sig["missing_baselines"] == 0:
        emit("MILESTONE: all 5 local baselines finished -- local training cells can now start.")
        state["baselines_done_reported"] = True

    # 2b. First local training cell launched.
    if not state["first_local_cell_reported"] and sig["first_local_cell"]:
        emit("MILESTONE: first local (GPU) training cell launched -- local-source square training underway.")
        state["first_local_cell_reported"] = True

    # 3a. Every +15 completed cells.
    done = q.get("done", 0)
    milestone = (done // COMPLETED_STEP) * COMPLETED_STEP
    if milestone >= COMPLETED_STEP and milestone > state["last_completed_milestone"]:
        emit(f"PROGRESS: {done}/{q.get('total')} square cells complete "
             f"(tinker {q.get('tinker_done')}/{q.get('tinker_total')}, "
             f"local {q.get('local_done')}/{q.get('local_total')}, "
             f"running={q.get('running')} failed={q.get('failed')}).")
        state["last_completed_milestone"] = milestone

    # 3b. Local queue drained.
    if (not state["local_drained_reported"] and q.get("local_total", 0) > 0
            and q.get("local_done") == q.get("local_total")):
        emit(f"MILESTONE: LOCAL queue fully drained -- all {q.get('local_total')} local-source cells done.")
        state["local_drained_reported"] = True

    # 3c. Whole square done.
    if (not state["square_done_reported"] and q.get("total", 0) > 0
            and q.get("done") == q.get("total")):
        emit(f"MILESTONE: SQUARE COMPLETE -- all {q.get('total')} cells done. "
             f"Final tinker submit total={sig['submit_total']}.")
        state["square_done_reported"] = True

    # 4. Tinker submit-total milestones.
    st = sig["submit_total"]
    for m in SUBMIT_MILESTONES:
        if st >= m and m not in state["submit_milestones_reported"]:
            emit(f"TINKER-SPEND: running submit total crossed {m} (now {st}). "
                 f"Each cell = 1 SFT + 1 DPO submission (paid).")
            state["submit_milestones_reported"].append(m)

    # 5. GPU collision risk.
    collisions = gpu_collisions()
    for c in collisions:
        # dedupe by (gpu, sorted foreign pids)
        key = c.split(" foreign=")[0] + "|" + c.split(" foreign=")[1]
        if key not in state["reported_collisions"]:
            emit(f"GPU-COLLISION RISK: {c} -- our job is co-resident with a foreign user on that card.")
            state["reported_collisions"].append(key)
    # forget collisions that have cleared, so a later recurrence re-alerts
    live_keys = {c.split(" foreign=")[0] + "|" + c.split(" foreign=")[1] for c in collisions}
    state["reported_collisions"] = [k for k in state["reported_collisions"] if k in live_keys]

    return state


def main() -> None:
    emit(f"watcher online (poll every {POLL_SECONDS}s; silent unless a meaningful event fires).")
    state = load_watch_state()
    while True:
        try:
            state = check(state)
            save_watch_state(state)
        except Exception as e:
            emit(f"watcher self-error (non-fatal): {type(e).__name__}: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
