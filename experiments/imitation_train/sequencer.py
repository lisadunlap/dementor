#!/usr/bin/env python3
"""Detached, restartable daemon that trains the 12x12 imitation "square".

Computes the missing {sft,dpo}_chatbot_arena_<source>_as_<target>_seed42 cells for the 12 core
models (config.yaml roster, imitation: core) and drives two workers concurrently:
  - TINKER-source cells are submitted remotely via dementor.training.matrix (no local GPU).
  - LOCAL-source cells run one at a time per free, lease-won GPU in a CUDA_VISIBLE_DEVICES-pinned
    subprocess; the local-model baselines they depend on are generated first, opportunistically.

The concerns are split across focused modules:
  runtime.py   -- portable, env-driven config + logging (imported first by everyone).
  worklist.py  -- registry-driven worklist + the single locked shared-state dict.
  gpus.py      -- GPU usability polling + js_park block policy + shared-lease arbitration.
  workers.py   -- the tinker + local worker loops and their subprocess launchers.
  sequencer.py -- this file: the entrypoint that wires them together.

Launched via `setsid nohup PY sequencer.py >> sequencer.log 2>&1 &` so it outlives the launching
session. See README.md for env vars and operation.
"""
from __future__ import annotations

import os
import threading
import time
import traceback

import workers
import worklist
from runtime import log


def main() -> None:
    log("=" * 80)
    log(f"sequencer starting, pid={os.getpid()}")
    worklist.backup_registry_once()

    worklist.init_state()
    counts = worklist.backend_counts()
    log(f"[init] missing cells: {sum(counts.values())} total "
        f"({counts.get('tinker', 0)} tinker-source, {counts.get('local', 0)} local-source)")

    workers.build_all_data_once()

    stop_event = threading.Event()
    t_tinker = threading.Thread(
        target=workers.tinker_worker_loop, args=(stop_event,), daemon=False, name="tinker-worker")
    t_local = threading.Thread(
        target=workers.local_worker_loop, args=(stop_event,), daemon=False, name="local-worker")
    t_tinker.start()
    t_local.start()

    try:
        while True:
            time.sleep(60)
            counts = worklist.queue_counts()
            if counts and set(counts) == {"done"}:
                log("[main] ALL CELLS DONE -- sequencer idling (still alive for monitoring)")
            elif not counts:
                log("[main] queue empty (nothing missing) -- sequencer idling")
    except KeyboardInterrupt:
        log("[main] KeyboardInterrupt -- stopping workers")
        stop_event.set()
        t_tinker.join(timeout=30)
        t_local.join(timeout=30)


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except Exception as e:
            log(f"[fatal] top-level exception, restarting in 30s: {e}\n{traceback.format_exc()}")
            time.sleep(30)
