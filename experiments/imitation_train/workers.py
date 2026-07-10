#!/usr/bin/env python3
"""The two worker loops that fill the imitation square.

tinker_worker_loop (remote, no local GPU): submits TINKER-source cells via dementor.training.matrix
(launch_sft / launch_dpo) with bounded concurrency.

local_worker_loop (local GPUs): runs LOCAL-source cells one at a time per free, lease-won card via
`dementor-matrix launch-local-cell` in a CUDA_VISIBLE_DEVICES-pinned subprocess, generates the
local-model baselines those cells depend on, gracefully vacates any of our jobs stranded on a
now-off-limits 0-3 card, and self-heals a raced-out registry write.

Both loops read/write cell status only through worklist's locked accessors.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import gpus
import worklist
from gpus import gpu_lease
from runtime import (
    BLOCK_GPUS,
    COEXIST,
    DATASET,
    DISK_PATH,
    ENOSPC_RETRY_BACKOFF_SEC,
    GPU_POLL_INTERVAL,
    HEARTBEAT_INTERVAL,
    LEASE_GPUS,
    LEASE_HOLDER,
    LOCAL_LOG_DIR,
    MIN_FREE_DISK_GB,
    MP_SOURCE_SLUGS,
    PY,
    REPO,
    SEED,
    SQUARE_DATASETS,
    SUSTAINED_POLLS,
    TINKER_PARALLEL,
    log,
)

_submit_count_lock = threading.Lock()
_submit_count = {"n": 0}


def bump_submit_count(n: int = 1) -> int:
    with _submit_count_lock:
        _submit_count["n"] += n
        return _submit_count["n"]


def free_disk_gb() -> float:
    try:
        st = os.statvfs(DISK_PATH)
        return st.f_bavail * st.f_frsize / 1e9
    except Exception:
        return float("inf")  # fail-open: don't wedge the daemon on a statvfs hiccup


_last_build_sig = {"sig": None}


def build_all_data_once() -> None:
    """Build the square's SFT/DPO CSVs (chatbot_arena ONLY) from available baselines.

    Scoped to SQUARE_DATASETS so we don't rebuild gsm8k/oasst1/writingprompts every poll. Skips
    entirely when disk is low (a build write would just ENOSPC-spam) and de-dupes: only does real
    work when the set of available baselines actually changed since last time."""
    from dementor.training import matrix

    if free_disk_gb() < MIN_FREE_DISK_GB:
        return  # too tight to write CSVs safely; retry a later poll once space frees
    # Cheap change-detection: which baseline files exist right now (chatbot_arena).
    try:
        sig = tuple(sorted(
            p.name for p in (matrix.BASELINES_DIR / DATASET).glob("*_train.csv")
        ))
    except Exception:
        sig = None
    if sig is not None and sig == _last_build_sig["sig"]:
        return  # nothing new since last build -> skip (idempotent, avoids per-poll churn)
    try:
        n1 = matrix.build_sft_data(dry_run=False, datasets=SQUARE_DATASETS)
        n2 = matrix.build_dpo_data(dry_run=False, datasets=SQUARE_DATASETS)
        _last_build_sig["sig"] = sig
        log(f"[data] built {n1} SFT + {n2} DPO CSVs (chatbot_arena only)")
    except Exception as e:
        log(f"[error] build_all_data_once failed: {e}")


def _cell_hit_enospc(slug: str) -> bool:
    """True if this cell's log tail shows a 'No space left on device' / [Errno 28] failure."""
    try:
        tail = (LOCAL_LOG_DIR / f"{slug}.log").read_text(errors="replace")[-6000:]
    except Exception:
        return False
    return "No space left on device" in tail or "Errno 28" in tail or "errno 28" in tail


def cleanup_cell_artifacts(slug: str, rec: dict) -> None:
    """After a cell registers its adapters, delete its regeneratable DPO preference JSONL (~250 MB)
    and empty TRL _trainer dirs, keeping only the LoRA adapters. Prevents per-cell footprint creep
    across 55+ cells."""
    import shutil
    from dementor.training import matrix

    name = f"{rec['source_slug']}_as_{rec['target_slug']}_seed{SEED}"
    for d in (
        matrix.DPO_OUTPUT_DIR / DATASET / name / "preference_artifacts",
        matrix.DPO_OUTPUT_DIR / DATASET / name / "_trainer",
        matrix.SFT_OUTPUT_DIR / DATASET / name / "_trainer",
    ):
        try:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


# ============================================================================
# Tinker worker (remote, no local GPU)
# ============================================================================


def run_tinker_cell(slug: str, rec: dict) -> None:
    from dementor.training import matrix

    cell = matrix.Cell(source=rec["source"], target=rec["target"], dataset=DATASET, seed=SEED)
    try:
        if rec.get("need_sft"):
            if not matrix.sft_data_path(cell).exists():
                log(f"[tinker] {slug}: SFT data not built yet -- leaving pending")
                worklist.set_status(slug, "pending")
                return
            n = bump_submit_count()
            log(f"[tinker] submitting SFT {slug} (running submit total={n})")
            res = matrix.launch_sft(cells=[cell], dry_run=False, parallel=1)
            jobs = res.get("jobs", [])
            if not jobs:
                log(f"[tinker] {slug}: SFT skipped by launch_sft (missing-data/already-registered) -- leaving pending")
                worklist.set_status(slug, "pending")
                return
            if jobs[0].get("error"):
                raise RuntimeError(f"SFT failed: {jobs[0]['error']}")
            log(f"[tinker] SFT done {slug}")
            worklist.set_status(slug, "running", need_sft=False)
        if rec.get("need_dpo"):
            if not matrix.dpo_data_path(cell).exists():
                log(f"[tinker] {slug}: DPO data not built yet -- leaving pending")
                worklist.set_status(slug, "pending")
                return
            n = bump_submit_count()
            log(f"[tinker] submitting DPO {slug} (running submit total={n})")
            res = matrix.launch_dpo(cells=[cell], dry_run=False, parallel=1)
            jobs = res.get("jobs", [])
            if not jobs:
                log(f"[tinker] {slug}: DPO skipped by launch_dpo (missing-data/missing-sft/already-registered) -- leaving pending")
                worklist.set_status(slug, "pending")
                return
            if jobs[0].get("error"):
                raise RuntimeError(f"DPO failed: {jobs[0]['error']}")
            log(f"[tinker] DPO done {slug}")
        worklist.set_status(slug, "done")
        log(f"[tinker] CELL COMPLETE {slug}")
    except Exception as e:
        worklist.set_status(slug, "failed", error=str(e))
        log(f"[error] tinker cell {slug} failed: {e}\n{traceback.format_exc()}")


def tinker_worker_loop(stop_event: threading.Event) -> None:
    log("[tinker] worker loop starting")
    while not stop_event.is_set():
        try:
            worklist.refresh_from_registry()
            pending = worklist.get_pending("tinker")
            if not pending:
                log("[tinker] no pending tinker cells (queue empty or all in-flight) -- sleeping")
                time.sleep(HEARTBEAT_INTERVAL)
                continue
            log(f"[tinker] dispatching {len(pending)} pending tinker cells, parallel={TINKER_PARALLEL}")
            for slug, _rec in pending:
                worklist.set_status(slug, "running")
            with ThreadPoolExecutor(max_workers=TINKER_PARALLEL) as pool:
                futs = [pool.submit(run_tinker_cell, slug, rec) for slug, rec in pending]
                for f in futs:
                    f.result()
            n_bounced = sum(
                1 for slug, _ in pending
                if worklist.get_rec(slug) and worklist.get_rec(slug)["status"] == "pending"
            )
            log(f"[tinker] batch complete ({n_bounced}/{len(pending)} left pending -- data not ready yet)")
            if n_bounced == len(pending):
                # Nothing in this whole batch was actually submittable (baselines/data still
                # missing) -- avoid busy-looping while we wait for baseline-gen to catch up.
                time.sleep(HEARTBEAT_INTERVAL)
        except Exception as e:
            log(f"[error] tinker_worker_loop: {e}\n{traceback.format_exc()}")
            time.sleep(30)


# ============================================================================
# Local worker: baseline generation + GPU-pinned SFT/DPO
# ============================================================================


def baseline_path(model_id: str) -> Path:
    from dementor.training import matrix

    return matrix.baseline_path(model_id, DATASET)


def local_core_model_ids() -> list[str]:
    return [m["id"] for m in worklist.core_models() if m["backend"] == "local"]


def missing_baselines() -> list[str]:
    return [mid for mid in local_core_model_ids() if not baseline_path(mid).exists()]


def launch_baseline_subprocess(gpu_idx: int, model_id: str) -> subprocess.Popen:
    slug = model_id.replace("/", "_")
    log_file = LOCAL_LOG_DIR / f"baseline_{slug}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    cmd = [
        PY, "-u", "-m", "dementor.training.matrix", "generate-target-responses",
        "--models", model_id, "--datasets", DATASET, "--parallel", "1",
    ]
    f = open(log_file, "a")
    f.write(f"\n=== launch {datetime.now().isoformat()} on GPU{gpu_idx} ===\n")
    f.flush()
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=f, stderr=subprocess.STDOUT)
    log(f"[local] launched baseline-gen for {model_id} on GPU{gpu_idx} (pid={proc.pid}), log={log_file}")
    return proc


def launch_cell_subprocess(gpu_idx: int, slug: str, rec: dict, secondary_gpu=None) -> subprocess.Popen:
    log_file = LOCAL_LOG_DIR / f"{slug}.log"
    env = os.environ.copy()
    if secondary_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = f"{gpu_idx},{secondary_gpu}"
        env["DEMENTOR_MP"] = "1"  # backend shards the model across both cards (device_map)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    cmd = [
        PY, "-u", "-m", "dementor.training.matrix", "launch-local-cell",
        "--source", rec["source"], "--target", rec["target"],
        "--dataset", DATASET, "--seed", str(SEED), "--phase", "all",
    ]
    f = open(log_file, "a")
    f.write(f"\n=== launch {datetime.now().isoformat()} on GPU{gpu_idx} ===\n")
    f.flush()
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=f, stderr=subprocess.STDOUT)
    log(f"[local] launched cell {slug} on GPU{gpu_idx} (pid={proc.pid}), log={log_file}")
    return proc


def self_heal_registry(slug: str, rec: dict) -> None:
    """If launch-local-cell exited 0 but the registry write got raced out (two local subprocesses
    finishing close enough together to race the JSON read-modify-write), repair it from the
    deterministic output directory."""
    from dementor.training import matrix
    from dementor.training.common import record_adapter_mapping

    keys = worklist.load_registry_keys()
    name = f"{rec['source_slug']}_as_{rec['target_slug']}_seed{SEED}"
    if rec.get("need_sft") and f"sft_{slug}" not in keys:
        out_dir = matrix.SFT_OUTPUT_DIR / DATASET / name
        if (out_dir / "adapter_config.json").exists():
            log(f"[self-heal] registering missing sft_{slug} from {out_dir}")
            record_adapter_mapping(
                f"sft_{slug}", str(out_dir), matrix.DATA / "tinker_adapters.json",
                metadata={"backend": "local", "checkpoint_path": str(out_dir), "base_model": rec["source"]},
            )
    if rec.get("need_dpo") and f"dpo_{slug}" not in keys:
        out_dir = matrix.DPO_OUTPUT_DIR / DATASET / name
        if (out_dir / "adapter_config.json").exists():
            log(f"[self-heal] registering missing dpo_{slug} from {out_dir}")
            record_adapter_mapping(
                f"dpo_{slug}", str(out_dir), matrix.DATA / "tinker_adapters.json",
                metadata={"backend": "local", "checkpoint_path": str(out_dir), "base_model": rec["source"]},
            )


def _terminate_proc(proc: subprocess.Popen, *, grace: float = 20.0) -> None:
    """Cleanly stop ONE of our own subprocesses (SIGTERM, then SIGKILL). Only ever called on procs
    WE spawned (tracked in running_cells/running_baselines) -- never on a foreign/js_park pid."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < grace:
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass


def vacate_block_gpus(running_cells: dict, running_baselines: dict, usability: dict, mp_secondary: dict) -> None:
    """Graceful-vacate safety net: if any of OUR tracked jobs is sitting on a GPU 0-3 that the
    current policy marks off-limits (js_park block active/clearing), stop it cleanly and RE-QUEUE the
    cell as pending (not failed) so it retrains later on the lease pool. Baselines are just killed
    (they regenerate). Never touches js_park's processes. In steady state this is a no-op."""
    for gpu_idx, (proc, slug) in list(running_cells.items()):
        if gpu_idx not in BLOCK_GPUS:
            continue
        ok, reason = usability.get(gpu_idx, (False, "unknown"))
        if ok:
            continue  # 0-3 currently re-opened (sustained-clear) -> leave it
        _terminate_proc(proc)
        del running_cells[gpu_idx]
        mp_secondary.pop(gpu_idx, None)  # release the 2nd card if this was a model-parallel cell
        worklist.set_status(slug, "pending", pid=None, gpu=None)
        log(f"[vacate] RE-QUEUED cell {slug}: was on GPU{gpu_idx} (js_park block: {reason}); "
            f"stopped our subprocess and marked pending for retrain on the lease pool")
    for gpu_idx, (proc, model_id) in list(running_baselines.items()):
        if gpu_idx not in BLOCK_GPUS:
            continue
        ok, _reason = usability.get(gpu_idx, (False, "unknown"))
        if ok:
            continue
        _terminate_proc(proc)
        del running_baselines[gpu_idx]
        log(f"[vacate] stopped baseline-gen {model_id} on GPU{gpu_idx} (js_park block); will regenerate")


def local_worker_loop(stop_event: threading.Event) -> None:
    log("[local] worker loop starting")
    running_baselines: dict[int, tuple[subprocess.Popen, str]] = {}  # gpu_idx -> (proc, model_id)
    running_cells: dict[int, tuple[subprocess.Popen, str]] = {}  # gpu_idx -> (proc, slug)
    mp_secondary: dict[int, int] = {}  # primary_gpu -> secondary_gpu (model-parallel cells hold 2 cards)
    idle_counts: dict[int, int] = {}  # gpu_idx -> consecutive usable polls (sustained-idle gate; coexistence)
    last_heartbeat = 0.0

    if COEXIST:
        try:
            reclaimed = gpu_lease.reap()  # drop any stale leases left by a prior crashed sequencer run
            log(f"[local] gpu_lease coexistence ON: holder={LEASE_HOLDER} cards={sorted(LEASE_GPUS)} "
                f"sustained_polls={SUSTAINED_POLLS} lease_root={gpu_lease.LOCK_ROOT} reaped_stale={reclaimed}")
        except Exception as e:
            log(f"[local] gpu_lease.reap warning: {e}")
    else:
        log("[local] gpu_lease coexistence OFF (SEQ_COEXIST=0): immediate-grab, no lease")

    while not stop_event.is_set():
        try:
            # 1. Reap finished subprocesses first.
            for gpu_idx, (proc, model_id) in list(running_baselines.items()):
                rc = proc.poll()
                if rc is not None:
                    del running_baselines[gpu_idx]
                    gpus.lease_release(gpu_idx)  # free the card's lease for the other sustained-idle daemons
                    if rc == 0:
                        log(f"[local] baseline-gen COMPLETE {model_id} on GPU{gpu_idx}")
                    else:
                        log(f"[local] baseline-gen FAILED {model_id} on GPU{gpu_idx} (rc={rc}) -- will retry next cycle")

            for gpu_idx, (proc, slug) in list(running_cells.items()):
                rc = proc.poll()
                if rc is not None:
                    del running_cells[gpu_idx]
                    sec = mp_secondary.pop(gpu_idx, None)  # release 2nd card for a model-parallel cell
                    gpus.lease_release(gpu_idx)  # free the card's lease for the other sustained-idle daemons
                    if sec is not None:
                        gpus.lease_release(sec)
                    rec = worklist.get_rec(slug)
                    if rec is None:
                        continue
                    if rc == 0:
                        self_heal_registry(slug, rec)
                        keys = worklist.load_registry_keys()
                        still_missing = (rec.get("need_sft") and f"sft_{slug}" not in keys) or (
                            rec.get("need_dpo") and f"dpo_{slug}" not in keys
                        )
                        if still_missing:
                            worklist.set_status(slug, "failed", error="subprocess exited 0 but registry entry missing after self-heal")
                            log(f"[error] {slug}: subprocess exited 0 but registry entry missing after self-heal")
                        else:
                            worklist.set_status(slug, "done")
                            cleanup_cell_artifacts(slug, rec)
                            log(f"[local] CELL COMPLETE {slug} on GPU{gpu_idx}")
                    elif _cell_hit_enospc(slug):
                        # Transient shared-disk ENOSPC -- NEVER terminal. Re-queue pending with a
                        # backoff so it retries once space frees (the launch guard also throttles).
                        n = rec.get("enospc_retries", 0) + 1
                        worklist.set_status(slug, "pending", enospc_retries=n,
                                            retry_after=time.time() + ENOSPC_RETRY_BACKOFF_SEC, error=None)
                        log(f"[disk] local cell {slug} hit ENOSPC on GPU{gpu_idx} "
                            f"(retry #{n}) -> re-queued pending (backoff {ENOSPC_RETRY_BACKOFF_SEC:.0f}s)")
                    else:
                        worklist.set_status(slug, "failed", error=f"launch-local-cell exited rc={rc}")
                        log(f"[error] local cell {slug} FAILED on GPU{gpu_idx} (rc={rc})")

            worklist.refresh_from_registry()

            # 2. What GPUs are usable right now?
            usability = gpus.gpu_usability()

            # 2b. Graceful-vacate: pull any of our jobs off a now-off-limits 0-3 card and re-queue
            # the cell (pending, not failed) so it retrains on the lease pool.
            vacate_block_gpus(running_cells, running_baselines, usability, mp_secondary)

            busy_idxs = set(running_baselines) | set(running_cells) | set(mp_secondary.values())
            # Sustained-idle gate (coexistence): a card must read usable for SUSTAINED_POLLS
            # consecutive polls before we treat it as free -- this yields transient frees to the
            # steering roster's no-lease first-idle scheduler. Cards busy with OUR jobs reset to 0;
            # not-usable cards reset to 0. Disabled when SEQ_COEXIST=0.
            for idx, (ok, _reason) in usability.items():
                if idx in busy_idxs or not ok:
                    idle_counts[idx] = 0
                else:
                    idle_counts[idx] = idle_counts.get(idx, 0) + 1
            free_gpus = [
                idx for idx, (ok, _reason) in usability.items()
                if ok and idx not in busy_idxs
                and (not COEXIST or idle_counts.get(idx, 0) >= SUSTAINED_POLLS)
            ]

            # 2c. Disk guard: never LAUNCH new local work while free space is below the floor --
            # throttle (wait) instead of writing into a full disk and ENOSPC-failing.
            free_gb = free_disk_gb()
            if free_gb < MIN_FREE_DISK_GB and free_gpus:
                if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:  # throttle this log
                    log(f"[disk] free={free_gb:.1f}GB < {MIN_FREE_DISK_GB:.0f}GB floor -> "
                        f"holding {len(free_gpus)} idle GPU(s), not launching until space frees")
                free_gpus = []

            # 3. Prioritize missing baselines (everything downstream depends on them).
            need_baselines = [
                m for m in missing_baselines() if m not in {mid for _, mid in running_baselines.values()}
            ]
            for gpu_idx in list(free_gpus):
                if not need_baselines:
                    break
                if not gpus.lease_claim(gpu_idx):
                    free_gpus.remove(gpu_idx)  # another lease-holding daemon won this card -> yield it
                    continue
                model_id = need_baselines.pop(0)
                proc = launch_baseline_subprocess(gpu_idx, model_id)
                running_baselines[gpu_idx] = (proc, model_id)
                free_gpus.remove(gpu_idx)
                idle_counts[gpu_idx] = 0

            # 4. Refresh data CSVs (chatbot_arena only) then dispatch local cells.
            if free_gpus:
                build_all_data_once()
                pending_local = worklist.get_pending("local")
                from dementor.training import matrix

                now_ts = time.time()
                consumed: set = set()  # GPUs claimed this pass (a model-parallel cell claims 2)
                for gpu_idx in list(free_gpus):
                    if gpu_idx in consumed:
                        continue
                    for slug, rec in pending_local:
                        cur = worklist.get_rec(slug)
                        if cur is None or cur["status"] != "pending":
                            continue
                        if cur.get("retry_after", 0) > now_ts:
                            continue  # ENOSPC backoff still in effect -- try again later
                        cell = matrix.Cell(
                            source=rec["source"], target=rec["target"], dataset=DATASET, seed=SEED
                        )
                        sft_csv = matrix.sft_data_path(cell)
                        dpo_csv = matrix.dpo_data_path(cell)
                        if not (sft_csv.exists() and dpo_csv.exists()):
                            continue  # data not ready yet (baseline still missing/pending) -- skip for now
                        if rec.get("source_slug") in MP_SOURCE_SLUGS:
                            # student too big for one 80GB card -> shard across 2 GPUs. Needs a 2nd
                            # free card this pass; otherwise defer (prefer a smaller cell / next cycle).
                            sec = next((g for g in free_gpus if g != gpu_idx and g not in consumed), None)
                            if sec is None:
                                continue
                            # MP needs BOTH cards' leases; if we can't win both, release what we grabbed.
                            if not gpus.lease_claim(gpu_idx):
                                consumed.add(gpu_idx)  # another daemon owns this card -> can't use it
                                break
                            if not gpus.lease_claim(sec):
                                gpus.lease_release(gpu_idx)
                                continue  # 2nd card lost; maybe a single-card cell can still use gpu_idx
                            proc = launch_cell_subprocess(gpu_idx, slug, rec, secondary_gpu=sec)
                            running_cells[gpu_idx] = (proc, slug)
                            mp_secondary[gpu_idx] = sec
                            consumed.add(gpu_idx)
                            consumed.add(sec)
                            idle_counts[gpu_idx] = 0
                            idle_counts[sec] = 0
                            log(f"[local] model-parallel cell {slug} on GPU{gpu_idx}+{sec}")
                        else:
                            if not gpus.lease_claim(gpu_idx):
                                consumed.add(gpu_idx)  # another daemon owns this card -> yield it
                                break
                            proc = launch_cell_subprocess(gpu_idx, slug, rec)
                            running_cells[gpu_idx] = (proc, slug)
                            consumed.add(gpu_idx)
                            idle_counts[gpu_idx] = 0
                        worklist.set_status(slug, "running", pid=proc.pid, gpu=gpu_idx)
                        break

            # 5. Heartbeat.
            now = time.time()
            if now - last_heartbeat > HEARTBEAT_INTERVAL:
                last_heartbeat = now
                counts = worklist.queue_counts()
                log(
                    f"[heartbeat] queue={counts} running_cells={len(running_cells)} "
                    f"missing_baselines={len(missing_baselines())} running_baselines={len(running_baselines)} "
                    f"free_disk={free_disk_gb():.1f}GB gpu_status={usability} "
                    f"submitted_tinker_total={_submit_count['n']}"
                )

            time.sleep(GPU_POLL_INTERVAL)
        except Exception as e:
            log(f"[error] local_worker_loop: {e}\n{traceback.format_exc()}")
            time.sleep(30)
