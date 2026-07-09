#!/usr/bin/env python3
"""Detached, restartable daemon that trains the 12x12 imitation "square".

Computes the missing {sft,dpo}_chatbot_arena_{source}_as_{target}_seed42 cells for the
12 core models (dementor-ethan config.yaml roster, imitation: core), then:
  - TINKER-source cells: submitted remotely via dementor.training.matrix (launch_sft /
    launch_dpo), bounded concurrency via a thread pool. No local GPU needed.
  - LOCAL-source cells: run one at a time per free GPU via
    `dementor-matrix launch-local-cell` in a CUDA_VISIBLE_DEVICES-pinned subprocess.
    A GPU is usable iff: index != 4 (compute-prohibited); memory.used < 1500 MiB; no
    compute process owned by a user other than $USER; AND it is not inside js_park's
    active 0-3 block. GPUs 0-3 are treated as js_park's (he cycles vLLM across them all
    night) and stay OFF-LIMITS to local training while any foreign user is on ANY of
    0/1/2/3 -- re-opening only after 0-3 has been continuously foreign-free for a
    sustained window (BLOCK_SUSTAINED_CLEAR_SECONDS). If one of our jobs is ever found on
    an off-limits 0-3 card it is gracefully stopped and its cell re-queued (see
    vacate_block_gpus). In practice local work stays on GPUs 5/6/7.
  - Local-model baselines (needed before any local cell's SFT/DPO data can be built)
    are generated opportunistically on whatever GPU frees up first, ahead of training.

State model: a SINGLE shared in-memory dict (SHARED_STATE), guarded by one lock, is the
only source of truth during a run; it is persisted to queue_state.json on every mutation.
This matters: the tinker worker (thread) and the local-GPU worker (thread) both mutate
cell status concurrently, and earlier versions of this daemon had each call an
independent load-from-disk/save-to-disk cycle -- a classic lost-update race where one
thread's stale snapshot could silently flip another thread's "running"/"done" cell back
to "pending", triggering a duplicate (paid) Tinker resubmission. The pid-liveness
"was this actually still running" reconciliation only ever runs ONCE, at process startup
(covering a daemon restart, e.g. after an OOM-kill or session death) -- never on the
periodic mid-run refreshes, which only add newly-missing cells / retire newly-completed
ones by consulting the registry (the real source of truth for "done").

Everything is logged to sequencer.log with timestamps. Launched via
`setsid nohup PY sequencer.py >> sequencer.log 2>&1 &` so it outlives the launching
session.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------- portable config
# All roots are ENV-OVERRIDABLE with our-box values as defaults (see PARTNER_SETUP.md). Canonical
# shared env vars (identical names on the eval/steering halves): DEMENTOR_REPO, DEMENTOR_DATA,
# DEMENTOR_HF_HOME, DEMENTOR_GPUS. The live campaign runs from its own state dir, so a fresh clone
# with defaults gets an independent DEMENTOR_DATA/imitation_train state and never clobbers it.
def _env(name, default):
    v = os.environ.get(name)
    return v if v else default

def _gpu_set(name, default):
    # Distinguish UNSET (use default) from set-to-empty (a partner disabling a policy, e.g.
    # DEMENTOR_BLOCK_GPUS="" -> no block cards). Empty string must NOT fall back to the default.
    raw = os.environ.get(name)
    if raw is None:
        raw = default
    return {int(x) for x in raw.split(",") if str(x).strip()}

_HERE = Path(__file__).resolve().parent                          # experiments/imitation_train
REPO = Path(_env("DEMENTOR_REPO", str(_HERE.parent.parent)))     # repo root: two levels up
PY = _env("DEMENTOR_PY", sys.executable)                         # interpreter for training subprocesses
DATA_ROOT = Path(_env("DEMENTOR_DATA", str(REPO / "data")))      # big-disk data/outputs root (may be symlink)
# State (queue_state.json, local_logs/, markers). Default a dedicated dir under DEMENTOR_DATA; point
# DEMENTOR_TRAIN_STATE at an existing campaign dir to RESUME it (e.g. our /data exp_imitation_square).
STATE_DIR = Path(_env("DEMENTOR_TRAIN_STATE", str(DATA_ROOT / "imitation_train")))
QUEUE_PATH = STATE_DIR / "queue_state.json"
# Registry the training writes/reads MUST match dementor.training.matrix.DATA (= DEMENTOR_DATA/data).
REGISTRY_PATH = DATA_ROOT / "tinker_adapters.json"
REGISTRY_BACKUP_MARKER = STATE_DIR / "registry_backup_done.marker"
LOCAL_LOG_DIR = STATE_DIR / "local_logs"
DATASET = "chatbot_arena"
SEED = int(os.environ.get("SEED", "42"))  # multi-seed: wrapper sets SEED=43,44 for robustness expansion
TINKER_PARALLEL = 4
GPU_POLL_INTERVAL = 30.0
HEARTBEAT_INTERVAL = 120.0
# GPU4 is compute-prohibited on OUR box; NEVER usable. Env DEMENTOR_FORBIDDEN_GPUS (default "4");
# a partner with no banned card sets it empty.
FORBIDDEN_GPUS = _gpu_set("DEMENTOR_FORBIDDEN_GPUS", "4")
# Slugs whose LOCAL training must shard across 2 GPUs (device_map model-parallel) because the
# student is too big for one 80GB card. gemma-4-31b (256K-vocab, untied lm_head VLM text tower)
# needs it. granite-4-h-small (32B-A9B GraniteMoeHybrid) is INTENTIONALLY single-GPU: 32.7B params
# bf16 ~= 65GB fits one 80GB H100 with gradient_checkpointing (tied 100K-vocab embeddings; all 72
# experts resident but that's still ~65GB) -- like the other 32B locals. If it ever OOMs on one
# card, add "granite-4-h-small" here to give its cells the 2-GPU model-parallel path.
MP_SOURCE_SLUGS = {"gemma-4-31b", "llama-3.3-70b"}  # students too big for one 80GB card -> device_map across 2 GPUs
MEM_FREE_THRESHOLD_MIB = 1500.0
ME = os.environ.get("USER", "ethantsliu")

# --- disk pressure policy -------------------------------------------------------
# /data is a SHARED, chronically-near-full 28T disk (other users too). When it hits
# 0 bytes free, adapter/data writes fail with OSError [Errno 28] and a cell that would
# otherwise succeed dies. Two defenses: (1) a proactive guard -- never LAUNCH a local
# cell/baseline while free space is below MIN_FREE_DISK_GB, so we throttle instead of
# failing; (2) reactive resilience -- a cell that dies with ENOSPC is RE-QUEUED (pending
# with backoff), never marked terminally failed, so transient shared-disk pressure can't
# permanently kill it. We also scope data-building to the square's only dataset and prune
# each finished cell's regeneratable DPO preference_artifacts to keep our footprint small.
DISK_PATH = str(DATA_ROOT)          # free-space probed on the data/outputs disk
MIN_FREE_DISK_GB = 20.0             # don't launch new local work below this
ENOSPC_RETRY_BACKOFF_SEC = 180.0    # re-queued ENOSPC cell won't relaunch before this
SQUARE_DATASETS = ["chatbot_arena"]  # the square is chatbot_arena-only; don't build the others

# --- js_park block policy -------------------------------------------------------
# GPUs 0-3 are js_park's active block: he cycles vLLM across them all night, so a card
# there that momentarily shows no foreign process is NOT safe to grab -- if he returns
# while our ~30 GB LoRA cell holds it, his run OOMs. So we treat 0-3 as OFF-LIMITS to
# local training whenever js_park (any foreign user) is present on ANY of 0/1/2/3, and
# only re-open them once 0-3 has been continuously foreign-free for a SUSTAINED window.
# Tinker (remote) cells are unaffected -- this only gates local-GPU scheduling.
# js_park's active block on OUR box. Env DEMENTOR_BLOCK_GPUS (default "0,1,2,3"); a partner on a
# DEDICATED box (no foreign block) sets it empty to disable this politeness gate.
BLOCK_GPUS = _gpu_set("DEMENTOR_BLOCK_GPUS", "0,1,2,3")
BLOCK_SUSTAINED_CLEAR_SECONDS = 600.0  # 0-3 must be foreign-free this long before we trust them
_block_state = {"clear_since": None}   # monotonic-ish ts when 0-3 last became (and stayed) clear

sys.path.insert(0, str(REPO))
# Canonical DEMENTOR_HF_HOME (default our-box path) -> standard HF_HOME the child training procs read.
os.environ.setdefault("HF_HOME", _env("DEMENTOR_HF_HOME", "/data/ethantsliu/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# --- Shared GPU lease (coexistence with erosion_daemon / retry_pc_fails / steering roster) -------
# GPUs 5/6/7 are shared with several other sustained-idle daemons that arbitrate "who gets a freed
# card" via an atomic mkdir lease (gpu_lease.py). Without taking that lease this daemon could watch
# the same freed card go idle and launch a ~65GB cell on top of another daemon's job -> OOM/race.
# Imported from the sibling in-repo imitation_safety package so training + erosion daemons share ONE
# lease namespace (gpu_lease.LOCK_ROOT is env-derived from the same DEMENTOR_DATA on a given box).
sys.path.insert(0, str(REPO / "experiments" / "imitation_safety"))
import gpu_lease  # noqa: E402  (shared atomic GPU-lease lock)

# Cards this daemon may lease-arbitrate. Canonical DEMENTOR_GPUS (default 5,6,7 on our box; a partner
# on a dedicated box sets DEMENTOR_GPUS=0,1,2,3). erosion/retry daemons share exactly these.
LEASE_GPUS = _gpu_set("DEMENTOR_GPUS", "5,6,7")
LEASE_HOLDER = "granite_imit"   # our label written into the lease meta
# Coexistence mode (default ON). When ON, before launching local work on a 5/6/7 card the daemon (a)
# requires the card to read usable for SUSTAINED_POLLS consecutive polls -- so the steering roster's
# first-idle scheduler (which takes NO lease) wins transient frees between its own jobs, i.e. we YIELD
# to the roster -- AND (b) wins gpu_lease.try_claim(card) to beat the other lease-holding daemons;
# the lease is released the moment our job on that card is reaped. Set SEQ_COEXIST=0 to restore the
# old immediate-grab / no-lease behavior. This is a strict politeness/safety layer: it never changes
# WHICH cells run, only the timing of grabbing a card, so it is safe to leave on for every seed/model.
COEXIST = os.environ.get("SEQ_COEXIST", "1") not in ("0", "false", "False", "")
SUSTAINED_POLLS = int(os.environ.get("SEQ_SUSTAINED_POLLS", "3"))
# Local-only mode (default OFF). When ON, the queue is restricted to backend==local cells so the
# tinker worker never touches Tinker-only-source cells (no local source weights -> can't train
# locally). Set IMIT_LOCAL_ONLY=1 for the granite LOCAL run; left OFF, other seeds/models keep
# processing their tinker cells exactly as before (behavior unchanged for them).
LOCAL_ONLY = os.environ.get("IMIT_LOCAL_ONLY", "0") in ("1", "true", "True")

LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_lock = threading.Lock()
_state_lock = threading.RLock()
_submit_count_lock = threading.Lock()
_submit_count = {"n": 0}

SHARED_STATE: dict = {}  # slug -> record; guarded by _state_lock; single source of truth


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    with _log_lock:
        print(line, flush=True)


def bump_submit_count(n: int = 1) -> int:
    with _submit_count_lock:
        _submit_count["n"] += n
        return _submit_count["n"]


# ============================================================================
# Registry / config helpers
# ============================================================================


def backup_registry_once() -> None:
    if REGISTRY_BACKUP_MARKER.exists():
        return
    if REGISTRY_PATH.exists():
        backup_path = STATE_DIR / f"tinker_adapters.backup_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        backup_path.write_text(REGISTRY_PATH.read_text())
        log(f"[backup] {REGISTRY_PATH} -> {backup_path}")
    REGISTRY_BACKUP_MARKER.write_text(datetime.now().isoformat())


def load_registry_keys() -> set:
    if REGISTRY_PATH.exists():
        try:
            return set(json.loads(REGISTRY_PATH.read_text()).keys())
        except Exception as e:
            log(f"[warn] failed to read registry: {e}")
            return set()
    return set()


def core_models() -> list[dict]:
    from dementor import config

    return [m for m in config.roster() if m.get("imitation") == "core"]


def compute_queue() -> dict:
    """Fresh registry-driven view of every still-missing core-square cell. Pure function
    of config.yaml + the registry; carries no run-time status (caller merges that in)."""
    models = core_models()
    ids = [m["id"] for m in models]
    slug_of = {m["id"]: m["slug"] for m in models}
    backend_of = {m["id"]: m["backend"] for m in models}
    keys = load_registry_keys()
    cells = {}
    for s in ids:
        for t in ids:
            if s == t:
                continue
            ss, ts = slug_of[s], slug_of[t]
            slug = f"{DATASET}_{ss}_as_{ts}_seed{SEED}"
            need_sft = f"sft_{slug}" not in keys
            need_dpo = f"dpo_{slug}" not in keys
            if not need_sft and not need_dpo:
                continue
            cells[slug] = {
                "source": s,
                "target": t,
                "source_slug": ss,
                "target_slug": ts,
                "backend": backend_of[s],
                "need_sft": need_sft,
                "need_dpo": need_dpo,
            }
    if LOCAL_ONLY:
        # Drop Tinker-source cells entirely (no local source weights -> untrainable here). Keeps the
        # tinker worker idle and prevents it from submitting paid Tinker jobs during a local-only run.
        cells = {slug: rec for slug, rec in cells.items() if rec["backend"] == "local"}
    return cells


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _persist_locked() -> None:
    """Caller MUST hold _state_lock."""
    tmp = QUEUE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(SHARED_STATE, indent=2, sort_keys=True))
    os.replace(tmp, QUEUE_PATH)


def init_state() -> None:
    """Run exactly ONCE, at daemon startup. Loads any queue_state.json left by a prior
    (possibly killed) run of this same daemon, reconciles it against the registry
    (ground truth for "done") and pid-liveness (ground truth for "actually still
    running"), and seeds SHARED_STATE. This is the ONLY place a "running" status is
    ever reset to "pending" based on staleness -- mid-run refreshes never do this.
    """
    fresh = compute_queue()
    prior = {}
    if QUEUE_PATH.exists():
        try:
            prior = json.loads(QUEUE_PATH.read_text())
        except Exception as e:
            log(f"[warn] failed to read prior queue state: {e}")

    with _state_lock:
        SHARED_STATE.clear()
        for slug, rec in fresh.items():
            prev = prior.get(slug)
            rec["status"] = "pending"
            if prev:
                status = prev.get("status")
                if status == "running":
                    pid = prev.get("pid")
                    if pid and _pid_alive(pid):
                        rec.update({k: v for k, v in prev.items() if k not in ("need_sft", "need_dpo")})
                        log(f"[resume] {slug}: prior run (pid={pid}) still alive -- keeping status=running")
                    else:
                        log(f"[resume] {slug}: prior run (pid={prev.get('pid')}) not alive -> retry")
                elif status == "failed":
                    rec["prior_error"] = prev.get("error")
                    log(f"[resume] {slug}: prior attempt failed ({prev.get('error')}) -> retry")
            SHARED_STATE[slug] = rec
        n_prior_done = sum(1 for r in prior.values() if r.get("status") == "done")
        _persist_locked()
    log(
        f"[init] {len(fresh)} cells still missing from registry "
        f"(prior state file had {len(prior)} tracked cells, {n_prior_done} of which are now "
        f"fully resolved and dropped from the active queue)"
    )


def refresh_from_registry() -> None:
    """Mid-run refresh: consult the registry + config for (a) cells that are now fully
    done (drop/mark done) and (b) cells whose data dependencies just became available
    (need_sft/need_dpo flags). NEVER resets an in-flight "running" status -- only the
    registry (via self-heal / job completion) or the owning worker thread does that.
    """
    fresh = compute_queue()
    with _state_lock:
        for slug, new_rec in fresh.items():
            if slug not in SHARED_STATE:
                new_rec["status"] = "pending"
                SHARED_STATE[slug] = new_rec
                log(f"[queue] new cell appeared: {slug}")
            else:
                cur = SHARED_STATE[slug]
                cur["need_sft"] = new_rec["need_sft"]
                cur["need_dpo"] = new_rec["need_dpo"]
        for slug, rec in list(SHARED_STATE.items()):
            if slug not in fresh and rec.get("status") != "done":
                rec["status"] = "done"
                log(f"[queue] {slug}: now fully covered by registry -> done")
        _persist_locked()


def get_pending(backend: str) -> list[tuple[str, dict]]:
    with _state_lock:
        return [
            (slug, dict(rec))
            for slug, rec in SHARED_STATE.items()
            if rec.get("backend") == backend and rec.get("status") == "pending"
        ]


def get_rec(slug: str) -> dict | None:
    with _state_lock:
        rec = SHARED_STATE.get(slug)
        return dict(rec) if rec is not None else None


def set_status(slug: str, status: str, **extra) -> None:
    with _state_lock:
        rec = SHARED_STATE.get(slug)
        if rec is None:
            return
        rec["status"] = status
        rec.update(extra)
        _persist_locked()


def queue_counts() -> dict:
    with _state_lock:
        from collections import Counter

        return dict(Counter(r["status"] for r in SHARED_STATE.values()))


# ============================================================================
# GPU polling
# ============================================================================


def _nvidia_smi(args: list[str]) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi"] + args, capture_output=True, text=True, timeout=20, check=True
    ).stdout.strip()
    return [l for l in out.splitlines() if l.strip()]


def _proc_owner(pid: int) -> str | None:
    try:
        out = subprocess.run(
            ["ps", "-o", "user:32=", "-p", str(pid)], capture_output=True, text=True, timeout=10
        )
        u = out.stdout.strip()
        return u or None
    except Exception:
        return None


def gpu_usability() -> dict:
    """idx -> (usable: bool, reason: str)."""
    try:
        mem_lines = _nvidia_smi(["--query-gpu=index,memory.used", "--format=csv,noheader,nounits"])
        uuid_lines = _nvidia_smi(["--query-gpu=index,uuid", "--format=csv,noheader"])
        app_lines = _nvidia_smi(
            ["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"]
        )
    except Exception as e:
        log(f"[warn] nvidia-smi query failed: {e}")
        return {}

    mem_used = {}
    for line in mem_lines:
        idx_s, used_s = [x.strip() for x in line.split(",")]
        mem_used[int(idx_s)] = float(used_s)

    uuid_to_idx = {}
    for line in uuid_lines:
        idx_s, uuid = [x.strip() for x in line.split(",", 1)]
        uuid_to_idx[uuid] = int(idx_s)

    foreign_gpus = set()
    for line in app_lines:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        pid_s, uuid = parts[0], parts[1]
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        idx = uuid_to_idx.get(uuid)
        if idx is None:
            continue
        owner = _proc_owner(pid)
        if owner and owner != ME:
            foreign_gpus.add(idx)
            log(f"[gpu] GPU{idx} has a foreign process (pid={pid}, user={owner}) -> skip")

    # js_park block gate: is ANY foreign user currently on ANY of GPUs 0-3? If so, the
    # whole 0-3 block is off-limits and the sustained-clear timer resets. Only once 0-3
    # has been continuously foreign-free for BLOCK_SUSTAINED_CLEAR_SECONDS do we re-open
    # it -- so a card that momentarily blinks free mid-cycle is still treated as his.
    now = time.time()
    block_has_foreign = any(idx in foreign_gpus for idx in BLOCK_GPUS)
    if block_has_foreign:
        if _block_state["clear_since"] is not None:
            log("[gpu] js_park present on GPUs 0-3 -> block OFF-LIMITS, resetting sustained-clear timer")
        _block_state["clear_since"] = None
    elif _block_state["clear_since"] is None:
        _block_state["clear_since"] = now  # just went clear; start the sustained-clear timer
        log(f"[gpu] GPUs 0-3 now foreign-free; must stay clear {BLOCK_SUSTAINED_CLEAR_SECONDS:.0f}s before re-opening")
    clear_for = (now - _block_state["clear_since"]) if _block_state["clear_since"] is not None else 0.0
    block_allowed = clear_for >= BLOCK_SUSTAINED_CLEAR_SECONDS

    result = {}
    for idx, used in sorted(mem_used.items()):
        if idx in FORBIDDEN_GPUS:
            result[idx] = (False, "prohibited")
        elif idx in BLOCK_GPUS and not block_allowed:
            # js_park's block: unusable unless 0-3 has been sustained-clear long enough.
            if block_has_foreign:
                result[idx] = (False, "js_park-block (active)")
            else:
                result[idx] = (False, f"js_park-block (clearing {clear_for:.0f}/{BLOCK_SUSTAINED_CLEAR_SECONDS:.0f}s)")
        elif idx in foreign_gpus:
            result[idx] = (False, "foreign-user")
        elif used >= MEM_FREE_THRESHOLD_MIB:
            result[idx] = (False, f"busy mem_used={used:.0f}MiB")
        else:
            result[idx] = (True, f"free mem_used={used:.0f}MiB")
    return result


# ============================================================================
# Tinker worker (remote, no local GPU)
# ============================================================================


def free_disk_gb() -> float:
    try:
        st = os.statvfs(DISK_PATH)
        return st.f_bavail * st.f_frsize / 1e9
    except Exception:
        return float("inf")  # fail-open: don't wedge the daemon on a statvfs hiccup


_last_build_sig = {"sig": None}


def build_all_data_once() -> None:
    """Build the square's SFT/DPO CSVs (chatbot_arena ONLY) from available baselines.

    Scoped to SQUARE_DATASETS so we don't rebuild gsm8k/oasst1/writingprompts every poll
    (that was pure disk + log churn -- the square is chatbot_arena-only). Skips entirely
    when disk is low (a build write would just ENOSPC-spam) and de-dupes: only logs/does
    real work when the set of available baselines actually changed since last time."""
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
    """After a cell registers its adapters, delete its regeneratable DPO preference
    JSONL (~250 MB) and empty TRL _trainer dirs, keeping only the LoRA adapters. Prevents
    per-cell footprint creep across 55+ cells."""
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


def run_tinker_cell(slug: str, rec: dict) -> None:
    from dementor.training import matrix

    cell = matrix.Cell(source=rec["source"], target=rec["target"], dataset=DATASET, seed=SEED)
    try:
        if rec.get("need_sft"):
            if not matrix.sft_data_path(cell).exists():
                log(f"[tinker] {slug}: SFT data not built yet -- leaving pending")
                set_status(slug, "pending")
                return
            n = bump_submit_count()
            log(f"[tinker] submitting SFT {slug} (running submit total={n})")
            res = matrix.launch_sft(cells=[cell], dry_run=False, parallel=1)
            jobs = res.get("jobs", [])
            if not jobs:
                log(f"[tinker] {slug}: SFT skipped by launch_sft (missing-data/already-registered) -- leaving pending")
                set_status(slug, "pending")
                return
            if jobs[0].get("error"):
                raise RuntimeError(f"SFT failed: {jobs[0]['error']}")
            log(f"[tinker] SFT done {slug}")
            set_status(slug, "running", need_sft=False)
        if rec.get("need_dpo"):
            if not matrix.dpo_data_path(cell).exists():
                log(f"[tinker] {slug}: DPO data not built yet -- leaving pending")
                set_status(slug, "pending")
                return
            n = bump_submit_count()
            log(f"[tinker] submitting DPO {slug} (running submit total={n})")
            res = matrix.launch_dpo(cells=[cell], dry_run=False, parallel=1)
            jobs = res.get("jobs", [])
            if not jobs:
                log(f"[tinker] {slug}: DPO skipped by launch_dpo (missing-data/missing-sft/already-registered) -- leaving pending")
                set_status(slug, "pending")
                return
            if jobs[0].get("error"):
                raise RuntimeError(f"DPO failed: {jobs[0]['error']}")
            log(f"[tinker] DPO done {slug}")
        set_status(slug, "done")
        log(f"[tinker] CELL COMPLETE {slug}")
    except Exception as e:
        set_status(slug, "failed", error=str(e))
        log(f"[error] tinker cell {slug} failed: {e}\n{traceback.format_exc()}")


def tinker_worker_loop(stop_event: threading.Event) -> None:
    log("[tinker] worker loop starting")
    while not stop_event.is_set():
        try:
            refresh_from_registry()
            pending = get_pending("tinker")
            if not pending:
                log("[tinker] no pending tinker cells (queue empty or all in-flight) -- sleeping")
                time.sleep(HEARTBEAT_INTERVAL)
                continue
            log(f"[tinker] dispatching {len(pending)} pending tinker cells, parallel={TINKER_PARALLEL}")
            for slug, _rec in pending:
                set_status(slug, "running")
            with ThreadPoolExecutor(max_workers=TINKER_PARALLEL) as pool:
                futs = [pool.submit(run_tinker_cell, slug, rec) for slug, rec in pending]
                for f in futs:
                    f.result()
            n_bounced = sum(1 for slug, _ in pending if get_rec(slug) and get_rec(slug)["status"] == "pending")
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
    return [m["id"] for m in core_models() if m["backend"] == "local"]


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


def _lease_try_claim(gpu_idx: int) -> bool:
    """Win the shared GPU lease for a 5/6/7 card before launching (yields to any other lease-holding
    daemon that already owns it). Returns True (may launch) for non-lease cards or when coexistence
    is disabled -- so this is a no-op for the 0-3 path and when SEQ_COEXIST=0."""
    if not COEXIST or gpu_idx not in LEASE_GPUS:
        return True
    try:
        return gpu_lease.try_claim(gpu_idx, holder=LEASE_HOLDER)
    except Exception as e:
        log(f"[lease] try_claim GPU{gpu_idx} error ({e}) -> treating as busy, yielding")
        return False


def _lease_release(gpu_idx: int) -> None:
    """Release our lease on a card iff we hold it (safe no-op otherwise). Called when a job is reaped."""
    if not COEXIST or gpu_idx not in LEASE_GPUS:
        return
    try:
        gpu_lease.release(gpu_idx, holder=LEASE_HOLDER)
    except Exception as e:
        log(f"[lease] release GPU{gpu_idx} error ({e})")


def self_heal_registry(slug: str, rec: dict) -> None:
    """If launch-local-cell exited 0 but the registry write got raced out (two local
    subprocesses -- separate OS processes -- finishing close enough together to race
    the JSON read-modify-write), repair it from the deterministic output directory."""
    from dementor.training import matrix
    from dementor.training.common import record_adapter_mapping

    keys = load_registry_keys()
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
    """Cleanly stop ONE of our own subprocesses by object (SIGTERM, then SIGKILL). Only
    ever called on procs WE spawned (tracked in running_cells/running_baselines) -- never
    on a foreign/js_park pid."""
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
    """Graceful-vacate safety net: if any of OUR tracked jobs is sitting on a GPU 0-3 that
    the current policy marks off-limits (js_park block active/clearing), stop it cleanly and
    RE-QUEUE the cell as pending (not failed) so it retrains later on 5/6/7. Baselines are
    just killed (they regenerate). Never touches js_park's processes. In steady state this is
    a no-op -- the scheduler already refuses to launch onto blocked cards -- but it recovers
    cleanly if a job was placed on 0-3 by an earlier daemon build/policy."""
    for gpu_idx, (proc, slug) in list(running_cells.items()):
        if gpu_idx not in BLOCK_GPUS:
            continue
        ok, reason = usability.get(gpu_idx, (False, "unknown"))
        if ok:
            continue  # 0-3 currently re-opened (sustained-clear) -> leave it
        _terminate_proc(proc)
        del running_cells[gpu_idx]
        mp_secondary.pop(gpu_idx, None)  # release the 2nd card if this was a model-parallel cell
        set_status(slug, "pending", pid=None, gpu=None)
        log(f"[vacate] RE-QUEUED cell {slug}: was on GPU{gpu_idx} (js_park block: {reason}); "
            f"stopped our subprocess and marked pending for retrain on 5/6/7")
    for gpu_idx, (proc, model_id) in list(running_baselines.items()):
        if gpu_idx not in BLOCK_GPUS:
            continue
        ok, _reason = usability.get(gpu_idx, (False, "unknown"))
        if ok:
            continue
        _terminate_proc(proc)
        del running_baselines[gpu_idx]
        log(f"[vacate] stopped baseline-gen {model_id} on GPU{gpu_idx} (js_park block); will regenerate on 5/6/7")


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
                    _lease_release(gpu_idx)  # free the card's lease for the other sustained-idle daemons
                    if rc == 0:
                        log(f"[local] baseline-gen COMPLETE {model_id} on GPU{gpu_idx}")
                    else:
                        log(f"[local] baseline-gen FAILED {model_id} on GPU{gpu_idx} (rc={rc}) -- will retry next cycle")

            for gpu_idx, (proc, slug) in list(running_cells.items()):
                rc = proc.poll()
                if rc is not None:
                    del running_cells[gpu_idx]
                    sec = mp_secondary.pop(gpu_idx, None)  # release 2nd card for a model-parallel cell
                    _lease_release(gpu_idx)  # free the card's lease for the other sustained-idle daemons
                    if sec is not None:
                        _lease_release(sec)
                    rec = get_rec(slug)
                    if rec is None:
                        continue
                    if rc == 0:
                        self_heal_registry(slug, rec)
                        keys = load_registry_keys()
                        still_missing = (rec.get("need_sft") and f"sft_{slug}" not in keys) or (
                            rec.get("need_dpo") and f"dpo_{slug}" not in keys
                        )
                        if still_missing:
                            set_status(slug, "failed", error="subprocess exited 0 but registry entry missing after self-heal")
                            log(f"[error] {slug}: subprocess exited 0 but registry entry missing after self-heal")
                        else:
                            set_status(slug, "done")
                            cleanup_cell_artifacts(slug, rec)
                            log(f"[local] CELL COMPLETE {slug} on GPU{gpu_idx}")
                    elif _cell_hit_enospc(slug):
                        # Transient shared-disk ENOSPC -- NEVER terminal. Re-queue pending with a
                        # backoff so it retries once space frees (the launch guard also throttles).
                        n = rec.get("enospc_retries", 0) + 1
                        set_status(slug, "pending", enospc_retries=n,
                                   retry_after=time.time() + ENOSPC_RETRY_BACKOFF_SEC, error=None)
                        log(f"[disk] local cell {slug} hit ENOSPC on GPU{gpu_idx} "
                            f"(retry #{n}) -> re-queued pending (backoff {ENOSPC_RETRY_BACKOFF_SEC:.0f}s)")
                    else:
                        set_status(slug, "failed", error=f"launch-local-cell exited rc={rc}")
                        log(f"[error] local cell {slug} FAILED on GPU{gpu_idx} (rc={rc})")

            refresh_from_registry()

            # 2. What GPUs are usable right now?
            usability = gpu_usability()

            # 2b. Graceful-vacate: pull any of our jobs off a now-off-limits 0-3 card and
            # re-queue the cell (pending, not failed) so it retrains on 5/6/7.
            vacate_block_gpus(running_cells, running_baselines, usability, mp_secondary)

            busy_idxs = set(running_baselines) | set(running_cells) | set(mp_secondary.values())
            # Sustained-idle gate (coexistence): a card must read usable for SUSTAINED_POLLS consecutive
            # polls before we treat it as free -- this yields transient frees to the steering roster's
            # first-idle scheduler (it takes no lease and grabs immediately). Cards busy with OUR jobs
            # reset to 0; not-usable cards reset to 0. Disabled when SEQ_COEXIST=0.
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

            # 2c. Disk guard: never LAUNCH new local work while free space is below the
            # floor -- throttle (wait) instead of writing into a full disk and ENOSPC-failing.
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
                if not _lease_try_claim(gpu_idx):
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
                pending_local = get_pending("local")
                from dementor.training import matrix

                now_ts = time.time()
                consumed: set = set()  # GPUs claimed this pass (a model-parallel cell claims 2)
                for gpu_idx in list(free_gpus):
                    if gpu_idx in consumed:
                        continue
                    for slug, rec in pending_local:
                        cur = get_rec(slug)
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
                            if not _lease_try_claim(gpu_idx):
                                consumed.add(gpu_idx)  # another daemon owns this card -> can't use it
                                break
                            if not _lease_try_claim(sec):
                                _lease_release(gpu_idx)
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
                            if not _lease_try_claim(gpu_idx):
                                consumed.add(gpu_idx)  # another daemon owns this card -> yield it
                                break
                            proc = launch_cell_subprocess(gpu_idx, slug, rec)
                            running_cells[gpu_idx] = (proc, slug)
                            consumed.add(gpu_idx)
                            idle_counts[gpu_idx] = 0
                        set_status(slug, "running", pid=proc.pid, gpu=gpu_idx)
                        break

            # 5. Heartbeat.
            now = time.time()
            if now - last_heartbeat > HEARTBEAT_INTERVAL:
                last_heartbeat = now
                counts = queue_counts()
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


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    log("=" * 80)
    log(f"sequencer starting, pid={os.getpid()}")
    backup_registry_once()

    init_state()
    n_tinker = sum(1 for r in SHARED_STATE.values() if r["backend"] == "tinker")
    n_local = sum(1 for r in SHARED_STATE.values() if r["backend"] == "local")
    log(f"[init] missing cells: {len(SHARED_STATE)} total ({n_tinker} tinker-source, {n_local} local-source)")

    build_all_data_once()

    stop_event = threading.Event()
    t_tinker = threading.Thread(target=tinker_worker_loop, args=(stop_event,), daemon=False, name="tinker-worker")
    t_local = threading.Thread(target=local_worker_loop, args=(stop_event,), daemon=False, name="local-worker")
    t_tinker.start()
    t_local.start()

    try:
        while True:
            time.sleep(60)
            counts = queue_counts()
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
