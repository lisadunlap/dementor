#!/usr/bin/env python3
"""Registry-driven worklist + shared run state for the imitation square.

The worklist is a pure function of config.yaml + the adapter registry: every still-missing
{sft,dpo}_chatbot_arena_<source>_as_<target>_seed<SEED> cell for the 12 core models.

Run state lives in a SINGLE shared in-memory dict (SHARED_STATE), guarded by one lock, persisted to
queue_state.json on every mutation. This is the only source of truth during a run: the tinker worker
(thread) and the local-GPU worker (thread) both mutate cell status concurrently, so one shared,
locked dict avoids the lost-update race where a stale snapshot flips a "running"/"done" cell back to
"pending" (which would trigger a duplicate, paid Tinker resubmission).

The pid-liveness "was this actually still running" reconciliation runs ONCE, at startup (init_state) --
never on the periodic mid-run refreshes (refresh_from_registry), which only add newly-missing cells
and retire newly-completed ones by consulting the registry (the real source of truth for "done").
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from threading import RLock

from runtime import (
    DATASET,
    LOCAL_ONLY,
    QUEUE_PATH,
    REGISTRY_BACKUP_MARKER,
    REGISTRY_PATH,
    SEED,
    STATE_DIR,
    log,
)

_state_lock = RLock()
SHARED_STATE: dict = {}  # slug -> record; guarded by _state_lock; single source of truth


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


# ============================================================================
# Shared state
# ============================================================================


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
    (possibly killed) run, reconciles it against the registry (ground truth for "done") and
    pid-liveness (ground truth for "actually still running"), and seeds SHARED_STATE. This is the
    ONLY place a "running" status is ever reset to "pending" based on staleness.
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
    """Mid-run refresh: consult the registry + config for (a) cells that are now fully done
    (mark done) and (b) cells whose data dependencies just became available (need_sft/need_dpo
    flags). NEVER resets an in-flight "running" status -- only the registry (via self-heal / job
    completion) or the owning worker thread does that.
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
        return dict(Counter(r["status"] for r in SHARED_STATE.values()))


def backend_counts() -> dict:
    """{backend -> n} over all tracked cells (for the startup summary)."""
    with _state_lock:
        return dict(Counter(r["backend"] for r in SHARED_STATE.values()))
