#!/usr/bin/env python3
"""GPU visibility, usability policy, and shared-lease arbitration.

`gpu_usability()` is the single "which cards may we launch on right now" oracle. A card is usable
iff: it isn't compute-prohibited (FORBIDDEN_GPUS), it isn't inside js_park's active 0-3 block (unless
that block has been sustained-clear long enough), no foreign user holds a compute process on it, and
its memory is below the free threshold.

`lease_claim` / `lease_release` wrap the shared atomic GPU lease (gpu_lease, imported from the sibling
imitation_safety package) so this training daemon shares ONE lease namespace with the erosion daemons
and never launches a ~65GB cell on top of another daemon's job.
"""
from __future__ import annotations

import subprocess
import sys
import time

from runtime import (
    BLOCK_GPUS,
    BLOCK_SUSTAINED_CLEAR_SECONDS,
    COEXIST,
    FORBIDDEN_GPUS,
    LEASE_GPUS,
    LEASE_HOLDER,
    ME,
    MEM_FREE_THRESHOLD_MIB,
    REPO,
    log,
)

# Imported from the sibling in-repo imitation_safety package so training + erosion daemons share ONE
# lease namespace (gpu_lease.LOCK_ROOT is env-derived from the same DEMENTOR_DATA on a given box).
sys.path.insert(0, str(REPO / "experiments" / "imitation_safety"))
import gpu_lease  # noqa: E402  (shared atomic GPU-lease lock)

_block_state = {"clear_since": None}   # monotonic-ish ts when 0-3 last became (and stayed) clear


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

    # js_park block gate: is ANY foreign user currently on ANY of GPUs 0-3? If so, the whole 0-3
    # block is off-limits and the sustained-clear timer resets. Only once 0-3 has been continuously
    # foreign-free for BLOCK_SUSTAINED_CLEAR_SECONDS do we re-open it.
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


def lease_claim(gpu_idx: int) -> bool:
    """Win the shared GPU lease for a lease-pool card before launching (yields to any other
    lease-holding daemon that already owns it). Returns True (may launch) for non-lease cards or when
    coexistence is disabled -- so this is a no-op for the 0-3 path and when SEQ_COEXIST=0."""
    if not COEXIST or gpu_idx not in LEASE_GPUS:
        return True
    try:
        return gpu_lease.try_claim(gpu_idx, holder=LEASE_HOLDER)
    except Exception as e:
        log(f"[lease] try_claim GPU{gpu_idx} error ({e}) -> treating as busy, yielding")
        return False


def lease_release(gpu_idx: int) -> None:
    """Release our lease on a card iff we hold it (safe no-op otherwise). Called when a job is reaped."""
    if not COEXIST or gpu_idx not in LEASE_GPUS:
        return
    try:
        gpu_lease.release(gpu_idx, holder=LEASE_HOLDER)
    except Exception as e:
        log(f"[lease] release GPU{gpu_idx} error ({e})")
