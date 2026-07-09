#!/usr/bin/env python
"""Shared atomic GPU-lease lock for the dementor sustained-idle daemons (GPUs 5/6/7).

WHY: several *sustained-idle* daemons coexist on the shared H100 box -- the local imitation
erosion daemon (erosion_daemon.py), the PC_FAILS retry poller (retry_pc_fails.py), and later the
granite imitation training launcher. Each claims a card only after it reads idle for N consecutive
polls. Without arbitration, two of them can BOTH watch the same freed card go idle for 3 polls and
BOTH launch a job on it -> OOM / race. This lock makes "who gets the freed card" a single atomic
winner.

DESIGN:
  * The claim primitive is `os.mkdir(LOCK_ROOT/gpu<N>)`. POSIX mkdir is atomic: with two concurrent
    callers, exactly ONE succeeds and every other gets FileExistsError. That is the whole race-freedom
    guarantee -- no flock, no pidfile-truncation window.
  * The winner writes meta.json (pid + holder label + unix ts) INSIDE the lease dir. Release removes
    the lease dir (only if we still own it, checked by pid).
  * STALE reclaim: a lease is stale if its holder pid is dead (os.kill(pid,0) raises) OR it is older
    than max_age (backstop for a hung/zombie holder). Reclaim is serialized by a per-card reclaim
    mutex (also an atomic mkdir on "gpu<N>.reclaim"): exactly one reclaimer enters, RE-verifies
    staleness, rmtree's the dead lease, then races a fresh mkdir claim. Because removal happens only
    inside the mutex and the actual claim is still the atomic mkdir, the reclaim path can't double-book
    or let a straggler move a just-recreated live lease aside.

COEXISTENCE WITH THE ROSTER: roster_queue.py uses FIRST-idle (no sustained gate) and does NOT take a
lease, so it always wins transient frees between its own jobs. This lease ONLY arbitrates AMONG the
sustained-idle daemons -- it never blocks the roster. A daemon must still (a) confirm the card is
genuinely idle (its own util+free-mem thresholds) AND (b) win the lease here, before launching.

USAGE (in a sustained-idle daemon):
    import gpu_lease
    ...
    if idle_count[g] >= sustained_polls and gpu_lease.try_claim(g, holder="erosion_daemon"):
        running[g] = (launch(...), item)      # launch ONLY after winning the lease
    ...
    # when the job for gpu g is reaped:
    gpu_lease.release(g, holder="erosion_daemon")

The GRANITE IMITATION TRAINING launcher (a future sustained-idle daemon) MUST use the SAME pattern:
    sys.path.insert(0, os.path.dirname(__file__)); import gpu_lease  # this package dir
    if idle_count[g] >= sustained_polls and gpu_lease.try_claim(g, holder="granite_imitation"):
        launch training on GPU g            # only after confirming idle AND winning the lease
    ...
    gpu_lease.release(g, holder="granite_imitation")   # when that training job exits
For a 2-card model-parallel granite job, try_claim BOTH cards and release BOTH (see erosion_daemon.py
MP path); if you can't win both, release whatever you grabbed and retry next poll.

CLI (inspection / manual ops):
    python gpu_lease.py list                  # show current leases (and stale flag)
    python gpu_lease.py claim <gpu> [holder]  # try to claim (for testing)
    python gpu_lease.py release <gpu>         # release a lease held by THIS pid
    python gpu_lease.py reap                  # drop any stale leases now
"""
import os, sys, json, time, shutil, errno

# Fixed absolute root so EVERY daemon (regardless of its own cwd) shares the same lock namespace.
# Env-derivable to stay path-portable; the default MIRRORS erosion_common.WORK_ROOT so all daemons
# on a box share one lease namespace as long as they inherit the same DEMENTOR_* env. gpu_lease is
# imported standalone (it must NOT import erosion_common), so the default is recomputed here.
def _default_lease_root():
    root = os.environ.get("DEMENTOR_IMITATION_ROOT")
    if not root:
        data = os.environ.get("DEMENTOR_DATA")
        if not data:
            repo = os.environ.get("DEMENTOR_REPO") or os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            data = os.path.join(repo, "data")
        root = os.path.join(data, "imitation_safety")
    return os.path.join(root, "gpu_leases")

LOCK_ROOT = os.environ.get("DEMENTOR_GPU_LEASE_ROOT") or _default_lease_root()

# Backstop for a hung/zombie holder whose pid is somehow still alive. The holder pid is a long-lived
# DAEMON, so a lease stays valid for the whole (possibly multi-hour) job it guards; pid-death is the
# primary/immediate stale signal, max_age is only the safety net. Keep it well above any single job.
DEFAULT_MAX_AGE = 6 * 3600  # 6 hours

# Grace window (s) for a just-created lease dir whose meta.json isn't written yet. The claim is the
# atomic os.mkdir; the winner writes meta microseconds later. Without a grace window a CONCURRENT
# reclaimer would see the dir with no (or a half-written) meta, mistake it for abandoned/corrupt, and
# reclaim it out from under the winner -> double-book. Any dir younger than this is treated as a fresh
# claim-in-progress, never stale. Meta writes are atomic (temp + os.replace) so a reader never sees a
# partial file; this window only covers the mkdir->write gap.
CLAIM_GRACE = 10

# The per-card reclaim mutex is held only for the microseconds of a rmtree+mkdir. If a reclaimer
# crashes mid-hold it would wedge reclaim forever, so a mutex older than this is force-cleared.
RECLAIM_MUTEX_TTL = 60


def _lease_path(gpu):
    return os.path.join(LOCK_ROOT, f"gpu{gpu}")


def _meta_path(gpu):
    return os.path.join(_lease_path(gpu), "meta.json")


def _pid_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except (ValueError, TypeError):
        return False
    return True


def _read_meta(gpu):
    try:
        with open(_meta_path(gpu)) as f:
            return json.load(f)
    except FileNotFoundError:
        return None  # dir may exist but meta not yet written (claim in progress) -> see _is_stale
    except Exception:
        return {}  # corrupt/partial meta -> treated with the grace window in _is_stale


def _dir_age(gpu):
    try:
        return time.time() - os.stat(_lease_path(gpu)).st_mtime
    except OSError:
        return None


def _is_stale(gpu, max_age):
    """A held lease is stale if its holder pid is dead OR it is older than max_age. Returns False if
    the lease dir doesn't exist. A dir whose meta is missing/corrupt is treated as a fresh
    claim-in-progress (NOT stale) while younger than CLAIM_GRACE -- this closes the mkdir->write race
    so a concurrent reclaimer can't steal a lease the instant after another process won the mkdir."""
    if not os.path.isdir(_lease_path(gpu)):
        return False
    meta = _read_meta(gpu)
    if not meta:  # meta.json missing or unreadable
        age = _dir_age(gpu)
        if age is None:
            return False  # dir vanished under us
        return age > CLAIM_GRACE  # abandoned mid-claim (winner crashed) -> reclaimable after grace
    if not _pid_alive(meta.get("pid")):
        return True
    if (time.time() - float(meta.get("ts", 0))) > max_age:
        return True
    return False


def _write_meta(gpu, holder):
    """Write meta atomically (temp + os.replace) so a concurrent reader never sees a partial file."""
    meta = {"pid": os.getpid(), "holder": holder, "ts": time.time(),
            "ts_h": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": gpu}
    tmp = _meta_path(gpu) + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, _meta_path(gpu))
    return meta


def _claim_once(gpu, holder):
    """Atomic single attempt: win iff os.mkdir succeeds. Returns True on win, False if already held."""
    os.makedirs(LOCK_ROOT, exist_ok=True)
    try:
        os.mkdir(_lease_path(gpu))
    except FileExistsError:
        return False
    except OSError as e:
        if e.errno == errno.EEXIST:
            return False
        raise
    _write_meta(gpu, holder)
    return True


def _reclaim(gpu, max_age, claim_holder=None):
    """Serialized reclaim of a stale lease, guarded by the per-card reclaim mutex.

    Only one process enters the mutex; it RE-verifies staleness (so it never removes a lease that
    became fresh), rmtree's the dead lease, then -- if claim_holder is given -- races a fresh atomic
    mkdir claim. Returns True iff (claim requested and won) or (no claim requested and a stale lease
    was dropped). Any other contender is either blocked on the mutex or arbitrated by the final mkdir,
    so no double-book is possible.
    """
    rmutex = _lease_path(gpu) + ".reclaim"
    os.makedirs(LOCK_ROOT, exist_ok=True)
    try:
        os.mkdir(rmutex)
    except FileExistsError:
        # Force-clear a mutex left by a crashed reclaimer (held far longer than its microsecond life).
        try:
            if time.time() - os.stat(rmutex).st_mtime > RECLAIM_MUTEX_TTL:
                os.rmdir(rmutex)
                os.mkdir(rmutex)
            else:
                return False  # another reclaimer is mid-reclaim; retry on the next poll
        except OSError:
            return False
    except OSError:
        return False
    try:
        if not _is_stale(gpu, max_age):
            return False  # became fresh (already reclaimed by someone) -> don't touch it
        shutil.rmtree(_lease_path(gpu), ignore_errors=True)
        if claim_holder is None:
            return True
        return _claim_once(gpu, claim_holder)
    finally:
        try:
            os.rmdir(rmutex)
        except OSError:
            pass


def try_claim(gpu, holder="daemon", max_age=DEFAULT_MAX_AGE):
    """Try to atomically acquire the lease for `gpu`. Returns True iff THIS process now holds it.

    Never blocks. If the lease is free -> win it via atomic mkdir. If held by a live, fresh holder ->
    False. If held by a stale holder (dead pid / older than max_age / abandoned mid-claim) -> reclaim
    it under the reclaim mutex and race for it.
    """
    if _claim_once(gpu, holder):
        return True
    if not _is_stale(gpu, max_age):
        return False
    return _reclaim(gpu, max_age, claim_holder=holder)


def owns(gpu):
    """True iff the lease for `gpu` exists and is held by THIS process (pid match)."""
    meta = _read_meta(gpu)
    return bool(meta) and meta.get("pid") == os.getpid()


def release(gpu, holder=None):
    """Release the lease for `gpu` iff THIS process holds it (pid match). Safe no-op otherwise, so a
    lease that was already reclaimed + re-taken by another daemon is never clobbered."""
    if owns(gpu):
        shutil.rmtree(_lease_path(gpu), ignore_errors=True)
        return True
    return False


def status(max_age=DEFAULT_MAX_AGE):
    """List of dicts describing every current lease dir (for inspection)."""
    out = []
    if not os.path.isdir(LOCK_ROOT):
        return out
    for name in sorted(os.listdir(LOCK_ROOT)):
        p = os.path.join(LOCK_ROOT, name)
        if not os.path.isdir(p) or not name.startswith("gpu") or ".stale." in name:
            continue
        try:
            gpu = int(name[3:])
        except ValueError:
            continue
        meta = _read_meta(gpu) or {}
        out.append({"gpu": gpu, "pid": meta.get("pid"), "holder": meta.get("holder"),
                    "ts_h": meta.get("ts_h"), "alive": _pid_alive(meta.get("pid")),
                    "stale": _is_stale(gpu, max_age)})
    return out


def reap(max_age=DEFAULT_MAX_AGE):
    """Drop every stale lease now (without claiming). Returns the list of reclaimed gpu ids."""
    reclaimed = []
    for st in status(max_age):
        if st["stale"] and _reclaim(st["gpu"], max_age, claim_holder=None):
            reclaimed.append(st["gpu"])
    return reclaimed


def _cli():
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        for st in status():
            flag = "STALE" if st["stale"] else ("alive" if st["alive"] else "dead?")
            print(f"GPU{st['gpu']}: holder={st['holder']} pid={st['pid']} [{flag}] since {st['ts_h']}")
        if not status():
            print("(no active leases)")
        return
    cmd = sys.argv[1]
    if cmd == "claim":
        gpu = int(sys.argv[2]); holder = sys.argv[3] if len(sys.argv) > 3 else "cli"
        print("WON" if try_claim(gpu, holder=holder) else "BUSY")
    elif cmd == "release":
        gpu = int(sys.argv[2])
        print("released" if release(gpu) else "not owned by this pid")
    elif cmd == "reap":
        print("reclaimed:", reap() or "(none)")
    else:
        print(__doc__)


if __name__ == "__main__":
    _cli()
