#!/usr/bin/env python
"""Resumable, GPU-aware driver for the IMITATION safety-erosion sweep.

Models the steering roster_queue / retry-poller pattern: it is a POLITE, low-priority scheduler
that coexists with the steering roster on the shared box.

  * Candidate cards come from ``DEMENTOR_GPUS``.
  * A card is claimed as soon as it is idle (mem.used < --mem-max AND util <= --util-max). This box
    is dedicated to the imitation-safety sweep, so the default gate is one 5s poll rather than the
    older shared-roster 3x30s sustained-idle delay.
  * BASELINES run first (each disguise adapter's erosion delta needs its base-model baseline).
  * Llama-3.3-70B uses three 80 GB cards; other configured MP models use two.
  * By default an ``ERROR.json`` is left for explicit inspection. ``--retry-errors`` retries each
    such item at most once in the current supervisor process and reuses its stage checkpoints.

Usage:
  erosion_daemon.py --dry-run                 # print worklist + GPU snapshot, launch nothing
  erosion_daemon.py [--max-prompts N] [--benchmarks a,b] [--once] [--include-baselines-only]
                    [--util-max U] [--mem-max M] [--sustained-polls P] [--interval S]
"""
import os, sys, time, argparse, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erosion_common as EC
import gpu_lease  # shared atomic GPU-lease lock (arbitrates among the sustained-idle daemons)
import daemon_common as DC  # shared gpu_stat / dlog factory / HF-offline env builder

GPUS = EC.GPUS   # env DEMENTOR_GPUS (default 5,6,7; partner 4xH100 box: DEMENTOR_GPUS=0,1,2,3)
LEASE_HOLDER = "erosion_daemon"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_erosion_item.py")
# Cached models run fully offline (avoids the HF Xet download hang); pre-set HF_HUB_OFFLINE=0 to
# let an uncached base (e.g. Ministral-8B) fetch on first use.
BASE_ENV = DC.hf_offline_env(EC.REPO, EC.HF_HOME, EC.HF_HUB_CACHE)

# Per-base generation batch override.  The global --gen-batch (32) is calibrated for the big
# bases: phi-4 (14B) already sits at ~64GB / 80GB at batch 32, and the 31B+ bases are memory-
# bound, so they must stay at 32.  But the <=8B bases leave ~40GB of the card idle at batch 32,
# so they double to 64 (still inside the proven ~64GB envelope, since max_new_tokens=256 caps the
# KV peak).  Bases NOT listed here fall back to the global --gen-batch => zero behaviour change.
# run_erosion_item halves the batch on OOM, so an over-estimate self-heals rather than crashing.
GEN_BATCH_BY_BASE = {
    "adamo1139/aya-expanse-8b-ungated": 64,     # 8B
    "google/gemma-4-E4B-it": 64,                # ~4B effective (matformer)
    "meta-llama/Llama-3.1-8B-Instruct": 64,     # 8B
    "mistralai/Ministral-8B-Instruct-2410": 64, # 8B
    "allenai/OLMo-3-7B-Instruct": 64,           # 7B
}


dlog = DC.make_dlog(os.path.join(HERE, "logs", "daemon.log"))


def generated(item_id, benchmarks, require_fidelity=False):
    d = os.path.join(EC.WORK, item_id)
    safety_done = all(os.path.exists(os.path.join(d, b, "all_gens.csv")) for b in benchmarks)
    if not safety_done or not require_fidelity or item_id.startswith("baseline_"):
        return safety_done
    import fidelity_common as FC
    return FC.gens_done(item_id)


def done(item_id, retry_errors=False, generation_only=False, benchmarks=EC.DEFAULT_BENCHMARKS,
         require_fidelity=False):
    d = os.path.join(EC.WORK, item_id)
    if generation_only:
        return generated(item_id, benchmarks, require_fidelity=require_fidelity)
    return os.path.exists(os.path.join(d, "metrics.json")) or (
        not retry_errors and os.path.exists(os.path.join(d, "ERROR.json"))
    )


gpu_stat = DC.gpu_stat


def external_running_ids(worklist):
    """Detect erosion workers already running outside this supervisor.

    This lets us restart the daemon to change scheduling knobs without duplicating in-flight items
    whose parent was an older daemon process.
    """
    try:
        out = subprocess.run(["ps", "-eo", "cmd"], stdout=subprocess.PIPE, text=True).stdout
    except Exception:
        return set()
    return {w["id"] for w in worklist if f"run_erosion_item.py {w['id']}" in out}


def external_reserved_gpus():
    """Cards declared by any live erosion worker, including workers from older supervisors.

    Generation workers unload their base before judging. Memory-only idle detection otherwise sees
    that transition as a free card and launches a colliding job. The worker's CUDA declaration is
    the reservation for its entire process lifetime.
    """
    reserved = set()
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,cmd="], stdout=subprocess.PIPE, text=True, check=True
        ).stdout
    except Exception:
        return reserved
    for line in out.splitlines():
        if "run_erosion_item.py" not in line:
            continue
        try:
            pid = int(line.strip().split(None, 1)[0])
            payload = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
        except (OSError, ValueError, IndexError):
            continue
        for entry in payload:
            if not entry.startswith(b"CUDA_VISIBLE_DEVICES="):
                continue
            for value in entry.split(b"=", 1)[1].decode().split(","):
                try:
                    reserved.add(int(value))
                except ValueError:
                    pass
    return reserved


def required_gpus(item):
    """Physical 80 GB card count proven for the local generation backend."""
    if item.get("base_model") == "meta-llama/Llama-3.3-70B-Instruct":
        return 3
    return 2 if item.get("needs_mp") else 1


def launch(item, gpus, extra):
    env = dict(BASE_ENV, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in gpus))
    env.setdefault("RTL_JUDGE_BATCH_SIZE", "32")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if len(gpus) > 1:
        env["DEMENTOR_MP"] = "1"
    lg = open(os.path.join(HERE, "logs", f"item_{item['id']}.log"), "a")
    cmd = [EC.PY, RUNNER, item["id"]] + extra
    # Per-base gen-batch override (argparse takes the LAST --gen-batch, so appending wins over the
    # global one already in `extra`).  Only the small bases in GEN_BATCH_BY_BASE are bumped.
    gb = GEN_BATCH_BY_BASE.get(item.get("base_model"))
    if gb is not None:
        cmd += ["--gen-batch", str(gb)]
    dlog(f"LAUNCH {item['id']} ({item['kind']}) on GPU{gpus} mp={len(gpus) > 1}"
         + (f" gen_batch={gb}" if gb is not None else ""))
    return subprocess.Popen(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT)


def select_items(adapters, baselines, requested_csv):
    """Restrict a daemon run to explicit, locally resolvable item ids."""
    if not requested_csv:
        return adapters, baselines
    requested = {item.strip() for item in requested_csv.split(",") if item.strip()}
    available = {item["id"] for item in adapters + baselines}
    unknown = sorted(requested - available)
    if unknown:
        raise SystemExit("unknown or non-local item ids: " + ", ".join(unknown))
    return (
        [item for item in adapters if item["id"] in requested],
        [item for item in baselines if item["id"] in requested],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-prompts-per-benchmark", "--max-prompts", dest="max_prompts",
                    type=int, default=EC.DEFAULT_MAX_PROMPTS,
                    help="deterministic stratified subsample size per benchmark (default 200; <=0 = full)")
    ap.add_argument("--subsample-seed", type=int, default=EC.DEFAULT_SUBSAMPLE_SEED)
    ap.add_argument("--benchmarks", default=",".join(EC.DEFAULT_BENCHMARKS))
    ap.add_argument("--seed", default="seed42")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--util-max", type=int, default=5)
    ap.add_argument("--mem-max", type=int, default=5000)
    ap.add_argument("--sustained-polls", type=int, default=1)
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--gen-batch", type=int, default=int(os.environ.get("DEMENTOR_EROSION_GEN_BATCH", "32")),
                    help="generation batch size passed to run_erosion_item.py")
    ap.add_argument("--retry-errors", action="store_true",
                    help="retry pre-existing ERROR checkpoints once during this supervisor run")
    ap.add_argument("--generate-only", action="store_true",
                    help="checkpoint missing generations; use batched local judging afterward")
    ap.add_argument("--also-fidelity", action="store_true",
                    help="generate held-out fidelity responses in each adapter's existing model load")
    ap.add_argument("--items", default=None,
                    help="comma-separated item ids to consider (default: the full local worklist)")
    args = ap.parse_args()

    adapters, baselines = EC.build_worklist(seed=args.seed, local_only=True)
    adapters, baselines = select_items(adapters, baselines, args.items)
    # baselines FIRST (erosion deltas depend on them), then adapters
    worklist = baselines + adapters
    extra = ["--benchmarks", args.benchmarks, "--max-prompts", str(args.max_prompts),
             "--gen-batch", str(args.gen_batch),
             "--subsample-seed", str(args.subsample_seed)]
    if args.generate_only:
        extra.append("--generate-only")
    if args.also_fidelity:
        extra.append("--also-fidelity")
    selected_benchmarks = [b for b in args.benchmarks.split(",") if b]

    def item_done(item_id):
        return done(
            item_id,
            retry_errors=args.retry_errors,
            generation_only=args.generate_only,
            benchmarks=selected_benchmarks,
            require_fidelity=args.also_fidelity,
        )

    if args.dry_run:
        print(f"worklist: {len(baselines)} baselines + {len(adapters)} adapters = {len(worklist)} items")
        nmp = [w["id"] for w in worklist if required_gpus(w) > 1]
        print(f"model-parallel items ({len(nmp)}): {nmp}")
        todo = [w["id"] for w in worklist if not item_done(w["id"])]
        print(f"remaining (not done): {len(todo)}")
        for w in worklist[:12]:
            print(f"  {'DONE' if item_done(w['id']) else 'todo':4s} {w['kind']:8s} {w['id']}")
        if len(worklist) > 12:
            print(f"  ... (+{len(worklist)-12} more)")
        print("GPU snapshot (5/6/7):")
        for g in GPUS:
            u, m = gpu_stat(g)
            print(f"  GPU{g}: util={u}% mem_used={m}MB")
        print("\n[dry-run] nothing launched.")
        return

    reclaimed = gpu_lease.reap()  # clear any stale leases from a prior crashed run
    dlog(f"daemon start items={len(worklist)} ({len(baselines)} baselines + {len(adapters)} adapters) "
         f"max_prompts={args.max_prompts} util_max={args.util_max}% mem_max={args.mem_max}MB "
         f"sustained_polls={args.sustained_polls} interval={args.interval}s gen_batch={args.gen_batch} "
         f"lease_root={gpu_lease.LOCK_ROOT} reaped_stale={reclaimed}")
    idle = {g: 0 for g in GPUS}
    running = {}   # gpu -> (Popen, id)
    mp_job = None  # (Popen, id, [g1,g2])
    failed_this_run = set()
    while True:
        for g, (p, iid) in list(running.items()):
            if p.poll() is not None:
                dlog(f"done {iid} on GPU{g} rc={p.returncode}")
                if p.returncode:
                    failed_this_run.add(iid)
                del running[g]
                gpu_lease.release(g, holder=LEASE_HOLDER)  # free the card's lease for other daemons
        if mp_job and mp_job[0].poll() is not None:
            dlog(f"done MP {mp_job[1]} rc={mp_job[0].returncode}")
            if mp_job[0].returncode:
                failed_this_run.add(mp_job[1])
            for g in mp_job[2]:
                gpu_lease.release(g, holder=LEASE_HOLDER)
            mp_job = None

        external = external_running_ids(worklist)
        inflight = {iid for _, iid in running.values()} | ({mp_job[1]} if mp_job else set()) | external
        todo = [w for w in worklist
                if not item_done(w["id"])
                and w["id"] not in inflight
                and w["id"] not in failed_this_run]
        if not todo and not running and not mp_job and not external:
            dlog("ALL DONE"); break

        busy = (set(running) | (set(mp_job[2]) if mp_job else set())
                | external_reserved_gpus())
        for g in GPUS:
            if g in busy:
                idle[g] = 0; continue
            u, m = gpu_stat(g)
            idle[g] = idle[g] + 1 if (u <= args.util_max and m < args.mem_max) else 0
        avail = [g for g in GPUS if g not in busy and idle[g] >= args.sustained_polls]

        # MP items first. Must WIN every required lease before launching.
        mp_todo = [w for w in todo if required_gpus(w) > 1]
        if mp_job is None and mp_todo and len(avail) >= required_gpus(mp_todo[0]):
            g2 = avail[:required_gpus(mp_todo[0])]
            claimed = []
            for g in g2:
                if gpu_lease.try_claim(g, holder=LEASE_HOLDER):
                    claimed.append(g)
                else:
                    break  # another sustained-idle daemon holds this card -> yield
            if len(claimed) == len(g2):
                mp_job = (launch(mp_todo[0], g2, extra), mp_todo[0]["id"], g2)
                for g in g2:
                    idle[g] = 0
                # Remove every claimed card. Llama-70B claims three; slicing by two could launch a
                # single-card worker onto its third card in the same scheduler pass.
                avail = [g for g in avail if g not in claimed]
                inflight.add(mp_todo[0]["id"])
            else:
                for g in claimed:  # couldn't get both -> release what we grabbed, retry next poll
                    gpu_lease.release(g, holder=LEASE_HOLDER)
        # single-card items -- claim the lease before launching; skip a card another daemon won.
        for g in avail:
            single = [w for w in todo if required_gpus(w) == 1 and w["id"] not in inflight]
            if not single:
                break
            if not gpu_lease.try_claim(g, holder=LEASE_HOLDER):
                continue  # lost the lease race to another sustained-idle daemon -> yield this card
            w = single[0]
            running[g] = (launch(w, [g], extra), w["id"])
            inflight.add(w["id"])
            idle[g] = 0

        if args.once:
            dlog(f"[--once] running={ {g: i for g,(_,i) in running.items()} } mp={mp_job[1] if mp_job else None}")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
