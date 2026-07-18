#!/usr/bin/env python
"""Resumable, GPU-aware driver for the IMITATION safety-erosion sweep (GPUs 5/6/7 only).

Models the steering roster_queue / retry-poller pattern: it is a POLITE, low-priority scheduler
that coexists with the steering roster on the shared box.

  * GPUs 5/6/7 only (GPU4 prohibited; 0-3 belong to others).
  * A card is claimed as soon as it is idle (mem.used < --mem-max AND util <= --util-max). This box
    is dedicated to the imitation-safety sweep, so the default gate is one 5s poll rather than the
    older shared-roster 3x30s sustained-idle delay.
  * BASELINES run first (each disguise adapter's erosion delta needs its base-model baseline).
  * Big base models (granite-4 32B, gpt-oss-120b) run MODEL-PARALLEL on 2 idle cards (DEMENTOR_MP=1);
    everything else is single-card.
  * done(item) = work/<id>/metrics.json OR work/<id>/ERROR.json (an errored item is not retried
    forever).  Fully resumable: re-running the daemon picks up where it left off.

Usage:
  erosion_daemon.py --dry-run                 # print worklist + GPU snapshot, launch nothing
  erosion_daemon.py [--max-prompts N] [--benchmarks a,b] [--once] [--include-baselines-only]
                    [--util-max U] [--mem-max M] [--sustained-polls P] [--interval S]
"""
import os, sys, time, argparse, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erosion_common as EC
import gpu_lease  # shared atomic GPU-lease lock (arbitrates among the sustained-idle daemons)

GPUS = EC.GPUS   # env DEMENTOR_GPUS (default 5,6,7; partner 4xH100 box: DEMENTOR_GPUS=0,1,2,3)
LEASE_HOLDER = "erosion_daemon"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_erosion_item.py")
BASE_ENV = dict(os.environ, HF_HOME=EC.HF_HOME,
                HF_HUB_CACHE=EC.HF_HUB_CACHE, HF_HUB_DISABLE_XET="1",
                PYTHONPATH=EC.REPO)
# Cached models run fully offline (avoids the HF Xet download hang); pre-set HF_HUB_OFFLINE=0 to
# let an uncached base (e.g. Ministral-8B) fetch on first use.
BASE_ENV.setdefault("HF_HUB_OFFLINE", "1")

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


def dlog(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(os.path.join(HERE, "logs", "daemon.log"), "a").write(line + "\n")


def done(item_id):
    d = os.path.join(EC.WORK, item_id)
    return os.path.exists(os.path.join(d, "metrics.json")) or os.path.exists(os.path.join(d, "ERROR.json"))


def gpu_stat(g):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits", "-i", str(g)],
                             stdout=subprocess.PIPE, text=True).stdout.strip().splitlines()[0]
        util, mem = [int(x.strip()) for x in out.split(",")]
        return util, mem
    except Exception:
        return 100, 999999


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


def launch(item, gpus, extra):
    env = dict(BASE_ENV, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in gpus))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-prompts-per-benchmark", "--max-prompts", dest="max_prompts",
                    type=int, default=EC.DEFAULT_MAX_PROMPTS,
                    help="deterministic stratified subsample size per benchmark (default 300; <=0 = full)")
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
    args = ap.parse_args()

    adapters, baselines = EC.build_worklist(seed=args.seed, local_only=True)
    # baselines FIRST (erosion deltas depend on them), then adapters
    worklist = baselines + adapters
    extra = ["--benchmarks", args.benchmarks, "--max-prompts", str(args.max_prompts),
             "--gen-batch", str(args.gen_batch),
             "--subsample-seed", str(args.subsample_seed)]

    if args.dry_run:
        print(f"worklist: {len(baselines)} baselines + {len(adapters)} adapters = {len(worklist)} items")
        nmp = [w["id"] for w in worklist if w["needs_mp"]]
        print(f"model-parallel items ({len(nmp)}): {nmp}")
        todo = [w["id"] for w in worklist if not done(w["id"])]
        print(f"remaining (not done): {len(todo)}")
        for w in worklist[:12]:
            print(f"  {'DONE' if done(w['id']) else 'todo':4s} {w['kind']:8s} {w['id']}")
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
    while True:
        for g, (p, iid) in list(running.items()):
            if p.poll() is not None:
                dlog(f"done {iid} on GPU{g} rc={p.returncode}")
                del running[g]
                gpu_lease.release(g, holder=LEASE_HOLDER)  # free the card's lease for other daemons
        if mp_job and mp_job[0].poll() is not None:
            dlog(f"done MP {mp_job[1]} rc={mp_job[0].returncode}")
            for g in mp_job[2]:
                gpu_lease.release(g, holder=LEASE_HOLDER)
            mp_job = None

        external = external_running_ids(worklist)
        inflight = {iid for _, iid in running.values()} | ({mp_job[1]} if mp_job else set()) | external
        todo = [w for w in worklist if not done(w["id"]) and w["id"] not in inflight]
        if not todo and not running and not mp_job and not external:
            dlog("ALL DONE"); break

        busy = set(running) | (set(mp_job[2]) if mp_job else set())
        for g in GPUS:
            if g in busy:
                idle[g] = 0; continue
            u, m = gpu_stat(g)
            idle[g] = idle[g] + 1 if (u <= args.util_max and m < args.mem_max) else 0
        avail = [g for g in GPUS if g not in busy and idle[g] >= args.sustained_polls]

        # MP items first (need 2 sustained-idle cards). Must WIN the lease on BOTH before launching.
        mp_todo = [w for w in todo if w["needs_mp"]]
        if mp_job is None and len(avail) >= 2 and mp_todo:
            g2 = avail[:2]
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
                avail = avail[2:]
                inflight.add(mp_todo[0]["id"])
            else:
                for g in claimed:  # couldn't get both -> release what we grabbed, retry next poll
                    gpu_lease.release(g, holder=LEASE_HOLDER)
        # single-card items -- claim the lease before launching; skip a card another daemon won.
        for g in avail:
            single = [w for w in todo if not w["needs_mp"] and w["id"] not in inflight]
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
