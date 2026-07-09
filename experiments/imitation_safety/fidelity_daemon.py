#!/usr/bin/env python
"""Resumable, LOW-PRIORITY orchestrator for the IMITATION FIDELITY sweep.

Explicitly SUBORDINATE to the running safety-erosion sweep (erosion_daemon.py + tinker_erosion.py),
which is the primary result.  Two independent concerns run concurrently:

  REMOTE + CPU (no GPU, never contends with erosion) -- the bulk (572 tinker adapters + 28 tinker
    refs):
      * one long-lived `gen-tinker` subprocess samples every tinker disguise adapter + tinker target
        reference on Tinker's servers.  It uses FEWER sample-workers than the erosion sampler
        (--sample-workers, default 16 vs erosion's 64) so it never starves erosion's Tinker throughput.
      * a periodic `score --scorer embed --device cpu` subprocess scores every newly-ready adapter on
        CPU (all-MiniLM-L6-v2), then rebuilds the fidelity CSV.  Zero GPU.

  LOCAL GPU (92 local PEFT adapters + 11 local refs, all chatbot_arena) -- lease-gated + STRICTER idle
    gate than erosion so erosion wins every freed card:
      * a card is taken only when GENUINELY IDLE for --sustained-polls consecutive polls (default 8,
        vs erosion's 3) AND its shared gpu_lease (holder='fidelity') is won.  We hold the lease for
        just --local-batch items then release, so erosion can preempt between items.
      * big local targets (gemma-4-31b, granite) need 2 cards (DEMENTOR_MP) -- handled by gen-local.

done(item) = its gens.csv (generation) / fidelity_<scorer>.json (scoring) exists -> fully resumable.
The daemon holds NO GPU itself; it only supervises subprocesses.

Usage:
  fidelity_daemon.py [--scorer embed] [--sample-workers 16] [--local-batch 2]
                     [--util-max 5 --mem-max 5000 --sustained-polls 8 --interval 45]
                     [--no-local]        # remote+CPU only (safest: literally never touches a card)
                     [--once] [--dry-run]
Launch detached (setsid) -- see the __main__ hint / the launch command in the task report.
"""
import os, sys, time, json, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity_common as FC
import gpu_lease

GPUS = FC.EC.GPUS   # env DEMENTOR_GPUS (default 5,6,7; partner 4xH100 box: DEMENTOR_GPUS=0,1,2,3)
LEASE_HOLDER = "fidelity"
PY = FC.PY
EVAL = os.path.join(HERE, "fidelity_eval.py")
LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
DLOG = os.path.join(LOG_DIR, "fidelity_daemon.log")

BASE_ENV = dict(os.environ, HF_HOME=FC.EC.HF_HOME,
                HF_HUB_CACHE=FC.EC.HF_HUB_CACHE, HF_HUB_DISABLE_XET="1",
                PYTHONPATH=FC.REPO)
BASE_ENV.setdefault("HF_HUB_OFFLINE", "1")


def dlog(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        open(DLOG, "a").write(line + "\n")
    except Exception:
        pass


def gpu_stat(g):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits", "-i", str(g)],
                             stdout=subprocess.PIPE, text=True).stdout.strip().splitlines()[0]
        return [int(x.strip()) for x in out.split(",")]
    except Exception:
        return 100, 999999


def _popen(argv, gpus=None):
    env = dict(BASE_ENV)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    lg = open(os.path.join(LOG_DIR, "fidelity_sub.log"), "a")
    return subprocess.Popen([PY, EVAL] + argv, env=env, stdout=lg, stderr=subprocess.STDOUT)


def counts(worklist, scorer):
    refs = FC.reference_items(worklist)
    tk = [it for it in worklist if it["backend"] == "tinker"]
    loc = [it for it in worklist if it["backend"] == "local"]
    tk_ref = [r for r in refs if r["backend"] == "tinker"]
    loc_ref = [r for r in refs if r["backend"] == "local"]
    remote_gen_todo = [it for it in tk if not FC.gens_done(it["id"])] + \
                      [r for r in tk_ref if not FC.ref_done(r["dataset"], r["target"])]
    local_gen_todo = [it for it in loc if not FC.gens_done(it["id"])] + \
                     [r for r in loc_ref if not FC.ref_done(r["dataset"], r["target"])]
    score_todo = [it for it in worklist
                  if FC.gens_done(it["id"]) and FC.ref_done(it["dataset"], it["target"])
                  and not FC.scored(it["id"], scorer)]
    return remote_gen_todo, local_gen_todo, score_todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="embed", choices=["embed", "judge"])
    ap.add_argument("--sample-workers", type=int, default=16,
                    help="tinker sample workers (LOWER than erosion's 64 to stay subordinate)")
    ap.add_argument("--max-prompts", type=int, default=FC.HELDOUT_N)
    ap.add_argument("--local-batch", type=int, default=2,
                    help="local items to process per lease-hold before releasing the card")
    ap.add_argument("--util-max", type=int, default=5)
    ap.add_argument("--mem-max", type=int, default=5000)
    ap.add_argument("--sustained-polls", type=int, default=8,
                    help="consecutive idle polls before taking a card (STRICTER than erosion's 3)")
    ap.add_argument("--interval", type=int, default=45)
    ap.add_argument("--score-every", type=int, default=600, help="seconds between CPU score passes")
    ap.add_argument("--no-local", action="store_true", help="remote+CPU only; never touch a GPU")
    ap.add_argument("--seed", default="all")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    worklist = FC.build_fidelity_worklist(seed=args.seed)
    rem, loc, sc = counts(worklist, args.scorer)
    dlog(f"fidelity daemon start: {len(worklist)} adapters  remote_gen_todo={len(rem)} "
         f"local_gen_todo={len(loc)} score_todo={len(sc)} scorer={args.scorer} "
         f"no_local={args.no_local} sustained={args.sustained_polls} workers={args.sample_workers}")
    if args.dry_run:
        for g in GPUS:
            u, m = gpu_stat(g)
            print(f"  GPU{g}: util={u}% mem={m}MB")
        print("[dry-run] nothing launched.")
        return

    reclaimed = gpu_lease.reap()
    dlog(f"reaped stale leases: {reclaimed}")
    gen_tinker_p = None      # long-lived remote sampler
    score_p = None           # periodic CPU scorer
    local_p = None           # lease-gated local-gen (holds `local_gpus`)
    local_gpus = []
    last_score = 0
    idle = {g: 0 for g in GPUS}

    while True:
        rem, loc, sc = counts(worklist, args.scorer)

        # ---- remote sampler (no GPU): keep one alive while remote gens remain -------------------
        if rem and (gen_tinker_p is None or gen_tinker_p.poll() is not None):
            gen_tinker_p = _popen(["gen-tinker", "--sample-workers", str(args.sample_workers),
                                   "--max-prompts", str(args.max_prompts)])
            dlog(f"[remote] (re)launched gen-tinker pid={gen_tinker_p.pid} remaining={len(rem)}")

        # ---- CPU scorer: periodic, only when idle scored-todo exists and no scorer running -------
        if score_p is not None and score_p.poll() is not None:
            score_p = None
            _popen(["build-csv", "--scorer", args.scorer, "--seed", args.seed]).wait()  # refresh CSV
        if score_p is None and sc and (time.time() - last_score) > args.score_every:
            score_p = _popen(["score", "--scorer", args.scorer, "--device", "cpu"])
            last_score = time.time()
            dlog(f"[score] launched embed scorer pid={score_p.pid} todo={len(sc)}")

        # ---- local GPU gen: lease-gated, stricter idle gate than erosion --------------------------
        if not args.no_local:
            if local_p is not None and local_p.poll() is not None:
                dlog(f"[local] gen-local rc={local_p.returncode}; releasing {local_gpus}")
                for g in local_gpus:
                    gpu_lease.release(g, holder=LEASE_HOLDER)
                local_p, local_gpus = None, []
                score_p = None
                last_score = 0  # force a score pass soon after new local gens land
            if local_p is None and loc:
                need_mp = any(it.get("needs_mp") or it.get("target_needs_mp") for it in loc[:1])
                busy = set(local_gpus)
                for g in GPUS:
                    if g in busy:
                        idle[g] = 0
                        continue
                    u, m = gpu_stat(g)
                    idle[g] = idle[g] + 1 if (u <= args.util_max and m < args.mem_max) else 0
                avail = [g for g in GPUS if idle[g] >= args.sustained_polls]
                want = 2 if need_mp else 1
                if len(avail) >= want:
                    claimed = []
                    for g in avail[:want]:
                        if gpu_lease.try_claim(g, holder=LEASE_HOLDER):
                            claimed.append(g)
                        else:
                            break
                    if len(claimed) == want:
                        local_gpus = claimed
                        local_p = _popen(["gen-local", "--limit", str(args.local_batch),
                                          "--max-prompts", str(args.max_prompts)], gpus=claimed)
                        for g in claimed:
                            idle[g] = 0
                        dlog(f"[local] LAUNCH gen-local pid={local_p.pid} on GPU{claimed} "
                             f"(lease held) remaining_local={len(loc)}")
                    else:
                        for g in claimed:
                            gpu_lease.release(g, holder=LEASE_HOLDER)

        # ---- termination -------------------------------------------------------------------------
        remote_running = gen_tinker_p is not None and gen_tinker_p.poll() is None
        local_running = local_p is not None and local_p.poll() is None
        score_running = score_p is not None and score_p.poll() is None
        all_done = (not rem and not sc and (args.no_local or not loc)
                    and not remote_running and not local_running and not score_running)
        if all_done:
            _popen(["build-csv", "--scorer", args.scorer, "--seed", args.seed]).wait()
            dlog("ALL DONE (remote+local gens + scoring complete; final CSV built)")
            return
        if args.once:
            dlog(f"[--once] remote_running={remote_running} local_running={local_running} "
                 f"score_running={score_running}")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
