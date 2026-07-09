#!/usr/bin/env python
"""Resumable, GPU-polite driver for the LOCAL track of the PROMPT-RUNG safety-erosion sweep
(GPUs 5/6/7 only).  Same sustained-idle + shared-GPU-lease pattern as erosion_daemon.py, so it
coexists with the steering roster AND the running SFT/DPO erosion/fidelity daemons at EQUAL-OR-LOWER
priority (it only takes a card that is genuinely idle AND whose lease it wins).

  * GPUs 5/6/7 only (GPU4 prohibited; 0-3 belong to others).
  * A card is claimed only when GENUINELY IDLE (mem<--mem-max AND util<=--util-max) for
    --sustained-polls consecutive polls, then only if gpu_lease.try_claim wins it.
  * NO baselines here -- the prompt rung REUSES the SFT/DPO rung's baselines (EC.WORK/baseline_<slug>).
    Worklist = local-source (method x pair) items only.
  * done(item) = work_prompt_erosion/<id>/metrics.json OR ERROR.json.  Fully resumable.

Usage:
  prompt_erosion_daemon.py --dry-run
  prompt_erosion_daemon.py [--methods m1,m2] [--max-prompts N] [--benchmarks a,b] [--once]
                           [--util-max U] [--mem-max M] [--sustained-polls P] [--interval S]
"""
import os, sys, time, argparse, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erosion_common as EC
import prompt_erosion_common as PC
import gpu_lease

GPUS = EC.GPUS   # env DEMENTOR_GPUS (default 5,6,7; partner 4xH100 box: DEMENTOR_GPUS=0,1,2,3)
LEASE_HOLDER = "prompt_erosion"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_prompt_erosion_item.py")
BASE_ENV = dict(os.environ, HF_HOME=EC.HF_HOME,
                HF_HUB_CACHE=EC.HF_HUB_CACHE, HF_HUB_DISABLE_XET="1",
                PYTHONPATH=EC.REPO)
BASE_ENV.setdefault("HF_HUB_OFFLINE", "1")


def dlog(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(os.path.join(HERE, "logs", "prompt_daemon.log"), "a").write(line + "\n")


def gpu_stat(g):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits", "-i", str(g)],
                             stdout=subprocess.PIPE, text=True).stdout.strip().splitlines()[0]
        util, mem = [int(x.strip()) for x in out.split(",")]
        return util, mem
    except Exception:
        return 100, 999999


def launch(item, gpus, extra):
    env = dict(BASE_ENV, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in gpus))
    if len(gpus) > 1:
        env["DEMENTOR_MP"] = "1"
    lg = open(os.path.join(HERE, "logs", f"pitem_{item['id']}.log"), "a")
    cmd = [EC.PY, RUNNER, item["id"]] + extra
    dlog(f"LAUNCH {item['id']} (method={item['method']}) on GPU{gpus} mp={len(gpus) > 1}")
    return subprocess.Popen(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(PC.METHODS),
                    help="comma methods (default all 5): just_name_it,random_sampling,stylistic,behavioral,contrastive")
    ap.add_argument("--max-prompts-per-benchmark", "--max-prompts", dest="max_prompts",
                    type=int, default=PC.DEFAULT_MAX_PROMPTS)
    ap.add_argument("--subsample-seed", type=int, default=PC.DEFAULT_SUBSAMPLE_SEED)
    ap.add_argument("--benchmarks", default=",".join(EC.DEFAULT_BENCHMARKS))
    ap.add_argument("--max-new-tokens", type=int, default=PC.DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--util-max", type=int, default=5)
    ap.add_argument("--mem-max", type=int, default=5000)
    ap.add_argument("--sustained-polls", type=int, default=3)
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    worklist = PC.build_worklist(methods=methods, backend="local")
    extra = ["--benchmarks", args.benchmarks, "--max-prompts", str(args.max_prompts),
             "--subsample-seed", str(args.subsample_seed), "--max-new-tokens", str(args.max_new_tokens)]

    if args.dry_run:
        from collections import Counter
        print(f"LOCAL prompt-erosion worklist: {len(worklist)} (method x pair) items "
              f"({len(set((w['source'],w['target']) for w in worklist))} pairs x {len(methods)} methods)")
        print("by method:", dict(Counter(w["method"] for w in worklist)))
        print("by source:", dict(Counter(w["source"] for w in worklist)))
        todo = [w["id"] for w in worklist if not PC.done(w["id"])]
        print(f"remaining (not done): {len(todo)} / {len(worklist)}")
        # baseline availability (reused from the SFT/DPO rung)
        miss = sorted({w["source"] for w in worklist if PC.baseline_metrics(w["source"]) is None})
        print(f"baselines MISSING for sources (erosion NaN until computed by erosion_daemon): {miss}")
        for w in worklist[:10]:
            print(f"  {'DONE' if PC.done(w['id']) else 'todo':4s} {w['id']}")
        print("GPU snapshot (5/6/7):")
        for g in GPUS:
            u, m = gpu_stat(g)
            print(f"  GPU{g}: util={u}% mem_used={m}MB")
        print("\n[dry-run] nothing launched.")
        return

    reclaimed = gpu_lease.reap()
    dlog(f"prompt-erosion daemon start items={len(worklist)} methods={methods} "
         f"max_prompts={args.max_prompts} util_max={args.util_max}% mem_max={args.mem_max}MB "
         f"lease_root={gpu_lease.LOCK_ROOT} reaped_stale={reclaimed}")
    idle = {g: 0 for g in GPUS}
    running = {}   # gpu -> (Popen, id)
    mp_job = None  # (Popen, id, [g1,g2])
    while True:
        for g, (p, iid) in list(running.items()):
            if p.poll() is not None:
                dlog(f"done {iid} on GPU{g} rc={p.returncode}")
                del running[g]
                gpu_lease.release(g, holder=LEASE_HOLDER)
        if mp_job and mp_job[0].poll() is not None:
            dlog(f"done MP {mp_job[1]} rc={mp_job[0].returncode}")
            for g in mp_job[2]:
                gpu_lease.release(g, holder=LEASE_HOLDER)
            mp_job = None

        inflight = {iid for _, iid in running.values()} | ({mp_job[1]} if mp_job else set())
        todo = [w for w in worklist if not PC.done(w["id"]) and w["id"] not in inflight]
        if not todo and not running and not mp_job:
            dlog("ALL DONE"); break

        busy = set(running) | (set(mp_job[2]) if mp_job else set())
        for g in GPUS:
            if g in busy:
                idle[g] = 0; continue
            u, m = gpu_stat(g)
            idle[g] = idle[g] + 1 if (u <= args.util_max and m < args.mem_max) else 0
        avail = [g for g in GPUS if g not in busy and idle[g] >= args.sustained_polls]

        mp_todo = [w for w in todo if w["needs_mp"]]
        if mp_job is None and len(avail) >= 2 and mp_todo:
            g2 = avail[:2]
            claimed = []
            for g in g2:
                if gpu_lease.try_claim(g, holder=LEASE_HOLDER):
                    claimed.append(g)
                else:
                    break
            if len(claimed) == len(g2):
                mp_job = (launch(mp_todo[0], g2, extra), mp_todo[0]["id"], g2)
                for g in g2:
                    idle[g] = 0
                avail = avail[2:]
                inflight.add(mp_todo[0]["id"])
            else:
                for g in claimed:
                    gpu_lease.release(g, holder=LEASE_HOLDER)
        for g in avail:
            single = [w for w in todo if not w["needs_mp"] and w["id"] not in inflight]
            if not single:
                break
            if not gpu_lease.try_claim(g, holder=LEASE_HOLDER):
                continue
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
