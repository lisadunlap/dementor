#!/usr/bin/env python
"""Resumable, lease-coordinated poller for the STEERING MULTI-BENCHMARK dissociation eval.

WHY THIS EXISTS: the RDO roster (roster_queue.py -> run_rdo_model.py) only produces the AdvBench
cone verdict (run_rdo_model stage 6 -> <slug>/eval/metrics.json). The 9-benchmark dissociation eval
(run_benchmark_eval.py -> <slug>/eval_<bench>/metrics.json + benchmarks_summary.json) is NOT wired to
auto-run. This daemon fills that gap: it watches repl80_rdo/<slug>/ for models the roster has
CONE-SCORED (selected_cone.pt + vectors_ml.pt present) and, on a genuinely-idle 5/6/7 card it wins the
shared GPU lease for, launches `run_benchmark_eval.py <slug> --max-prompts 300` for each.

It is a SIBLING sustained-idle daemon that mirrors erosion_daemon.py + retry_pc_fails.py:
  * GPUs 5/6/7 only (GPU4 prohibited; 0-3 belong to js_park).
  * A card is claimed only when GENUINELY IDLE (util <= --util-max AND mem.used < --mem-max) for
    --sustained-polls CONSECUTIVE polls -- so a card the roster momentarily frees between its own
    jobs is left for the roster (roster_queue uses FIRST-idle and takes no lease, so it always wins
    transient frees). This daemon only takes a card the roster has truly released.
  * Before launching it must WIN the shared atomic mkdir lease (gpu_lease.py) on the card, so it
    can't double-book a freed card against the erosion daemon / retry poller / granite launcher.
    Releases the lease when the job for that card is reaped.
  * BIG models run MODEL-PARALLEL on 2 idle cards (DEMENTOR_MP=1 + CUDA_VISIBLE_DEVICES=a,b), which
    cone_eval.py consumes via rdo_port.load_model_mp_aware (device_map="auto"). Must WIN the lease on
    BOTH cards before launching; if it can't get both, it releases whatever it grabbed and retries.
    MP set = worklist needs_mp==true UNION the explicitly-oversized slugs (mixtral 47B bf16 does NOT
    fit one 80GB card; the rest are belt-and-suspenders). Everything else is single-card.

READY (a model is eligible): repl80_rdo/<slug>/ has selected_cone.pt AND vectors_ml.pt (exactly the
two prereqs run_benchmark_eval.py checks) AND <slug> is a real worklist slug (so <slug>_retry5/ and
non-model dirs are skipped -- run_benchmark_eval looks the slug up in rdo_worklist.json and would
sys.exit on an unknown one).
DONE (a model is finished, skipped): repl80_rdo/<slug>/benchmarks_summary.json exists. run_benchmark_eval
itself is benchmark-granular resumable (skips any eval_<bench>/metrics.json already present), so a run
interrupted before it wrote the summary is simply re-launched and resumes where it stopped.

Runs FOREVER as a poller (like retry_pc_fails.py) -- new models keep becoming ready as the roster
scores them, so it re-scans every poll; --once does a single scheduling pass, --dry-run just prints
the ready/pending/done split + GPU + lease snapshot and launches nothing.

Usage:
    benchmark_eval_daemon.py --dry-run        # scan + print status, launch nothing
    benchmark_eval_daemon.py                  # run as a daemon poller
    benchmark_eval_daemon.py --once           # one scheduling pass then exit (for testing)
Tunables: --max-prompts --util-max --mem-max --sustained-polls --interval
"""
import os, sys, time, json, argparse, subprocess

# Shared atomic GPU-lease lock lives with the imitation-erosion daemon; import it by absolute path so
# THIS poller, the erosion daemon, the retry poller and the granite launcher all arbitrate over the
# SAME lock namespace (prevents two sustained-idle daemons launching on the same freed card).
HERE = os.path.dirname(os.path.abspath(__file__))   # steering package dir (run_benchmark_eval lives here)
sys.path.insert(0, HERE)
import steer_config as CFG
gpu_lease = CFG.import_gpu_lease()   # SHARED lease from the imitation package (reused, not duplicated)

ROOT = CFG.WORK_ROOT   # worklist + per-model <slug>/ outputs + logs (env DEMENTOR_STEER_WORK)
PY = CFG.PY
RUNNER = os.path.join(HERE, "run_benchmark_eval.py")
WORKLIST = CFG.worklist_path()   # WORK_ROOT copy if present, else the committed repo copy
GPUS = CFG.GPUS
LEASE_HOLDER = "benchmark_eval_daemon"

# BIG models -> model-parallel on 2 cards. Source of truth is the worklist needs_mp flag; we add the
# explicitly-oversized slugs the task calls out (mixtral 47B bf16 ~94GB genuinely needs 2 cards; the
# 31-35B ones are conservative -- MP just wastes a card, never OOMs). Everything else runs single-card.
MP_EXTRA = {"gpt-oss-120b", "mixtral-8x7b", "gemma-4-31b", "granite-4-h-small", "olmo-3.1-32b"}


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(os.path.join(ROOT, "benchmark_eval_daemon.log"), "a").write(line + "\n")


def load_specs():
    wl = json.load(open(WORKLIST))
    return {m["slug"]: m for m in wl["models"]}


def is_mp(slug, spec):
    return bool(spec.get("needs_mp")) or slug in MP_EXTRA


def cone_scored(slug):
    """A model is CONE-SCORED / ready iff it has BOTH prereqs run_benchmark_eval.py requires."""
    d = os.path.join(ROOT, slug)
    return (os.path.exists(os.path.join(d, "selected_cone.pt"))
            and os.path.exists(os.path.join(d, "vectors_ml.pt")))


def bench_done(slug):
    """Finished when the summary marker exists (written after run_benchmark_eval's benchmark loop)."""
    return os.path.exists(os.path.join(ROOT, slug, "benchmarks_summary.json"))


def ready_models(specs):
    """Worklist slugs that are cone-scored but not yet benchmark-eval'd, in worklist order."""
    return [s for s in specs if cone_scored(s) and not bench_done(s)]


def gpu_stat(g):
    """(utilization.gpu %, memory.used MB); conservative fallback (busy) on error."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits", "-i", str(g)],
                             stdout=subprocess.PIPE, text=True).stdout.strip().splitlines()[0]
        util, mem = [int(x.strip()) for x in out.split(",")]
        return util, mem
    except Exception:
        return 100, 999999


def launch(slug, gpus, max_prompts):
    """Launch run_benchmark_eval.py for `slug` on `gpus` (1 card, or 2 with DEMENTOR_MP=1). The child
    passes os.environ straight through to cone_eval.py, so CUDA_VISIBLE_DEVICES + DEMENTOR_MP set here
    reach cone_eval's rdo_port.load_model_mp_aware."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in gpus),
               HF_HUB_DISABLE_XET="1")  # guard against the HF Xet download hang
    if len(gpus) > 1:
        env["DEMENTOR_MP"] = "1"
    else:
        env.pop("DEMENTOR_MP", None)
    lg = open(os.path.join(ROOT, f"benchmark_eval_{slug}.log"), "a")
    cmd = [PY, RUNNER, slug, "--max-prompts", str(max_prompts)]
    log(f"LAUNCH benchmark_eval {slug} on GPU{gpus} mp={len(gpus) > 1} max_prompts={max_prompts}")
    return subprocess.Popen(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT)


def scan_report(specs):
    ready = ready_models(specs)
    scored = [s for s in specs if cone_scored(s)]
    done = [s for s in scored if bench_done(s)]
    print(f"worklist slugs: {len(specs)} | cone-scored: {len(scored)} | "
          f"benchmark-eval done: {len(done)} | READY now: {len(ready)}")
    print("READY (cone-scored, benchmark-eval pending):")
    for s in ready:
        print(f"  - {s:22s} {'[MP/2-card]' if is_mp(s, specs[s]) else '[single-card]'}")
    if done:
        print(f"already benchmark-eval'd: {done}")
    pend = [s for s in specs if not cone_scored(s) and not bench_done(s)]
    print(f"pending cone-scoring by roster (not ready yet): {len(pend)}")
    print("GPU idleness snapshot (5/6/7):")
    for g in GPUS:
        u, m = gpu_stat(g)
        print(f"  GPU{g}: util={u}% mem_used={m}MB")
    print("GPU leases:")
    st = gpu_lease.status()
    for s in st:
        print(f"  GPU{s['gpu']}: holder={s['holder']} pid={s['pid']} "
              f"{'STALE' if s['stale'] else ('alive' if s['alive'] else 'dead?')}")
    if not st:
        print("  (no active leases)")
    return ready


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="scan + print status, launch nothing")
    ap.add_argument("--once", action="store_true", help="one scheduling pass then exit")
    ap.add_argument("--max-prompts", type=int, default=300, help="prompts/benchmark (default 300)")
    ap.add_argument("--util-max", type=int, default=5, help="max utilization.gpu %% to count a card idle")
    ap.add_argument("--mem-max", type=int, default=5000, help="max memory.used MB to count a card idle")
    ap.add_argument("--sustained-polls", type=int, default=3,
                    help="consecutive idle polls required before claiming a card (roster wins transient frees)")
    ap.add_argument("--interval", type=int, default=30, help="poll interval seconds")
    args = ap.parse_args()

    specs = load_specs()

    if args.dry_run:
        scan_report(specs)
        print("\n[dry-run] no GPU jobs launched.")
        return

    reclaimed = gpu_lease.reap()  # clear stale leases from a prior crashed run before we start
    log(f"benchmark_eval daemon start max_prompts={args.max_prompts} util_max={args.util_max}% "
        f"mem_max={args.mem_max}MB sustained_polls={args.sustained_polls} interval={args.interval}s "
        f"lease_root={gpu_lease.LOCK_ROOT} reaped_stale={reclaimed}")
    idle = {g: 0 for g in GPUS}
    running = {}   # gpu -> (Popen, slug)   single-card jobs
    mp_job = None  # (Popen, slug, [g1, g2])  one 2-card job at a time
    while True:
        # reap finished single-card jobs -> release their leases
        for g, (p, slug) in list(running.items()):
            if p.poll() is not None:
                log(f"done {slug} on GPU{g} rc={p.returncode}")
                del running[g]
                gpu_lease.release(g, holder=LEASE_HOLDER)
        # reap the finished MP job -> release BOTH leases
        if mp_job and mp_job[0].poll() is not None:
            log(f"done MP {mp_job[1]} rc={mp_job[0].returncode}")
            for g in mp_job[2]:
                gpu_lease.release(g, holder=LEASE_HOLDER)
            mp_job = None

        inflight = {s for _, s in running.values()} | ({mp_job[1]} if mp_job else set())
        todo = [s for s in ready_models(specs) if s not in inflight]

        # update per-card sustained-idle counters (a card we're using never counts as idle)
        busy = set(running) | (set(mp_job[2]) if mp_job else set())
        for g in GPUS:
            if g in busy:
                idle[g] = 0; continue
            u, m = gpu_stat(g)
            idle[g] = idle[g] + 1 if (u <= args.util_max and m < args.mem_max) else 0
        avail = [g for g in GPUS if g not in busy and idle[g] >= args.sustained_polls]

        # MP items first (need 2 sustained-idle cards). WIN the lease on BOTH before launching.
        mp_todo = [s for s in todo if is_mp(s, specs[s])]
        if mp_job is None and len(avail) >= 2 and mp_todo:
            g2 = avail[:2]
            claimed = []
            for g in g2:
                if gpu_lease.try_claim(g, holder=LEASE_HOLDER):
                    claimed.append(g)
                else:
                    break  # another sustained-idle daemon holds this card -> yield
            if len(claimed) == len(g2):
                slug = mp_todo[0]
                mp_job = (launch(slug, g2, args.max_prompts), slug, g2)
                for g in g2:
                    idle[g] = 0
                avail = avail[2:]
                inflight.add(slug)
            else:
                for g in claimed:  # couldn't get both -> release what we grabbed, retry next poll
                    gpu_lease.release(g, holder=LEASE_HOLDER)

        # single-card items -- claim the lease before launching; skip a card another daemon won.
        for g in avail:
            single = [s for s in todo if not is_mp(s, specs[s]) and s not in inflight]
            if not single:
                break
            if not gpu_lease.try_claim(g, holder=LEASE_HOLDER):
                continue  # lost the lease race to another sustained-idle daemon -> yield this card
            slug = single[0]
            running[g] = (launch(slug, [g], args.max_prompts), slug)
            inflight.add(slug)
            idle[g] = 0

        if args.once:
            log(f"[--once] running={ {g: s for g,(_,s) in running.items()} } "
                f"mp={mp_job[1] if mp_job else None} todo={todo}")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
