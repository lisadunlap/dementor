#!/usr/bin/env python
"""Low-priority PC_FAILS retry poller.

Re-runs any model whose FIRST-PASS RDO verdict is PC_FAILS with a STRONGER refusal cone so the
positive control can be established. Diagnosis (qwen2.5-14b): at cone dim<=4 / beta<=1.4 the
refusal cone was under-powered -- the model rerouted to alternate refusal phrasing
("Certainly not!") instead of producing harm. Fix = stronger cone (dim->5, extended betas).

Each retry launches run_rdo_model.py with:
    RDO_MIN_DIM=2  RDO_MAX_DIM=5  RDO_BETAS=0.6,1.0,1.4,2.0,3.0  RDO_OUT_SUFFIX=_retry5
so output lands in <slug>_retry5/ and every stage (cone/select/eval) re-runs fresh WITHOUT
colliding with or skipping the first-pass artifacts in <slug>/.

Coexistence with the first-pass roster (roster_queue.py + roster_worker.sh) -- MUST NOT DISTURB:
  * GPUs 5/6/7 only.
  * Strictly low priority. A card is claimed only when it is GENUINELY IDLE
    (utilization.gpu <= UTIL_MAX AND memory.used < MEM_MAX) AND its static roster worker has
    exited AND it has read idle for SUSTAINED_POLLS *consecutive* polls. The sustained-idle gate
    is the key: the first-pass roster (roster_queue) claims a freed card on its very first idle
    poll, so it always wins the transient free window between first-pass jobs -- the retry poller
    only takes a card that has stayed idle, i.e. one the first pass no longer wants.
  * At most MAX_CONCURRENT retries in flight (default 1).
  * This is a *sibling* process -- it never reads or mutates roster_queue's live state.
  * GPU-lease lock (gpu_lease.py, shared with the imitation-erosion daemon): before launching on a
    sustained-idle card the poller must WIN an atomic mkdir lease on it, and release it when the retry
    exits. This arbitrates ONLY among the sustained-idle daemons (erosion daemon, this poller, the
    granite imitation launcher) so two of them can't both claim the same freed card. roster_queue
    still uses first-idle and does NOT take the lease, so it always wins transient frees.

Resumable: a retry is "done" when <slug>_retry5/eval/metrics.json OR <slug>_retry5/ERROR.json
exists. New PC_FAILS produced by the still-running first pass are picked up on later polls.

Usage:
    python retry_pc_fails.py --dry-run     # scan + print PC_FAILS / retry status, launch nothing
    python retry_pc_fails.py               # run as a daemon poller (only fires on idle cards)
    python retry_pc_fails.py --once        # one scheduling pass then exit (for testing)
Tunables: --util-max --mem-max --sustained-polls --interval --max-concurrent
"""
import os, sys, time, json, glob, argparse, subprocess

# Shared atomic GPU-lease lock lives with the imitation-erosion daemon; import it by absolute path so
# this poller and the erosion daemon arbitrate over the SAME lock namespace (prevents two
# sustained-idle daemons from both launching on the same freed card).
HERE = os.path.dirname(os.path.abspath(__file__))   # steering package dir (run_rdo_model lives here)
sys.path.insert(0, HERE)
import steer_config as CFG
gpu_lease = CFG.import_gpu_lease()   # SHARED lease from the imitation package (reused, not duplicated)

ROOT = CFG.WORK_ROOT   # metrics scan + retry <slug>_retry5/ outputs + logs (env DEMENTOR_STEER_WORK)
PY = CFG.PY
GPUS = CFG.GPUS
LEASE_HOLDER = "retry_pc_fails"

# Stronger-cone retry knobs -> consumed by run_rdo_model.py via env-var overrides.
RETRY_MIN_DIM = "2"
RETRY_MAX_DIM = "5"
RETRY_BETAS = "0.6,1.0,1.4,2.0,3.0"
SUFFIX = "_retry5"

BASE_ENV = dict(os.environ, **CFG.hf_env(offline=True))


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(os.path.join(ROOT, "retry_pc_fails.log"), "a").write(line + "\n")


def pc_fails():
    """Base-slug dirs whose first-pass eval/metrics.json has verdict==PC_FAILS (retry dirs skipped
    so a retry that is itself PC_FAILS never loops forever)."""
    out = []
    for mj in glob.glob(os.path.join(ROOT, "*", "eval", "metrics.json")):
        slug = os.path.basename(os.path.dirname(os.path.dirname(mj)))
        if slug.endswith(SUFFIX):
            continue
        try:
            if json.load(open(mj)).get("verdict") == "PC_FAILS":
                out.append(slug)
        except Exception:
            continue
    return sorted(set(out))


def retry_done(slug):
    d = os.path.join(ROOT, slug + SUFFIX)
    return os.path.exists(os.path.join(d, "eval", "metrics.json")) or os.path.exists(os.path.join(d, "ERROR.json"))


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


def static_worker_alive(g):
    return subprocess.run(["pgrep", "-f", f"roster_worker.sh {g}"], stdout=subprocess.DEVNULL).returncode == 0


def launch(slug, gpu):
    d = os.path.join(ROOT, slug + SUFFIX); os.makedirs(d, exist_ok=True)
    env = dict(BASE_ENV, CUDA_VISIBLE_DEVICES=str(gpu),
               RDO_MIN_DIM=RETRY_MIN_DIM, RDO_MAX_DIM=RETRY_MAX_DIM,
               RDO_BETAS=RETRY_BETAS, RDO_OUT_SUFFIX=SUFFIX)
    lg = open(os.path.join(ROOT, f"retry_{slug}{SUFFIX}.log"), "a")
    log(f"LAUNCH retry {slug} -> {slug}{SUFFIX} on GPU{gpu} (dim {RETRY_MIN_DIM}..{RETRY_MAX_DIM} betas={RETRY_BETAS})")
    # base slug as argv (run_rdo_model looks spec up by slug); suffix/dim/betas come from env.
    return subprocess.Popen([PY, os.path.join(HERE, "run_rdo_model.py"), slug], env=env, stdout=lg,
                            stderr=subprocess.STDOUT)


def scan_report():
    fails = pc_fails()
    print(f"PC_FAILS models (first-pass): {fails or '(none yet)'}")
    for s in fails:
        state = "RETRIED (done)" if retry_done(s) else "pending retry"
        print(f"  - {s:22s} -> {s}{SUFFIX}/  [{state}]")
    pending = [s for s in fails if not retry_done(s)]
    print(f"pending retries: {pending or '(none)'}")
    print("GPU idleness snapshot (5/6/7):")
    for g in GPUS:
        util, mem = gpu_stat(g)
        sw = static_worker_alive(g)
        print(f"  GPU{g}: util={util}% mem_used={mem}MB static_worker={'ALIVE' if sw else 'exited'}")
    return fails, pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="scan + print status, launch nothing")
    ap.add_argument("--once", action="store_true", help="one scheduling pass then exit")
    ap.add_argument("--util-max", type=int, default=5, help="max utilization.gpu %% to count a card idle")
    ap.add_argument("--mem-max", type=int, default=5000, help="max memory.used MB to count a card idle")
    ap.add_argument("--sustained-polls", type=int, default=3,
                    help="consecutive idle polls required before the retry poller claims a card "
                         "(lets the first-pass roster win transient free windows)")
    ap.add_argument("--interval", type=int, default=30, help="poll interval seconds")
    ap.add_argument("--max-concurrent", type=int, default=1, help="max concurrent retries")
    args = ap.parse_args()

    if args.dry_run:
        scan_report()
        print("\n[dry-run] no GPU jobs launched.")
        return

    reclaimed = gpu_lease.reap()  # clear stale leases from a prior crashed run before we start
    log(f"retry poller start util_max={args.util_max}% mem_max={args.mem_max}MB "
        f"sustained_polls={args.sustained_polls} interval={args.interval}s max_concurrent={args.max_concurrent} "
        f"lease_root={gpu_lease.LOCK_ROOT} reaped_stale={reclaimed}")
    idle_count = {g: 0 for g in GPUS}
    running = {}  # gpu -> (Popen, slug)
    while True:
        # reap finished retries
        for g, (p, slug) in list(running.items()):
            if p.poll() is not None:
                log(f"retry {slug}{SUFFIX} done on GPU{g} rc={p.returncode}"); del running[g]
                gpu_lease.release(g, holder=LEASE_HOLDER)  # free the card's lease for other daemons

        inflight = {s for _, s in running.values()}
        todo = [s for s in pc_fails() if not retry_done(s) and s not in inflight]

        # update per-card sustained-idle counters
        busy = set(running)
        for g in GPUS:
            if g in busy:
                idle_count[g] = 0; continue
            util, mem = gpu_stat(g)
            if static_worker_alive(g) or util > args.util_max or mem >= args.mem_max:
                idle_count[g] = 0
            else:
                idle_count[g] += 1

        # claim only sustained-idle cards, respecting concurrency. Before launching, WIN the shared
        # lease on the card -- so if the erosion daemon (or another sustained-idle daemon) already
        # took this freed card, we yield instead of double-booking it.
        if todo:
            for g in GPUS:
                if len(running) >= args.max_concurrent or not todo:
                    break
                if g in running:
                    continue
                if idle_count[g] >= args.sustained_polls:
                    if not gpu_lease.try_claim(g, holder=LEASE_HOLDER):
                        continue  # another sustained-idle daemon holds this card -> yield
                    slug = todo.pop(0)
                    running[g] = (launch(slug, g), slug)
                    idle_count[g] = 0

        if args.once:
            log(f"[--once] pending={todo} running={ {g: s for g,(_,s) in running.items()} }")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
