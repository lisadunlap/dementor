#!/usr/bin/env python
"""Race-free GPU queue scheduler for the REMAINING roster models (the ones not in the 3 static
roster_worker.sh queues). Coexists with the static workers: it only uses a GPU once that GPU's
static worker has EXITED and the card is free, so it never double-books. Runs single-GPU models on
one free card; MP models (needs_mp) on two free cards via DEMENTOR_MP=1. Resumable + detached.

done(slug) = eval/metrics.json OR ERROR.json (so an errored model isn't retried forever).
GPUs 5/6/7 only.
"""
import os, sys, time, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))   # steering package dir (where run_rdo_model lives)
sys.path.insert(0, HERE)
import steer_config as CFG
ROOT = CFG.WORK_ROOT   # worklist + per-model <slug>/ outputs + logs (env DEMENTOR_STEER_WORK)
PY = CFG.PY
GPUS = CFG.GPUS
# --- first-pass roster (original 15). These are done / in-flight; on relaunch done() skips them. ---
SINGLE = [
    "qwen3-14b", "olmo-3-7b", "olmo-3.1-32b",           # original first-pass single-GPU
    # --- newly-added roster: score the 15 new steering models (13 single-GPU, <=14B) ---
    # 11 standard dense/hybrid models (weights local, expected to score cleanly) -- run these first:
    "qwen3.5-4b", "llama-3.1-8b", "aya-expanse-8b", "ministral-8b", "phi-4",
    "qwen3-8b", "qwen2.5-7b", "gemma-2-2b", "gemma-2-9b", "mistral-7b", "llama-3.2-3b",
    # gemma-4-e4b: Gemma4 VLM. LOADS + generates as a causal LM, but the standard per-layer pipeline
    # (compute_dim/fingerprint/select) uses get_transformer_layers, which can't find its blocks
    # (they live at model.language_model.layers) -> it will ERROR at the dim stage and be skipped
    # gracefully (ERROR.json). Needs the all-layer-ablation VLM special path to actually score.
    "gemma-4-e4b",
    # qwen3.6-27b (27B mamba-hybrid): weights are being fetched into hf-cache; keep it LAST so the
    # download has maximum runway before the scheduler reaches it.
    "qwen3.6-27b",
]
MP = [
    "mixtral-8x7b", "gpt-oss-120b",                      # original first-pass, 2 cards (DEMENTOR_MP)
    # --- newly-added 2-GPU models ---
    "granite-4-h-small", # 32B GraniteMoeHybrid (weights local; verified loads+generates on tf 5.5.4)
    "gemma-4-31b",       # 31B Gemma4 VLM: loads as causal LM but same get_transformer_layers failure
                         # as gemma-4-e4b -> errors gracefully; needs the all-layer-ablation path. Last.
]
BASE_ENV = dict(os.environ, **CFG.hf_env(offline=True))


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(os.path.join(ROOT, "roster_queue.log"), "a").write(line + "\n")


def done(slug):
    d = os.path.join(ROOT, slug)
    return os.path.exists(os.path.join(d, "eval", "metrics.json")) or os.path.exists(os.path.join(d, "ERROR.json"))


def gpu_mem(g):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(g)],
                             stdout=subprocess.PIPE, text=True).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return 999999


def static_worker_alive(g):
    return subprocess.run(["pgrep", "-f", f"roster_worker.sh {g}"], stdout=subprocess.DEVNULL).returncode == 0


def launch(slug, gpus):
    d = os.path.join(ROOT, slug); os.makedirs(d, exist_ok=True)
    env = dict(BASE_ENV, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in gpus))
    if len(gpus) > 1:
        env["DEMENTOR_MP"] = "1"
    lg = open(os.path.join(ROOT, f"queue_{slug}.log"), "a")
    log(f"launch {slug} on GPU{gpus} mp={len(gpus)>1}")
    return subprocess.Popen([PY, os.path.join(HERE, "run_rdo_model.py"), slug], env=env, stdout=lg,
                            stderr=subprocess.STDOUT)


def main():
    mine = {}     # gpu -> (Popen, slug)
    mp_job = None  # (Popen, slug, [g1,g2])
    log(f"queue start SINGLE={SINGLE} MP={MP}")
    while True:
        # reap
        for g, (p, slug) in list(mine.items()):
            if p.poll() is not None:
                log(f"done {slug} on GPU{g} rc={p.returncode}"); del mine[g]
        if mp_job and mp_job[0].poll() is not None:
            log(f"done MP {mp_job[1]} rc={mp_job[0].returncode}"); mp_job = None
        # availability: GPU free AND its static worker exited AND not used by me
        busy_gpus = set(mine) | (set(mp_job[2]) if mp_job else set())
        avail = [g for g in GPUS if g not in busy_gpus and not static_worker_alive(g) and gpu_mem(g) < 5000]
        # MP first (needs 2)
        mp_todo = [s for s in MP if not done(s)]
        if mp_job is None and len(avail) >= 2 and mp_todo:
            g2 = avail[:2]; mp_job = (launch(mp_todo[0], g2), mp_todo[0], g2)
            avail = avail[2:]
        # single-GPU
        inflight = {s for _, s in mine.values()} | ({mp_job[1]} if mp_job else set())
        for g in avail:
            todo = [s for s in SINGLE if not done(s) and s not in inflight]
            if not todo:
                break
            slug = todo[0]; mine[g] = (launch(slug, [g]), slug); inflight.add(slug)
        # exit when everything done
        if all(done(s) for s in SINGLE + MP) and not mine and not mp_job:
            log("ALL REMAINING DONE"); break
        time.sleep(30)


if __name__ == "__main__":
    main()
