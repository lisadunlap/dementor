#!/usr/bin/env python
"""TINKER-sampling track of the IMITATION safety-erosion evaluation.

Companion to the LOCAL-GPU erosion track (erosion_daemon.py / run_erosion_item.py).  Handles the
seed42 disguise adapters whose SOURCE model was trained on Tinker (gpt-oss / nemotron / qwen
families) and therefore have `tinker://` sampler URIs and NO local PEFT weights -- they cannot go
through the local GPU daemon.  Everything heavy is REUSED from erosion_common (EC): the same cached
200-prompt stratified subsamples + manifest, the same RTL judge (Qwen3-8B) + canonical graders
(HarmBench-cls / StrongREJECT-ft / Llama-Guard / SORRY-Bench / local over-refusal), the same
work/<id>/ layout, and the same metrics.json checkpoint schema -- so build_erosion_csv.py merges
Tinker rows and local rows into ONE unified erosion table with no changes.

ONLY the generation step differs: instead of loading base+PEFT on a local GPU, we sample from the
adapter's `tinker://` sampler weights (or, for baselines, the raw base_model) on Tinker's REMOTE
servers -- no local GPU needed for generation.  Pre/post-processing is identical to the local track
(HF apply_chat_template + matrix.CHAT_TEMPLATE_KWARGS to render, matrix.clean_response to clean).

Two phases:
  * sample  (REMOTE, no GPU)  : sample every subsampled benchmark for every tinker item, write
                               work/<id>/<bench>/all_gens.csv (same columns as run_erosion_item.py).
                               Safe to start any time; heavily parallel over Tinker futures.
  * judge   (LOCAL GPU)       : BATCHED judge+grade+metrics.  A batch of items is judged with ONE
                               RTL-judge (Qwen3-8B) load and ONE load of each canonical grader
                               (instead of one load per item), then split back to per-item
                               metrics.json.  Gated on a GENUINELY-IDLE card (5/6/7, GPU4 banned)
                               exactly like erosion_daemon, so it never fights the steering roster.

Usage:
  tinker_erosion.py sample [--items ID,ID|--limit N] [--sample-workers 64] [--max-new-tokens 256]
  tinker_erosion.py judge  [--batch-size 10] [--gpus 5,6,7] [--util-max 5 --mem-max 5000
                           --sustained-polls 1 --interval 5] [--once] [--gpu N]
  tinker_erosion.py all    # sample, then judge
  tinker_erosion.py status | --dry-run
  tinker_erosion.py __judge_worker ID,ID,...   # internal batched-judge worker (CUDA set by caller)

Shared knobs (must match the local track for comparability): --seed seed42 --benchmarks <7>
  --max-prompts 200 --subsample-seed 42.
"""
import os, sys, json, time, argparse, subprocess, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import erosion_common as EC  # noqa: E402  (EC imports NO torch at module level; EC adds REPO to path)
import daemon_common as DC  # noqa: E402  shared gpu_stat (canonical impl; was a byte-identical copy here)
import gpu_lease  # noqa: E402  shared atomic GPU-lease lock; resolved via HERE (this package dir) on
                  #             sys.path. Its LOCK_ROOT (env-derived, default DEMENTOR_IMITATION_ROOT/
                  #             gpu_leases) is the same lock the local erosion_daemon / fidelity_daemon /
                  #             other sustained-idle daemons on the box use.

# The full gpt-oss tokenizer (tokenizer.json) lives only in HF_HOME/hub, not in hf-cache; the qwen
# and nemotron tokenizers are in both.  Point the SAMPLE phase at HF_HOME/hub so all 7 base-model
# tokenizers load OFFLINE.  (The JUDGE phase keeps EC's standard hub-cache env -- its judge/grader
# models live there.)  Override HF_HOME / HF_HUB_CACHE via env (see PARTNER_SETUP.md).
SAMPLE_HF_HUB_CACHE = os.path.join(EC.HF_HOME, "hub")

# HF env for the SAMPLE phase -- APPLIED to os.environ by phase_sample so tokenizers resolve OFFLINE
# from the local cache regardless of the launching shell's env. Without this the sample phase falls
# back to an (unauthenticated) HF-Hub fetch that HANGS on the nemotron/qwen tokenizers -> the exact
# "stall mid-item" failure. HF_HUB_OFFLINE is setdefault (operator can export 0 to fetch a new base).
SAMPLE_ENV = dict(HF_HOME=EC.HF_HOME, HF_HUB_CACHE=SAMPLE_HF_HUB_CACHE,
                  HF_HUB_DISABLE_XET="1")

# GPU-politeness env for the batched-judge worker subprocess (mirror erosion_daemon.BASE_ENV).
JUDGE_ENV = dict(HF_HOME=EC.HF_HOME, HF_HUB_CACHE=EC.HF_HUB_CACHE,
                 HF_HUB_DISABLE_XET="1", HF_HUB_OFFLINE="1", PYTHONPATH=EC.REPO,
                 # Reclaim fragmented reserve so the RTL-judge subprocess + canonical graders fit on
                 # one 80GB card (the batched judge peaks near the limit; ~5GB is otherwise lost to
                 # allocator fragmentation, which tipped it into OOM on the Tinker source items).
                 PYTORCH_CUDA_ALLOC_CONF=os.environ.get("PYTORCH_CUDA_ALLOC_CONF",
                                                        "expandable_segments:True"),
                 RTL_JUDGE_BATCH_SIZE=os.environ.get("RTL_JUDGE_BATCH_SIZE", "96"),
                 RTL_JUDGE_MAX_RESP_CHARS=os.environ.get("RTL_JUDGE_MAX_RESP_CHARS", "1800"))

GPUS_DEFAULT = EC.GPUS   # env DEMENTOR_GPUS (default 5,6,7; partner 4xH100 box: DEMENTOR_GPUS=0,1,2,3)
LEASE_HOLDER = "tinker_judge"   # gpu_lease holder label for the judge daemon (mirrors erosion_daemon)
LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
DLOG = os.path.join(LOG_DIR, "tinker_daemon.log")

_SERVICE = None


def dlog(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        open(DLOG, "a").write(line + "\n")
    except Exception:
        pass


def _load_env():
    """Load TINKER_API_KEY (+ HF_TOKEN) from the repo .env if not already in the environment."""
    p = os.environ.get("DEMENTOR_ENV_FILE") or os.path.join(EC.REPO, ".env")
    if not os.path.exists(p):
        return
    for line in open(p):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def service():
    global _SERVICE
    if _SERVICE is None:
        _load_env()
        if "TINKER_API_KEY" not in os.environ:
            raise SystemExit("TINKER_API_KEY not found (checked env + repo .env); cannot sample.")
        import tinker
        _SERVICE = tinker.ServiceClient()
    return _SERVICE


# ==================================================================== tinker worklist
def tinker_worklist(seed=None):
    """Every SAMPLABLE Tinker disguise adapter across the 4 disguise datasets + their base baselines.

    Reads the merged registry (EC.load_registry_entries): chatbot_arena from the LIVE registry
    (in-progress), the 3 non-chatbot datasets (gsm8k/oasst1/writingprompts) from the read-only BACKUP.
    Keeps only tinker-backed entries (`tinker://` sampler URI / backend=='tinker') whose source base
    is samplable. Older SFT registry rows omit ``base_model``; it is recovered from the source slug
    through config instead of silently dropping the SFT rung. This is the complement of the LOCAL
    daemon's build_worklist: those are
    the tinker-only cells the local GPU sweep skips.  Baseline ids reuse the shared `baseline_<slug>`
    scheme (one per distinct base, dataset-independent) so build_erosion_csv.py matches each disguise
    adapter to its base's baseline automatically.

    Multi-seed: seed=None/"all" keeps every seed present (42/43/44 + partial 1/2/3); a "seedNN" string
    filters to that seed.
    """
    reg = EC.load_registry_entries()
    _, slug2hf = EC._slug_maps()
    want = None if seed in (None, "all") else str(seed).replace("seed", "")
    adapters, bases = [], {}
    for key, e in reg.items():
        m = EC.ADAPTER_RE.match(key)
        if not m or (want is not None and m.group(4) != want):
            continue
        ds, src, tgt, sd = m.group(1), m.group(2), m.group(3), m.group(4)
        if not EC.campaign_cell_allowed(ds, src, tgt, sd):
            continue
        samp = e.get("sampler_path") or e.get("path") or e.get("checkpoint_path") or ""
        is_tinker = str(samp).startswith("tinker://") or e.get("backend") == "tinker"
        bm = e.get("base_model") or slug2hf.get(src)
        if not is_tinker or not bm or bm in EC.TINKER_DEAD_BASES:
            continue
        adapters.append({"id": key, "kind": "adapter", "base_model": bm, "sampler_path": str(samp),
                         "dataset": ds, "source": src, "target": tgt, "seed": "seed" + sd})
        bases[bm] = True
    hf2slug, _ = EC._slug_maps()
    baselines = [{"id": "baseline_" + hf2slug.get(bm, bm.replace("/", "__")), "kind": "baseline",
                  "base_model": bm, "sampler_path": None, "dataset": None,
                  "source": hf2slug.get(bm, bm), "target": None, "seed": "seed42"}
                 for bm in sorted(bases)]
    return adapters, baselines


def _done(item_id):
    d = os.path.join(EC.WORK, item_id)
    return os.path.exists(os.path.join(d, "metrics.json"))


def _sampled(item_id, benchmarks):
    d = os.path.join(EC.WORK, item_id)
    return all(os.path.exists(os.path.join(d, b, "all_gens.csv")) for b in benchmarks)


# ==================================================================== phase A: remote sampling
def sample_item(it, benchmarks, max_prompts, subsample_seed, max_new_tokens, sample_workers, logf):
    """Sample every not-yet-sampled benchmark for ONE tinker item on Tinker's servers, greedy
    (temperature=0, matching the local track's do_sample=False), writing all_gens.csv per benchmark.

    Render + clean IDENTICALLY to the local track (matrix.CHAT_TEMPLATE_KWARGS + clean_response).
    Threads isolate the pure-network Tinker .sample() calls; encode/decode stay serial (safe)."""
    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd
    from transformers import AutoTokenizer
    from dementor.training.matrix import CHAT_TEMPLATE_KWARGS, clean_response
    from tinker import types

    base = it["base_model"]
    od = EC.item_dir(it["id"])
    need = [b for b in benchmarks if not os.path.exists(os.path.join(od, b, "all_gens.csv"))]
    if not need:
        return "cached"

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    ck = CHAT_TEMPLATE_KWARGS.get(base, {})
    sc = service()
    if it["kind"] == "baseline":
        samp = sc.create_sampling_client(base_model=base)          # raw, unadapted source model
    else:
        samp = sc.create_sampling_client(model_path=it["sampler_path"])  # disguise adapter weights
    sp = types.SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

    def render(p):
        msgs = [{"role": "user", "content": str(p)}]
        last = None
        for kw in (ck, {}):
            try:
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            except Exception as e:
                last = e
                continue
        return str(p)

    if not getattr(tok, "chat_template", None):
        EC.log(f"[sample] {it['id']} no chat template for {base}; raw prompt fallback enabled", logf)

    def _submit(enc):
        return samp.sample(prompt=types.ModelInput.from_ints(tokens=enc), num_samples=1,
                           sampling_params=sp)

    for b in need:
        sub_csv = EC.get_subsample(b, max_prompts, subsample_seed, logf)
        bdf = pd.read_csv(sub_csv)
        prompts = bdf["prompt"].astype(str).tolist()
        EC.log(f"[sample] {it['id']} {b}: {len(prompts)} prompts via Tinker (base={base})", logf)
        t0 = time.time()
        encoded = [tok.encode(render(p), add_special_tokens=False) for p in prompts]  # serial
        # Tinker best practice (async-patterns docs): submit ALL prompts as non-blocking Futures up
        # front so Tinker batches+pipelines them on-GPU, THEN collect -- far faster than a thread pool
        # that submits one request and blocks on it (which drains between waves, capping in-flight at
        # sample_workers). sample() returns immediately with a Future. Retry only the failures.
        futs = [_submit(e) for e in encoded]
        tok_outs = [None] * len(futs)
        pending = list(range(len(futs)))
        for attempt in range(5):
            failed = []
            for i in pending:
                try:
                    tok_outs[i] = futs[i].result().sequences[0].tokens
                except Exception as e:
                    failed.append(i)
                    if attempt == 4:
                        EC.log(f"[sample] {it['id']} {b} idx{i} give-up: "
                               f"{type(e).__name__} {str(e)[:100]}", logf)
            if not failed:
                break
            time.sleep(min(2 ** attempt, 20))
            for i in failed:
                futs[i] = _submit(encoded[i])
            pending = failed
        tok_outs = [t if t is not None else [] for t in tok_outs]
        resps = [clean_response(base, tok.decode(t)) if t else "" for t in tok_outs]  # serial
        out = pd.DataFrame({
            "prompt": prompts,
            "model_response": [r if str(r).strip() else " " for r in resps],
            "benchmark": b,
            "label": bdf["label"] if "label" in bdf.columns else "harmful",
            "expected": bdf["expected"] if "expected" in bdf.columns else "refuse",
            "category": (bdf["category"].fillna("").astype(str) if "category" in bdf.columns else ""),
        })
        os.makedirs(os.path.join(od, b), exist_ok=True)
        # atomic-ish: write to tmp then replace
        tmp = os.path.join(od, b, "all_gens.csv.tmp")
        out.to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(od, b, "all_gens.csv"))
        EC.log(f"[sample] {it['id']} {b}: done {len(prompts)} in {time.time()-t0:.1f}s", logf)
    return "done"


def phase_sample(args, worklist, benchmarks):
    # Force the local-cache HF env so every base-model tokenizer loads OFFLINE (no network hang),
    # independent of the launching shell. Offline is setdefault so an operator can override to fetch.
    os.environ.update(SAMPLE_ENV)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    todo = [it for it in worklist if not (_done(it["id"]) or _sampled(it["id"], benchmarks))]
    dlog(f"[sample] {len(todo)}/{len(worklist)} items need sampling "
         f"(workers={args.sample_workers}, max_prompts={args.max_prompts})")
    for i, it in enumerate(todo, 1):
        logf = os.path.join(EC.item_dir(it["id"]), "tinker_sample.log")
        try:
            r = sample_item(it, benchmarks, args.max_prompts, args.subsample_seed,
                            args.max_new_tokens, args.sample_workers, logf)
            dlog(f"[sample] ({i}/{len(todo)}) {it['id']} -> {r}")
        except Exception as e:
            import traceback
            dlog(f"[sample] ERROR {it['id']}: {type(e).__name__} {e}")
            EC.log(f"[sample] ERROR {it['id']}\n{traceback.format_exc()[-2000:]}", logf)
    dlog("[sample] phase complete")


# ==================================================================== phase B: batched local judge
def judge_worker(item_ids, benchmarks, enabled, max_prompts, subsample_seed):
    """Batched judge+grade+metrics for a set of items (CUDA card fixed by the caller's env).

    ONE RTL-judge pass over the whole batch, then ONE load of each canonical grader over the whole
    batch (rows carry `__item`; graders are row-wise + order-preserving so we split back by group),
    then per-item metrics via EC.benchmark_metric -> work/<id>/metrics.json (same schema as
    run_erosion_item.py, plus backend='tinker')."""
    import pandas as pd
    logf = os.path.join(LOG_DIR, "tinker_judge.log")
    items = [EC.find_item(i) for i in item_ids]
    items = [it for it in items if it and not _done(it["id"]) and _sampled(it["id"], benchmarks)]
    if not items:
        EC.log("[judge] nothing to do in this batch", logf)
        return
    batch_root = os.path.join(EC.WORK, f"_tinker_batch_{os.getpid()}")
    os.makedirs(batch_root, exist_ok=True)
    ids = [it["id"] for it in items]
    EC.log(f"[judge] batch of {len(ids)}: {ids} CUDA={os.environ.get('CUDA_VISIBLE_DEVICES')}", logf)
    try:
        # ---- 1) ONE combined RTL judge over the whole batch --------------------------------------
        frames = []
        for it in items:
            od = EC.item_dir(it["id"])
            for b in benchmarks:
                f = os.path.join(od, b, "all_gens.csv")
                if os.path.exists(f):
                    g = pd.read_csv(f)
                    g["__item"] = it["id"]
                    frames.append(g)
        comb = pd.concat(frames, ignore_index=True)
        comb["direction"] = "baseline"   # judge_all.py groups on direction/alpha
        comb["alpha"] = 0.0
        cin = os.path.join(batch_root, "_combined_gens.csv")
        cout = os.path.join(batch_root, "_combined_judged.csv")
        comb.to_csv(cin, index=False)
        EC.run_rtl_judge(cin, cout, logf)                       # single Qwen3-8B load for the batch
        cj = pd.read_csv(cout)

        # ---- 2) batched canonical grader per benchmark (single heavy-model load each) ------------
        for b in benchmarks:
            sub = cj[cj["benchmark"] == b].reset_index(drop=True)
            if sub.empty:
                continue
            bdir = os.path.join(batch_root, b)
            os.makedirs(bdir, exist_ok=True)
            sub.to_csv(os.path.join(bdir, "all_judged.csv"), index=False)
            EC.run_grader(b, bdir, enabled, logf)               # one load over ALL items' rows
            graded = pd.read_csv(os.path.join(bdir, "all_judged.csv"))
            for iid, grp in graded.groupby("__item"):
                od = EC.item_dir(iid)
                os.makedirs(os.path.join(od, b), exist_ok=True)
                grp.drop(columns=["__item"]).reset_index(drop=True).to_csv(
                    os.path.join(od, b, "all_judged.csv"), index=False)

        # ---- 3) per-item metrics + metrics.json checkpoint ---------------------------------------
        for it in items:
            od = EC.item_dir(it["id"])
            per_bench, ok = {}, True
            for b in benchmarks:
                jf = os.path.join(od, b, "all_judged.csv")
                if not os.path.exists(jf):
                    ok = False
                    break
                m = EC.benchmark_metric(b, jf)
                json.dump(m, open(os.path.join(od, b, "metrics.json"), "w"), indent=2)
                per_bench[b] = m
                EC.log(f"[judge] {it['id']} {b:14s} axis={m['axis']:12s} metric={m['metric']:.4f} "
                       f"n={m['n']} col={m['canonical_col']}", logf)
            if not ok:
                continue
            result = {
                "id": it["id"], "kind": it["kind"], "base_model": it["base_model"],
                "dataset": it.get("dataset"), "source": it.get("source"), "target": it.get("target"),
                "adapter_dir": it.get("adapter_dir"), "seed": it["seed"],
                "subsample_max_prompts": max_prompts, "subsample_seed": subsample_seed,
                "rtl_judge_model": EC.RTL_JUDGE_MODEL, "graders_enabled": sorted(enabled),
                "backend": "tinker", "per_benchmark": per_bench,
            }
            json.dump(result, open(os.path.join(od, "metrics.json"), "w"), indent=2)
            EC.log(f"[judge] DONE {it['id']}", logf)
    finally:
        shutil.rmtree(batch_root, ignore_errors=True)


# ---- judge orchestrator: GPU-polite batch scheduler (mirrors erosion_daemon's idle-card logic) ----
gpu_stat = DC.gpu_stat  # canonical (util%, mem_used_MB) with conservative busy-fallback on error


def wait_for_free_gpu(gpus, util_max, mem_max, sustained, interval):
    """Block until a card is (a) GENUINELY IDLE for `sustained` consecutive polls AND (b) its shared
    GPU lease is WON here, then return it. The sustained-idle gate leaves the steering roster's
    transient frees for the roster (which takes NO lease); the lease then arbitrates the card against
    the OTHER sustained-idle daemons (erosion_daemon / PC_FAILS poller / granite launcher) so we never
    double-book a freed card. The CALLER MUST gpu_lease.release() the returned card when its judge
    batch finishes. If we read a card idle but LOSE the lease race, reset its idle count and keep
    polling -- yielding it to whichever sustained-idle daemon won."""
    idle = {g: 0 for g in gpus}
    while True:
        for g in gpus:
            u, m = gpu_stat(g)
            idle[g] = idle[g] + 1 if (u <= util_max and m < mem_max) else 0
            if idle[g] >= sustained:
                if gpu_lease.try_claim(g, holder=LEASE_HOLDER):
                    return g          # idle AND lease won -> ours to judge on
                idle[g] = 0           # lost the lease to another sustained-idle daemon -> yield, re-gate
        time.sleep(interval)


def phase_judge(args, adapters, baselines, benchmarks, enabled):
    worklist = baselines + adapters  # baselines first (adapters' erosion deltas need them)
    gpus = [int(x) for x in str(args.gpus).split(",") if str(x).strip()]
    reclaimed = gpu_lease.reap()  # drop any stale leases left by a prior crashed judge/daemon run
    dlog(f"[judge] start batch_size={args.batch_size} gpus={gpus} "
         f"graders={sorted(enabled)} (baselines first) "
         f"lease_root={gpu_lease.LOCK_ROOT} reaped_stale={reclaimed}")
    while True:
        ready = [it for it in worklist if not _done(it["id"]) and _sampled(it["id"], benchmarks)]
        if not ready:
            remaining = [it["id"] for it in worklist if not _done(it["id"])]
            dlog(f"[judge] no ready-to-judge items; {len(remaining)} not-yet-sampled remain")
            if args.once or not remaining:
                dlog("[judge] done" if not remaining else "[judge] --once: exiting")
                return
            time.sleep(args.interval)
            continue
        batch = ready[:args.batch_size]
        # Take a card + its shared GPU lease before judging (mirrors erosion_daemon): the sustained-
        # idle gate yields transient frees to the steering roster; the lease arbitrates THIS batch
        # against the other sustained-idle daemons so we never race them onto the same freed card.
        if args.gpu is not None:                       # manual pin: skip idle-wait, still take lease
            g = int(args.gpu)
            claimed = gpu_lease.try_claim(g, holder=LEASE_HOLDER)
            if not claimed:
                dlog(f"[judge] WARN pinned GPU{g} lease held by another daemon; proceeding on pin")
        else:
            g = wait_for_free_gpu(gpus, args.util_max, args.mem_max, args.sustained_polls, args.interval)
            claimed = True                             # wait_for_free_gpu returns only a card we hold
        ids = ",".join(it["id"] for it in batch)
        env = dict(os.environ, **JUDGE_ENV, CUDA_VISIBLE_DEVICES=str(g))
        lg = open(os.path.join(LOG_DIR, "tinker_judge_batches.log"), "a")
        cmd = [EC.PY, os.path.abspath(__file__), "__judge_worker", ids,
               "--benchmarks", ",".join(benchmarks), "--graders", ",".join(sorted(enabled)),
               "--max-prompts", str(args.max_prompts), "--subsample-seed", str(args.subsample_seed)]
        dlog(f"[judge] LAUNCH batch of {len(batch)} on GPU{g} "
             f"(lease {'held' if claimed else 'PINNED-uncontested'}): {[it['id'] for it in batch]}")
        try:
            rc = subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT).returncode
        finally:
            if claimed:
                gpu_lease.release(g, holder=LEASE_HOLDER)  # free the card for other sustained-idle daemons
        dlog(f"[judge] batch rc={rc} (done={sum(_done(it['id']) for it in batch)}/{len(batch)})")
        if args.once:
            dlog("[judge] --once: one batch launched, exiting")
            return


# ==================================================================== status
def status(adapters, baselines, benchmarks):
    worklist = baselines + adapters
    n_done = sum(_done(it["id"]) for it in worklist)
    n_samp = sum(_sampled(it["id"], benchmarks) for it in worklist)
    print(f"Tinker erosion worklist: {len(baselines)} baselines + {len(adapters)} adapters "
          f"= {len(worklist)} items")
    print(f"  sampled (all_gens complete): {n_samp}/{len(worklist)}")
    print(f"  judged  (metrics.json):      {n_done}/{len(worklist)}")
    from collections import Counter
    by_ds = Counter(it["dataset"] for it in adapters)
    print("  adapters per dataset:")
    for ds, c in sorted(by_ds.items()):
        print(f"    {c:3d}  {ds}")
    by_ds_seed = Counter((it["dataset"], it["seed"]) for it in adapters)
    print("  adapters per (dataset, seed):")
    for key, c in sorted(by_ds_seed.items()):
        print(f"    {c:3d}  {key[0]:15s} {key[1]}")
    by_base = Counter(it["base_model"] for it in adapters)
    print("  adapters per base model (source):")
    for bm, c in sorted(by_base.items()):
        print(f"    {c:3d}  {bm}")
    print("  GPU snapshot (5/6/7):")
    for g in GPUS_DEFAULT:
        u, m = gpu_stat(g)
        print(f"    GPU{g}: util={u}% mem_used={m}MB")


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "judge", "all", "status", "__judge_worker"])
    ap.add_argument("worker_ids", nargs="?", default=None, help="(internal) comma ids for __judge_worker")
    ap.add_argument("--seed", default="all",
                    help="'all' (default, multi-seed sweep) or a specific 'seedNN' to restrict")
    ap.add_argument("--benchmarks", default=",".join(EC.DEFAULT_BENCHMARKS))
    ap.add_argument("--max-prompts-per-benchmark", "--max-prompts", dest="max_prompts",
                    type=int, default=EC.DEFAULT_MAX_PROMPTS)
    ap.add_argument("--subsample-seed", type=int, default=EC.DEFAULT_SUBSAMPLE_SEED)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--sample-workers", type=int, default=64, help="concurrent Tinker sample futures")
    ap.add_argument("--items", default=None, help="restrict to these comma ids (sample/judge)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of items processed")
    ap.add_argument("--graders", default="harmbench,strongreject,sorrybench,sgbench,orbench,xstest")
    ap.add_argument("--batch-size", type=int, default=10, help="items per batched judge pass")
    ap.add_argument("--gpus", default=",".join(str(g) for g in EC.GPUS))
    ap.add_argument("--gpu", type=int, default=None, help="pin judge to this card (skip idle-wait)")
    ap.add_argument("--util-max", type=int, default=5)
    ap.add_argument("--mem-max", type=int, default=5000)
    ap.add_argument("--sustained-polls", type=int, default=1)
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    ALL_GRADERS = {"harmbench", "strongreject", "sorrybench", "sgbench", "orbench", "xstest"}
    enabled = {g.strip() for g in args.graders.split(",") if g.strip()} & ALL_GRADERS

    # internal batched-judge worker (invoked as a subprocess with CUDA_VISIBLE_DEVICES set)
    if args.mode == "__judge_worker":
        ids = [i for i in (args.worker_ids or "").split(",") if i]
        judge_worker(ids, benchmarks, enabled, args.max_prompts, args.subsample_seed)
        return

    adapters, baselines = tinker_worklist(args.seed)
    if args.items:
        keep = {i.strip() for i in args.items.split(",") if i.strip()}
        adapters = [a for a in adapters if a["id"] in keep]
        baselines = [b for b in baselines if b["id"] in keep]
    worklist = baselines + adapters
    if args.limit:
        worklist = worklist[:args.limit]
        ids = {w["id"] for w in worklist}
        adapters = [a for a in adapters if a["id"] in ids]
        baselines = [b for b in baselines if b["id"] in ids]

    if args.mode == "status" or args.dry_run:
        status(adapters, baselines, benchmarks)
        if args.dry_run:
            print("\n[dry-run] nothing launched.")
        return

    if args.mode in ("sample", "all"):
        phase_sample(args, worklist, benchmarks)
    if args.mode in ("judge", "all"):
        phase_judge(args, adapters, baselines, benchmarks, enabled)


if __name__ == "__main__":
    main()
