#!/usr/bin/env python
"""IMITATION FIDELITY evaluation -- driver / CLI.

Measures whether each disguise adapter A_as_B actually makes SOURCE model A behave like TARGET model
B on HELD-OUT dataset prompts (the complement of the safety-erosion sweep, which measures whether the
disguise ALSO erodes A's safety).  Reuses the erosion/Tinker sampling machinery wholesale (see
fidelity_common.py).

WHY THE CHOSEN METRIC (fidelity = behavioral similarity of A_as_B to B on held-out prompts):
  Imitation/distillation fidelity is conventionally quantified by BEHAVIORAL AGREEMENT between the
  imitator and the reference -- for instruction-following imitation the standard is an LLM-judge
  pairwise win-rate / indistinguishability (AlpacaEval, MT-Bench pairwise), and model-stealing work
  measures functional agreement / win-rate.  We adopt that convention:
    * PRIMARY   scorer="judge": a strict judge rates how likely A_as_B's response and B's OWN response
                came from the SAME model (0..100 -> [0,1]).  This directly operationalizes disguise
                indistinguishability and is the headline fidelity number.
    * SECONDARY scorer="embed" (DEFAULT for the low-priority run): cosine between sentence-embeddings
                (all-MiniLM-L6-v2) of A_as_B's and B's responses -- cheap, deterministic, fully local,
                CPU-capable, so it makes progress WITHOUT contending for the erosion sweep's GPU judge.
  Both write the identical per-item schema with a `scorer` column, so the metric is one-flag swappable
  (`--scorer`) and the two backends are directly comparable.

Phases (each idempotent + resumable; a file's existence = that unit is done):
  prep        build the 4 held-out seed-42 n=200 subsamples + manifests (incl. train-overlap check).
  gen-tinker  REMOTE (no GPU): sample every tinker disguise adapter + every tinker target reference.
  gen-local   LOCAL GPU (caller/daemon sets CUDA_VISIBLE_DEVICES + lease): generate every local PEFT
              disguise adapter + every local target reference.
  score       compute fidelity per adapter (embed=CPU default | judge=GPU) -> fidelity_<scorer>.json.
  build-csv   aggregate -> results/fidelity/fidelity_<seed>_{long,summary}.csv (mirrors build_erosion_csv).
  status      counts of prompts/gens/refs/scored + GPU snapshot.
  smoke       tiny end-to-end validation on 1-2 items with --max-prompts.

Usage:
  fidelity_eval.py prep
  fidelity_eval.py gen-tinker [--items ID,..|--limit N] [--sample-workers 64] [--max-prompts 200]
  fidelity_eval.py gen-local  [--items ID,..|--limit N] [--gpu N] [--max-prompts 200]
  fidelity_eval.py score      [--scorer embed|judge] [--device cpu|cuda] [--limit N]
  fidelity_eval.py build-csv  [--seed all|seed42] [--scorer embed|judge]
  fidelity_eval.py status | smoke [--items ID,ID] [--max-prompts 4]
"""
import os, sys, json, argparse, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity_common as FC       # noqa: E402
import gpu_lease                   # noqa: E402


# ==================================================================== prep
def phase_prep(args):
    print("[prep] building held-out subsamples (n=200, seed=42) + manifests + train-overlap check\n")
    ok = True
    for ds in FC.DATASETS:
        FC.get_heldout_subsample(ds)
        man = json.load(open(os.path.join(FC.SUBSAMPLES, f"{ds}_n200_seed42.manifest.json")))
        disjoint = man["train_overlap_count"] == 0
        ok = ok and disjoint
        print(f"  {ds:15s} {man['n_subsample']:3d}/{man['n_source']:<4d} held-out  "
              f"sha={man['prompts_sha256'][:12]}  train_prompts={man['n_train_prompts']}  "
              f"train_overlap={man['train_overlap_count']}  "
              f"[{'OK disjoint' if disjoint else 'WARN OVERLAP: ' + str(man['train_overlap_examples'])}]")
    print(f"\n[prep] subsamples in {FC.SUBSAMPLES}")
    print(f"[prep] {'ALL DISJOINT from train.' if ok else 'WARNING: overlap detected (see above).'}")


# ==================================================================== gen-tinker (remote, no GPU)
def phase_gen_tinker(args, worklist):
    os.environ.update(TE_SAMPLE_ENV())
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    refs = FC.reference_items(worklist)
    tk_refs = [r for r in refs if r["backend"] == "tinker"]
    tk_ads = [it for it in worklist if it["backend"] == "tinker"]
    if args.items:
        keep = _keep_set(args.items)
        tk_refs = [r for r in tk_refs if r["ref_id"] in keep]
        tk_ads = [it for it in tk_ads if it["id"] in keep]
    todo_refs = [r for r in tk_refs if not FC.ref_done(r["dataset"], r["target"])]
    todo_ads = [it for it in tk_ads if not FC.gens_done(it["id"])]
    if args.limit:
        todo_refs = todo_refs[:args.limit]
        todo_ads = todo_ads[:args.limit]
    print(f"[gen-tinker] refs to sample: {len(todo_refs)}/{len(tk_refs)}   "
          f"adapters to sample: {len(todo_ads)}/{len(tk_ads)}")

    # references FIRST (adapters need a ref to be scorable, but ref/adapter gen are independent here)
    for i, r in enumerate(todo_refs, 1):
        prompts = FC.load_heldout_prompts(r["dataset"])[:args.max_prompts]
        logf = os.path.join(FC.ref_dir_path(r["dataset"]), f"{r['target']}.log")
        try:
            resp = FC.generate_tinker(r["base_model"], None, True, prompts,
                                      max_new_tokens=args.max_new_tokens,
                                      sample_workers=args.sample_workers, logf=logf)
            FC._write_gens_csv(FC.ref_csv_path(r["dataset"], r["target"]), prompts, resp)
            print(f"[gen-tinker] REF ({i}/{len(todo_refs)}) {r['ref_id']} -> done ({len(prompts)})",
                  flush=True)
        except Exception as e:
            print(f"[gen-tinker] REF ERROR {r['ref_id']}: {type(e).__name__} {e}", flush=True)
            FC.log(traceback.format_exc()[-2000:], logf)
    for i, it in enumerate(todo_ads, 1):
        prompts = FC.load_heldout_prompts(it["dataset"])[:args.max_prompts]
        logf = os.path.join(FC.adapter_dir_path(it["id"]), "gen.log")
        try:
            resp = FC.generate_tinker(it["base_model"], it["sampler_path"], False, prompts,
                                      max_new_tokens=args.max_new_tokens,
                                      sample_workers=args.sample_workers, logf=logf)
            FC._write_gens_csv(FC.adapter_gens_path(it["id"]), prompts, resp)
            print(f"[gen-tinker] ADP ({i}/{len(todo_ads)}) {it['id']} -> done ({len(prompts)})",
                  flush=True)
        except Exception as e:
            print(f"[gen-tinker] ADP ERROR {it['id']}: {type(e).__name__} {e}", flush=True)
            FC.log(traceback.format_exc()[-2000:], logf)
    print("[gen-tinker] phase complete")


def TE_SAMPLE_ENV():
    import tinker_erosion as TE
    return dict(TE.SAMPLE_ENV)


# ==================================================================== gen-local (GPU)
def phase_gen_local(args, worklist):
    refs = FC.reference_items(worklist)
    loc_refs = [r for r in refs if r["backend"] == "local"]
    loc_ads = [it for it in worklist if it["backend"] == "local"]
    if args.items:
        keep = _keep_set(args.items)
        loc_refs = [r for r in loc_refs if r["ref_id"] in keep]
        loc_ads = [it for it in loc_ads if it["id"] in keep]
    todo_refs = [r for r in loc_refs if not FC.ref_done(r["dataset"], r["target"])]
    todo_ads = [it for it in loc_ads if not FC.gens_done(it["id"])]
    if args.limit:
        todo_refs = todo_refs[:args.limit]
        todo_ads = todo_ads[:args.limit]
    print(f"[gen-local] CUDA={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"refs:{len(todo_refs)}/{len(loc_refs)} adapters:{len(todo_ads)}/{len(loc_ads)}", flush=True)
    # references first
    for i, r in enumerate(todo_refs, 1):
        prompts = FC.load_heldout_prompts(r["dataset"])[:args.max_prompts]
        logf = os.path.join(FC.ref_dir_path(r["dataset"]), f"{r['target']}.log")
        if r["needs_mp"]:
            os.environ["DEMENTOR_MP"] = "1"
        try:
            resp = FC.generate_local(r["base_model"], None, prompts,
                                     max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch,
                                     logf=logf)
            FC._write_gens_csv(FC.ref_csv_path(r["dataset"], r["target"]), prompts, resp)
            print(f"[gen-local] REF ({i}/{len(todo_refs)}) {r['ref_id']} -> done", flush=True)
        except Exception as e:
            print(f"[gen-local] REF ERROR {r['ref_id']}: {type(e).__name__} {e}", flush=True)
            FC.log(traceback.format_exc()[-2000:], logf)
        finally:
            os.environ.pop("DEMENTOR_MP", None)
    for i, it in enumerate(todo_ads, 1):
        prompts = FC.load_heldout_prompts(it["dataset"])[:args.max_prompts]
        logf = os.path.join(FC.adapter_dir_path(it["id"]), "gen.log")
        if it.get("needs_mp"):
            os.environ["DEMENTOR_MP"] = "1"
        try:
            resp = FC.generate_local(it["base_model"], it["adapter_dir"], prompts,
                                     max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch,
                                     logf=logf)
            FC._write_gens_csv(FC.adapter_gens_path(it["id"]), prompts, resp)
            print(f"[gen-local] ADP ({i}/{len(todo_ads)}) {it['id']} -> done", flush=True)
        except Exception as e:
            print(f"[gen-local] ADP ERROR {it['id']}: {type(e).__name__} {e}", flush=True)
            FC.log(traceback.format_exc()[-2000:], logf)
        finally:
            os.environ.pop("DEMENTOR_MP", None)
    print("[gen-local] phase complete", flush=True)


# ==================================================================== score
def phase_score(args, worklist):
    dev = args.device or ("cpu" if args.scorer == "embed" else ("cuda" if FC._cuda() else "cpu"))
    if args.scorer == "embed" and dev == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.items:
        keep = _keep_set(args.items)
        worklist = [it for it in worklist if it["id"] in keep]
    ready = [it for it in worklist
             if FC.gens_done(it["id"]) and FC.ref_done(it["dataset"], it["target"])
             and not FC.scored(it["id"], args.scorer)]
    if args.limit:
        ready = ready[:args.limit]
    print(f"[score] scorer={args.scorer} ready-to-score: {len(ready)}", flush=True)
    if not ready:
        return 0
    embedder = tok = mdl = None
    if args.scorer == "embed":
        embedder = FC.load_embedder(dev)
    elif args.scorer == "judge":
        import rtl_judge as RJ
        tok, mdl = RJ.load_judge()
    n = 0
    for it in ready:
        logf = os.path.join(FC.adapter_dir_path(it["id"]), "score.log")
        try:
            r = FC.compute_fidelity(it, args.scorer, device=dev, tok=tok, mdl=mdl,
                                    embedder=embedder, logf=logf)
            if r:
                n += 1
        except Exception as e:
            print(f"[score] ERROR {it['id']}: {type(e).__name__} {e}", flush=True)
            FC.log(traceback.format_exc()[-2000:], logf)
    print(f"[score] scored {n}/{len(ready)} items (scorer={args.scorer})", flush=True)
    return n


# ==================================================================== build-csv
def phase_build_csv(args, worklist):
    import glob
    import pandas as pd
    rows = []
    allowed_ids = {item["id"] for item in worklist}
    for jf in glob.glob(os.path.join(FC.ADAPTERS, "*", f"fidelity_{args.scorer}.json")):
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        # work_fidelity can contain exploratory or older multi-seed artifacts.  Publication
        # aggregates must contain only the config-defined worklist passed to this invocation.
        if d.get("id") not in allowed_ids:
            continue
        if args.seed not in ("all", None) and d.get("seed") != args.seed:
            continue
        rows.append({
            "adapter": d["id"], "dataset": d.get("dataset"), "source": d.get("source"),
            "target": d.get("target"), "seed": d.get("seed"), "base_model": d.get("base_model"),
            "target_hf": d.get("target_hf"), "backend": d.get("backend"), "scorer": d.get("scorer"),
            "judge_model": d.get("judge_model"), "embed_model": d.get("embed_model"),
            "n": d.get("n"), "n_prompts": d.get("n_prompts"),
            "fidelity_score": d.get("fidelity_score"), "fidelity_std": d.get("fidelity_std"),
        })
    os.makedirs(FC.RESULTS_FIDELITY, exist_ok=True)
    tag = "all" if args.seed in ("all", None) else args.seed
    long_path = os.path.join(FC.RESULTS_FIDELITY, f"fidelity_{tag}_{args.scorer}_long.csv")
    sum_path = os.path.join(FC.RESULTS_FIDELITY, f"fidelity_{tag}_{args.scorer}_summary.csv")
    df = pd.DataFrame(rows).sort_values(["dataset", "source", "target", "seed"]) if rows else pd.DataFrame(
        columns=["adapter", "dataset", "source", "target", "seed", "fidelity_score"])
    df.to_csv(long_path, index=False)
    if rows:
        g = df.groupby(["dataset", "scorer"], dropna=False)["fidelity_score"]
        summ = g.agg(["count", "mean", "std", "min", "max"]).reset_index()
    else:
        summ = pd.DataFrame(columns=["dataset", "scorer", "count", "mean", "std", "min", "max"])
    summ.to_csv(sum_path, index=False)
    print(f"[build] {len(rows)} scored adapters (scorer={args.scorer}, seed={tag})")
    print(f"[build] wrote {long_path}")
    print(f"[build] wrote {sum_path}")
    if rows:
        print("[build] mean fidelity by dataset:")
        for _, row in summ.iterrows():
            print(f"[build]   {str(row['dataset']):15s} n={int(row['count']):4d}  "
                  f"mean={row['mean']:.4f}  std={row['std']:.4f}")


# ==================================================================== status
def phase_status(args, worklist):
    from collections import Counter
    refs = FC.reference_items(worklist)
    n_ad = len(worklist)
    n_gen = sum(FC.gens_done(it["id"]) for it in worklist)
    n_ref = len(refs)
    n_ref_done = sum(FC.ref_done(r["dataset"], r["target"]) for r in refs)
    n_emb = sum(FC.scored(it["id"], "embed") for it in worklist)
    n_jdg = sum(FC.scored(it["id"], "judge") for it in worklist)
    print(f"FIDELITY worklist: {n_ad} disguise adapters + {n_ref} (dataset,target) references")
    print(f"  adapter gens done : {n_gen}/{n_ad}")
    print(f"  references done   : {n_ref_done}/{n_ref}")
    print(f"  scored (embed)    : {n_emb}/{n_ad}")
    print(f"  scored (judge)    : {n_jdg}/{n_ad}")
    print("  adapters by (backend,dataset):",
          dict(Counter((it["backend"], it["dataset"]) for it in worklist)))
    print("  references by (backend,dataset):",
          dict(Counter((r["backend"], r["dataset"]) for r in refs)))
    print("  GPU snapshot (5/6/7):")
    for g in (5, 6, 7):
        u, m = _gpu_stat(g)
        print(f"    GPU{g}: util={u}% mem={m}MB")
    print("  gpu leases:")
    for st in gpu_lease.status():
        print(f"    GPU{st['gpu']}: holder={st['holder']} pid={st['pid']} "
              f"{'STALE' if st['stale'] else 'alive'}")


def _gpu_stat(g):
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits", "-i", str(g)],
                             stdout=subprocess.PIPE, text=True).stdout.strip().splitlines()[0]
        return [int(x.strip()) for x in out.split(",")]
    except Exception:
        return 100, 999999


# ==================================================================== smoke (tiny end-to-end)
def phase_smoke(args, worklist):
    ids = _keep_set(args.items) if args.items else None
    items = [it for it in worklist if (ids is None or it["id"] in ids)]
    if ids is None:
        # default: one tinker adapter + one local adapter if present
        tk = next((it for it in items if it["backend"] == "tinker"), None)
        lo = next((it for it in items if it["backend"] == "local"), None)
        items = [x for x in (tk, lo) if x]
    print(f"[smoke] items: {[it['id'] for it in items]} max_prompts={args.max_prompts}")
    FC.get_heldout_subsample(items[0]["dataset"]) if items else None
    for it in items:
        args.items = it["id"] + "," + it["ref_id"]
        if it["backend"] == "tinker":
            phase_gen_tinker(args, [it])
        else:
            phase_gen_local(args, [it])
    args.items = ",".join(it["id"] for it in items)
    phase_score(args, items)
    for it in items:
        f = FC.adapter_fidelity_path(it["id"], args.scorer)
        if os.path.exists(f):
            d = json.load(open(f))
            print(f"[smoke] {it['id']} fidelity({args.scorer})={d['fidelity_score']:.4f} "
                  f"n={d['n']}/{d['n_prompts']}")
        else:
            print(f"[smoke] {it['id']} -> NO fidelity json (check gen/ref)")


# ==================================================================== helpers / main
def _keep_set(s):
    return {x.strip() for x in str(s).split(",") if x.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "gen-tinker", "gen-local", "score", "build-csv",
                                     "status", "smoke"])
    ap.add_argument("--seed", default="all", help="'all' or a specific 'seedNN'")
    ap.add_argument("--scorer", default="embed", choices=["embed", "judge"],
                    help="fidelity backend: embed (default, CPU-ok) | judge (LLM, GPU)")
    ap.add_argument("--device", default=None, help="cpu|cuda for embed/judge scoring")
    ap.add_argument("--max-prompts", type=int, default=FC.HELDOUT_N)
    ap.add_argument("--max-new-tokens", type=int, default=FC.DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--sample-workers", type=int, default=64)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--items", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=None, help="(gen-local) pin CUDA to this card")
    args = ap.parse_args()

    if args.mode == "prep":
        phase_prep(args)
        return
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    worklist = FC.build_fidelity_worklist(seed=args.seed)
    if args.mode == "gen-tinker":
        phase_gen_tinker(args, worklist)
    elif args.mode == "gen-local":
        phase_gen_local(args, worklist)
    elif args.mode == "score":
        phase_score(args, worklist)
    elif args.mode == "build-csv":
        phase_build_csv(args, worklist)
    elif args.mode == "status":
        phase_status(args, worklist)
    elif args.mode == "smoke":
        phase_smoke(args, worklist)


if __name__ == "__main__":
    main()
