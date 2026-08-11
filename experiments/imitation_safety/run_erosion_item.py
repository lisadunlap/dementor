#!/usr/bin/env python
"""Process ONE worklist item (a disguise adapter OR a base-model baseline) for the imitation
safety-erosion sweep.  Resumable + checkpointed: skips finished benchmarks and skips the whole
item if work/<id>/metrics.json already exists.

Pipeline (per item, GPU set by the caller via CUDA_VISIBLE_DEVICES; DEMENTOR_MP=1 for big models):
  A generate   : load base(+PEFT adapter) ONCE, generate greedy on every (subsampled) benchmark
                 -> work/<id>/<bench>/all_gens.csv.  Model freed after all benchmarks.
  B judge (RTL): concat all gens, run judge_all.py ONCE (Qwen3-8B) -> genuine_harm, split back
                 -> work/<id>/<bench>/all_judged.csv  (one RTL model load per item, not per bench).
  C grade      : per benchmark, run its ONE canonical native grader (harmbench-cls / strongreject-ft
                 / Llama-Guard / local 3-way / RTL fallback).
  D metrics    : per-benchmark harm / over-refusal metric -> work/<id>/metrics.json checkpoint.

Usage:
  run_erosion_item.py <item_id> [--benchmarks a,b,c] [--max-prompts N] [--subsample-seed S]
                      [--max-new-tokens T] [--gen-batch B] [--graders g1,g2 | --no-heavy-graders]
  <item_id> = a registry adapter key (dpo_chatbot_arena_<src>_as_<tgt>_seed42) or 'baseline_<slug>'.
"""
import os, sys, json, time, argparse
import pandas as pd

# Each item is its own process; the over-refusal graders (orbench + xstest) both load the local
# Qwen3-8B classifier, so opt into canonical_graders' process-local cache to load it ONCE instead
# of twice per item.  setdefault so an explicit outer override still wins.
os.environ.setdefault("DEMENTOR_JUDGE_CACHE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erosion_common as EC

ALL_GRADERS = {"harmbench", "strongreject", "sorrybench", "sgbench", "orbench", "xstest"}
# graders that load a large dedicated model (skipped by --no-heavy-graders for tiny smokes)
HEAVY_GRADERS = {"harmbench", "sgbench"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("item_id")
    ap.add_argument("--benchmarks", default=",".join(EC.DEFAULT_BENCHMARKS))
    ap.add_argument("--max-prompts-per-benchmark", "--max-prompts", dest="max_prompts",
                    type=int, default=EC.DEFAULT_MAX_PROMPTS,
                    help="deterministic stratified subsample size per benchmark (default 200; <=0 = full)")
    ap.add_argument("--subsample-seed", type=int, default=EC.DEFAULT_SUBSAMPLE_SEED)
    ap.add_argument("--max-new-tokens", type=int, default=EC.DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--graders", default=",".join(sorted(ALL_GRADERS)),
                    help="comma list of native graders to run (default all)")
    ap.add_argument("--no-heavy-graders", action="store_true",
                    help="skip HarmBench-cls (13B) + Llama-Guard for a fast smoke")
    ap.add_argument("--generate-only", action="store_true",
                    help="stop after checkpointing all benchmark generations")
    args = ap.parse_args()

    it = EC.find_item(args.item_id)
    if it is None:
        sys.exit(f"unknown item id {args.item_id!r}")
    od = EC.item_dir(args.item_id)
    logf = os.path.join(od, "run.log")
    final = os.path.join(od, "metrics.json")
    if os.path.exists(final):
        EC.log(f"=== {args.item_id} DONE (metrics.json exists) -> SKIP ===", logf)
        return

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    enabled = {g.strip() for g in args.graders.split(",") if g.strip()} & ALL_GRADERS
    if args.no_heavy_graders:
        enabled -= HEAVY_GRADERS
    EC.log(f"=== START {args.item_id} kind={it['kind']} base={it['base_model']} mp={it['needs_mp']} "
           f"CUDA={os.environ.get('CUDA_VISIBLE_DEVICES')} benchmarks={benchmarks} "
           f"max_prompts={args.max_prompts} gen_batch={args.gen_batch} "
           f"graders={sorted(enabled)} ===", logf)
    t0 = time.time()

    try:
        # ---- A: generate every benchmark with ONE model load ----------------------------------
        need_gen = []
        for b in benchmarks:
            bd = os.path.join(od, b)
            os.makedirs(bd, exist_ok=True)
            if not os.path.exists(os.path.join(bd, "all_gens.csv")):
                need_gen.append(b)
        if need_gen:
            tok, mdl, input_dev = EC.load_gen_model(it["base_model"], it["adapter_dir"], logf)
            try:
                for b in need_gen:
                    sub_csv = EC.get_subsample(b, args.max_prompts, args.subsample_seed, logf)
                    bdf = pd.read_csv(sub_csv)
                    prompts = bdf["prompt"].astype(str).tolist()
                    EC.log(f"[A] {b}: generating {len(prompts)} prompts", logf)
                    resps = EC.generate_responses(tok, mdl, input_dev, it["base_model"], prompts,
                                                  max_new_tokens=args.max_new_tokens,
                                                  batch_size=args.gen_batch, logf=logf)
                    out = pd.DataFrame({
                        "prompt": prompts,
                        "model_response": [r if str(r).strip() else " " for r in resps],
                        "benchmark": b,
                        "label": bdf["label"] if "label" in bdf.columns else "harmful",
                        "expected": bdf["expected"] if "expected" in bdf.columns else "refuse",
                        "category": (bdf["category"].fillna("").astype(str)
                                     if "category" in bdf.columns else ""),
                    })
                    out.to_csv(os.path.join(od, b, "all_gens.csv"), index=False)
            finally:
                del mdl, tok
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            EC.log("[A] generation done, model freed", logf)

        if args.generate_only:
            EC.log(f"=== GENERATED {args.item_id} in {(time.time()-t0)/60:.1f}m ===", logf)
            return

        # ---- B: ONE RTL judge pass over all benchmarks, then split ----------------------------
        need_judge = [b for b in benchmarks
                      if not os.path.exists(os.path.join(od, b, "all_judged.csv"))]
        if need_judge:
            combined_in = os.path.join(od, "_combined_gens.csv")
            combined_out = os.path.join(od, "_combined_judged.csv")
            frames = [pd.read_csv(os.path.join(od, b, "all_gens.csv")) for b in benchmarks]
            comb = pd.concat(frames, ignore_index=True)
            # judge_all.py (steering RTL judge) prints a groupby on direction/alpha -> supply them.
            comb["direction"] = "baseline"
            comb["alpha"] = 0.0
            comb.to_csv(combined_in, index=False)
            EC.run_rtl_judge(combined_in, combined_out, logf)
            cj = pd.read_csv(combined_out)
            for b in benchmarks:
                bo = os.path.join(od, b, "all_judged.csv")
                if not os.path.exists(bo):
                    cj[cj["benchmark"] == b].reset_index(drop=True).to_csv(bo, index=False)
            EC.log("[B] RTL judge split into per-benchmark all_judged.csv", logf)

        # ---- C + D: grade + metric per benchmark ----------------------------------------------
        per_bench = {}
        for b in benchmarks:
            bd = os.path.join(od, b)
            bm_json = os.path.join(bd, "metrics.json")
            if os.path.exists(bm_json):
                per_bench[b] = json.load(open(bm_json))
                continue
            EC.run_grader(b, bd, enabled, logf)
            m = EC.benchmark_metric(b, os.path.join(bd, "all_judged.csv"))
            json.dump(m, open(bm_json, "w"), indent=2)
            per_bench[b] = m
            EC.log(f"[D] {b:14s} axis={m['axis']:12s} metric={m['metric']:.4f} "
                   f"n={m['n']} col={m['canonical_col']}", logf)

        result = {
            "id": args.item_id, "kind": it["kind"], "base_model": it["base_model"],
            "dataset": it.get("dataset"), "source": it.get("source"), "target": it.get("target"),
            "adapter_dir": it["adapter_dir"], "seed": it["seed"],
            "subsample_max_prompts": args.max_prompts, "subsample_seed": args.subsample_seed,
            "rtl_judge_model": EC.RTL_JUDGE_MODEL, "graders_enabled": sorted(enabled),
            "per_benchmark": per_bench,
        }
        final_tmp = final + f".tmp.{os.getpid()}"
        with open(final_tmp, "w") as fh:
            json.dump(result, fh, indent=2)
        os.replace(final_tmp, final)
        error_path = os.path.join(od, "ERROR.json")
        if os.path.exists(error_path):
            os.unlink(error_path)
        EC.log(f"=== DONE {args.item_id} in {(time.time()-t0)/60:.1f}m ===", logf)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        EC.log(f"=== ERROR {args.item_id}: {e} ===\n{tb[-2500:]}", logf)
        json.dump({"id": args.item_id, "status": "error", "error": str(e), "tb": tb[-2000:]},
                  open(os.path.join(od, "ERROR.json"), "w"), indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
