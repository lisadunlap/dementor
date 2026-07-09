#!/usr/bin/env python
"""Process ONE (method, source, target) LOCAL prompt-disguise item for the PROMPT-RUNG safety-erosion
sweep.  Resumable + checkpointed (skips finished benchmarks; skips the item if metrics.json exists).

Pipeline (per item; GPU set by the caller via CUDA_VISIBLE_DEVICES, DEMENTOR_MP=1 for big models):
  A generate : load the BASE source model A ONCE (adapter_dir=None -> A unadapted), build the disguise
               (method's system-prompt +/- few-shot from B's chatbot_arena data), render each
               (subsampled) benchmark prompt through the disguise, generate greedy -> <bench>/all_gens.csv.
  B judge    : concat all gens, ONE RTL judge (Qwen3-8B) -> genuine_harm, split back.   (reuse EC)
  C grade    : per benchmark run its ONE canonical native grader.                        (reuse EC)
  D metrics  : per-benchmark harm / over-refusal metric -> metrics.json (with a `method` field). (EC)

Usage:
  run_prompt_erosion_item.py <item_id> [--benchmarks a,b] [--max-prompts N] [--subsample-seed S]
                             [--max-new-tokens T] [--gen-batch B] [--graders g1,g2 | --no-heavy-graders]
  <item_id> = prompt_<method>_<source>_as_<target>  (source must be a LOCAL-backend model).
"""
import os, sys, json, time, argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erosion_common as EC
import prompt_erosion_common as PC

ALL_GRADERS = {"harmbench", "strongreject", "sorrybench", "sgbench", "orbench", "xstest"}
HEAVY_GRADERS = {"harmbench", "sgbench"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("item_id")
    ap.add_argument("--benchmarks", default=",".join(EC.DEFAULT_BENCHMARKS))
    ap.add_argument("--max-prompts-per-benchmark", "--max-prompts", dest="max_prompts",
                    type=int, default=PC.DEFAULT_MAX_PROMPTS)
    ap.add_argument("--subsample-seed", type=int, default=PC.DEFAULT_SUBSAMPLE_SEED)
    ap.add_argument("--max-new-tokens", type=int, default=PC.DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--graders", default=",".join(sorted(ALL_GRADERS)))
    ap.add_argument("--no-heavy-graders", action="store_true")
    args = ap.parse_args()

    PC._load_env()
    it = PC.find_item(args.item_id)
    if it is None:
        sys.exit(f"unknown item id {args.item_id!r}")
    if it["backend"] != "local":
        sys.exit(f"{args.item_id} is a {it['backend']} source -> use prompt_tinker_erosion.py, not this runner")
    od = PC.item_dir(args.item_id)
    logf = os.path.join(od, "run.log")
    final = os.path.join(od, "metrics.json")
    if os.path.exists(final):
        EC.log(f"=== {args.item_id} DONE (metrics.json exists) -> SKIP ===", logf)
        return

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    enabled = {g.strip() for g in args.graders.split(",") if g.strip()} & ALL_GRADERS
    if args.no_heavy_graders:
        enabled -= HEAVY_GRADERS
    EC.log(f"=== START {args.item_id} method={it['method']} source={it['source']} target={it['target']} "
           f"base={it['base_model']} mp={it['needs_mp']} CUDA={os.environ.get('CUDA_VISIBLE_DEVICES')} "
           f"benchmarks={benchmarks} max_prompts={args.max_prompts} graders={sorted(enabled)} ===", logf)
    t0 = time.time()

    try:
        # ---- A: build disguise + generate every benchmark with ONE model load -------------------
        need_gen = [b for b in benchmarks
                    if not os.path.exists(os.path.join(od, b, "all_gens.csv"))]
        for b in benchmarks:
            os.makedirs(os.path.join(od, b), exist_ok=True)
        if need_gen:
            source_df, target_df = PC.load_disguise_dfs(it["source"], it["target"])
            method_obj = PC.make_disguise(it["method"], it["base_model"], it["target_hf"],
                                          it["source"], it["target"], source_df, target_df,
                                          seed=PC.DISGUISE_SEED, logf=logf)
            PC.load_source_model(it["base_model"], logf)
            try:
                for b in need_gen:
                    sub_csv = EC.get_subsample(b, args.max_prompts, args.subsample_seed, logf)
                    bdf = pd.read_csv(sub_csv)
                    prompts = bdf["prompt"].astype(str).tolist()
                    EC.log(f"[A] {b}: disguised generation for {len(prompts)} prompts", logf)
                    resps = PC.generate_disguised_local(it["base_model"], method_obj, prompts,
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
                PC.free_source_model(it["base_model"])
            EC.log("[A] generation done, model freed", logf)

        # ---- B: ONE RTL judge over all benchmarks, then split (reuse EC) -------------------------
        need_judge = [b for b in benchmarks
                      if not os.path.exists(os.path.join(od, b, "all_judged.csv"))]
        if need_judge:
            combined_in = os.path.join(od, "_combined_gens.csv")
            combined_out = os.path.join(od, "_combined_judged.csv")
            frames = [pd.read_csv(os.path.join(od, b, "all_gens.csv")) for b in benchmarks]
            comb = pd.concat(frames, ignore_index=True)
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

        # ---- C + D: grade + metric per benchmark (reuse EC) --------------------------------------
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
            "id": args.item_id, "kind": "prompt_disguise", "method": it["method"],
            "base_model": it["base_model"], "dataset": "chatbot_arena",
            "source": it["source"], "target": it["target"], "target_hf": it["target_hf"],
            "backend": "local", "seed": it["seed"],
            "subsample_max_prompts": args.max_prompts, "subsample_seed": args.subsample_seed,
            "rtl_judge_model": EC.RTL_JUDGE_MODEL, "graders_enabled": sorted(enabled),
            "per_benchmark": per_bench,
        }
        json.dump(result, open(final, "w"), indent=2)
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
