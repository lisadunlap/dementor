#!/usr/bin/env python
"""Per-model MULTI-BENCHMARK dissociation eval driver (runs LATER, on GPU; RDO roster owns 5/6/7).

For a model slug that already has selected_cone.pt + vectors_ml.pt (produced by run_rdo_model.py),
run cone_eval.py once per benchmark so we get, per model:
  * HARM axis (RTL judge + Arditi refusal substring), for cone(refusal) / fingerprint / random
    ablation, on EACH harmful benchmark: advbench, sorrybench, sgbench, harmbench, strongreject
  * OVER-REFUSAL axis (benign wrongly-refused), on: orbench_hard, orbench_80k, xstest
    (orbench_toxic is the harmful decoy control that SHOULD still be refused)
Optional native graders: HarmBench-cls on harmbench, StrongREJECT rubric on strongreject.

Writes repl80_rdo/<slug>/eval_<benchmark>/metrics.json.  Resumable (skips finished benchmarks).
Does NOT touch run_rdo_model.py's advbench eval/ dir.

Usage: run_benchmark_eval.py <slug> [--benchmarks a,b,c] [--betas ...] [--max-prompts N]
       CUDA_VISIBLE_DEVICES set by caller.
"""
import os, sys, json, time, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))   # steering package dir (port/ lives here)
PORT = os.path.join(HERE, "port")
sys.path.insert(0, HERE)
import steer_config as CFG
ROOT = CFG.WORK_ROOT   # worklist + per-model <slug>/ outputs (env DEMENTOR_STEER_WORK)
PY = CFG.PY

# EVALUATION ORDER: the 5 STANDARD benchmarks run FIRST so their results land first, then the
# 2025 newcomers (OR-Bench + SG-Bench) run LAST.  orbench_toxic is the harmful decoy control that
# SHOULD stay refused; orbench_80k is the large over-refusal split -- both ride at the tail with
# the OR-Bench family.
STANDARD = ["advbench", "harmbench", "strongreject", "xstest", "sorrybench"]  # 5 standard, FIRST
TAIL = ["orbench_hard", "orbench_toxic", "orbench_80k", "sgbench"]  # OR-Bench family + SG-Bench, LAST
# PHASE 1 (2026-07-09): standard-5 ONLY, so all 32 models clear the core refusal benchmarks before
# any model spends a card on the slow OR-Bench grader (~48min/model). PHASE 2: restore
# `DEFAULT = STANDARD + TAIL` and delete the per-model benchmarks_summary.json markers so the daemon
# re-runs and cone_eval fills in the cached-skipping TAIL benchmarks.
DEFAULT = STANDARD
# CANONICAL native grader per benchmark -- 100% LOCAL in the default path (ZERO OpenAI calls).
# The OpenAI cross-checks stay wired but DORMANT behind USE_OPENAI_GRADERS=1.
#   advbench     -> refusal-substring ASR (canonical in cone_eval; no extra flag)
#   harmbench    -> cais/HarmBench-Llama-2-13b-cls (local)
#   strongreject -> qylu4156 fine-tuned gemma-2b judge (local); [dormant] GPT rubric cross-check
#   sorrybench   -> sorry-bench ft-mistral-7b (HF-gated 403) -> LOCAL RTL-judge fallback
#   sgbench      -> meta-llama/Llama-Guard-3-8B (local) -> LOCAL RTL-judge fallback
#   orbench_*    -> LOCAL 3-way refusal classifier on the RTL-judge model (Qwen3-8B)
#   xstest       -> string-match (in cone_eval) + LOCAL 3-way classifier (Qwen3-8B)
NATIVE = {
    "harmbench": ["--harmbench-grader"],
    "strongreject": ["--strongreject-grader", "--strongreject-rubric"],
    "sorrybench": ["--sorrybench-grader"],
    "sorrybench_variants": ["--sorrybench-grader"],
    "sgbench": ["--sgbench-grader"],
    "orbench_hard": ["--orbench-grader"],
    "orbench_80k": ["--orbench-grader"],
    "orbench_toxic": ["--orbench-grader"],
    "xstest": ["--xstest-grader"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--benchmarks", default=",".join(DEFAULT))
    ap.add_argument("--betas", default="0.6,1.0,1.4")
    # 200 is the standard for new steering cells and exactly harmonized rescores. Some legacy base
    # cells retain disclosed native n=300/n=100 denominators because their cached generations do
    # not contain the complete newer prompt selection; see base_steering_coverage.json.
    ap.add_argument("--max-prompts", type=int, default=200)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--adapter", default=None,
                    help="LoRA imitation adapter to merge into the base before steering "
                         "(cone/fingerprint/random stay the STOCK model's directions). "
                         "REQUIRES --outdir: adapter runs must not land in the canonical "
                         "roster tree, or rebuild_steering_tables would fold them into the "
                         "base model's cells.")
    ap.add_argument("--cone", default=None,
                    help="override the cone .pt (default: WORK_ROOT/<slug>/selected_cone.pt). "
                         "Use with --outdir to evaluate an alternative cone -- e.g. a wider-k "
                         "retry fit -- without overwriting the canonical roster cell.")
    ap.add_argument("--vectors-ml", dest="vectors_ml", default=None,
                    help="override vectors_ml.pt (default: WORK_ROOT/<slug>/vectors_ml.pt)")
    ap.add_argument("--no-controls", action="store_true",
                    help="Cone + baseline arms ONLY: skip the fingerprint/random single-direction "
                         "controls. Those are the only arms that depend on DEMENTOR_ABLATE_LAYER "
                         "(the cone is ablated at EVERY layer), so when the derivation depth is "
                         "still undecided they are the only arms that would have to be redone. "
                         "Emits the 4 layer-independent arms now and defers the 6 layer-dependent "
                         "ones to a single pass at the chosen depth, in a SEPARATE --outdir (parts "
                         "are cached as '<direction>_b<beta>.csv' with no layer tag, so reusing an "
                         "outdir would silently serve stale-layer controls).")
    ap.add_argument("--outdir", default=None,
                    help="base output dir for eval_<bench>/ (default: WORK_ROOT/<slug>). "
                         "Cone and vectors are still read from WORK_ROOT/<slug>.")
    args = ap.parse_args()
    if args.adapter and not args.outdir:
        sys.exit("--adapter requires --outdir (keep adapter runs out of the canonical roster tree)")

    wl = CFG.load_worklist()   # repo/scratch fallback + per-box model-path resolution
    spec = next((m for m in wl["models"] if m["slug"] == args.slug), None)
    if spec is None:
        sys.exit(f"unknown slug {args.slug}")
    src = os.path.join(ROOT, args.slug)            # where the STOCK artifacts live
    od = args.outdir or src                        # where THIS run's outputs go
    os.makedirs(od, exist_ok=True)
    cone = args.cone or os.path.join(src, "selected_cone.pt")
    vml = args.vectors_ml or os.path.join(src, "vectors_ml.pt")
    if (args.cone or args.vectors_ml) and not args.outdir:
        sys.exit("--cone/--vectors-ml require --outdir (do not overwrite the canonical cell)")
    prereqs = (cone,) if args.no_controls else (cone, vml)
    for f in prereqs:
        if not os.path.exists(f):
            sys.exit(f"missing prerequisite {f} (run run_rdo_model.py {args.slug} first)")

    # Adaptive generation batch: small dense models have ample VRAM headroom at 80GB, so a bigger
    # batch saturates the card (single-model, single-card jobs otherwise sit ~50% util). Big/MoE
    # models (full weights resident) and MP models are memory-bound -> keep the conservative 16.
    # Only auto-tune when the caller left the default (an explicit --gen-batch still wins).
    if args.gen_batch == 16 and not spec.get("needs_mp"):
        pb = spec.get("params_b") or 8
        args.gen_batch = 48 if pb <= 14 else (24 if pb <= 20 else 16)
        print(f"[{args.slug}] adaptive gen-batch -> {args.gen_batch} (params_b={pb})", flush=True)

    env = dict(os.environ)
    # A bare box may launch this once with HF_HUB_OFFLINE=0 to populate its cache,
    # but a queued experiment must retain an explicit parent offline policy.  Do
    # not silently turn a cache miss into a network fetch between resumable cells.
    offline = os.environ.get("HF_HUB_OFFLINE", "1") in ("1", "true", "True")
    env.update(CFG.hf_env(offline=offline))  # HF_HOME/HF_HUB_CACHE/HF_HUB_DISABLE_XET/PYTHONPATH
    # propagate HF_TOKEN / OPENAI_API_KEY (gated local judges + OpenAI canonical graders) from repo .env
    repo_env = CFG.REPO_ENV
    if os.path.exists(repo_env):
        for line in open(repo_env):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    results = {}
    for bench in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        # advbench canonical result already lives in eval/; mirror to eval_advbench for a uniform table.
        out = os.path.join(od, f"eval_{bench}")
        mp = os.path.join(out, "metrics.json")
        if os.path.exists(mp):
            print(f"[{args.slug}] {bench}: cached", flush=True)
            results[bench] = json.load(open(mp)); continue
        cmd = [PY, os.path.join(PORT, "cone_eval.py"), "--model", spec["path"], "--cone", cone,
               "--out", out, "--betas", args.betas,
               "--benchmark", bench, "--gen-batch", str(args.gen_batch)]
        if not args.no_controls:
            cmd += ["--vectors-ml", vml]
        if args.adapter:
            cmd += ["--adapter", args.adapter]
        if args.max_prompts:
            cmd += ["--max-prompts", str(args.max_prompts)]
        cmd += NATIVE.get(bench, [])
        t0 = time.time()
        print(f"[{args.slug}] {bench}: START {' '.join(cmd[-6:])}", flush=True)
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        open(os.path.join(od, f"bench_{bench}.log"), "w").write(r.stdout or "")
        if r.returncode != 0 or not os.path.exists(mp):
            print(f"[{args.slug}] {bench}: FAILED rc={r.returncode} (see bench_{bench}.log)", flush=True)
            continue
        m = json.load(open(mp)); results[bench] = m
        print(f"[{args.slug}] {bench}: DONE {(time.time()-t0)/60:.1f}m axes={m.get('axes')} "
              f"verdict={m.get('verdict')} cone={m.get('refusal_matched')} fp={m.get('fingerprint_matched')} "
              f"rnd={m.get('random_matched')} over_refusal_cone={m.get('over_refusal_cone')}", flush=True)
    json.dump(results, open(os.path.join(od, "benchmarks_summary.json"), "w"), indent=2)
    print(f"[{args.slug}] wrote benchmarks_summary.json ({len(results)} benchmarks)", flush=True)


if __name__ == "__main__":
    main()
