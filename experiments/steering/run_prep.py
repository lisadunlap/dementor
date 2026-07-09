#!/usr/bin/env python
"""Model-intrinsic direction PREP for the RDO roster (does NOT depend on the validated cone):
per model runs stage 1-2 (benign + fingerprint M-vs-llama + random @ L14) and stage 3 (DIM
diff-of-means). Loops the worklist prep_order, resumable (skips models whose dim/ + vectors_ml.pt
already exist). Cone training / eval are handled later by run_rdo_model.py once validation passes.

Usage: run_prep.py [slug1 slug2 ...]   (default: worklist prep_order; skips needs_mp models)
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_rdo_model as RRM
ROOT = RRM.ROOT   # worklist + per-model <slug>/ outputs (env DEMENTOR_STEER_WORK)


def prep_one(spec):
    slug = spec["slug"]
    od = os.path.join(ROOT, slug); os.makedirs(od, exist_ok=True)
    dim_ok = os.path.exists(os.path.join(od, "dim", "direction.pt"))
    fp_ok = os.path.exists(os.path.join(od, "vectors_ml.pt"))
    if dim_ok and fp_ok:
        RRM.log(od, f"[prep] {slug} already prepped -> SKIP"); return True
    RRM.log(od, f"=== PREP {slug} path={spec['path']} CUDA={os.environ.get('CUDA_VISIBLE_DEVICES')} ===")
    t0 = time.time()
    try:
        RRM.stage_benign_fingerprint(spec, od)
        RRM.stage_dim(spec, od)
        RRM.log(od, f"=== PREP DONE {slug} in {(time.time()-t0)/60:.1f}m ===")
        return True
    except Exception as e:
        import traceback
        RRM.log(od, f"=== PREP ERROR {slug}: {e} ===\n{traceback.format_exc()[-2000:]}")
        return False


def main():
    wl = RRM.CFG.load_worklist()   # repo/scratch fallback + per-box model-path resolution
    by_slug = {m["slug"]: m for m in wl["models"]}
    slugs = sys.argv[1:] or wl["prep_order"]
    print(f"[prep] queue: {slugs}", flush=True)
    for slug in slugs:
        spec = by_slug.get(slug)
        if spec is None:
            print(f"[prep] unknown slug {slug}"); continue
        if spec.get("needs_mp"):
            print(f"[prep] {slug} needs_mp -> skip on single GPU"); continue
        prep_one(spec)
    print("[prep] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
