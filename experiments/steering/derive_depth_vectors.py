#!/usr/bin/env python
"""Derive fingerprint + random directions at RELATIVE depths (25% / 50% / 75%).

Robustness companion to the campaign's fixed absolute grid {4, 8, 14, 20}: that grid
is inherited from prior work on 32-layer models and lands anywhere from 17.5% to
87.5% of depth across the 28-model roster. This script derives the same M-vs-reference
diff-of-means fingerprint at three depth-proportional layers per model, so the
depth-sweep evals can report a worst-case-over-derivation-depth bound.

Conventions are IDENTICAL to run_model.stage_derive -- same benign.csv / reference
CSV pairing, same derive_steering_vector pooling, same norm-matched random with
generator seed (100 + L) -- so a depth layer that happens to coincide with an
existing grid layer produces byte-identical vectors; those are copied from
vectors_ml.pt instead of re-derived.

Output: <RDO>/<slug>/vectors_depths.pt   {"vectors": {L: {fingerprint, random}},
                                          "meta": {..., "depth_fracs": {L: frac}}}

Usage: derive_depth_vectors.py <slug>   (CUDA_VISIBLE_DEVICES set by caller)
"""
import os
import sys
import json

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import steer_config as CFG          # noqa: E402
import run_model as RM              # noqa: E402  (LAYERS, batch_sizes, EXP)
sys.path.insert(0, os.path.join(CFG.REPO, ""))
from dementor.steering.steering_rung import derive_steering_vector  # noqa: E402

RDO = os.environ.get("DEMENTOR_STEER_WORK", "/data/ethantsliu/exp_steer_safety/repl80_rdo")
REPL80 = "/data/ethantsliu/exp_steer_safety/repl80"
DEPTH_FRACS = (0.25, 0.50, 0.75)


def spec_for(slug):
    wl = json.load(open(os.path.join(HERE, "rdo_worklist.json")))
    for m in wl["models"]:
        if m["slug"] == slug:
            return m
    raise SystemExit(f"unknown slug {slug}")


def n_layers(path):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    nl = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    if nl is None:
        tc = getattr(cfg, "text_config", None)
        nl = getattr(tc, "num_hidden_layers", None) if tc is not None else None
    if not nl:
        raise SystemExit(f"cannot determine depth for {path}")
    return int(nl)


def main():
    slug = sys.argv[1]
    spec = spec_for(slug)
    out_path = os.path.join(RDO, slug, "vectors_depths.pt")
    if os.path.exists(out_path):
        print(f"[depths] cached {out_path}")
        return

    nl = n_layers(spec["path"])
    # depth layer = frac of nl, clamped to a valid hookable index
    want = {max(1, min(nl - 1, round(f * nl))): f for f in DEPTH_FRACS}
    print(f"[depths] {slug}: {nl} layers -> " +
          ", ".join(f"L{L} ({int(100*f)}%)" for L, f in sorted(want.items())))

    ml_path = os.path.join(RDO, slug, "vectors_ml.pt")
    ml = torch.load(ml_path, map_location="cpu", weights_only=False)
    have = ml["vectors"]

    vecs, meta = {}, {"source": spec["path"], "ref": spec["ref"], "n_layers": nl,
                      "depth_fracs": {int(L): f for L, f in want.items()},
                      "derived_from": "vectors_ml.pt reuse + fresh derive",
                      "convention": "run_model.stage_derive (seed 100+L random)"}

    missing = [L for L in want if L not in have]
    for L in want:
        if L in have:
            vecs[L] = {k: v.float() for k, v in have[L].items()}
            print(f"[depths]   L{L}: reused from vectors_ml.pt")

    if missing:
        ref_csv = {"llama": "gen/llama_benign.csv", "qwen": "train_benign.csv"}[spec["ref"]]
        mben = pd.read_csv(os.path.join(REPL80, slug, "benign.csv"))[
            ["prompt", "model_response"]].rename(columns={"model_response": "m"})
        ref = pd.read_csv(RM.EXP + ref_csv)[["prompt", "model_response"]].rename(
            columns={"model_response": "ref"})
        m = mben.merge(ref, on="prompt").dropna().drop_duplicates("prompt").reset_index(drop=True)
        prompts = m["prompt"].astype(str).tolist()
        bs = RM.batch_sizes(spec["params_b"])["derive"]
        print(f"[depths]   deriving L{missing} (n={len(prompts)}, bs={bs})")
        vl = derive_steering_vector(spec["path"], None, prompts,
                                    source_responses=m["m"].astype(str).tolist(),
                                    target_responses=m["ref"].astype(str).tolist(),
                                    layers=missing, batch_size=bs)
        for L in missing:
            g = torch.Generator().manual_seed(100 + L)
            vr = torch.randn(vl[L].shape, generator=g)
            vr = vr / vr.norm() * vl[L].norm()
            vecs[L] = {"fingerprint": vl[L].float(), "random": vr.float()}

    for L in vecs:
        meta[f"norm_fp_L{L}"] = float(vecs[L]["fingerprint"].norm())
    torch.save({"vectors": vecs, "meta": meta}, out_path)
    print(f"[depths] wrote {out_path}  norms=" +
          ", ".join("L%d:%.3f" % (L, meta['norm_fp_L%d' % L]) for L in sorted(vecs)))


if __name__ == "__main__":
    main()
