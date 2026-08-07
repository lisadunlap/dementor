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
sys.path.insert(0, os.path.join(HERE, "port"))
# This script LOADS MODELS, so it needs the same compat shims as every other model-loading entry
# point (rdo_port / cone_eval / compute_dim / run_rdo_model). Its wire-in list simply missed this
# file. Without it: gpt-oss-20b ships no chat_template and apply_chat_template hard-fails
# ("tokenizer.chat_template is not set"), granite-4-h-small's Mamba-2 mixer tries to resolve fused
# kernels from the hub and raises OfflineModeIsEnabled under HF_HUB_OFFLINE=1, and nemotron-nano's
# generate() indexes a None cache_position. Must precede the first from_pretrained call, which it
# does -- the shims hook the AutoTokenizer/AutoModel classmethods at call time.
import rdo_compat                   # noqa: E402,F401  (chat_template / kernel / generation shims)
import steer_config as CFG          # noqa: E402
import run_model as RM              # noqa: E402  (LAYERS, batch_sizes, EXP)
sys.path.insert(0, os.path.join(CFG.REPO, ""))
from dementor.steering.steering_rung import derive_steering_vector  # noqa: E402

RDO = os.environ.get("DEMENTOR_STEER_WORK", "/data/ethantsliu/exp_steer_safety/repl80_rdo")
REPL80 = os.environ.get("DEMENTOR_STEER_REPL80", "/data/ethantsliu/exp_steer_safety/repl80")


def _benign_csv(slug: str) -> str:
    """Locate <slug>/benign.csv, preferring the repl80 tree and falling back to the RDO tree.

    repl80 holds benign.csv for only 15 models; repl80_rdo holds 40, including llama-3.3-70b -- the
    model where the depth sweep matters most (fixed layer 14 lands in its first third). Where both
    trees have the file they are BYTE-IDENTICAL for all 15 overlapping models, so the fallback picks
    up the same authoritative artifact rather than a regenerated one: no provenance change.
    """
    primary = os.path.join(REPL80, slug, "benign.csv")
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(RDO, slug, "benign.csv")
    if os.path.exists(fallback):
        print(f"[depths] benign.csv not in repl80; using identical RDO-tree copy: {fallback}")
        return fallback
    raise FileNotFoundError(
        f"no benign.csv for {slug} in {REPL80} or {RDO} -- it is a recorded derivation input and "
        f"must be transferred, never regenerated (that would change the fingerprint's provenance)")
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
        mben = pd.read_csv(_benign_csv(slug))[
            ["prompt", "model_response"]].rename(columns={"model_response": "m"})
        ref = pd.read_csv(RM.EXP + ref_csv)[["prompt", "model_response"]].rename(
            columns={"model_response": "ref"})
        m = mben.merge(ref, on="prompt").dropna().drop_duplicates("prompt").reset_index(drop=True)
        prompts = m["prompt"].astype(str).tolist()
        # DEMENTOR_DERIVE_BS overrides the params_b-derived batch. The table is a per-family
        # heuristic, and a MoE with all experts resident (gpt-oss-20b: 20B -> bs=4) can still fill an
        # 80GB card and OOM during pooling. Lowering the batch only costs wall-clock: the derivation
        # is a one-off, and pooling is order-independent, so the vectors are unchanged.
        bs = int(os.environ.get("DEMENTOR_DERIVE_BS") or RM.batch_sizes(spec["params_b"])["derive"])
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
