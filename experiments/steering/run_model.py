#!/usr/bin/env python
"""Benign-generation + fingerprint-derivation stages, imported by run_rdo_model.py (stages 1-2).

run_rdo_model.py reuses two stages from this module when a model's cached artifacts are absent:
  stage_benign : M's greedy benign responses on train_benign prompts (reuse an existing CSV if the
                 spec names one via `benign_reuse`) -> <od>/benign.csv.
  stage_derive : identity FINGERPRINT_M = diff-of-means(M-benign vs reference-benign) in M's residual
                 space at layers [4,8,14,20], plus a per-layer norm-matched random control ->
                 <od>/vectors_ml.pt. Layers are clamped to the model's depth for shallow models.

Not a standalone entry point: it exposes these two helpers for the RDO cone driver. The method
mirrors the validated qwen2.5-7b single-model derive pipeline (SAME operator, SAME layers).
"""
import os, sys, gc, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)               # so `import steer_config` works standalone + when imported
import steer_config as CFG             # central path/env config (env-overridable, in-repo defaults)
# Steering INPUT DATA dir (train_benign.csv / gen/*_benign.csv). Env DEMENTOR_STEER_DATA;
# default: the in-repo experiments/steering/data/ dir shipped alongside this module (live /data fallback).
EXP = CFG.STEER_DATA.rstrip("/") + "/"
LAYERS = [4, 8, 14, 20]


def batch_sizes(pb):
    if pb <= 4:   return dict(gen=32, derive=16, geom=32)
    if pb <= 9:   return dict(gen=24, derive=8,  geom=16)
    if pb <= 15:  return dict(gen=16, derive=6,  geom=12)
    if pb <= 22:  return dict(gen=12, derive=4,  geom=8)
    if pb <= 30:  return dict(gen=8,  derive=3,  geom=6)
    return dict(gen=6, derive=2, geom=4)


def log(od, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(od, "run.log"), "a") as f:
        f.write(line + "\n")


def free():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def render_prompt(tok, p):
    """Chat-template render with add_generation_prompt; force non-thinking when supported."""
    msgs = [{"role": "user", "content": str(p)}]
    for kw in ({"enable_thinking": False}, {"reasoning_effort": "low"}, {}):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError:
            continue
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# =========================================================================
# STAGE 1 : benign responses of M
# =========================================================================
def stage_benign(spec, od):
    import pandas as pd, torch
    from dementor.steering._common import load_causal_lm
    out = os.path.join(od, "benign.csv")
    if os.path.exists(out):
        log(od, f"[A benign] cached {out}")
        return
    reuse = spec.get("benign_reuse")
    tb = pd.read_csv(EXP + "train_benign.csv")
    prompts = tb["prompt"].astype(str).tolist()
    if reuse:
        src = pd.read_csv(EXP + reuse)[["prompt", "model_response"]]
        m = pd.DataFrame({"prompt": prompts}).merge(src, on="prompt", how="left")
        m["model_response"] = m["model_response"].fillna("").replace("", " ")
        m.to_csv(out, index=False)
        log(od, f"[A benign] reused {reuse} n={len(m)}")
        return
    log(od, f"[A benign] generating M benign n={len(prompts)} path={spec['path']}")
    tok, mdl, dev = load_causal_lm(spec["path"], padding_side="left", device="cuda", dtype="auto")
    bs = batch_sizes(spec["params_b"])["gen"]
    rendered = [render_prompt(tok, p) for p in prompts]
    resps = [""] * len(prompts)
    with torch.no_grad():
        for s in range(0, len(rendered), bs):
            enc = tok(rendered[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = mdl.generate(**enc, max_new_tokens=320, do_sample=False, use_cache=False,
                             pad_token_id=tok.pad_token_id)
            for j, t in enumerate(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                resps[s + j] = t.strip()
    df = pd.DataFrame({"prompt": prompts, "model_response": resps})
    df["model_response"] = df["model_response"].fillna("").replace("", " ")
    df.to_csv(out, index=False)
    log(od, f"[A benign] wrote {out} empty={int((df.model_response.str.strip()=='').sum())}")
    del mdl, tok
    free()


# =========================================================================
# STAGE 2 : derive fingerprint_M + random control
# =========================================================================
def stage_derive(spec, od):
    import pandas as pd, torch
    from dementor.steering.steering_rung import derive_steering_vector
    out = os.path.join(od, "vectors_ml.pt")
    if os.path.exists(out):
        log(od, f"[B derive] cached {out}")
        return
    ref_csv = {"llama": "gen/llama_benign.csv", "qwen": "train_benign.csv"}[spec["ref"]]
    mben = pd.read_csv(os.path.join(od, "benign.csv"))[["prompt", "model_response"]].rename(columns={"model_response": "m"})
    ref = pd.read_csv(EXP + ref_csv)[["prompt", "model_response"]].rename(columns={"model_response": "ref"})
    m = mben.merge(ref, on="prompt").dropna().drop_duplicates("prompt").reset_index(drop=True)
    prompts = m["prompt"].astype(str).tolist()
    bs = batch_sizes(spec["params_b"])["derive"]
    # Clamp fingerprint layers to the model's actual depth so shallow models load (e.g. olmoe-1b-7b
    # has only 16 layers vs the hardcoded L20). No-op for models with >=21 layers. ABLATE_LAYER (14),
    # which cone_eval reads, survives for any model with >=15 layers.
    try:
        from transformers import AutoConfig
        _cfg = AutoConfig.from_pretrained(spec["path"], trust_remote_code=True)
        _nl = getattr(_cfg, "num_hidden_layers", None) or getattr(_cfg, "n_layer", None)
    except Exception:
        _nl = None
    use_layers = LAYERS if not _nl else sorted({max(0, min(int(L), int(_nl) - 1)) for L in LAYERS})
    if use_layers != LAYERS:
        log(od, f"[B derive] clamped fingerprint layers {LAYERS} -> {use_layers} (model depth {_nl})")
    log(od, f"[B derive] fingerprint = M-vs-{spec['ref']} ; aligned n={len(prompts)} derive_bs={bs}")
    vl = derive_steering_vector(spec["path"], None, prompts,
                                source_responses=m["m"].astype(str).tolist(),
                                target_responses=m["ref"].astype(str).tolist(),
                                layers=use_layers, batch_size=bs)
    vecs, meta = {}, {"source": spec["path"], "ref": spec["ref"], "ref_csv": ref_csv,
                      "layers": use_layers, "n_prompts": len(prompts)}
    for L in use_layers:
        g = torch.Generator().manual_seed(100 + L)
        vr = torch.randn(vl[L].shape, generator=g)
        vr = vr / vr.norm() * vl[L].norm()
        vecs[L] = {"fingerprint": vl[L].float(), "random": vr.float()}
        meta[f"norm_fp_L{L}"] = float(vl[L].norm())
    torch.save({"vectors": vecs, "meta": meta}, out)
    norms = ", ".join("L%d:%.2f" % (L, meta["norm_fp_L%d" % L]) for L in use_layers)
    log(od, "[B derive] wrote %s norms={%s}" % (out, norms))
    free()
