#!/usr/bin/env python3
"""Does the fingerprint direction we ablate correspond to what imitation adapters do?

The paper has two halves that do not currently touch: an activation-space result
(ablating the fingerprint direction leaves refusal intact) and a weight-space result
(imitation fine-tuning barely erodes safety). The bridge is an empirical question:

    Is the direction the LoRA adapter actually moves the model along the same
    direction the steering experiment ablates?

For a base model M and an imitation adapter A, holding the *text* fixed and changing
only the weights, the adapter-induced representational shift at layer L is

    s_L = mean_i [ h_L^{M+A}(x_i) - h_L^{M}(x_i) ]

over the benign contrast set x_i (the same prompts and base-model responses used to
derive the fingerprint). We report three quantities per (adapter, layer):

  cos(s, fingerprint)  -- does imitation move the model along the ablated axis?
  cos(s, random)       -- the null floor for that comparison.
  align(s, cone)       -- ||B B^T s|| / ||s||, the fraction of the adapter's shift that
                          lies inside the refusal cone. This is the sharper safety
                          question: does weight-level imitation traverse refusal-coding
                          dimensions at all? Compare against sqrt(k/d), the analytic
                          floor for a random direction.

Ministral-8B is the informative base: it is the only in-grid source with substantial
measured erosion (+3.37 pp), so if imitation ever moves a model along refusal-coding
dimensions, it should be visible here.

Usage:
    python experiments/steering/adapter_shift_bridge.py --base ministral-8b --limit 4
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import pandas as pd
import torch

RDO = os.environ.get("DEMENTOR_RDO_DIR", "/data/ethantsliu/exp_steer_safety/repl80_rdo")
HF_ORG = "dementor-research"
# Local weights, keyed by steering slug. Falls back to the HF id in vectors_ml.pt meta.
LOCAL_BASE = {
    "ministral-8b": "/data/ethantsliu/models_dl/Ministral-8B-Instruct-2410",
    "qwen3.5-4b": "/data/ethantsliu/models_dl/Qwen3.5-4B",
    "qwen3.6-35b": "/data/ethantsliu/models_dl/Qwen3.6-35B-A3B",
    "nemotron-nano": "/data/ethantsliu/models_dl/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "gpt-oss-120b": "/data/ethantsliu/models_dl/gpt-oss-120b",
}

# The steering roster and the imitation roster use different slugs for the same model.
# The run directory is keyed by the STEERING slug; the HuggingFace adapter repos are named
# with the IMITATION slug (dpo_<dataset>_<imit-slug>_as_<target>_<seed>).  Anything not
# listed here uses the same string for both.
IMIT_SLUG = {
    "nemotron-nano": "nemotron-nano-30b-a3b",
    "qwen3.6-35b": "qwen3.6-35b-a3b",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cone_align(vec: torch.Tensor, basis: torch.Tensor) -> float:
    v = vec.float()
    if v.norm() == 0:
        return float("nan")
    q, _ = torch.linalg.qr(basis.float().T)
    return float((q @ (q.T @ v)).norm() / v.norm())


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    if a.norm() == 0 or b.norm() == 0:
        return float("nan")
    return float((a @ b) / (a.norm() * b.norm()))


@torch.no_grad()
def mean_hidden(model, tok, rows: list[tuple[str, str]], layers: list[int], device: str) -> dict:
    """Mean last-token hidden state per layer over (prompt, response) pairs."""
    acc = {L: None for L in layers}
    n = 0
    for prompt, response in rows:
        msgs = [{"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": str(response)}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            text = f"{prompt}\n\n{response}"
        ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(
            model.device if device == "auto" else device)
        out = model(**ids, output_hidden_states=True)
        for L in layers:
            h = out.hidden_states[L][0, -1, :].detach().float().cpu()
            acc[L] = h if acc[L] is None else acc[L] + h
        n += 1
        del out
    return {L: (v / n) for L, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="ministral-8b", help="steering slug of the base model")
    ap.add_argument("--limit", type=int, default=4, help="how many adapters to evaluate")
    ap.add_argument("--n-prompts", type=int, default=120)
    ap.add_argument("--device", default="cuda", help="'cuda', 'cuda:N', or 'auto' for model-parallel")
    ap.add_argument("--imit-slug", default=None,
                    help="imitation-roster slug used in the adapter repo names, when it differs "
                         "from the steering slug (default: IMIT_SLUG map, else --base)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    slug = args.base
    run_dir = os.path.join(RDO, slug)
    out_path = args.out or os.path.join(run_dir, "adapter_shift_bridge.json")

    vec = torch.load(os.path.join(run_dir, "vectors_ml.pt"), map_location="cpu", weights_only=False)
    cone = torch.load(os.path.join(run_dir, "selected_cone.pt"), map_location="cpu", weights_only=False)
    basis, k = cone["basis"], cone["basis"].shape[0]
    layers = sorted(vec["vectors"].keys())
    base_id = LOCAL_BASE.get(slug, vec["meta"]["source"])
    log(f"base={slug} weights={base_id} layers={layers} cone_dim={k}")

    bcsv = os.path.join(run_dir, "benign.csv")
    df = pd.read_csv(bcsv).dropna(subset=["prompt", "model_response"]).head(args.n_prompts)
    rows = list(zip(df["prompt"], df["model_response"]))
    log(f"contrast set: {len(rows)} (prompt, response) pairs from {bcsv}")

    from huggingface_hub import HfApi
    api = HfApi()
    islug = args.imit_slug or IMIT_SLUG.get(slug, slug)
    cands = [m.id for m in api.list_models(author=HF_ORG, search=f"{islug}_as")]
    # Keep only adapters where this model is the fine-tuned SOURCE (dpo_<ds>_<SRC>_as_<TGT>).
    # Anchored on both sides so `qwen3.6-35b` does not also match `qwen3.6-35b-a3b`.
    cands = [c for c in cands if f"_{islug}_as_" in c]
    cands = sorted(cands)[: args.limit]
    if not cands:
        sys.exit(f"no adapters found on {HF_ORG} with {slug} as source")
    log(f"adapters ({len(cands)}): " + ", ".join(c.split('/')[-1] for c in cands))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log("loading base model ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.bfloat16, device_map=args.device)
    base.eval()
    log("computing base hidden means ...")
    h_base = mean_hidden(base, tok, rows, layers, args.device)

    results = {"base": slug, "base_weights": base_id, "cone_dim": k,
               "hidden_dim": int(basis.shape[1]), "n_prompts": len(rows),
               "analytic_cone_floor": round(math.sqrt(k / basis.shape[1]), 6),
               "adapters": {}}

    for repo in cands:
        name = repo.split("/")[-1]
        log(f"--- {name}")
        try:
            merged = PeftModel.from_pretrained(base, repo)
            merged.eval()
            h_ad = mean_hidden(merged, tok, rows, layers, args.device)
            per_layer = {}
            for L in layers:
                shift = h_ad[L] - h_base[L]
                fp, rnd = vec["vectors"][L]["fingerprint"], vec["vectors"][L].get("random")
                per_layer[str(L)] = {
                    "shift_norm": round(float(shift.norm()), 6),
                    "base_norm": round(float(h_base[L].norm()), 6),
                    "rel_shift": round(float(shift.norm() / h_base[L].norm()), 6),
                    "cos_shift_fingerprint": round(cos(shift, fp), 6),
                    "cos_shift_random": round(cos(shift, rnd), 6) if rnd is not None else None,
                    "align_shift_cone": round(cone_align(shift, basis), 6),
                }
                log(f"    L{L:<3d} |s|/|h|={per_layer[str(L)]['rel_shift']:.4f} "
                    f"cos(s,fp)={per_layer[str(L)]['cos_shift_fingerprint']:+.4f} "
                    f"cos(s,rand)={per_layer[str(L)]['cos_shift_random']:+.4f} "
                    f"align(s,cone)={per_layer[str(L)]['align_shift_cone']:.4f}")
            results["adapters"][name] = per_layer
            # Unload the adapter so the next one starts from clean base weights.
            merged.unload()
            del merged
        except Exception as exc:
            log(f"    !! failed: {exc}")
            results["adapters"][name] = {"error": str(exc)}
            # A partially-attached adapter leaves LoRA weights (and a stale peft_config) on the
            # base, and the NEXT from_pretrained then stacks on top of it -- every subsequent
            # shift would be measured against a contaminated reference.  Rebuild from disk.
            del base
            gc.collect()
            torch.cuda.empty_cache()
            base = AutoModelForCausalLM.from_pretrained(
                base_id, dtype=torch.bfloat16, device_map=args.device)
            base.eval()
            log("    (base reloaded after failure)")
        # from_pretrained leaves `peft_config` behind even after a clean unload(); drop it so the
        # next load starts from a bare base rather than warning about multiple adapters.
        if hasattr(base, "peft_config"):
            del base.peft_config
        gc.collect()
        torch.cuda.empty_cache()
        with open(out_path, "w") as fh:      # checkpoint after every adapter
            json.dump(results, fh, indent=2)

    log(f"wrote {out_path}")
    ok = {k_: v for k_, v in results["adapters"].items() if "error" not in v}
    if ok:
        for L in map(str, layers):
            cf = [v[L]["cos_shift_fingerprint"] for v in ok.values()]
            cr = [v[L]["cos_shift_random"] for v in ok.values() if v[L]["cos_shift_random"] is not None]
            ca = [v[L]["align_shift_cone"] for v in ok.values()]
            log(f"SUMMARY L{L}: cos(s,fp)={sum(cf)/len(cf):+.4f}  "
                f"cos(s,rand)={sum(cr)/len(cr):+.4f}  align(s,cone)={sum(ca)/len(ca):.4f}  "
                f"(cone floor {results['analytic_cone_floor']:.4f})")


if __name__ == "__main__":
    main()
