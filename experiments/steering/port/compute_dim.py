#!/usr/bin/env python
"""Model-agnostic DIM (diff-of-means refusal direction) computer, for roster models NOT supported by
their refusal_direction pipeline (only gemma/qwen/llama). Mirrors Arditi generate+select but
simplified: last-token diff-of-means per candidate layer, layer selected by which single-direction
all-layer ablation most erodes harmful-val bypass while keeping harmless bypass sane.

Produces the same artifacts rdo_port consumes:
    <out>/direction.pt            (raw diff-of-means at best layer; its norm = alpha)
    <out>/direction_metadata.json {"pos": -1, "layer": L}

Usage: compute_dim.py --model <path> --out <dir> [--family auto] [--n-train 400 --n-val 64]
"""
import os, sys, json, argparse
import torch

_PORT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PORT)
sys.path.insert(0, os.path.dirname(_PORT))    # steering package dir (steer_config)
import steer_config as CFG
import rdo_compat  # transformers-5.5.4 compat shims (LossKwargs / chat_template / nemotron gen)
sys.path.insert(0, CFG.REPO)  # repo root for `dementor` imports (env DEMENTOR_REPO)
from dementor.steering._common import load_causal_lm, get_transformer_layers
import rdo_port as RP


def last_tok_means(model, tok, prompts, layers_idx, device, bs):
    tok.padding_side = "left"
    sums = {L: None for L in layers_idx}; n = 0
    with torch.no_grad():
        for s in range(0, len(prompts), bs):
            batch = prompts[s:s + bs]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            for L in layers_idx:
                v = hs[L + 1][:, -1, :].double().cpu().sum(0)
                sums[L] = v if sums[L] is None else sums[L] + v
            n += len(batch)
    return {L: (sums[L] / n).float() for L in layers_idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--family", default="auto")
    ap.add_argument("--splits-dir", default=CFG.SPLITS_DIR)
    ap.add_argument("--n-train", type=int, default=400); ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--gen-batch", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "direction.pt")):
        RP.log("[compute_dim] cached -> SKIP"); return

    tok, model, device = RP.load_model_mp_aware(args.model)  # MP-aware (device_map when DEMENTOR_MP=1)
    model.requires_grad_(False)
    layers = get_transformer_layers(model); n_layers = len(layers)
    family = RP.family_of(args.model, args.family)
    render = RP.make_render(family, tok)
    refusal_toks = RP.refusal_token_ids(family, tok)
    RP.log(f"[compute_dim] family={family} n_layers={n_layers} refusal_toks={refusal_toks}")

    import json as _json
    ht = _json.load(open(os.path.join(args.splits_dir, "harmful_train.json")))
    hl = _json.load(open(os.path.join(args.splits_dir, "harmless_train.json")))
    hv = _json.load(open(os.path.join(args.splits_dir, "harmful_val.json")))
    lv = _json.load(open(os.path.join(args.splits_dir, "harmless_val.json")))
    h_ins = [d["instruction"] for d in ht]; l_ins = [d["instruction"] for d in hl][:len(h_ins)]
    hv_ins = [d["instruction"] for d in hv]; lv_ins = [d["instruction"] for d in lv][:len(hv_ins)]

    # filter train by bypass
    hp = [render(x) for x in h_ins]; lp = [render(x) for x in l_ins]
    hs_ = RP.bypass_scores(model, tok, hp, refusal_toks, device, args.gen_batch)
    ls_ = RP.bypass_scores(model, tok, lp, refusal_toks, device, args.gen_batch)
    hs_l = hs_.tolist(); ls_l = ls_.tolist()
    fh = [i for i, s in enumerate(hs_l) if s > 0]
    fl = [i for i, s in enumerate(ls_l) if s < 0]
    # fallback for models with weak/atypical refusal-token signal (e.g. reasoning distills):
    # if the sign filter keeps too few, rank by bypass score instead so diff-of-means is well-defined
    if len(fh) < 64:
        RP.log(f"[compute_dim] weak harmful refusal signal (kept {len(fh)}); using top-bypass harmful")
        fh = sorted(range(len(hs_l)), key=lambda i: -hs_l[i])[:args.n_train]
    else:
        fh = fh[:args.n_train]
    if len(fl) < 64:
        RP.log(f"[compute_dim] weak harmless signal (kept {len(fl)}); using lowest-bypass harmless")
        fl = sorted(range(len(ls_l)), key=lambda i: ls_l[i])[:args.n_train]
    else:
        fl = fl[:args.n_train]
    hp_f = [hp[i] for i in fh]; lp_f = [lp[i] for i in fl]
    RP.log(f"[compute_dim] train kept {len(hp_f)}/{len(lp_f)}")

    # candidate layers: ~10 across depth (skip very first/last)
    grid = sorted(set(int(round(f * (n_layers - 1))) for f in
                      [i / 12 for i in range(2, 12)]))
    grid = [L for L in grid if 0 < L < n_layers - 1]
    hm = last_tok_means(model, tok, hp_f, grid, device, args.gen_batch)
    lm = last_tok_means(model, tok, lp_f, grid, device, args.gen_batch)
    dirs = {L: (hm[L] - lm[L]) for L in grid}

    # select layer by ablation bypass on val
    hvp = [render(x) for x in hv_ins][:args.n_val]
    lvp = [render(x) for x in lv_ins][:args.n_val]
    ops = RP.ConeOps(model)
    best_L, best_score = None, None
    with torch.no_grad():
        for L in grid:
            d = dirs[L].to(device)
            ops.ablate_on = True; ops.add_on = False; ops.set_direction(d)
            hb = RP.bypass_scores(model, tok, hvp, refusal_toks, device, args.gen_batch).mean().item()
            lb = RP.bypass_scores(model, tok, lvp, refusal_toks, device, args.gen_batch).mean().item()
            ops.ablate_on = False
            # want low harmful bypass (erosion), but not destroying harmless (lb not too high)
            score = hb + 0.5 * max(0.0, lb)
            RP.log(f"[compute_dim] L{L}: harmful_bypass={hb:+.3f} harmless_bypass={lb:+.3f} score={score:+.3f}")
            if best_score is None or score < best_score:
                best_score, best_L = score, L
    ops.remove()
    direction = dirs[best_L]
    json.dump({"pos": -1, "layer": int(best_L)}, open(os.path.join(args.out, "direction_metadata.json"), "w"), indent=2)
    torch.save(direction, os.path.join(args.out, "direction.pt"))
    RP.log(f"[compute_dim] DONE best_layer={best_L} |dir|={float(direction.norm()):.3f}")


if __name__ == "__main__":
    main()
