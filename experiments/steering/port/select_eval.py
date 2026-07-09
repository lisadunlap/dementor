#!/usr/bin/env python
"""Select the RDO cone dim (max harmful-val ablation erosion) from a cones/ dir, then run the full
RTL-judged coherence-matched erosion eval (cone_eval.py) on the selected dim. Used for validation
(their cone vs port cone) and reusable by the roster driver.

Usage: select_eval.py --model <path> --cones-dir <dir> --out <dir> --vectors-ml <pt> [--family auto]
"""
import os, sys, json, glob, shutil, subprocess, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))     # steering package dir (steer_config)
import steer_config as CFG
sys.path.insert(0, CFG.REPO)  # repo root for `dementor` imports (env DEMENTOR_REPO)
import rdo_port as RP
from dementor.steering._common import load_causal_lm, get_transformer_layers
from cone_eval import register_cone
PY = sys.executable


def select_dim(model_path, cones_dir, out, family):
    sel = os.path.join(out, "selected_cone.pt")
    if os.path.exists(sel):
        return json.load(open(os.path.join(out, "selection.json")))["selected_dim"]
    splits = CFG.SPLITS_DIR
    hv = [d["instruction"] for d in json.load(open(os.path.join(splits, "harmful_val.json")))][:64]
    lv = [d["instruction"] for d in json.load(open(os.path.join(splits, "harmless_val.json")))][:64]
    tok, model, dev = load_causal_lm(model_path, padding_side="left", device="cuda", dtype="auto")
    model.requires_grad_(False)
    fam = RP.family_of(model_path, family)
    render = RP.make_render(fam, tok); rtoks = RP.refusal_token_ids(fam, tok)
    hvp = [render(x) for x in hv]; lvp = [render(x) for x in lv]
    base_h = RP.bypass_scores(model, tok, hvp, rtoks, dev, 16).mean().item()
    results = {}
    for f in sorted(glob.glob(os.path.join(cones_dir, "cone_dim_*.pt"))):
        k = int(os.path.basename(f).split("_")[-1].split(".")[0])
        basis = torch.load(f, map_location="cpu")["basis"].float()
        handles = register_cone(model, basis, 1.0)
        try:
            hb = RP.bypass_scores(model, tok, hvp, rtoks, dev, 16).mean().item()
            lb = RP.bypass_scores(model, tok, lvp, rtoks, dev, 16).mean().item()
        finally:
            for h in handles:
                h.remove()
        results[k] = {"harmful_bypass_base": base_h, "harmful_bypass_ablated": hb,
                      "harmless_bypass_ablated": lb, "erosion": base_h - hb}
        RP.log(f"[select] dim {k}: harmful_bypass {base_h:+.3f}->{hb:+.3f} (erosion {base_h-hb:+.3f}) harmless_abl {lb:+.3f}")
    del model, tok
    import gc; gc.collect(); torch.cuda.empty_cache()
    ok = {k: v for k, v in results.items() if v["harmless_bypass_ablated"] < 2.0}
    pool = ok or results
    best = max(pool, key=lambda k: pool[k]["erosion"])
    shutil.copy(os.path.join(cones_dir, f"cone_dim_{best}.pt"), os.path.join(out, "selected_cone.pt"))
    json.dump({"selected_dim": best, "per_dim": results}, open(os.path.join(out, "selection.json"), "w"), indent=2)
    RP.log(f"[select] selected dim {best}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--cones-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--vectors-ml", default=None)
    ap.add_argument("--family", default="auto"); ap.add_argument("--betas", default="0.6,1.0,1.4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    select_dim(args.model, args.cones_dir, args.out, args.family)
    cmd = [PY, os.path.join(HERE, "cone_eval.py"), "--model", args.model,
           "--cone", os.path.join(args.out, "selected_cone.pt"), "--out", args.out, "--betas", args.betas]
    if args.vectors_ml and os.path.exists(args.vectors_ml):
        cmd += ["--vectors-ml", args.vectors_ml]
    RP.log(f"[select_eval] running cone_eval: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
