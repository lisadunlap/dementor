#!/usr/bin/env python
"""Per-model RDO-cone roster driver (task: RDO refusal cone as positive control vs single-dir
fingerprint + random). Resumable; each stage skipped if its artifact exists. Writes to
repl80_rdo/<slug>/.

Stages:
  1 benign      : reuse repl80/<slug>/benign.csv if present, else generate (run_model.stage_benign).
  2 fingerprint : reuse repl80/<slug>/vectors_ml.pt (fingerprint M-vs-llama + random @ L14) if
                  present, else derive (run_model.stage_derive).
  3 dim         : compute_dim.py -> dim/direction.pt (+metadata). model-agnostic diff-of-means.
  4 cone        : rdo_port.py -> cones/cone_dim_{min..max}.pt.
  5 select      : pick cone dim maximizing harmful-val ablation erosion with retain-KL < 0.1 (cheap,
                  no judge) -> selected_cone.pt.
  6 eval        : cone_eval.py (gen cone+fingerprint+random betas -> RTL judge -> coherence-matched
                  verdict) -> eval/metrics.json.

Usage: run_rdo_model.py <slug>   (reads rdo_worklist.json; CUDA_VISIBLE_DEVICES set by caller)
"""
import os, sys, json, time, subprocess, shutil, glob
import torch

HERE = os.path.dirname(os.path.abspath(__file__))   # steering package dir (port/ + sibling scripts)
PORT = os.path.join(HERE, "port")
sys.path.insert(0, HERE)
import steer_config as CFG
ROOT = CFG.WORK_ROOT   # worklist + per-model <slug>/ outputs (env DEMENTOR_STEER_WORK)
REPL80 = CFG.REPL80    # fingerprint/benign reuse source (env DEMENTOR_REPL80)
PY = CFG.PY
sys.path.insert(0, REPL80)
sys.path.insert(0, PORT)
import rdo_compat  # transformers-5.5.4 compat shims; installs LossKwargs/chat_template/nemotron
                   # patches for the in-process prep stages (benign, fingerprint, select).

# Env-var overrides (backward-compatible: defaults below == original hard-coded values, so the
# already-running first-pass roster is UNTOUCHED). A retry sets RDO_MAX_DIM / RDO_BETAS for a
# stronger refusal cone and RDO_OUT_SUFFIX to write to a sibling dir (<slug><suffix>/) instead of
# colliding with / skipping the first-pass artifacts in <slug>/.
MIN_DIM = int(os.environ.get("RDO_MIN_DIM", "2"))
MAX_DIM = int(os.environ.get("RDO_MAX_DIM", "4"))
BETAS = os.environ.get("RDO_BETAS", "0.6,1.0,1.4")
OUT_SUFFIX = os.environ.get("RDO_OUT_SUFFIX", "")


def log(od, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(od, "rdo_run.log"), "a") as f:
        f.write(line + "\n")


def run(cmd, env=None, logf=None):
    e = dict(os.environ)
    e.update(HF_HOME=CFG.HF_HOME, HF_HUB_CACHE=CFG.HF_HUB_CACHE,
             HF_HUB_DISABLE_XET="1", HF_HUB_OFFLINE="1")
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if logf:
        open(logf, "a").write(r.stdout or "")
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed rc={r.returncode}: {' '.join(cmd)}\n--- tail ---\n{(r.stdout or '')[-2500:]}")
    return r.stdout


def stage_benign_fingerprint(spec, od):
    """Reuse repl80 benign.csv + vectors_ml.pt when present; else derive via run_model stages."""
    slug = spec["slug"]
    src = os.path.join(REPL80, slug)
    ben = os.path.join(od, "benign.csv"); vml = os.path.join(od, "vectors_ml.pt")
    for fn, dst in (("benign.csv", ben), ("vectors_ml.pt", vml)):
        s = os.path.join(src, fn)
        if os.path.exists(s) and not os.path.exists(dst):
            shutil.copy(s, dst)
            log(od, f"[1-2] reused repl80/{slug}/{fn}")
    if os.path.exists(ben) and os.path.exists(vml):
        return
    # derive from scratch with run_model stages
    import run_model as RM
    rmspec = {"slug": slug, "path": spec["path"], "arch": spec.get("arch", "dense"),
              "ref": spec.get("ref", "llama"), "params_b": spec["params_b"],
              "benign_reuse": spec.get("benign_reuse")}
    if not os.path.exists(ben):
        log(od, "[1] deriving benign (run_model.stage_benign)")
        RM.stage_benign(rmspec, od)
    if not os.path.exists(vml):
        log(od, "[2] deriving fingerprint (run_model.stage_derive)")
        RM.stage_derive(rmspec, od)


def stage_dim(spec, od):
    ddir = os.path.join(od, "dim")
    if os.path.exists(os.path.join(ddir, "direction.pt")):
        log(od, "[3] dim cached"); return
    log(od, "[3] computing DIM (compute_dim.py)")
    run([PY, os.path.join(PORT, "compute_dim.py"), "--model", spec["path"], "--out", ddir,
         "--family", spec.get("family", "auto")], logf=os.path.join(od, "rdo_run.log"))


def stage_cone(spec, od):
    cones = os.path.join(od, "cones")
    if os.path.exists(os.path.join(cones, f"cone_dim_{MAX_DIM}.pt")):
        log(od, "[4] cone cached"); return
    log(od, f"[4] training RDO cone (rdo_port.py) dims {MIN_DIM}..{MAX_DIM}")
    run([PY, os.path.join(PORT, "rdo_port.py"), "--model", spec["path"],
         "--dim-dir", os.path.join(od, "dim"), "--out-dir", od,
         "--min-cone-dim", str(MIN_DIM), "--max-cone-dim", str(MAX_DIM),
         "--family", spec.get("family", "auto")], logf=os.path.join(od, "rdo_run.log"))


def stage_select(spec, od):
    """Cheap dim selection: for each cone dim, ablate subspace on harmful-val -> mean bypass (lower
    = more erosion); retain-KL on harmless-val. Pick max-erosion dim with retain sane. Writes
    selected_cone.pt (copy of chosen cone_dim_k.pt) + selection.json."""
    sel = os.path.join(od, "selected_cone.pt")
    if os.path.exists(sel):
        log(od, "[5] selection cached"); return
    log(od, "[5] selecting cone dim (harmful-val ablation bypass)")
    sys.path.insert(0, PORT)
    import rdo_port as RP
    from dementor.steering._common import load_causal_lm, get_transformer_layers
    import json as _json
    splits = CFG.SPLITS_DIR
    hv = [d["instruction"] for d in _json.load(open(os.path.join(splits, "harmful_val.json")))][:64]
    lv = [d["instruction"] for d in _json.load(open(os.path.join(splits, "harmless_val.json")))][:64]
    tok, model, dev = load_causal_lm(spec["path"], padding_side="left", device="cuda", dtype="auto")
    model.requires_grad_(False)
    family = RP.family_of(spec["path"], spec.get("family", "auto"))
    render = RP.make_render(family, tok); rtoks = RP.refusal_token_ids(family, tok)
    hvp = [render(x) for x in hv]; lvp = [render(x) for x in lv]
    from cone_eval import register_cone
    base_h = RP.bypass_scores(model, tok, hvp, rtoks, dev, 16).mean().item()
    results = {}
    for f in sorted(glob.glob(os.path.join(od, "cones", "cone_dim_*.pt"))):
        k = int(os.path.basename(f).split("_")[-1].split(".")[0])
        basis = torch.load(f, map_location="cpu")["basis"].float()
        handles = register_cone(model, basis, 1.0)
        try:
            hb = RP.bypass_scores(model, tok, hvp, rtoks, dev, 16).mean().item()
            lb = RP.bypass_scores(model, tok, lvp, rtoks, dev, 16).mean().item()
        finally:
            for h in handles:
                h.remove()
        results[k] = {"harmful_bypass_ablated": hb, "harmless_bypass_ablated": lb,
                      "erosion": base_h - hb}
        log(od, f"[5] dim {k}: harmful_bypass {base_h:+.3f}->{hb:+.3f} (erosion {base_h-hb:+.3f}) harmless {lb:+.3f}")
    del model, tok
    import gc; gc.collect(); torch.cuda.empty_cache()
    # pick max erosion with harmless_bypass not blown up (< 2.0)
    ok = {k: v for k, v in results.items() if v["harmless_bypass_ablated"] < 2.0}
    pool = ok or results
    best = max(pool, key=lambda k: pool[k]["erosion"])
    shutil.copy(os.path.join(od, "cones", f"cone_dim_{best}.pt"), sel)
    _json.dump({"selected_dim": best, "base_harmful_bypass": base_h, "per_dim": results},
               open(os.path.join(od, "selection.json"), "w"), indent=2)
    log(od, f"[5] selected cone dim {best}")


def stage_eval(spec, od):
    ev = os.path.join(od, "eval")
    if os.path.exists(os.path.join(ev, "metrics.json")):
        log(od, "[6] eval cached"); return
    log(od, "[6] erosion eval + judge + verdict (cone_eval.py)")
    run([PY, os.path.join(PORT, "cone_eval.py"), "--model", spec["path"],
         "--cone", os.path.join(od, "selected_cone.pt"), "--out", ev,
         "--vectors-ml", os.path.join(od, "vectors_ml.pt"), "--betas", BETAS],
        logf=os.path.join(od, "rdo_run.log"))


def main():
    slug = sys.argv[1]
    wl = json.load(open(os.path.join(ROOT, "rdo_worklist.json")))
    spec = next((m for m in wl["models"] if m["slug"] == slug), None)
    if spec is None:
        print(f"unknown slug {slug}"); sys.exit(2)
    # od carries OUT_SUFFIX; every stage takes `od` as an arg, so cone/dim/select/eval all write
    # under this dir. The fingerprint/benign REUSE source is repl80/<base-slug> (spec["slug"]),
    # which is independent of the suffix -- so a retry gets fresh cone/select/eval in its own dir.
    od = os.path.join(ROOT, slug + OUT_SUFFIX); os.makedirs(od, exist_ok=True)
    final = os.path.join(od, "eval", "metrics.json")
    if os.path.exists(final):
        log(od, f"=== {slug} DONE (metrics.json exists) -> SKIP ==="); return
    log(od, f"=== START {slug} path={spec['path']} CUDA={os.environ.get('CUDA_VISIBLE_DEVICES')} ===")
    t0 = time.time()
    try:
        stage_benign_fingerprint(spec, od)
        stage_dim(spec, od)
        stage_cone(spec, od)
        stage_select(spec, od)
        stage_eval(spec, od)
        m = json.load(open(final))
        log(od, f"=== DONE {slug} in {(time.time()-t0)/60:.1f}m verdict={m['verdict']} "
                f"cone={m['refusal_matched']} fp={m['fingerprint_matched']} rnd={m['random_matched']} ===")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(od, f"=== ERROR {slug}: {e} ===\n{tb[-2500:]}")
        json.dump({"slug": slug, "status": "error", "error": str(e), "tb": tb[-2000:]},
                  open(os.path.join(od, "ERROR.json"), "w"), indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
