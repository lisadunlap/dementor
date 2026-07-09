#!/usr/bin/env python
"""Collect the RDO-cone roster dissociation table from repl80_rdo/<slug>/eval/metrics.json + the
qwen2.5-7b validation summary. Numeric aggregates only."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steer_config as CFG
ROOT = CFG.WORK_ROOT   # per-model <slug>/eval/metrics.json + worklist (env DEMENTOR_STEER_WORK)
WL = CFG.load_worklist(resolve=False)   # arch/slug only; no model-path resolution needed
ARCH = {m["slug"]: m.get("arch", "dense") for m in WL["models"]}

rows = []
for md in sorted(glob.glob(os.path.join(ROOT, "*"))):
    slug = os.path.basename(md)
    mp = os.path.join(md, "eval", "metrics.json")
    if not os.path.isfile(mp):
        continue
    m = json.load(open(mp))
    sel = None
    sp = os.path.join(md, "selection.json")
    if os.path.exists(sp):
        sel = json.load(open(sp)).get("selected_dim")
    rows.append(dict(slug=slug, arch=ARCH.get(slug, "?"), dim=sel,
                     base=m.get("baseline_harm"), cone=m.get("refusal_matched"),
                     fp=m.get("fingerprint_matched"), rnd=m.get("random_matched"),
                     verdict=m.get("verdict")))

print(f"\n{'model':20s}{'arch':11s}{'dim':>4}{'base':>7}{'cone@.85':>9}{'fp@.85':>8}{'rnd@.85':>8}  verdict")
print("-" * 82)
def f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and x == x else str(x)
for r in rows:
    print(f"{r['slug']:20s}{r['arch']:11s}{str(r['dim']):>4}{f(r['base']):>7}{f(r['cone']):>9}"
          f"{f(r['fp']):>8}{f(r['rnd']):>8}  {r['verdict']}")
from collections import Counter
print("\nTALLY:", dict(Counter(r["verdict"] for r in rows)))

# persist durable table (survives interruption)
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print(f"{'model':20s}{'arch':11s}{'dim':>4}{'base':>7}{'cone@.85':>9}{'fp@.85':>8}{'rnd@.85':>8}  verdict")
    print("-" * 82)
    for r in rows:
        print(f"{r['slug']:20s}{r['arch']:11s}{str(r['dim']):>4}{f(r['base']):>7}{f(r['cone']):>9}"
              f"{f(r['fp']):>8}{f(r['rnd']):>8}  {r['verdict']}")
    print("\nTALLY:", dict(Counter(r["verdict"] for r in rows)))
open(os.path.join(ROOT, "FINAL_RDO_TABLE.txt"), "w").write(buf.getvalue())
json.dump(rows, open(os.path.join(ROOT, "FINAL_RDO_TABLE.json"), "w"), indent=2)

vs = os.path.join(ROOT, "_validation", "VALIDATION_SUMMARY.json")
if os.path.exists(vs):
    v = json.load(open(vs))
    print("\n=== qwen2.5-7b VALIDATION (their cone vs port) ===")
    print("  known single-dir:", v.get("known_single_dir"))
    print("  cone erosion  their:", v.get("cone_erosion_their"), " port:", v.get("cone_erosion_port"),
          " |diff|:", v.get("erosion_abs_diff"))
    print("  port fingerprint:", v.get("port_fingerprint_matched"), " port random:", v.get("port_random_matched"))
    if v.get("subspace"):
        for k, e in v["subspace"]["per_dim"].items():
            print(f"  subspace dim {k}: principal_cos={[round(c,3) for c in e['principal_cos']]} mean={e['mean']:.3f}")
    print("  MATCH:", v.get("MATCH"), " subspace_ok:", v.get("subspace_ok"))
