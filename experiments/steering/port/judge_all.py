#!/usr/bin/env python
"""Run the validated RTL judge (model = $RTL_JUDGE_MODEL) on all steered generations.
genuine_harm = REFUSE_THEN_LEAK OR FULL_COMPLY. Writes judged/all_judged.csv.

SHARED judge infra: this + rtl_judge.py (co-located) + canonical_graders.py + cone_eval.py are the
judge/grader modules used by BOTH the steering cone eval AND the imitation erosion/fidelity judging.
Path-portable: rtl_judge.py is imported from this dir (env DEMENTOR_RTL_JUDGE_DIR overrides); the
default IN/OUT root comes from DEMENTOR_STEER_ROOT, but cone_eval always sets IN_CSV/OUT_CSV.
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # co-located rtl_judge.py
_rtl_dir = os.environ.get("DEMENTOR_RTL_JUDGE_DIR")
if _rtl_dir:
    sys.path.insert(0, _rtl_dir)   # optional external rtl_judge (env override)
import rtl_judge as RJ

W = os.environ.get("DEMENTOR_STEER_ROOT", "/data/ethantsliu/exp_steer_safety").rstrip("/") + "/"
IN = os.environ.get("IN_CSV", W + "gen/all_gens.csv")
OUT = os.environ.get("OUT_CSV", W + "judged/all_judged.csv")
print("[judge] model =", RJ.JUDGE, "IN=", IN)
df = pd.read_csv(IN)
df["model_response"] = df["model_response"].astype(str)
print(f"[judge] rows={len(df)} cells={df.groupby(['direction','alpha']).ngroups}")
tok, mdl = RJ.load_judge()
batch_size = int(os.environ.get("RTL_JUDGE_BATCH_SIZE", "64"))
max_resp_chars = int(os.environ.get("RTL_JUDGE_MAX_RESP_CHARS", "3000"))
print(f"[judge] batch_size={batch_size} max_resp_chars={max_resp_chars}")
df = RJ.annotate(df, tok, mdl, batch_size=batch_size, max_resp_chars=max_resp_chars)
df.to_csv(OUT, index=False)
nun = int(df["rtl_code"].isna().sum())
print(f"[judge] wrote all_judged.csv n={len(df)} unparsed={nun}")
print("[judge] rtl_label dist:", df["rtl_label"].value_counts().to_dict())
