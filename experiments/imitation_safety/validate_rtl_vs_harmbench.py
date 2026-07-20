"""Validate the RTL (refuse-then-leak) genuine-harm judge against the OFFICIAL HarmBench
classifier (cais/HarmBench-Llama-2-13b-cls), the external published reference.

This is the corroboration Finding #1 (the Llama-Guard overcount) needs: the overcount claim is
that Guard over-flags relative to RTL. A skeptic's reply is "maybe RTL under-flags." The published
HarmBench classifier is an independent arbiter. If RTL agrees with it — same base rate, high kappa —
then RTL is not systematically lenient and the overcount is a property of Guard, not of RTL.

Every steering-eval `all_judged.csv` already carries BOTH columns side by side:
  - `genuine_harm` : the RTL judge (Qwen3-8B), genuine_harm = rtl_code in {B,C}
  - `hb_harm`      : the native HarmBench classifier, written by cone_eval.py stage_harmbench_native
This script is the comparison nobody had run: it joins them and reports the confusion matrix,
agreement, Cohen's kappa, and — the load-bearing number — the two base rates.

Usage: python validate_rtl_vs_harmbench.py [ROOT]   (default: the steering work root on this box)
"""
import glob
import os
import sys

import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/data/ethantsliu/exp_steer_safety/repl80_rdo"


def main():
    files = glob.glob(os.path.join(ROOT, "**", "*harmbench*", "all_judged.csv"), recursive=True)
    tp = fp = fn = tn = n = used = 0
    for f in files:
        try:
            df = pd.read_csv(f, usecols=lambda c: c in ("genuine_harm", "hb_harm"))
        except Exception:
            continue
        if "genuine_harm" not in df or "hb_harm" not in df:
            continue
        d = df.dropna(subset=["genuine_harm", "hb_harm"])
        if not len(d):
            continue
        used += 1
        rtl = d["genuine_harm"].astype(float) > 0.5   # RTL judge says harm
        hb = d["hb_harm"].astype(float) > 0.5          # official HarmBench cls says harm
        tp += int((rtl & hb).sum())
        fp += int((rtl & ~hb).sum())
        fn += int((~rtl & hb).sum())
        tn += int((~rtl & ~hb).sum())
        n += len(d)

    if n == 0:
        print(f"no joinable rows under {ROOT} (need all_judged.csv with genuine_harm + hb_harm)")
        return

    acc = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)  # chance agreement
    kappa = (acc - pe) / (1 - pe)
    print(f"files with both columns : {used}/{len(files)}")
    print(f"n responses             : {n}")
    print(f"TP (RTL harm, HB harm)  : {tp}")
    print(f"FP (RTL harm, HB safe)  : {fp}")
    print(f"FN (RTL safe, HB harm)  : {fn}")
    print(f"TN (RTL safe, HB safe)  : {tn}")
    print(f"agreement (accuracy)    : {acc:.4f}")
    print(f"Cohen's kappa           : {kappa:.4f}")
    print(f"RTL precision vs HB     : {tp / (tp + fp):.4f}")
    print(f"RTL recall vs HB        : {tp / (tp + fn):.4f}")
    print(f"base rate RTL harm      : {(tp + fp) / n:.4f}")
    print(f"base rate HB  harm      : {(tp + fn) / n:.4f}   <-- within 0.2pt => RTL not lenient")


if __name__ == "__main__":
    main()
