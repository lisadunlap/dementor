#!/usr/bin/env python3
"""GPU-free reproduction of the judge-reliability numbers reported in the paper.

Reads only committed CSV/JSON artifacts and recomputes every measurement-validation
figure from scratch: pairwise Cohen's kappa with contingency tables and bootstrap CIs,
Llama-Guard's precision/recall against each content-aware judge, and the per-benchmark
overcount ratios.

Why there is no rank correlation here: all three judges emit a BINARY label
(guard_unsafe, hb_harm, genuine_harm are each {0,1}). A Spearman rho on "judge severity
scores" is undefined for binary variables -- on 0/1 data it collapses to the phi
coefficient and carries no information beyond the 2x2 table. Cohen's kappa plus the
contingency tables is the correct reporting choice, and is what this script emits.

Usage:
    python experiments/imitation_safety/validation_analysis_template.py [--latex]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Repo-relative default; override with DEMENTOR_SAFETY_DIR.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SAFETY_DIR = os.environ.get("DEMENTOR_SAFETY_DIR", os.path.join(REPO, "data", "results", "safety"))

THREE_WAY = os.path.join(SAFETY_DIR, "overcount_three_way.csv")
VARIANCE = os.path.join(SAFETY_DIR, "erosion_variance_stats.json")

# Column -> display name. genuine_harm is the RTL (refuse-then-leak) judge's verdict.
JUDGES = {"guard_unsafe": "Llama-Guard", "hb_harm": "HarmBench-cls", "genuine_harm": "RTL"}
CONTENT_AWARE = ["hb_harm", "genuine_harm"]


def contingency(a: np.ndarray, b: np.ndarray) -> dict:
    """2x2 counts for two binary label vectors."""
    return {
        "both": int(((a == 1) & (b == 1)).sum()),
        "a_only": int(((a == 1) & (b == 0)).sum()),
        "b_only": int(((a == 0) & (b == 1)).sum()),
        "neither": int(((a == 0) & (b == 0)).sum()),
    }


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Cohen's kappa for two binary raters. Returns (kappa, p_observed, p_expected)."""
    n = len(a)
    c = contingency(a, b)
    p_o = (c["both"] + c["neither"]) / n
    p_yes = ((c["both"] + c["a_only"]) / n) * ((c["both"] + c["b_only"]) / n)
    p_no = ((c["b_only"] + c["neither"]) / n) * ((c["a_only"] + c["neither"]) / n)
    p_e = p_yes + p_no
    return (p_o - p_e) / (1 - p_e), p_o, p_e


def kappa_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for kappa. Seeded, so the CI is reproducible."""
    rng = np.random.default_rng(seed)
    n = len(a)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = cohens_kappa(a[idx], b[idx])[0]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def binary_confusion(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Treat `truth` as the reference judge and `pred` as the classifier under test."""
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


def load() -> pd.DataFrame:
    if not os.path.exists(THREE_WAY):
        sys.exit(f"missing artifact: {THREE_WAY}\nSet DEMENTOR_SAFETY_DIR to the results/safety dir.")
    df = pd.read_csv(THREE_WAY)
    for col in JUDGES:
        vals = set(pd.unique(df[col].dropna()))
        if not vals <= {0, 1}:
            sys.exit(f"{col} is not binary (values {sorted(vals)}); the kappa reporting assumes binary judges.")
        df[col] = df[col].astype(int)
    return df


def report(df: pd.DataFrame) -> dict:
    out: dict = {"n": len(df), "rates": {}, "kappa": {}, "guard_vs": {}, "per_benchmark": {}}

    print(f"n = {len(df)} responses, three judges, identical inputs\n")

    print("== flag rates ==")
    for col, name in JUDGES.items():
        rate = df[col].mean()
        out["rates"][name] = rate
        print(f"  {name:14s} {rate * 100:6.3f}%")
    g, h, r = (df[c].mean() for c in ("guard_unsafe", "hb_harm", "genuine_harm"))
    out["ratio_guard_hb"], out["ratio_guard_rtl"] = g / h, g / r
    print(f"  overcount: Guard/HarmBench = {g / h:.2f}x, Guard/RTL = {r and g / r:.2f}x\n")

    print("== pairwise Cohen's kappa (2x2 contingency, bootstrap 95% CI) ==")
    cols = list(JUDGES)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            ca, cb = cols[i], cols[j]
            a, b = df[ca].to_numpy(), df[cb].to_numpy()
            k, p_o, p_e = cohens_kappa(a, b)
            lo, hi = kappa_ci(a, b)
            c = contingency(a, b)
            key = f"{JUDGES[ca]} vs {JUDGES[cb]}"
            out["kappa"][key] = {"kappa": k, "ci": [lo, hi], "p_o": p_o, "p_e": p_e, **c}
            print(f"  {key:32s} kappa={k:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
            print(f"      p_o={p_o:.4f} p_e={p_e:.4f} | both={c['both']} "
                  f"{JUDGES[ca]}-only={c['a_only']} {JUDGES[cb]}-only={c['b_only']} neither={c['neither']}")
    print()

    print("== Llama-Guard against each content-aware judge (Guard = classifier under test) ==")
    for ref in CONTENT_AWARE:
        cm = binary_confusion(df["guard_unsafe"].to_numpy(), df[ref].to_numpy())
        out["guard_vs"][JUDGES[ref]] = cm
        print(f"  vs {JUDGES[ref]:14s} precision={cm['precision']:.4f} recall={cm['recall']:.4f} "
              f"F1={cm['f1']:.4f}")
        print(f"      over-flags {cm['fp']} benign, and MISSES {cm['fn']} of "
              f"{cm['tp'] + cm['fn']} genuinely harmful responses")
    print()

    print("== per-benchmark overcount ==")
    print(f"  {'benchmark':14s} {'n':>6s} {'Guard%':>8s} {'HB%':>7s} {'RTL%':>7s} {'G/HB':>7s} {'G/RTL':>7s}")
    for bench, grp in df.groupby("benchmark"):
        gb, hb, rb = grp.guard_unsafe.mean(), grp.hb_harm.mean(), grp.genuine_harm.mean()
        row = {"n": len(grp), "guard": gb, "hb": hb, "rtl": rb,
               "g_hb": gb / hb if hb else float("nan"), "g_rtl": gb / rb if rb else float("nan")}
        out["per_benchmark"][bench] = row
        print(f"  {bench:14s} {len(grp):6d} {gb * 100:7.2f}% {hb * 100:6.2f}% {rb * 100:6.2f}% "
              f"{row['g_hb']:6.2f}x {row['g_rtl']:6.2f}x")
    print()

    if os.path.exists(VARIANCE):
        v = json.load(open(VARIANCE))
        dec = v.get("variance_decomposition_pct", {})
        out["variance"] = {"n_adapters": v.get("n_adapters"), **dec,
                           "mean_erosion_pp": v.get("overall_mean_erosion", 0) * 100,
                           "pct_safer": v.get("pct_adapters_got_safer")}
        print("== erosion variance decomposition (single RTL-scored metric) ==")
        print(f"  n_adapters={v.get('n_adapters')}  mean={v.get('overall_mean_erosion', 0) * 100:.3f} pp  "
              f"safer={v.get('pct_adapters_got_safer'):.1f}%")
        print(f"  source={dec.get('source'):.2f}%  target={dec.get('target'):.2f}%  "
              f"dataset={dec.get('dataset'):.2f}%")
        print("  NOTE: this partitions ONE judge's metric. A by-judge decomposition would require\n"
              "  HarmBench-classifier scores over the matrix responses, which are not on disk;\n"
              "  do not report a per-judge split without computing it.\n")
    return out


def latex(out: dict) -> str:
    k = out["kappa"]
    rows = []
    for key, v in k.items():
        rows.append(f"{key} & {v['kappa']:.3f} & [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] & "
                    f"{v['both']} & {v['a_only']} & {v['b_only']} & {v['neither']} \\\\")
    body = "\n".join(rows)
    g_rtl = out["guard_vs"]["RTL"]
    g_hb = out["guard_vs"]["HarmBench-cls"]
    return f"""% auto-generated by validation_analysis_template.py -- do not hand-edit
\\begin{{table}}[t]
\\centering\\small
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
Judge pair & $\\kappa$ & 95\\% CI & Both & A only & B only & Neither \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Pairwise agreement between the three judges on the same $n={out['n']}$ responses.
The two content-aware judges agree substantially ($\\kappa={k['HarmBench-cls vs RTL']['kappa']:.3f}$);
Llama-Guard agrees with neither ($\\kappa={k['Llama-Guard vs HarmBench-cls']['kappa']:.3f}$ and
${k['Llama-Guard vs RTL']['kappa']:.3f}$). All three judges emit binary labels, so no rank
correlation is defined; $\\kappa$ with the contingency counts is reported instead.
Against the content-aware judges Llama-Guard attains precision
{g_hb['precision']:.3f}/{g_rtl['precision']:.3f} and recall
{g_hb['recall']:.3f}/{g_rtl['recall']:.3f}, i.e.\\ it both over-flags benign responses and
misses {g_hb['fn']}/{g_rtl['fn']} genuinely harmful ones.}}
\\label{{tab:judge_agreement}}
\\end{{table}}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latex", action="store_true", help="also emit a LaTeX table")
    ap.add_argument("--json", metavar="PATH", help="write the computed values as JSON")
    args = ap.parse_args()

    out = report(load())
    if args.latex:
        print(latex(out))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
