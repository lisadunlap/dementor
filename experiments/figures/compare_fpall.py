#!/usr/bin/env python
"""Campaign (fingerprint/random @ layer 14) vs all-layers variant, side by side.

The campaign ablates the refusal CONE at every decoder layer but the fingerprint and
random single directions at ABLATE_LAYER only -- a ~600x difference in application
sites. That invites "your treatment arm was weaker than your positive control, so of
course it did nothing". The fpall variant removes the asymmetry: the single directions
go through register_cone() as a normalised [1,hidden] basis, which is numerically the
same operator (verified max abs diff 0.0), applied at every layer.

This script reports what that changes. It reads campaign cells and variant cells via
load_cells(variant=...) so the two populations can never be mixed -- see the comment in
load_cells; eval_<bm>_fpall/ sorts after eval_<bm>/ and would silently overwrite it.

The roster is ALWAYS the campaign roster. Roster membership is defined by the positive
control, which this experiment does not touch, so recomputing it from variant cells
would be wrong.

Usage:  compare_fpall.py [--min-coverage N]
"""
import argparse
import importlib.util
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "rb", os.path.join(HERE, "rebuild_steering_figures.py"))
RB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RB)


def paired(camp, var, ros):
    """-> {slug: {bm: (campaign_cell, variant_cell)}} for roster models only."""
    out = {}
    for s in ros:
        for b, v in var.get(s, {}).items():
            if b not in RB.HARM_BENCHMARKS:
                continue
            c = camp.get(s, {}).get(b)
            if c is None:
                continue
            out.setdefault(s, {})[b] = (c, v)
    return out


def per_model(pairs, arm):
    """Mean over each model's CLEAN harm benchmarks, campaign vs variant.

    Selection is on the CAMPAIGN verdict, so the same cells are averaged on both
    sides and the comparison stays paired.

    Do NOT read the campaign verdict as also applying to the variant. CLEAN requires
    two things -- the cone erodes safety AND the fingerprint's excess over random is
    <= 0.05 -- so the verdict depends on the fingerprint arm too, and the variant can
    flip to INCONCLUSIVE while the campaign cell is CLEAN. That is a RESULT, not a
    filtering artifact: it means the all-layers operator moved the fingerprint arm
    away from its control. Such cells are counted here and flagged separately by
    variant_verdicts(); they must never be silently dropped.
    """
    rows = {}
    for s, bms in pairs.items():
        o = [c[arm] for c, _ in bms.values() if c["verdict"] == "CLEAN" and c[arm] is not None]
        n = [v[arm] for c, v in bms.values() if c["verdict"] == "CLEAN" and v[arm] is not None]
        if o and n and len(o) == len(n):
            rows[s] = (float(np.mean(o)), float(np.mean(n)), len(o))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-coverage", type=int, default=1,
                    help="only report models with at least this many variant benchmarks")
    args = ap.parse_args()

    camp = RB.load_cells()
    var = RB.load_cells(variant="fpall")
    ros = RB.roster(camp)
    pairs = {s: b for s, b in paired(camp, var, ros).items() if len(b) >= args.min_coverage}

    total = sum(len(b) for b in pairs.values())
    print(f"roster {len(ros)} models; variant cells paired: {total} "
          f"across {len(pairs)} models (of {len(ros)*len(RB.HARM_BENCHMARKS)} possible)\n")

    print(f"{'model':<22}{'n':>3}{'fp L14':>9}{'fp ALL':>9}{'rand L14':>10}{'rand ALL':>10}{'gap L14':>9}{'gap ALL':>9}")
    fp_o, fp_n, rd_o, rd_n = per_model(pairs, "fp"), None, per_model(pairs, "rand"), None
    for s in sorted(fp_o, key=lambda x: -abs(fp_o[x][0])):
        fo, fn, n = fp_o[s]
        ro, rn, _ = rd_o[s]
        print(f"{s:<22}{n:>3}{fo:>+9.2f}{fn:>+9.2f}{ro:>+10.2f}{rn:>+10.2f}"
              f"{abs(fo-ro):>9.2f}{abs(fn-rn):>9.2f}")

    # Verdict flips are the headline diagnostic of this experiment: a campaign cell
    # that is CLEAN but whose variant is not means the stronger operator pushed the
    # fingerprint arm away from its random control on that cell.
    flips = [(s, b, c["verdict"], v["verdict"])
             for s, bms in pairs.items() for b, (c, v) in bms.items()
             if c["verdict"] == "CLEAN" and v["verdict"] != "CLEAN"]
    print(f"\nverdict flips (campaign CLEAN -> variant not CLEAN): {len(flips)} of {total} paired cells")
    for s, b, o, n in sorted(flips):
        print(f"   {s:<22}{b:<14} {o} -> {n}")

    if not fp_o:
        return
    FO = [v[0] for v in fp_o.values()]; FN = [v[1] for v in fp_o.values()]
    RO = [v[0] for v in rd_o.values()]; RN = [v[1] for v in rd_o.values()]
    print(f"\n{'MEAN over reported models':<25}{np.mean(FO):>+6.3f}{np.mean(FN):>+9.3f}"
          f"{np.mean(RO):>+10.3f}{np.mean(RN):>+10.3f}"
          f"{abs(np.mean(FO)-np.mean(RO)):>9.3f}{abs(np.mean(FN)-np.mean(RN)):>9.3f}")

    if len(FO) >= 6:
        from scipy.stats import wilcoxon
        try:
            w = wilcoxon(FN, RN)
            print(f"\nfingerprint vs random, ALL-LAYERS operator: "
                  f"n={len(FN)} W={w.statistic:.0f} p={w.pvalue:.4f}")
            print("(campaign reference at full roster: n=24 W=122 p=0.63)")
        except ValueError as e:
            print(f"\nWilcoxon not computable: {e}")
    else:
        print(f"\n[{len(FO)} models paired -- need >=6 before the Wilcoxon is meaningful]")


if __name__ == "__main__":
    main()
