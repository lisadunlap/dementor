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
import json
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


def per_model_joint(pairs):
    """Joint fpall arm means over identical fpall-positive-control-passing benchmarks.

    The fpall cone, fingerprint, and random arms all come from the variant cell. Selection uses the
    variant's positive-control verdict; reusing the campaign cone or gate would mix runs and can
    retain fpall cells whose positive control is invalid.
    """
    rows = {}
    for s, bms in pairs.items():
        paired_values = [
            (v["cone"], c["fp"], v["fp"], c["rand"], v["rand"])
            for c, v in bms.values()
            if v["verdict"] in RB.POSITIVE_CONTROL_PASS
            and all(
                value is not None
                for value in (v["cone"], c["fp"], v["fp"], c["rand"], v["rand"])
            )
        ]
        if paired_values:
            cone, fp_old, fp_new, random_old, random_new = zip(*paired_values)
            rows[s] = {
                "cone": float(np.mean(cone)),
                "fp_old": float(np.mean(fp_old)),
                "fp_new": float(np.mean(fp_new)),
                "random_old": float(np.mean(random_old)),
                "random_new": float(np.mean(random_new)),
                "n": len(paired_values),
            }
    return rows


def primary_fpall_rows(variant_cells):
    """Same-cell arm means for the complete cohort passing the declared fpall gate."""
    complete_gated = [
        model
        for model in RB.roster(variant_cells)
        if all(benchmark in variant_cells.get(model, {}) for benchmark in RB.HARM_BENCHMARKS)
    ]
    rows = {}
    for model in complete_gated:
        same_cells = [
            cell
            for benchmark, cell in variant_cells[model].items()
            if benchmark in RB.HARM_BENCHMARKS
            and cell["verdict"] in RB.POSITIVE_CONTROL_PASS
            and all(cell[arm] is not None for arm in ("cone", "fp", "rand"))
        ]
        if same_cells:
            rows[model] = {
                arm: float(np.mean([cell[arm] for cell in same_cells]))
                for arm in ("cone", "fp", "rand")
            }
            rows[model]["n"] = len(same_cells)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-coverage", type=int, default=1,
                    help="only report models with at least this many variant benchmarks")
    ap.add_argument("--json-out", default=None,
                    help="optional path for machine-readable paired comparison statistics")
    ap.add_argument("--tex-out", default=None,
                    help="optional path for generated LaTeX steering-result macros")
    args = ap.parse_args()

    camp = RB.load_cells()
    var = RB.load_cells(variant="fpall")
    ros = RB.roster(camp)
    pairs = {s: b for s, b in paired(camp, var, ros).items() if len(b) >= args.min_coverage}

    total = sum(len(b) for b in pairs.values())
    print(f"roster {len(ros)} models; variant cells paired: {total} "
          f"across {len(pairs)} models (of {len(ros)*len(RB.HARM_BENCHMARKS)} possible)\n")

    print(f"{'model':<22}{'n':>3}{'fp L14':>9}{'fp ALL':>9}{'rand L14':>10}{'rand ALL':>10}{'gap L14':>9}{'gap ALL':>9}")
    joint_rows = per_model_joint(pairs)
    for s in sorted(joint_rows, key=lambda x: -abs(joint_rows[x]["fp_old"])):
        row = joint_rows[s]
        fo, fn = row["fp_old"], row["fp_new"]
        ro, rn, n = row["random_old"], row["random_new"], row["n"]
        print(f"{s:<22}{n:>3}{fo:>+9.2f}{fn:>+9.2f}{ro:>+10.2f}{rn:>+10.2f}"
              f"{abs(fo-ro):>9.2f}{abs(fn-rn):>9.2f}")

    # Verdict flips are the headline diagnostic of this experiment: a campaign cell
    # that is CLEAN but whose variant is not means the stronger operator pushed the
    # fingerprint arm away from its random control on that cell.
    flips = [(s, b, c["verdict"], v["verdict"])
             for s, bms in pairs.items() for b, (c, v) in bms.items()
             if c["verdict"] in RB.POSITIVE_CONTROL_PASS
             and v["verdict"] not in RB.POSITIVE_CONTROL_PASS]
    print(f"\nverdict flips (campaign pass -> variant non-pass): {len(flips)} of {total} paired cells")
    for s, b, o, n in sorted(flips):
        print(f"   {s:<22}{b:<14} {o} -> {n}")

    primary_rows = primary_fpall_rows(var)
    if not primary_rows:
        return
    common_models = sorted(joint_rows)
    FO = [joint_rows[model]["fp_old"] for model in common_models]
    FN_DIAG = [joint_rows[model]["fp_new"] for model in common_models]
    RO = [joint_rows[model]["random_old"] for model in common_models]
    RN_DIAG = [joint_rows[model]["random_new"] for model in common_models]
    if common_models:
        print(f"\n{'MEAN over paired diagnostic':<27}{np.mean(FO):>+6.3f}{np.mean(FN_DIAG):>+9.3f}"
              f"{np.mean(RO):>+10.3f}{np.mean(RN_DIAG):>+10.3f}"
              f"{abs(np.mean(FO)-np.mean(RO)):>9.3f}"
              f"{abs(np.mean(FN_DIAG)-np.mean(RN_DIAG)):>9.3f}")

    primary_models = sorted(primary_rows)
    FN = [primary_rows[model]["fp"] for model in primary_models]
    RN = [primary_rows[model]["rand"] for model in primary_models]
    CN = [primary_rows[model]["cone"] for model in primary_models]
    contributing_cells = sum(primary_rows[model]["n"] for model in primary_models)
    print(
        f"\nPRIMARY FPALL: {len(primary_models)} complete gated models, "
        f"{contributing_cells} positive-control-passing cells; "
        f"cone={np.mean(CN):+.3f}, fingerprint={np.mean(FN):+.3f}, "
        f"random={np.mean(RN):+.3f}"
    )

    all_layer_test = None
    campaign_test_record = None
    if len(FN) >= 6:
        try:
            all_layer_test = RB.paired_signed_rank_exact(FN, RN)
            campaign_rows = RB.per_model(camp, ros)
            campaign_test_record = RB.paired_signed_rank_exact(
                [row["fp"] for row in campaign_rows.values()],
                [row["rand"] for row in campaign_rows.values()],
            )
            print(f"\nfingerprint vs random, ALL-LAYERS operator: "
                  f"n={len(primary_models)} W={all_layer_test['statistic']:.0f} "
                  f"p={all_layer_test['p_value']:.4f}")
            print("(campaign reference at full roster: "
                  f"n={len(campaign_rows)} W={campaign_test_record['statistic']:.1f} "
                  f"p={campaign_test_record['p_value']:.4f})")
        except ValueError as e:
            print(f"\nWilcoxon not computable: {e}")
    else:
        print(f"\n[{len(FN)} models paired -- need >=6 before the Wilcoxon is meaningful]")

    if args.json_out:
        output = {
            "min_coverage": args.min_coverage,
            "campaign_roster_models": len(ros),
            "models_with_variant_pairs": len(pairs),
            "complete_paired_cells": total,
            "contributing_paired_cells": contributing_cells,
            "reported_models": primary_models,
            "means_pp": {
                "paired_diagnostic_campaign_fingerprint": float(np.mean(FO)),
                "paired_diagnostic_fpall_fingerprint": float(np.mean(FN_DIAG)),
                "paired_diagnostic_campaign_random": float(np.mean(RO)),
                "paired_diagnostic_fpall_random": float(np.mean(RN_DIAG)),
                "primary_fpall_fingerprint": float(np.mean(FN)),
                "primary_fpall_random": float(np.mean(RN)),
                "primary_fpall_matched_cone": float(np.mean(CN)),
            },
            "fpall_fingerprint_vs_random_wilcoxon": all_layer_test,
            "campaign_fingerprint_vs_random_wilcoxon": campaign_test_record,
            "campaign_pass_to_variant_nonpass_cells": [
                {"model": s, "benchmark": b, "campaign": old, "fpall": new}
                for s, b, old, new in sorted(flips)
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
            handle.write("\n")
        print(f"wrote {args.json_out}")

    if args.tex_out:
        complete_base = sum(
            all(benchmark in camp.get(model, {}) for benchmark in RB.HARM_BENCHMARKS)
            for model in camp
        )
        complete_fpall = sum(
            all(benchmark in var.get(model, {}) for benchmark in RB.HARM_BENCHMARKS)
            for model in var
        )
        gated_complete_fpall = len(primary_models)
        campaign_rows = RB.per_model(camp, ros)
        cone_mean = float(np.mean([row["cone"] for row in campaign_rows.values()]))
        fingerprint_mean = float(np.mean([row["fp"] for row in campaign_rows.values()]))
        random_mean = float(np.mean([row["rand"] for row in campaign_rows.values()]))
        lines = [
            "% Generated by compare_fpall.py; do not edit.",
            f"\\newcommand{{\\BaseSteeringModels}}{{{complete_base}}}",
            f"\\newcommand{{\\BaseSteeringGatedModels}}{{{len(ros)}}}",
            f"\\newcommand{{\\FpallCompleteModels}}{{{complete_fpall}}}",
            f"\\newcommand{{\\FpallGatedModels}}{{{gated_complete_fpall}}}",
            f"\\newcommand{{\\FpallPairedCells}}{{{contributing_cells}}}",
            f"\\newcommand{{\\ConeMeanEffectPP}}{{{cone_mean:+.2f}}}",
            f"\\newcommand{{\\FingerprintMeanEffectPP}}{{{fingerprint_mean:+.2f}}}",
            f"\\newcommand{{\\RandomMeanEffectPP}}{{{random_mean:+.2f}}}",
            f"\\newcommand{{\\FingerprintRandomW}}{{{campaign_test_record['statistic']:.1f}}}",
            f"\\newcommand{{\\FingerprintRandomP}}{{{campaign_test_record['p_value']:.3f}}}",
            f"\\newcommand{{\\FpallFingerprintMeanEffectPP}}{{{float(np.mean(FN)):+.2f}}}",
            f"\\newcommand{{\\FpallRandomMeanEffectPP}}{{{float(np.mean(RN)):+.2f}}}",
            f"\\newcommand{{\\FpallConeMeanEffectPP}}{{{float(np.mean(CN)):+.2f}}}",
            f"\\newcommand{{\\FpallFingerprintRandomW}}{{{all_layer_test['statistic']:.0f}}}",
            f"\\newcommand{{\\FpallFingerprintRandomP}}{{{all_layer_test['p_value']:.3f}}}",
            "",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(args.tex_out)), exist_ok=True)
        with open(args.tex_out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        print(f"wrote {args.tex_out}")


if __name__ == "__main__":
    main()
