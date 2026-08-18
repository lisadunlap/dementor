#!/usr/bin/env python
"""Strict one-command rebuild of the configured imitation-safety campaign.

The command audits coverage before touching any aggregate artifact. It then rebuilds
the stage-aware CSVs, SFT and DPO statistics, exploratory and paper figures, and a
small TeX macro file consumed by the paper. It never launches evaluation or training.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import audit_erosion_coverage as AUDIT  # noqa: E402
import erosion_common as EC  # noqa: E402
from dementor import config  # noqa: E402


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)


def signed_pp(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def ci_pp(values: list[float]) -> str:
    return f"[{100.0 * values[0]:+.2f}, {100.0 * values[1]:+.2f}]"


def signed_ci(values: list[float], digits: int = 3) -> str:
    return f"[{values[0]:+.{digits}f}, {values[1]:+.{digits}f}]"


def macro(name: str, value: object) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="imitation_safety")
    parser.add_argument("--seed", default="seed42")
    parser.add_argument("--work-root", action="append", dest="work_roots")
    parser.add_argument(
        "--tex-output",
        type=Path,
        default=REPO / "docs" / "generated_campaign_results.tex",
    )
    parser.add_argument(
        "--paper-tex-output",
        type=Path,
        default=REPO / "paper" / "naz_aaai2027" / "generated_campaign_results.tex",
    )
    args = parser.parse_args()

    roots = args.work_roots or [EC.WORK]
    report = AUDIT.audit(args.campaign, roots)
    results = Path(EC.RESULTS_SAFETY)
    results.mkdir(parents=True, exist_ok=True)
    coverage_path = results / "erosion_coverage.json"
    coverage_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    incomplete = {
        stage: stage_report
        for stage, stage_report in report["stages"].items()
        if stage_report["complete"] != stage_report["expected"]
    }
    if incomplete:
        summary = ", ".join(
            f"{stage}={value['complete']}/{value['expected']}"
            for stage, value in incomplete.items()
        )
        raise SystemExit(
            f"refusing to regenerate from partial coverage ({summary}); see {coverage_path}"
        )

    build_args = [
        str(HERE / "build_erosion_csv.py"),
        "--campaign", args.campaign,
        "--seed", args.seed,
    ]
    for root in roots:
        build_args.extend(("--work-root", root))
    run(*build_args)

    summary_csv = results / f"erosion_{args.seed}_summary.csv"
    long_csv = results / f"erosion_{args.seed}_long.csv"
    stats_paths = {}
    for stage in ("sft", "dpo"):
        out = results / f"erosion_variance_stats_{stage}.json"
        stats_paths[stage] = out
        run(
            str(HERE / "variance_decomp.py"),
            "--stage", stage,
            "--summary-csv", str(summary_csv),
            "--long-csv", str(long_csv),
            "--output", str(out),
        )
        run(str(HERE / "plot_erosion.py"), "--seed", args.seed, "--stage", stage)
        run(
            str(HERE / "plot_erosion_paper.py"),
            "--seed", args.seed,
            "--stage", stage,
            "--outdir", str(results / f"figures_paper_{stage}"),
        )

    stats = {stage: json.loads(path.read_text()) for stage, path in stats_paths.items()}
    dpo = stats["dpo"]
    sft = stats["sft"]
    paired = dpo["paired_sft_to_dpo"]
    sft_target = sft["target_relative_safety"]
    dpo_target = dpo["target_relative_safety"]
    sft_target_ci = sft_target["bootstrap"]["intervals"]
    dpo_target_ci = dpo_target["bootstrap"]["intervals"]
    evaluation = config.campaign_evaluation(args.campaign)
    headline = {
        "campaign": args.campaign,
        "models": len(config.campaign_roster(args.campaign)),
        "datasets": len(config.campaign_dataset_names(args.campaign)),
        "adapters_per_stage": report["stages"]["sft"]["expected"],
        "evaluation": evaluation,
        "sft": sft,
        "dpo": dpo,
        "paired_sft_to_dpo": paired,
    }
    headline_path = results / "erosion_campaign_headlines.json"
    headline_path.write_text(json.dumps(headline, indent=2) + "\n", encoding="utf-8")
    target_relative = {
        "campaign": args.campaign,
        "metric": "mean harmful-compliance rate over five harm benchmarks",
        "sft": sft_target,
        "dpo": dpo_target,
    }
    target_outputs = {
        results / "target_safety_transfer.json",
        REPO / "paper" / "naz_aaai2027" / "target_safety_transfer_stats.json",
    }
    for target_output in target_outputs:
        target_output.parent.mkdir(parents=True, exist_ok=True)
        target_output.write_text(
            json.dumps(target_relative, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    lines = [
        "% Generated by experiments/imitation_safety/regenerate_campaign.py; do not edit.",
        macro("CampaignModels", headline["models"]),
        macro("CampaignDatasets", headline["datasets"]),
        macro("CampaignAdaptersPerStage", headline["adapters_per_stage"]),
        macro("CampaignEvaluationPrompts", evaluation["max_prompts"]),
        macro("CampaignEvaluationSeed", evaluation["subsample_seed"]),
        macro("CampaignMaxNewTokens", evaluation["max_new_tokens"]),
        macro("SFTMeanErosionPP", signed_pp(sft["overall_mean_erosion"])),
        macro("DPOMeanErosionPP", signed_pp(dpo["overall_mean_erosion"])),
        macro("SFTMeanErosionCI", ci_pp(sft["source_cluster_bootstrap_95ci"])),
        macro("DPOMeanErosionCI", ci_pp(dpo["source_cluster_bootstrap_95ci"])),
        macro("SFTPctLowerHarm", f"{sft['pct_adapters_lower_harm']:.1f}"),
        macro("DPOPctLowerHarm", f"{dpo['pct_adapters_lower_harm']:.1f}"),
        macro("SFTPctBelowMinusOne", f"{sft['pct_adapters_below_minus_1pp']:.1f}"),
        macro("DPOPctBelowMinusOne", f"{dpo['pct_adapters_below_minus_1pp']:.1f}"),
        macro("SFTPctAtLeastFive", f"{sft['pct_adapters_at_least_5pp']:.1f}"),
        macro("DPOPctAtLeastFive", f"{dpo['pct_adapters_at_least_5pp']:.1f}"),
        macro("SFTMedianErosionPP", signed_pp(sft["median_erosion"])),
        macro("DPOMedianErosionPP", signed_pp(dpo["median_erosion"])),
        macro("SFTMaxErosionPP", signed_pp(sft["max_erosion"])),
        macro("DPOMaxErosionPP", signed_pp(dpo["max_erosion"])),
        macro("SFTMeanOverRefusalDeltaPP", signed_pp(sft["mean_over_refusal_delta"])),
        macro("DPOMeanOverRefusalDeltaPP", signed_pp(dpo["mean_over_refusal_delta"])),
        macro("SFTSourceEtaPct", f"{sft['variance_decomposition_pct']['source']:.1f}"),
        macro("SFTTargetEtaPct", f"{sft['variance_decomposition_pct']['target']:.1f}"),
        macro("SFTDatasetEtaPct", f"{sft['variance_decomposition_pct']['dataset']:.1f}"),
        macro("DPOSourceEtaPct", f"{dpo['variance_decomposition_pct']['source']:.1f}"),
        macro("DPOTargetEtaPct", f"{dpo['variance_decomposition_pct']['target']:.1f}"),
        macro("DPODatasetEtaPct", f"{dpo['variance_decomposition_pct']['dataset']:.1f}"),
        macro("PairedStageCells", paired["n_pairs"]),
        macro("PairedStageDeltaPP", signed_pp(paired["mean_delta"])),
        macro("PairedStageDeltaCI", ci_pp(paired["source_cluster_bootstrap_95ci"])),
        macro("PairedStageMedianPP", signed_pp(paired["median_delta"])),
        macro("PairedStagePctDPOHigher", f"{paired['pct_dpo_more_erosive']:.1f}"),
        macro("SFTEroderSources", len(sft["eroders"])),
        macro("DPOEroderSources", len(dpo["eroders"])),
        macro("SFTTargetAlignmentSlope", f"{sft_target['target_alignment_slope']:+.3f}"),
        macro("DPOTargetAlignmentSlope", f"{dpo_target['target_alignment_slope']:+.3f}"),
        macro("SFTTargetAlignmentCI", signed_ci(sft_target_ci["target_alignment_slope"])),
        macro("DPOTargetAlignmentCI", signed_ci(dpo_target_ci["target_alignment_slope"])),
        macro("SFTTargetDistanceReductionPP", signed_pp(sft_target["mean_target_distance_reduction"])),
        macro("DPOTargetDistanceReductionPP", signed_pp(dpo_target["mean_target_distance_reduction"])),
        macro("SFTTargetDistanceReductionCI", ci_pp(sft_target_ci["mean_target_distance_reduction"])),
        macro("DPOTargetDistanceReductionCI", ci_pp(dpo_target_ci["mean_target_distance_reduction"])),
        macro("SFTTargetCloserPct", f"{sft_target['pct_nonzero_gap_cells_closer_to_target']:.1f}"),
        macro("DPOTargetCloserPct", f"{dpo_target['pct_nonzero_gap_cells_closer_to_target']:.1f}"),
        macro(
            "SFTSaferTargetChangePP",
            signed_pp(sft_target["strata"]["safer_target"]["mean_adapter_change"]),
        ),
        macro(
            "DPOSaferTargetChangePP",
            signed_pp(dpo_target["strata"]["safer_target"]["mean_adapter_change"]),
        ),
        macro(
            "SFTMoreHarmfulTargetChangePP",
            signed_pp(sft_target["strata"]["more_harmful_target"]["mean_adapter_change"]),
        ),
        macro(
            "DPOMoreHarmfulTargetChangePP",
            signed_pp(dpo_target["strata"]["more_harmful_target"]["mean_adapter_change"]),
        ),
    ]
    generated_tex = "\n".join(lines) + "\n"
    for tex_output in {args.tex_output, args.paper_tex_output}:
        tex_output.parent.mkdir(parents=True, exist_ok=True)
        tex_output.write_text(generated_tex, encoding="utf-8")
    print(f"[regenerate] wrote {coverage_path}")
    print(f"[regenerate] wrote {headline_path}")
    for target_output in target_outputs:
        print(f"[regenerate] wrote {target_output}")
    print(f"[regenerate] wrote {args.tex_output}")
    print(f"[regenerate] wrote {args.paper_tex_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
