#!/usr/bin/env python3
"""Generate audited campaign-level fidelity statistics and manuscript macros.

The analysis is deliberately cell-complete: both scorers must contain the exact configured
1,104 adapters and every result must have a 200/200 denominator.  The embedding baseline compares
each unadapted source reference with the target reference on the same held-out prompts, so gains do
not mistake generic response similarity for imitation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fidelity_common as FC  # noqa: E402

KEYS = ["dataset", "source", "target", "seed"]


def stage(adapter: str) -> str:
    if adapter.startswith("self_sft_"):
        return "self_sft"
    if adapter.startswith("sft_"):
        return "sft"
    if adapter.startswith("dpo_"):
        return "dpo"
    raise ValueError(f"unrecognized adapter stage: {adapter}")


def load_scores(path: str, scorer: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["stage"] = frame["adapter"].map(stage)
    expected = {"sft": 528, "dpo": 528, "self_sft": 48}
    counts = frame.groupby("stage").size().to_dict()
    if counts != expected:
        raise RuntimeError(f"{scorer} stage coverage {counts}, expected {expected}")
    bad = frame[(frame["scorer"] != scorer) | (frame["n_prompts"] != 200) | (frame["n"] != 200)]
    if len(bad):
        sample = bad[["adapter", "n", "n_prompts"]].head(10).to_dict("records")
        raise RuntimeError(f"{scorer} has {len(bad)} non-200/200 cells: {sample}")
    return frame


def reference_embedding_baseline(device: str) -> pd.DataFrame:
    """Return 528 unadapted-source-to-target reference similarities."""
    model = FC.load_embedder(device)
    worklist = FC.build_fidelity_worklist(seed="all")
    roster = sorted({item["source"] for item in worklist})
    vectors = {}
    prompts = {}
    for dataset in FC.DATASETS:
        for slug in roster:
            path = FC.ref_csv_path(dataset, slug)
            frame = pd.read_csv(path)
            if len(frame) != 200:
                raise RuntimeError(f"reference {dataset}/{slug} has {len(frame)} rows")
            prompts[(dataset, slug)] = frame["prompt"].astype(str).tolist()
            vectors[(dataset, slug)] = model.encode(
                frame["model_response"].astype(str).tolist(), normalize_embeddings=True,
                batch_size=64, show_progress_bar=False,
            )
    rows = []
    for dataset in FC.DATASETS:
        for source in roster:
            for target in roster:
                if source == target:
                    continue
                if prompts[(dataset, source)] != prompts[(dataset, target)]:
                    raise RuntimeError(f"prompt mismatch for {dataset}: {source} vs {target}")
                score = np.sum(vectors[(dataset, source)] * vectors[(dataset, target)], axis=1)
                rows.append({"dataset": dataset, "source": source, "target": target,
                             "seed": "seed42", "unadapted_similarity": float(np.mean(score))})
    return pd.DataFrame(rows)


def correlation_record(x, y) -> dict:
    p = pearsonr(x, y)
    s = spearmanr(x, y)
    # SciPy <1.11 returns a plain ``(statistic, pvalue)`` tuple, while newer
    # releases expose the same values as named attributes.  The campaign
    # analysis must regenerate under either environment.
    p_stat = getattr(p, "statistic", p[0])
    p_value = getattr(p, "pvalue", p[1])
    s_stat = getattr(s, "statistic", s[0])
    s_value = getattr(s, "pvalue", s[1])
    return {"n": int(len(x)), "pearson_r": float(p_stat),
            "pearson_p": float(p_value), "spearman_rho": float(s_stat),
            "spearman_p": float(s_value)}


def analyze(embed: pd.DataFrame, judge: pd.DataFrame, baseline: pd.DataFrame,
            safety_path: str) -> dict:
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {"total_adapters": 1104, "sft": 528, "dpo": 528, "self_sft": 48,
                     "prompts_per_cell": 200},
        "stage_means": {},
    }
    for scorer, frame in (("embed", embed), ("judge", judge)):
        result["stage_means"][scorer] = {
            name: {"n": int(len(group)), "mean": float(group["fidelity_score"].mean()),
                   "std": float(group["fidelity_score"].std(ddof=1))}
            for name, group in frame.groupby("stage")
        }

    result["scorer_agreement"] = correlation_record(
        embed.sort_values("adapter")["fidelity_score"],
        judge.sort_values("adapter")["fidelity_score"],
    )

    result["unadapted_source_target_embed"] = {
        "n": int(len(baseline)), "mean": float(baseline["unadapted_similarity"].mean())}
    result["embed_gain_over_unadapted"] = {}
    for name in ("sft", "dpo"):
        cells = embed[embed["stage"] == name].merge(baseline, on=KEYS, validate="one_to_one")
        cells["gain"] = cells["fidelity_score"] - cells["unadapted_similarity"]
        result["embed_gain_over_unadapted"][name] = {
            "n": int(len(cells)), "mean": float(cells["gain"].mean()),
            "median": float(cells["gain"].median()),
            "pct_positive": float(100 * (cells["gain"] > 0).mean()),
            "by_dataset": {dataset: float(group["gain"].mean())
                           for dataset, group in cells.groupby("dataset")},
        }

    paired = {}
    for scorer, frame in (("embed", embed), ("judge", judge)):
        sft = frame[frame["stage"] == "sft"][KEYS + ["fidelity_score"]]
        dpo = frame[frame["stage"] == "dpo"][KEYS + ["fidelity_score"]]
        joined = sft.merge(dpo, on=KEYS, suffixes=("_sft", "_dpo"), validate="one_to_one")
        delta = joined["fidelity_score_dpo"] - joined["fidelity_score_sft"]
        paired[scorer] = {"n": int(len(delta)), "mean_dpo_minus_sft": float(delta.mean()),
                          "median_dpo_minus_sft": float(delta.median()),
                          "pct_dpo_higher": float(100 * (delta > 0).mean())}
    result["paired_stage_fidelity"] = paired

    safety = pd.read_csv(safety_path)
    harm = (safety[(safety["axis"] == "harm")
                   & safety["adapter"].str.startswith(("sft_", "dpo_"))]
            .groupby("adapter", as_index=False)["erosion"].mean()
            .rename(columns={"erosion": "mean_harm_erosion"}))
    result["fidelity_vs_harm"] = {}
    result["stage_delta_fidelity_vs_harm"] = {}
    for scorer, frame in (("embed", embed), ("judge", judge)):
        merged = frame[frame["stage"].isin(["sft", "dpo"])].merge(
            harm, on="adapter", validate="one_to_one")
        result["fidelity_vs_harm"][scorer] = {
            name: {
                "raw": correlation_record(group["fidelity_score"],
                                          group["mean_harm_erosion"]),
                "within_source_dataset": correlation_record(
                    group["fidelity_score"]
                    - group.groupby(["source", "dataset"])["fidelity_score"].transform("mean"),
                    group["mean_harm_erosion"]
                    - group.groupby(["source", "dataset"])["mean_harm_erosion"].transform("mean"),
                ),
            }
            for name, group in merged.groupby("stage")
        }
        sft = merged[merged["stage"] == "sft"][KEYS + ["fidelity_score", "mean_harm_erosion"]]
        dpo = merged[merged["stage"] == "dpo"][KEYS + ["fidelity_score", "mean_harm_erosion"]]
        deltas = sft.merge(dpo, on=KEYS, suffixes=("_sft", "_dpo"), validate="one_to_one")
        result["stage_delta_fidelity_vs_harm"][scorer] = correlation_record(
            deltas["fidelity_score_dpo"] - deltas["fidelity_score_sft"],
            deltas["mean_harm_erosion_dpo"] - deltas["mean_harm_erosion_sft"],
        )
    return result


def write_tex(result: dict, path: str) -> None:
    stage_means = result["stage_means"]
    gain = result["embed_gain_over_unadapted"]
    paired = result["paired_stage_fidelity"]
    agreement = result["scorer_agreement"]
    lines = [
        "% Generated by analyze_fidelity_campaign.py; do not edit.",
        "\\newcommand{\\FidelityAdapters}{1104}",
        "\\newcommand{\\SelfSFTAdapters}{48}",
        f"\\newcommand{{\\SFTEmbedFidelity}}{{{stage_means['embed']['sft']['mean']:.3f}}}",
        f"\\newcommand{{\\DPOEmbedFidelity}}{{{stage_means['embed']['dpo']['mean']:.3f}}}",
        f"\\newcommand{{\\SelfSFTEmbedFidelity}}{{{stage_means['embed']['self_sft']['mean']:.3f}}}",
        f"\\newcommand{{\\SFTJudgeFidelity}}{{{stage_means['judge']['sft']['mean']:.3f}}}",
        f"\\newcommand{{\\DPOJudgeFidelity}}{{{stage_means['judge']['dpo']['mean']:.3f}}}",
        f"\\newcommand{{\\SelfSFTJudgeFidelity}}{{{stage_means['judge']['self_sft']['mean']:.3f}}}",
        f"\\newcommand{{\\SFTEmbedGain}}{{{gain['sft']['mean']:+.3f}}}",
        f"\\newcommand{{\\DPOEmbedGain}}{{{gain['dpo']['mean']:+.3f}}}",
        f"\\newcommand{{\\SFTEmbedGainPositivePct}}{{{gain['sft']['pct_positive']:.1f}}}",
        f"\\newcommand{{\\DPOEmbedGainPositivePct}}{{{gain['dpo']['pct_positive']:.1f}}}",
        f"\\newcommand{{\\DPOminusSFTEmbedFidelity}}{{{paired['embed']['mean_dpo_minus_sft']:+.3f}}}",
        f"\\newcommand{{\\DPOminusSFTJudgeFidelity}}{{{paired['judge']['mean_dpo_minus_sft']:+.3f}}}",
        f"\\newcommand{{\\FidelityScorerCorrelation}}{{{agreement['pearson_r']:.3f}}}",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", default="data/results/fidelity/fidelity_all_embed_long.csv")
    parser.add_argument("--judge", default="data/results/fidelity/fidelity_all_judge_long.csv")
    parser.add_argument("--safety", default="data/results/safety/erosion_seed42_long.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline-cache",
                        default="data/results/fidelity/unadapted_source_target_embed.csv")
    parser.add_argument("--json-out",
                        default="data/results/fidelity/fidelity_campaign_headlines.json")
    parser.add_argument("--tex-out", default="paper/naz_aaai2027/generated_fidelity_results.tex")
    args = parser.parse_args()

    embed = load_scores(args.embed, "embed")
    judge = load_scores(args.judge, "judge")
    if os.path.exists(args.baseline_cache):
        baseline = pd.read_csv(args.baseline_cache)
    else:
        baseline = reference_embedding_baseline(args.device)
        os.makedirs(os.path.dirname(os.path.abspath(args.baseline_cache)), exist_ok=True)
        baseline.to_csv(args.baseline_cache, index=False)
    if len(baseline) != 528:
        raise RuntimeError(f"embedding baseline has {len(baseline)} cells, expected 528")
    result = analyze(embed, judge, baseline, args.safety)
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_tex(result, args.tex_out)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.json_out} and {args.tex_out}")


if __name__ == "__main__":
    main()
