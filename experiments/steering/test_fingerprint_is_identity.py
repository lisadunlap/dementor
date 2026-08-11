#!/usr/bin/env python
"""Test whether fingerprint ablation moves benign style toward the reference model.

The steering paper derives a benign cross-model provenance contrast in model M's residual stream.
Calling that contrast a complete identity representation would require evidence beyond its safety
dissociation. This diagnostic asks a narrower held-out question: on coherent benign XSTest rows,
does projection-ablation move M's response embedding toward the reference response more than a
matched random-direction ablation does?

For each shared prompt, the script compares cosine distance from M's baseline, fingerprint-ablated,
and random-ablated responses to the Llama-3.1-8B reference response. Rows failing the steering
coherence gate are removed, and the fingerprint-vs-random comparison is paired by prompt and model.
Failure to move response embeddings toward the reference beyond the random arm is evidence for the
bounded term ``benign provenance contrast`` rather than ``identity direction``.

Usage:
    python experiments/steering/test_fingerprint_is_identity.py [--beta 1.0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from dementor import config

R = os.environ.get("DEMENTOR_STEER_WORK", "/data/ethantsliu/exp_steer_safety/repl80_rdo")
ANCHOR = "llama-3.1-8b"


def summarize(rows: list[dict]) -> dict:
    """Build a JSON-serializable paired summary from per-model measurements."""
    fingerprint_deltas = [row["fingerprint_delta"] for row in rows]
    random_deltas = [row["random_delta"] for row in rows]
    result = {
        "n_models": len(rows),
        "mean_fingerprint_delta": statistics.mean(fingerprint_deltas),
        "mean_random_delta": statistics.mean(random_deltas),
        "fingerprint_gt_random": sum(
            fingerprint > random
            for fingerprint, random in zip(fingerprint_deltas, random_deltas)
        ),
        "wilcoxon_statistic": None,
        "wilcoxon_p_value": None,
    }
    try:
        from experiments.figures.rebuild_steering_figures import paired_signed_rank_exact

        test = paired_signed_rank_exact(fingerprint_deltas, random_deltas)
        result["wilcoxon_statistic"] = test["statistic"]
        result["wilcoxon_p_value"] = test["p_value"]
        result["wilcoxon_method"] = test["method"]
    except (ImportError, ValueError):
        result["wilcoxon_method"] = None
        pass
    return result


def coherent(df: pd.DataFrame) -> pd.DataFrame:
    r4 = pd.to_numeric(df.get("rep4"), errors="coerce")
    ppl = pd.to_numeric(df.get("ppl"), errors="coerce")
    word_count = df["model_response"].astype(str).str.split().str.len()
    return df[(r4 < 0.5) & (ppl < 100) & (word_count >= 5)]


def benign(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    data = pd.read_csv(path)
    if "expected" not in data.columns:
        return None
    data = data[data["expected"] == "comply"].copy()
    data["model_response"] = data["model_response"].astype(str)
    return coherent(data)[["prompt", "model_response"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", default="1.0")
    parser.add_argument("--min-prompts", type=int, default=25)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    anchor = benign(f"{R}/{ANCHOR}/eval_xstest/parts/baseline_b0.csv")
    if anchor is None or not len(anchor):
        sys.exit(f"no usable anchor benign rows for {ANCHOR}")
    print(f"anchor = {ANCHOR}, {len(anchor)} coherent benign rows\n")

    rows = []
    for directory in sorted(glob.glob(f"{R}/*/eval_xstest")):
        artifact_model = directory.split("/")[-2]
        if artifact_model == ANCHOR or artifact_model.endswith("_dim8"):
            continue
        base = benign(f"{directory}/parts/baseline_b0.csv")
        fingerprint = benign(f"{directory}/parts/fingerprint_b{args.beta}.csv")
        random = benign(f"{directory}/parts/random_b{args.beta}.csv")
        if any(frame is None or not len(frame) for frame in (base, fingerprint, random)):
            continue
        prompts = sorted(
            set(base["prompt"])
            & set(fingerprint["prompt"])
            & set(random["prompt"])
            & set(anchor["prompt"])
        )
        if len(prompts) < args.min_prompts:
            continue

        def vectors(frame: pd.DataFrame) -> np.ndarray:
            responses = (
                frame.drop_duplicates("prompt")
                .set_index("prompt")
                .loc[prompts, "model_response"]
            )
            return np.asarray(
                encoder.encode(list(responses), normalize_embeddings=True, show_progress_bar=False)
            )

        anchor_vectors = vectors(anchor)
        base_vectors = vectors(base)
        fingerprint_vectors = vectors(fingerprint)
        random_vectors = vectors(random)
        base_distance = float(np.mean(1 - np.sum(base_vectors * anchor_vectors, axis=1)))
        fingerprint_distance = float(
            np.mean(1 - np.sum(fingerprint_vectors * anchor_vectors, axis=1))
        )
        random_distance = float(np.mean(1 - np.sum(random_vectors * anchor_vectors, axis=1)))
        rows.append({
            "model": config.canonical_steering_slug(artifact_model),
            "n_prompts": len(prompts),
            "base_distance": base_distance,
            "fingerprint_distance": fingerprint_distance,
            "random_distance": random_distance,
            "fingerprint_delta": base_distance - fingerprint_distance,
            "random_delta": base_distance - random_distance,
        })

    if not rows:
        sys.exit("no model had all three arms with enough coherent benign rows")

    print(f"{'model':24s} {'n':>4s} {'d_base':>7s} {'d_fp':>7s} {'d_rnd':>7s} {'Δfp':>7s} {'Δrnd':>7s}")
    for row in sorted(rows, key=lambda row: -row["fingerprint_delta"]):
        print(
            f"{row['model']:24s} {row['n_prompts']:4d} {row['base_distance']:7.4f} "
            f"{row['fingerprint_distance']:7.4f} {row['random_distance']:7.4f} "
            f"{row['fingerprint_delta']:+7.4f} {row['random_delta']:+7.4f}"
        )

    aggregate = summarize(rows)
    print(f"\nn models = {aggregate['n_models']}")
    print(
        "  mean Δ embedding distance, fingerprint ablation : "
        f"{aggregate['mean_fingerprint_delta']:+.4f}"
    )
    print(
        "  mean Δ embedding distance, random ablation      : "
        f"{aggregate['mean_random_delta']:+.4f}"
    )
    print(
        "  fingerprint moves more than random in       : "
        f"{aggregate['fingerprint_gt_random']}/{aggregate['n_models']}"
    )
    if aggregate["wilcoxon_statistic"] is not None:
        print(
            "  Wilcoxon fingerprint vs random             : "
            f"W={aggregate['wilcoxon_statistic']:.0f}, "
            f"p={aggregate['wilcoxon_p_value']:.4g}"
        )
    print("\nREAD: Δ > 0 means ablation moved the response embedding toward the reference.")
    print("      The stronger claim requires Δfp > Δrnd, not merely Δfp > 0.")

    if args.json_out is not None:
        payload = {
            "schema_version": 1,
            "work_root": R,
            "anchor_model": ANCHOR,
            "encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "metric": "cosine_distance_between_response_embeddings",
            "beta": args.beta,
            "min_prompts": args.min_prompts,
            "per_model": sorted(rows, key=lambda row: row["model"]),
            "aggregate": aggregate,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
