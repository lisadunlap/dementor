#!/usr/bin/env python3
"""
Score responses with a trained stylometric attribution classifier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import load

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.judge.stylometric_utils import PosTagger, normalize_texts  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Score responses with a stylometric classifier.")
    parser.add_argument("--model-path", required=True, help="Path to stylometric_classifier.joblib.")
    parser.add_argument("--input", required=True, help="CSV with responses to score.")
    parser.add_argument("--text-col", default="model_response", help="Column with response text.")
    parser.add_argument("--prompt-col", default="prompt", help="Column with prompt text.")
    parser.add_argument(
        "--target-label",
        help="Optional label to include target probability as a column.",
    )
    parser.add_argument(
        "--output-probs",
        action="store_true",
        help="Include per-class probability columns for all labels.",
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    model_bundle = load(args.model_path)
    features = model_bundle["features"]
    clf = model_bundle["classifier"]
    label_encoder = model_bundle["label_encoder"]
    preprocess_config = model_bundle.get("preprocess_config", {})

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column '{args.text_col}' in {args.input}")

    texts = df[args.text_col].fillna("").astype(str).tolist()
    prompts = None
    if args.prompt_col in df.columns:
        prompts = df[args.prompt_col].fillna("").astype(str).tolist()
    if preprocess_config:
        texts = normalize_texts(texts, prompts, preprocess_config)
    X = features.transform(texts).toarray()
    probs = clf.predict_proba(X)
    preds = label_encoder.inverse_transform(probs.argmax(axis=1))

    df = df.copy()
    df["stylometric_pred_label"] = preds
    df["stylometric_pred_prob"] = probs.max(axis=1)

    if args.target_label:
        if args.target_label not in label_encoder.classes_:
            raise ValueError(
                f"Target label '{args.target_label}' not in classifier labels: "
                f"{', '.join(label_encoder.classes_)}"
            )
        target_idx = list(label_encoder.classes_).index(args.target_label)
        df["stylometric_target_prob"] = probs[:, target_idx]

    if args.output_probs:
        for idx, label in enumerate(label_encoder.classes_):
            col = f"stylometric_prob_{label}"
            df[col] = probs[:, idx]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved scores to {output_path}")


if __name__ == "__main__":
    main()
