#!/usr/bin/env python3
"""
Train and score a multi-view stylometric ensemble.

Ensemble components:
  1) Stylometric n-grams (char/word/POS)
  2) Structural + POS distribution features
  3) Word TF-IDF + SVD (LSA) + LogisticRegression
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.judge.stylometric_utils import (
    FormatFeatures,
    PosDistributionFeatures,
    StructureFeatures,
    ToneFeatures,
    normalize_texts,
)
from scripts.judge.legacy.train_stylometric_classifier import train_classifier


def _load_records(
    paths: List[Path],
    input_labels: Optional[List[str]],
    text_col: str,
    label_col: str,
    label_from_filename: bool,
    drop_error_prefixes: List[str],
    prompt_col: str,
) -> Tuple[List[str], List[str], List[Optional[str]]]:
    texts: List[str] = []
    labels: List[str] = []
    prompts: List[Optional[str]] = []

    for idx, path in enumerate(paths):
        df = pd.read_csv(path)
        if text_col not in df.columns:
            raise ValueError(f"Missing text column '{text_col}' in {path}")

        file_label = None
        if input_labels:
            file_label = input_labels[idx]

        label_series = None
        if label_col in df.columns:
            label_series = df[label_col].fillna("").astype(str)

        prompt_series = None
        if prompt_col in df.columns:
            prompt_series = df[prompt_col].fillna("").astype(str)

        for row_idx, text in enumerate(df[text_col].fillna("").astype(str)):
            text = text.strip()
            if not text:
                continue
            if any(text.startswith(prefix) for prefix in drop_error_prefixes):
                continue

            if file_label:
                label = file_label
            elif label_series is not None and label_series.iloc[row_idx]:
                label = label_series.iloc[row_idx]
            elif label_from_filename:
                label = path.stem
            else:
                raise ValueError(
                    f"No label available for {path} row {row_idx}. "
                    "Provide --input-labels, include a label column, "
                    "or enable --label-from-filename."
                )

            texts.append(text)
            labels.append(label)
            if prompt_series is not None:
                prompts.append(prompt_series.iloc[row_idx])
            else:
                prompts.append(None)

    return texts, labels, prompts


def _cap_per_label(
    texts: List[str],
    labels: List[str],
    prompts: List[Optional[str]],
    max_per_label: Optional[int],
    seed: int,
) -> Tuple[List[str], List[str], List[Optional[str]]]:
    if not max_per_label:
        return texts, labels, prompts

    indices_by_label: dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        indices_by_label[label].append(idx)

    rng = random.Random(seed)
    kept_indices: List[int] = []
    for label, indices in indices_by_label.items():
        if len(indices) > max_per_label:
            indices = rng.sample(indices, max_per_label)
        kept_indices.extend(indices)

    rng.shuffle(kept_indices)
    return (
        [texts[i] for i in kept_indices],
        [labels[i] for i in kept_indices],
        [prompts[i] for i in kept_indices],
    )


def _train_structure_classifier(
    texts: List[str],
    labels: List[str],
    spacy_model: str,
    seed: int,
) -> Tuple[FeatureUnion, LogisticRegression, LabelEncoder, dict]:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    features = FeatureUnion(
        [
            ("structure", StructureFeatures()),
            ("format", FormatFeatures()),
            ("pos_dist", PosDistributionFeatures(spacy_model)),
            ("tone", ToneFeatures()),
        ]
    )
    X = features.fit_transform(texts)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = label_encoder.inverse_transform(probs.argmax(axis=1))
    train_acc = (preds == labels).mean()
    train_log = log_loss(y, probs)
    metrics = {"train_accuracy": train_acc, "train_log_loss": train_log}
    return features, clf, label_encoder, metrics


def _train_lsa_classifier(
    texts: List[str],
    labels: List[str],
    seed: int,
) -> Tuple[Pipeline, LogisticRegression, LabelEncoder, dict]:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=5000,
    )
    svd = TruncatedSVD(n_components=200, random_state=seed)
    clf = LogisticRegression(
        max_iter=500,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=seed,
    )

    pipeline = Pipeline([("tfidf", tfidf), ("svd", svd)])
    X = pipeline.fit_transform(texts)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = label_encoder.inverse_transform(probs.argmax(axis=1))
    train_acc = (preds == labels).mean()
    train_log = log_loss(y, probs)
    metrics = {"train_accuracy": train_acc, "train_log_loss": train_log}
    return pipeline, clf, label_encoder, metrics


def _score_bundle(
    texts: List[str],
    features,
    clf,
    label_encoder: LabelEncoder,
) -> Tuple[np.ndarray, List[str]]:
    X = features.transform(texts)
    if hasattr(X, "toarray"):
        X = X.toarray()
    probs = clf.predict_proba(X)
    preds = label_encoder.inverse_transform(probs.argmax(axis=1))
    return probs, preds


def train_ensemble(args: argparse.Namespace) -> None:
    paths = [Path(p) for p in args.inputs]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    texts, labels, prompts = _load_records(
        paths,
        args.input_labels,
        args.text_col,
        args.label_col,
        args.label_from_filename,
        args.drop_error_prefix,
        args.prompt_col,
    )
    if not texts:
        raise ValueError("No training examples found after filtering.")

    preprocess_config = {
        "strip_prompt_echo": args.strip_prompt_echo,
        "strip_answer_prefix": args.strip_answer_prefix,
        "strip_leading_markers": args.strip_leading_markers,
        "strip_markdown": args.strip_markdown,
        "strip_punctuation": args.strip_punctuation,
        "strip_reasoning_phrases": args.strip_reasoning_phrases,
        "collapse_whitespace": args.collapse_whitespace,
        "lowercase": args.lowercase,
        "max_chars": args.max_chars if args.max_chars > 0 else None,
    }

    texts = normalize_texts(texts, prompts, preprocess_config)
    texts, labels, prompts = _cap_per_label(texts, labels, prompts, args.max_per_label, args.seed)

    # Classifier 1: stylometric n-grams
    features_1, clf_1, le_1 = train_classifier(
        texts,
        labels,
        args.spacy_model,
        args.max_features,
        args.seed,
        "gradient_boosting",
        (args.char_ngram_min, args.char_ngram_max),
        (args.word_ngram_min, args.word_ngram_max),
        (args.pos_ngram_min, args.pos_ngram_max),
        preprocess_config,
        True,
        True,
        True,
        False,
    )
    X1 = features_1.transform(texts).toarray()
    probs1 = clf_1.predict_proba(X1)
    preds1 = le_1.inverse_transform(probs1.argmax(axis=1))
    train_acc1 = (preds1 == labels).mean()
    train_log1 = log_loss(le_1.transform(labels), probs1)

    # Classifier 2: structure + POS distribution
    features_2, clf_2, le_2, metrics_2 = _train_structure_classifier(
        texts, labels, args.spacy_model, args.seed
    )

    # Classifier 3: LSA
    features_3, clf_3, le_3, metrics_3 = _train_lsa_classifier(
        texts, labels, args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ensemble_stylometric.joblib"
    dump(
        {
            "preprocess_config": preprocess_config,
            "label_encoder": le_1,
            "stylometric": {
                "features": features_1,
                "classifier": clf_1,
                "train_metrics": {
                    "train_accuracy": train_acc1,
                    "train_log_loss": train_log1,
                },
            },
            "structure": {
                "features": features_2,
                "classifier": clf_2,
                "train_metrics": metrics_2,
            },
            "lsa": {
                "features": features_3,
                "classifier": clf_3,
                "train_metrics": metrics_3,
            },
        },
        model_path,
    )

    label_counts = pd.Series(labels).value_counts().to_dict()
    metadata = {
        "inputs": [str(p) for p in paths],
        "text_col": args.text_col,
        "label_col": args.label_col,
        "label_from_filename": args.label_from_filename,
        "drop_error_prefixes": args.drop_error_prefix,
        "max_per_label": args.max_per_label,
        "seed": args.seed,
        "spacy_model": args.spacy_model,
        "max_features": args.max_features,
        "char_ngram_range": [args.char_ngram_min, args.char_ngram_max],
        "word_ngram_range": [args.word_ngram_min, args.word_ngram_max],
        "pos_ngram_range": [args.pos_ngram_min, args.pos_ngram_max],
        "preprocess_config": preprocess_config,
        "label_counts": label_counts,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved ensemble to {model_path}")


def score_ensemble(args: argparse.Namespace) -> None:
    bundle = load(args.model_path)
    preprocess_config = bundle["preprocess_config"]
    le = bundle["label_encoder"]

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column '{args.text_col}' in {args.input}")
    texts = df[args.text_col].fillna("").astype(str).tolist()
    prompts = df[args.prompt_col].fillna("").astype(str).tolist() if args.prompt_col in df.columns else None
    texts = normalize_texts(texts, prompts, preprocess_config)

    probs_1, preds_1 = _score_bundle(
        texts, bundle["stylometric"]["features"], bundle["stylometric"]["classifier"], le
    )
    probs_2, preds_2 = _score_bundle(
        texts, bundle["structure"]["features"], bundle["structure"]["classifier"], le
    )
    probs_3, preds_3 = _score_bundle(
        texts, bundle["lsa"]["features"], bundle["lsa"]["classifier"], le
    )

    df = df.copy()
    df["stylometric_pred_label"] = preds_1
    df["structure_pred_label"] = preds_2
    df["lsa_pred_label"] = preds_3

    # Aggregate probabilities
    probs_mean = (probs_1 + probs_2 + probs_3) / 3.0
    for idx, label in enumerate(le.classes_):
        df[f"ensemble_prob_{label}"] = probs_mean[:, idx]

    # Voting
    votes = np.vstack([preds_1, preds_2, preds_3]).T
    majority = []
    unanimous = []
    agree_count = []
    for row in votes:
        counts = pd.Series(row).value_counts()
        agree_count.append(int(counts.iloc[0]))
        majority.append(counts.idxmax())
        if counts.iloc[0] == 3:
            unanimous.append(counts.idxmax())
        else:
            unanimous.append("no_agreement")
    df["ensemble_majority_label"] = majority
    df["ensemble_unanimous_label"] = unanimous
    df["ensemble_agreement_count"] = agree_count

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved ensemble scores to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stylometric ensemble trainer/scorer.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train ensemble.")
    train.add_argument("--inputs", nargs="+", required=True, help="CSV files with model responses.")
    train.add_argument("--input-labels", nargs="*", help="Optional labels for each input file.")
    train.add_argument("--text-col", default="model_response", help="Column with response text.")
    train.add_argument("--prompt-col", default="prompt", help="Column with prompt text.")
    train.add_argument("--label-col", default="model", help="Column with labels if present.")
    train.add_argument(
        "--label-from-filename",
        action="store_true",
        help="Use filename stem as label if no label column/value.",
    )
    train.add_argument(
        "--drop-error-prefix",
        action="append",
        default=["ERROR:"],
        help="Drop rows whose text starts with this prefix (repeatable).",
    )
    train.add_argument(
        "--max-per-label",
        type=int,
        default=5000,
        help="Cap samples per label.",
    )
    train.add_argument("--seed", type=int, default=10, help="Random seed.")
    train.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model for POS tagging.",
    )
    train.add_argument("--max-features", type=int, default=2000, help="Max features per set.")
    train.add_argument("--max-chars", type=int, default=0, help="Truncate responses (0 = no limit).")
    train.add_argument("--strip-prompt-echo", action="store_true")
    train.add_argument("--strip-answer-prefix", action="store_true")
    train.add_argument("--strip-leading-markers", action="store_true")
    train.add_argument("--strip-markdown", action="store_true")
    train.add_argument("--strip-punctuation", action="store_true")
    train.add_argument("--strip-reasoning-phrases", action="store_true")
    train.add_argument("--collapse-whitespace", action="store_true")
    train.add_argument("--lowercase", action="store_true")
    train.add_argument("--char-ngram-min", type=int, default=3)
    train.add_argument("--char-ngram-max", type=int, default=5)
    train.add_argument("--word-ngram-min", type=int, default=2)
    train.add_argument("--word-ngram-max", type=int, default=4)
    train.add_argument("--pos-ngram-min", type=int, default=2)
    train.add_argument("--pos-ngram-max", type=int, default=4)
    train.add_argument("--output-dir", required=True)

    score = sub.add_parser("score", help="Score with ensemble.")
    score.add_argument("--model-path", required=True)
    score.add_argument("--input", required=True)
    score.add_argument("--text-col", default="model_response")
    score.add_argument("--prompt-col", default="prompt")
    score.add_argument("--output", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train_ensemble(args)
    elif args.command == "score":
        score_ensemble(args)


if __name__ == "__main__":
    main()
