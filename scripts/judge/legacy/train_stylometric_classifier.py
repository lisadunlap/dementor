#!/usr/bin/env python3
"""
Train a stylometric attribution classifier using char/word/POS n-grams.

This matches the "fingerprint" setup:
  - char n-grams: (2, 4)
  - word n-grams: (3, 5)
  - POS n-grams: (3, 5)
  - max_features per feature set: 2000
  - GradientBoostingClassifier with the paper's hyperparameters
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.judge.stylometric_utils import FormatFeatures, PosTagger, normalize_texts


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


def _apply_min_chars(
    texts: List[str],
    labels: List[str],
    prompts: List[Optional[str]],
    min_chars: Optional[int],
) -> Tuple[List[str], List[str], List[Optional[str]]]:
    if not min_chars:
        return texts, labels, prompts

    kept_texts: List[str] = []
    kept_labels: List[str] = []
    kept_prompts: List[Optional[str]] = []
    for text, label, prompt in zip(texts, labels, prompts):
        if len(text) >= min_chars:
            kept_texts.append(text)
            kept_labels.append(label)
            kept_prompts.append(prompt)

    return kept_texts, kept_labels, kept_prompts


def _dedupe_records(
    texts: List[str],
    labels: List[str],
    prompts: List[Optional[str]],
    dedupe_key: str,
    extra_texts: Optional[set[str]],
) -> Tuple[List[str], List[str], List[Optional[str]]]:
    seen: set[str] = set()
    kept_texts: List[str] = []
    kept_labels: List[str] = []
    kept_prompts: List[Optional[str]] = []

    for text, label, prompt in zip(texts, labels, prompts):
        if dedupe_key == "prompt_text" and prompt:
            key = f"{prompt}||{text}"
        else:
            key = text
        if key in seen:
            continue
        if extra_texts and key in extra_texts:
            continue
        seen.add(key)
        kept_texts.append(text)
        kept_labels.append(label)
        kept_prompts.append(prompt)

    return kept_texts, kept_labels, kept_prompts


def _length_match(
    texts: List[str],
    labels: List[str],
    prompts: List[Optional[str]],
    bins: int,
    seed: int,
) -> Tuple[List[str], List[str], List[Optional[str]]]:
    if bins <= 1:
        return texts, labels, prompts

    df = pd.DataFrame(
        {
            "text": texts,
            "label": labels,
            "prompt": prompts,
            "length": [len(t) for t in texts],
        }
    )
    df["bin"] = pd.qcut(df["length"], q=bins, duplicates="drop")
    rng = random.Random(seed)
    kept_rows = []
    for _, bin_df in df.groupby("bin"):
        counts = bin_df["label"].value_counts()
        min_count = counts.min()
        if min_count == 0:
            continue
        for label, label_df in bin_df.groupby("label"):
            sample_df = label_df.sample(
                n=min_count,
                random_state=rng.randint(0, 1_000_000),
                replace=False,
            )
            kept_rows.append(sample_df)
    if not kept_rows:
        return texts, labels, prompts
    kept_df = pd.concat(kept_rows).sample(
        frac=1.0, random_state=rng.randint(0, 1_000_000)
    )
    return (
        kept_df["text"].tolist(),
        kept_df["label"].tolist(),
        kept_df["prompt"].tolist(),
    )


def train_classifier(
    texts: List[str],
    labels: List[str],
    spacy_model: str,
    max_features: int,
    seed: int,
    classifier_type: str,
    char_ngram_range: Tuple[int, int],
    word_ngram_range: Tuple[int, int],
    pos_ngram_range: Tuple[int, int],
    preprocess_config: dict,
    include_char: bool,
    include_word: bool,
    include_pos: bool,
    include_format: bool,
) -> Tuple[FeatureUnion, object, LabelEncoder]:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    feature_list = []
    if include_char:
        feature_list.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=char_ngram_range,
                    max_features=max_features,
                ),
            )
        )
    if include_word:
        feature_list.append(
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=word_ngram_range,
                    max_features=max_features,
                ),
            )
        )
    if include_pos:
        feature_list.append(
            (
                "pos",
                Pipeline(
                    [
                        ("pos", PosTagger(spacy_model)),
                        (
                            "vec",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=pos_ngram_range,
                                max_features=max_features,
                            ),
                        ),
                    ]
                ),
            )
        )
    if include_format:
        feature_list.append(("format", FormatFeatures()))
    if not feature_list:
        raise ValueError("At least one of char/word/pos features must be enabled.")

    features = FeatureUnion(feature_list)

    X = features.fit_transform(texts).toarray()

    if classifier_type == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it with `pip install xgboost`."
            ) from exc

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.2,
            subsample=0.8,
            colsample_bytree=0.7,
            objective="multi:softprob",
            num_class=len(label_encoder.classes_),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
        )
    else:
        clf = GradientBoostingClassifier(
            learning_rate=0.2,
            n_estimators=90,
            max_depth=8,
            max_features="sqrt",
            subsample=0.8,
            random_state=seed,
            min_samples_leaf=30,
            min_samples_split=400,
        )
    clf.fit(X, y)

    return features, clf, label_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stylometric attribution classifier.")
    parser.add_argument("--inputs", nargs="+", required=True, help="CSV files with model responses.")
    parser.add_argument(
        "--input-labels",
        nargs="*",
        help="Optional labels for each input file (same order as --inputs).",
    )
    parser.add_argument("--text-col", default="model_response", help="Column with response text.")
    parser.add_argument("--prompt-col", default="prompt", help="Column with prompt text.")
    parser.add_argument("--label-col", default="model", help="Column with labels if present.")
    parser.add_argument(
        "--label-from-filename",
        action="store_true",
        help="Use filename stem as label if no label column/value.",
    )
    parser.add_argument(
        "--drop-error-prefix",
        action="append",
        default=["ERROR:"],
        help="Drop rows whose text starts with this prefix (repeatable).",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate normalized texts (or prompt+text) within the training set.",
    )
    parser.add_argument(
        "--dedupe-key",
        choices=["text", "prompt_text"],
        default="text",
        help="Key for deduplication.",
    )
    parser.add_argument(
        "--dedupe-against",
        nargs="*",
        help="Optional CSVs to drop any matching dedupe keys from training.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=5000,
        help="Cap samples per label (matches the paper default).",
    )
    parser.add_argument("--seed", type=int, default=10, help="Random seed.")
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model for POS tagging.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=2000,
        help="Max features per feature set.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=0,
        help="Drop responses shorter than this many characters after normalization.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Truncate responses to this many characters after normalization (0 = no limit).",
    )
    parser.add_argument(
        "--strip-leading-markers",
        action="store_true",
        help="Strip common leading discourse markers (e.g., 'Let's', 'To', 'First,').",
    )
    parser.add_argument(
        "--strip-markdown",
        action="store_true",
        help="Strip markdown formatting (headers, bullets, code fences, emphasis).",
    )
    parser.add_argument(
        "--strip-punctuation",
        action="store_true",
        help="Strip punctuation characters before feature extraction.",
    )
    parser.add_argument(
        "--strip-reasoning-phrases",
        action="store_true",
        help="Strip common step-by-step/solution lead-in phrases.",
    )
    parser.add_argument(
        "--strip-prompt-echo",
        action="store_true",
        help="Remove prompt prefix if the response starts with the prompt.",
    )
    parser.add_argument(
        "--strip-answer-prefix",
        action="store_true",
        help="Strip leading 'Answer:' or 'Final answer:' prefix.",
    )
    parser.add_argument(
        "--collapse-whitespace",
        action="store_true",
        help="Collapse all whitespace runs to single spaces.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase text before feature extraction.",
    )
    parser.add_argument(
        "--length-match-bins",
        type=int,
        default=0,
        help="If >1, length-match per label within quantile bins.",
    )
    parser.add_argument(
        "--disable-char",
        action="store_true",
        help="Disable char n-gram features.",
    )
    parser.add_argument(
        "--disable-word",
        action="store_true",
        help="Disable word n-gram features.",
    )
    parser.add_argument(
        "--disable-pos",
        action="store_true",
        help="Disable POS n-gram features.",
    )
    parser.add_argument(
        "--include-format-features",
        action="store_true",
        help="Include markdown/list structure features.",
    )
    parser.add_argument(
        "--char-ngram-min",
        type=int,
        default=3,
        help="Min char n-gram size.",
    )
    parser.add_argument(
        "--char-ngram-max",
        type=int,
        default=5,
        help="Max char n-gram size.",
    )
    parser.add_argument(
        "--word-ngram-min",
        type=int,
        default=2,
        help="Min word n-gram size.",
    )
    parser.add_argument(
        "--word-ngram-max",
        type=int,
        default=4,
        help="Max word n-gram size.",
    )
    parser.add_argument(
        "--pos-ngram-min",
        type=int,
        default=2,
        help="Min POS n-gram size.",
    )
    parser.add_argument(
        "--pos-ngram-max",
        type=int,
        default=4,
        help="Max POS n-gram size.",
    )
    parser.add_argument(
        "--classifier",
        choices=["gradient_boosting", "xgboost"],
        default="gradient_boosting",
        help="Classifier type to train.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save model artifacts.")
    args = parser.parse_args()

    if args.input_labels and len(args.input_labels) != len(args.inputs):
        raise ValueError("--input-labels must match the number of --inputs")

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
    texts, labels, prompts = _apply_min_chars(texts, labels, prompts, args.min_chars)

    extra_dedupe: Optional[set[str]] = None
    if args.dedupe_against:
        extra_dedupe = set()
        for path in args.dedupe_against:
            df = pd.read_csv(path)
            if args.text_col not in df.columns:
                raise ValueError(f"Missing text column '{args.text_col}' in {path}")
            prompt_series = None
            if args.prompt_col in df.columns:
                prompt_series = df[args.prompt_col].fillna("").astype(str).tolist()
            texts_against = df[args.text_col].fillna("").astype(str).tolist()
            normalized = normalize_texts(texts_against, prompt_series, preprocess_config)
            for text, prompt in zip(normalized, prompt_series or [None] * len(normalized)):
                if args.dedupe_key == "prompt_text" and prompt:
                    key = f"{prompt}||{text}"
                else:
                    key = text
                extra_dedupe.add(key)

    if args.dedupe:
        texts, labels, prompts = _dedupe_records(
            texts, labels, prompts, args.dedupe_key, extra_dedupe
        )

    if args.length_match_bins and args.length_match_bins > 1:
        texts, labels, prompts = _length_match(
            texts, labels, prompts, args.length_match_bins, args.seed
        )

    texts, labels, prompts = _cap_per_label(
        texts, labels, prompts, args.max_per_label, args.seed
    )

    features, clf, label_encoder = train_classifier(
        texts,
        labels,
        args.spacy_model,
        args.max_features,
        args.seed,
        args.classifier,
        (args.char_ngram_min, args.char_ngram_max),
        (args.word_ngram_min, args.word_ngram_max),
        (args.pos_ngram_min, args.pos_ngram_max),
        preprocess_config,
        not args.disable_char,
        not args.disable_word,
        not args.disable_pos,
        args.include_format_features,
    )

    X_train = features.transform(texts).toarray()
    train_probs = clf.predict_proba(X_train)
    train_preds = label_encoder.inverse_transform(train_probs.argmax(axis=1))
    train_acc = (train_preds == labels).mean()
    y_train = label_encoder.transform(labels)
    from sklearn.metrics import log_loss

    train_log_loss = log_loss(y_train, train_probs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "stylometric_classifier.joblib"
    dump(
        {
            "features": features,
            "classifier": clf,
            "label_encoder": label_encoder,
            "preprocess_config": preprocess_config,
            "train_metrics": {
                "train_accuracy": train_acc,
                "train_log_loss": train_log_loss,
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
        "length_match_bins": args.length_match_bins,
        "disable_char": args.disable_char,
        "disable_word": args.disable_word,
        "disable_pos": args.disable_pos,
        "include_format_features": args.include_format_features,
        "train_metrics": {
            "train_accuracy": train_acc,
            "train_log_loss": train_log_loss,
        },
        "classifier": args.classifier,
        "label_counts": label_counts,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved classifier to {model_path}")
    print("Label counts:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
