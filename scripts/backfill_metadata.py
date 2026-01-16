#!/usr/bin/env python3
"""
Backfill missing source_model/target_model metadata in legacy CSV runs.

Scans data/results for CSVs that have prompt/model_response style columns but
lack source/target metadata, infers model ids from filenames, and updates the
files in-place so the viewer can display them correctly.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

RESULTS_ROOT = Path("data/results")
_EXCLUDED_DIR_NAMES = {"scores", "metrics", "judgments"}
_EXCLUDED_SEGMENT_SUFFIXES = ("_scores", "_metrics", "_judgments")
_EXCLUDED_FILE_STEMS = {"scored", "scored_metrics"}

PROVIDER_MAP: Dict[str, str] = {
    "openai": "openai",
    "meta-llama": "meta-llama",
    "meta": "meta",
    "mistralai": "mistralai",
    "google": "google",
    "qwen": "Qwen",
    "qwen_qwen": "Qwen",
    "opengvlab": "OpenGVLab",
    "microsoft": "microsoft",
    "anthropic": "anthropic",
    "deepseek": "deepseek",
    "yi": "yi",
    "grok": "xai",
    "xai": "xai",
    "hf": "hf",
    "huggingface": "huggingface",
    "meta-llama3": "meta-llama3",
}


def _has_excluded_segment(path: Path) -> bool:
    for part in path.parts:
        normalized = part.lower()
        if normalized in _EXCLUDED_DIR_NAMES:
            return True
        if any(normalized.endswith(suffix) for suffix in _EXCLUDED_SEGMENT_SUFFIXES):
            return True
    return path.stem.lower() in _EXCLUDED_FILE_STEMS


def _normalize_model_id(raw: str) -> str:
    text = raw.strip("_- ")
    if not text:
        return ""
    lowered = text.lower()
    for key, canonical in PROVIDER_MAP.items():
        if lowered.startswith(key):
            remainder = text[len(key):]
            remainder = remainder.lstrip("_-/")
            return f"{canonical}/{remainder}" if remainder else canonical
    return text


def _infer_models_from_stem(stem: str) -> Tuple[str, str]:
    source = ""
    target = ""
    remainder = ""
    if "_as_" in stem:
        source, remainder = stem.split("_as_", 1)
        target = remainder
    if "_vs_" in stem:
        left, right = stem.split("_vs_", 1)
        if not source:
            source = left
        if target:
            target = target.split("_vs_", 1)[0]
        else:
            target = right
    source = _normalize_model_id(source)
    target = _normalize_model_id(target)
    return source or "", target or ""


def _needs_metadata(df: pd.DataFrame) -> bool:
    if "prompt" not in df.columns:
        return False
    if "model_response" not in df.columns and "target_response" not in df.columns:
        return False
    src_missing = "source_model" not in df.columns or df["source_model"].isna().all() or (df["source_model"] == "").all()
    tgt_missing = "target_model" not in df.columns or df["target_model"].isna().all() or (df["target_model"] == "").all()
    return src_missing or tgt_missing


def _insert_column(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if column in df.columns:
        df[column] = df[column].fillna(value)
        df.loc[df[column] == "", column] = value
        return df
    insert_after = None
    for base in ("target_response", "model_response", "prompt"):
        if base in df.columns:
            insert_after = base
            break
    if insert_after is None:
        df[column] = value
        return df
    idx = list(df.columns).index(insert_after) + 1
    before = df.columns[:idx]
    after = df.columns[idx:]
    new_cols = list(before) + [column] + list(after)
    df[column] = value
    return df[new_cols]


def _process_csv(path: Path) -> bool:
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if not _needs_metadata(df):
        return False
    source, target = _infer_models_from_stem(path.stem)
    source = source or "unknown"
    target = target or "unknown"
    updated = False
    if "source_model" not in df.columns or df["source_model"].isna().all() or (df["source_model"] == "").all():
        df = _insert_column(df, "source_model", source)
        updated = True
    if "target_model" not in df.columns or df["target_model"].isna().all() or (df["target_model"] == "").all():
        df = _insert_column(df, "target_model", target)
        updated = True
    if not updated:
        return False
    df.to_csv(path, index=False)
    return True


def iter_csvs(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.csv"):
        if _has_excluded_segment(path):
            continue
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing source/target metadata in disguise CSVs.")
    parser.add_argument("--root", default=str(RESULTS_ROOT), help="Root directory to scan (default: data/results).")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist.")

    total = 0
    updated = 0
    for csv_path in iter_csvs(root):
        total += 1
        if _process_csv(csv_path):
            updated += 1
            print(f"Updated {csv_path}")
    print(f"Processed {total} CSVs, added metadata to {updated}.")


if __name__ == "__main__":
    main()
