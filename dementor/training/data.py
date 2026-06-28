from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class FinetuneExample:
    """Container for a single prompt/response pair used for supervised fine-tuning."""

    prompt: str
    completion: str


@dataclass(frozen=True)
class PreferenceExample:
    """Container for a single prompt with preferred/rejected completions."""

    prompt: str
    chosen: str
    rejected: str


def load_csv_dataset(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    """Load a CSV file and verify the required columns are present."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns {set(required_columns)}. Missing: {missing}")
    return df


def sample_supervised_splits(
    df: pd.DataFrame,
    dataset_size: int,
    train_size: int,
    eval_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle and split a dataframe into train/eval slices of fixed size."""
    if train_size + eval_size != dataset_size:
        raise ValueError("train_size + eval_size must equal dataset_size.")
    if len(df) < dataset_size:
        raise ValueError(f"Requested dataset_size {dataset_size} but dataset only has {len(df)} rows.")
    sampled = df.sample(n=dataset_size, random_state=seed).reset_index(drop=True)
    train_df = sampled.iloc[:train_size].reset_index(drop=True)
    eval_df = sampled.iloc[train_size:].reset_index(drop=True)
    return train_df, eval_df


def sample_preference_splits(
    df: pd.DataFrame,
    dataset_size: Optional[int],
    train_size: int,
    eval_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle and split preference data, optionally subsampling first."""
    if dataset_size is not None and dataset_size < train_size + eval_size:
        raise ValueError("dataset_size must be at least train_size + eval_size.")

    if dataset_size is not None:
        if len(df) < dataset_size:
            raise ValueError(f"dataset_size={dataset_size} but CSV only has {len(df)} rows.")
        working = df.sample(n=dataset_size, random_state=seed).reset_index(drop=True)
    else:
        working = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    total_needed = train_size + eval_size
    if len(working) < total_needed:
        raise ValueError(
            f"Combined split size ({total_needed}) exceeds available rows ({len(working)}). "
            "Use smaller splits or collect more preference pairs."
        )
    train_df = working.iloc[:train_size].reset_index(drop=True)
    eval_df = working.iloc[train_size:train_size + eval_size].reset_index(drop=True)
    return train_df, eval_df


def dataframe_to_finetune_examples(
    df: pd.DataFrame,
    prompt_column: str,
    completion_column: str,
) -> list[FinetuneExample]:
    """Convert a dataframe into FinetuneExample rows."""
    examples: list[FinetuneExample] = []
    for row in df.itertuples(index=True):
        prompt_value = str(getattr(row, prompt_column))
        completion_value = str(getattr(row, completion_column))
        examples.append(FinetuneExample(prompt=prompt_value, completion=completion_value))
    return examples


def dataframe_to_preference_examples(
    df: pd.DataFrame,
    prompt_column: str,
    chosen_column: str,
    rejected_column: str,
) -> list[PreferenceExample]:
    """Convert a dataframe into PreferenceExample rows."""
    examples: list[PreferenceExample] = []
    for row in df.itertuples(index=True):
        prompt_value = str(getattr(row, prompt_column)).strip()
        chosen_value = str(getattr(row, chosen_column)).strip()
        rejected_value = str(getattr(row, rejected_column)).strip()
        examples.append(
            PreferenceExample(
                prompt=prompt_value,
                chosen=chosen_value,
                rejected=rejected_value,
            )
        )
    return examples


def write_preference_jsonl(
    path: Path,
    examples: Iterable[PreferenceExample],
    metadata: Optional[dict[str, object]] = None,
) -> None:
    """Write preference examples to JSONL, optionally attaching metadata per row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            record: dict[str, object] = {
                "prompt": ex.prompt,
                "chosen": ex.chosen,
                "rejected": ex.rejected,
            }
            if metadata:
                record["metadata"] = metadata
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_preference_csv(path: Path, examples: Iterable[PreferenceExample]) -> None:
    """Emit a CSV mirror of preference data for easier inspection."""
    df = pd.DataFrame(
        {
            "prompt": [ex.prompt for ex in examples],
            "chosen": [ex.chosen for ex in examples],
            "rejected": [ex.rejected for ex in examples],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_metadata_arg(metadata_arg: Optional[str]) -> Optional[dict[str, object]]:
    """Parse the JSON string provided via CLI metadata arguments."""
    if metadata_arg is None:
        return None
    try:
        return json.loads(metadata_arg)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for metadata: {exc}") from exc
