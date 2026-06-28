from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .data import (
    PreferenceExample,
    dataframe_to_preference_examples,
    load_csv_dataset,
    sample_preference_splits,
    write_preference_csv,
    write_preference_jsonl,
)


@dataclass(frozen=True)
class PreferenceDatasetConfig:
    """Input configuration describing the source CSV and desired splits."""

    dataset_csv: Path
    prompt_column: str = "prompt"
    chosen_column: str = "chosen_response"
    rejected_column: str = "rejected_response"
    dataset_size: Optional[int] = None
    train_size: int = 1200
    eval_size: int = 400
    seed: int = 42


@dataclass(frozen=True)
class PreferenceConfigStub:
    """Metadata recorded for the downstream Tinker preference config."""

    config_name: str
    model_name: str
    dpo_beta: float
    learning_rate: float


@dataclass(frozen=True)
class PreferenceDatasetArtifacts:
    """Collection of file paths written during preference prep."""

    train_jsonl: Path
    eval_jsonl: Path
    train_csv: Path
    eval_csv: Path
    config_path: Optional[Path]


def prepare_preference_examples(config: PreferenceDatasetConfig) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Convert the raw CSV into normalized PreferenceExample splits."""
    df = load_csv_dataset(
        path=config.dataset_csv,
        required_columns=[config.prompt_column, config.chosen_column, config.rejected_column],
    )
    train_df, eval_df = sample_preference_splits(
        df=df,
        dataset_size=config.dataset_size,
        train_size=config.train_size,
        eval_size=config.eval_size,
        seed=config.seed,
    )
    train_examples = dataframe_to_preference_examples(
        train_df,
        config.prompt_column,
        config.chosen_column,
        config.rejected_column,
    )
    eval_examples = dataframe_to_preference_examples(
        eval_df,
        config.prompt_column,
        config.chosen_column,
        config.rejected_column,
    )
    return train_examples, eval_examples


def write_config_stub(
    output_dir: Path,
    stub: PreferenceConfigStub,
    train_path: Path,
    eval_path: Path,
) -> Path:
    """Emit a reminder config for tinker_cookbook CLI launches."""
    config = {
        "log_relpath": str(output_dir / "logs"),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "renderer_name": "role_colon",
        "model_name": stub.model_name,
        "reference_model_name": stub.model_name,
        "learning_rate": stub.learning_rate,
        "dpo_beta": stub.dpo_beta,
    }
    config_path = output_dir / stub.config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        import json

        json.dump(config, handle, indent=2)
    return config_path


def write_preference_artifacts(
    *,
    output_dir: Path,
    train_examples: list[PreferenceExample],
    eval_examples: list[PreferenceExample],
    metadata: Optional[dict[str, object]],
    stub: Optional[PreferenceConfigStub],
) -> PreferenceDatasetArtifacts:
    """Write JSONL/CSV splits and optionally emit a config stub."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = output_dir / "train.jsonl"
    eval_jsonl = output_dir / "eval.jsonl"
    write_preference_jsonl(train_jsonl, train_examples, metadata)
    write_preference_jsonl(eval_jsonl, eval_examples, metadata)
    train_csv = output_dir / "train.csv"
    eval_csv = output_dir / "eval.csv"
    write_preference_csv(train_csv, train_examples)
    write_preference_csv(eval_csv, eval_examples)

    config_path: Optional[Path] = None
    if stub:
        config_path = write_config_stub(output_dir=output_dir, stub=stub, train_path=train_jsonl, eval_path=eval_jsonl)

    return PreferenceDatasetArtifacts(
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        train_csv=train_csv,
        eval_csv=eval_csv,
        config_path=config_path,
    )


def build_simple_preference_builder(train_path: Path, eval_path: Optional[Path]):
    """
    Construct the lightweight JSONL comparison builder used by the DPO CLI.

    Imports are scoped locally to avoid forcing heavy dependencies unless needed.
    """

    import chz
    import datasets
    from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
    from tinker_cookbook.preference.types import Comparison, LabeledComparison

    @chz.chz
    class SimplePreferenceJSONLBuilder(ComparisonDatasetBuilder):
        train_path: str
        eval_path: Optional[str] = None

        def get_train_and_test_datasets(self):
            def _load_jsonl(path_str: str):
                rows = []
                path_obj = Path(path_str)
                with path_obj.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                return datasets.Dataset.from_list(rows)

            train_dataset = _load_jsonl(self.train_path)
            eval_dataset = _load_jsonl(self.eval_path) if self.eval_path else None
            return train_dataset, eval_dataset

        def example_to_labeled_comparison(self, example: dict) -> Optional[LabeledComparison]:
            prompt = str(example.get("prompt", "")).strip()
            chosen = str(example.get("chosen", "")).strip()
            rejected = str(example.get("rejected", "")).strip()
            if not prompt or not chosen or not rejected:
                return None
            comparison = Comparison(
                prompt_conversation=[{"role": "user", "content": prompt}],
                completion_A=[{"role": "assistant", "content": chosen}],
                completion_B=[{"role": "assistant", "content": rejected}],
            )
            return LabeledComparison(comparison=comparison, label="A")

    return SimplePreferenceJSONLBuilder(
        train_path=str(train_path),
        eval_path=str(eval_path) if eval_path else None,
    )
