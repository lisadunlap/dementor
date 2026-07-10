"""The matrix ``Cell`` and the cell-iteration + per-cell data-path helpers.

A ``Cell`` is one (source, target, dataset, seed) point of the imitation matrix.
The ``iter_*`` helpers enumerate the cells; the ``*_path`` helpers map a cell to
the CSV/manifest locations under the directories declared in ``_constants``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ._constants import (
    DPO_DATA_DIR,
    MODEL_SLUG,
    MODELS,
    SEEDS,
    SELF_SFT_DATA_DIR,
    SFT_DATA_DIR,
    TRAIN_DATASETS,
)


@dataclass(frozen=True)
class Cell:
    source: str
    target: str
    dataset: str
    seed: int

    @property
    def slug(self) -> str:
        return f"{self.dataset}_{MODEL_SLUG[self.source]}_as_{MODEL_SLUG[self.target]}_seed{self.seed}"

    @property
    def llama_critical(self) -> bool:
        return self.source == "meta-llama/Llama-3.1-8B-Instruct"


def iter_cells() -> Iterable[Cell]:
    """All cross (source != target) cells over MODELS x datasets x seeds."""
    return [
        Cell(source=s, target=t, dataset=d, seed=seed)
        for s in MODELS for t in MODELS if s != t
        for d in TRAIN_DATASETS for seed in SEEDS
    ]


def iter_self_sft_cells(
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
) -> Iterable[Cell]:
    """Small post-training drift control: each model trains on its own outputs."""
    models = models or MODELS
    datasets = datasets or list(TRAIN_DATASETS)
    seeds = seeds or SEEDS
    return [
        Cell(source=model, target=model, dataset=dataset, seed=seed)
        for model in models
        for dataset in datasets
        for seed in seeds
    ]


def baseline_path(model: str, dataset: str) -> Path:
    # Resolve the directory through the package at call time so tests (and callers)
    # that monkeypatch ``matrix.BASELINES_DIR`` are still honoured after the split.
    from dementor.training import matrix as _matrix
    return _matrix.BASELINES_DIR / dataset / f"{MODEL_SLUG[model]}_train.csv"


def sft_data_path(cell: Cell) -> Path:
    return SFT_DATA_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_train.csv"


def self_sft_data_path(cell: Cell) -> Path:
    return SELF_SFT_DATA_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_self_train.csv"


def dpo_data_path(cell: Cell) -> Path:
    return DPO_DATA_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_pairs.csv"


def safety_sft_data_path(cell: Cell) -> Path:
    from dementor.training import matrix as _matrix  # honour monkeypatched matrix.SAFETY_SFT_DATA_DIR
    return _matrix.SAFETY_SFT_DATA_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_train.csv"


def safety_dpo_data_path(cell: Cell) -> Path:
    from dementor.training import matrix as _matrix  # honour monkeypatched matrix.SAFETY_DPO_DATA_DIR
    return _matrix.SAFETY_DPO_DATA_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_pairs.csv"


def _manifest_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".manifest.json")


def _unique_training_data_cells(cells: Iterable[Cell] | None = None) -> Iterable[Cell]:
    """Yield one cell per (source, target, dataset); training data is shared across seeds."""
    seen: set[tuple[str, str, str]] = set()
    for cell in (cells or iter_cells()):
        key = (cell.source, cell.target, cell.dataset)
        if key in seen:
            continue
        seen.add(key)
        yield cell
