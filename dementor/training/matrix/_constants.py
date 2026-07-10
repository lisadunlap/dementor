"""Filesystem layout and roster/dataset tables shared by every matrix subcommand.

All matrix output directories, safety-replay defaults, and the model/dataset
lookup tables are defined here once so the functional modules (target responses,
SFT, DPO, safety, push, ...) resolve against a single source of truth.

``MODELS``, ``MODEL_SLUG`` and ``CHAT_TEMPLATE_KWARGS`` are deliberately mutable
module-level state: census drivers (e.g. ``extra_model_census.py``) register
extra models by mutating ``CHAT_TEMPLATE_KWARGS`` in place. Consumers must import
and reference these exact objects rather than copying them, so mutations remain
visible everywhere.
"""
from __future__ import annotations

from pathlib import Path

from dementor import config

# NOTE: this module lives at dementor/training/matrix/_constants.py, so the repo
# root is four parents up (matrix -> training -> dementor -> repo root).
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
DATASETS = DATA / "datasets"
BASELINES_DIR = DATA / "model-responses" / "matrix_baselines"
SFT_DATA_DIR = DATA / "results" / "matrix" / "sft_data"
SFT_OUTPUT_DIR = DATA / "results" / "matrix" / "sft_runs"
SELF_SFT_DATA_DIR = DATA / "results" / "matrix" / "self_sft_data"
SELF_SFT_OUTPUT_DIR = DATA / "results" / "matrix" / "self_sft_runs"
DPO_DATA_DIR = DATA / "results" / "matrix" / "dpo_data"
DPO_OUTPUT_DIR = DATA / "results" / "matrix" / "dpo_runs"
SAFETY_SFT_DATA_DIR = DATA / "results" / "matrix" / "safety_sft_data"
SAFETY_SFT_OUTPUT_DIR = DATA / "results" / "matrix" / "safety_sft_runs"
SAFETY_DPO_DATA_DIR = DATA / "results" / "matrix" / "safety_dpo_data"
SAFETY_DPO_OUTPUT_DIR = DATA / "results" / "matrix" / "safety_dpo_runs"
PEFT_ADAPTER_DIR = DATA / "adapters" / "peft"
REFUSAL_DIR = DATA / "datasets" / "benchmarks" / "refusal"
DEFAULT_SAFETY_PROMPTS = REFUSAL_DIR / "advbench_harmful.csv"
DEFAULT_SAFETY_EXCLUDE_PROMPTS = REFUSAL_DIR / "eval_120h_80o_seed42.csv"
SAFETY_NATIVE_REFUSAL_DIR = ROOT / "results" / "safety" / "native_refusal"
DEFAULT_SAFETY_REPLAY_SIZE = 120
GENERIC_REFUSAL_RESPONSE = "I can't help with that request."
GENERIC_NONREFUSAL_RESPONSE = "I can help with that request."

# Derived from config.yaml (the single source of truth). MODELS is the config `roster`
# (10 models — but per config.yaml's REPRODUCIBILITY note only qwen3.6-27b, gpt-oss-20b,
# nemotron-nano-30b + llama-3.1-8b[legacy] are executed; the other 7 are a planned scale-up,
# so unfiltered matrix runs will dispatch unvalidated cells). The slug/chat maps also cover
# roster_legacy so existing adapters and the B2 cell-subset drivers still resolve. Module-level MUTABLE dicts on
# purpose — extra_model_census.py registers extra models by mutating CHAT_TEMPLATE_KWARGS.
MODELS: list[str] = [m["id"] for m in config.roster()]
MODEL_SLUG: dict[str, str] = {m["id"]: m["slug"] for m in config.roster(include_legacy=True)}
CHAT_TEMPLATE_KWARGS: dict[str, dict] = {
    m["id"]: dict(m.get("chat_template_kwargs", {})) for m in config.roster(include_legacy=True)
}

# Train splits + per-dataset SFT templates (from config.yaml).
TRAIN_DATASETS: dict[str, Path] = {
    n: config.resolve_path(config.dataset(n)["train_csv"]) for n in config.dataset_names()
}
DATASET_TEMPLATES: dict[str, tuple[str, str]] = {
    n: (config.dataset(n)["prompt_template"], config.dataset(n)["completion_template"])
    for n in config.dataset_names()
}

SEEDS: list[int] = config.seeds()
