"""Matrix dispatcher for the config-defined imitation campaign.

The named campaign in ``config.yaml`` supplies the model cohort, datasets, seeds,
and training/control stages. Use ``dementor-plan`` or ``list-cells`` to inspect the
resolved matrix without launching work.

Subcommands:
  generate-target-responses: generate baseline responses on TRAIN splits for use as SFT completions
  build-sft-data: join (train_prompt, target_response) pairs into SFT CSVs
  launch-sft: submit SFT jobs to Tinker (one per cell × seed)
  list-cells: print the matrix without launching anything

This package was split out of the former single ``matrix.py`` module. Its public
surface is unchanged: every constant, function, class and CLI that used to live at
``dementor.training.matrix`` is re-exported here at the exact same path. Internal
implementation lives in the ``_*`` submodules:

  _constants        filesystem layout + roster/dataset tables
  _models           clean_response / source_id_of / renderer_for
  _cells            Cell + cell iteration + per-cell data-path helpers
  _concurrency      retry + thread-pool dispatch scaffolding
  _configs          backend-aware SFT/DPO workflow-config builders
  _target_responses baseline target-response generation (SFT completions)
  _sft              SFT data builders + SFT job launcher + self-SFT monitor
  _dpo              DPO data builder + checkpoint backfill + DPO launchers
  _safety           safety-constrained (imitation + refusal replay) data builders
  _push             adapter export to PEFT + push to HuggingFace Hub
  _local_cell       single local (gemma-4) cell launcher for accelerate/torchrun
  _cli              argparse dispatcher (``main``)
"""
from __future__ import annotations

# Historical top-level imports. Kept here so the module namespace
# (dementor.training.matrix.<name>) stays byte-for-byte identical to the former
# single-file module for anything that referenced these off it.
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Matplotlib backend must be set BEFORE pyplot or any plotting helper imports.
# Worker threads in ThreadPoolExecutor can't use the GUI MacOS backend.
import matplotlib
matplotlib.use("Agg")

import pandas as pd

from dementor import config

from ._constants import (
    BASELINES_DIR,
    CHAT_TEMPLATE_KWARGS,
    DATA,
    DATASETS,
    DATASET_TEMPLATES,
    DEFAULT_SAFETY_EXCLUDE_PROMPTS,
    DEFAULT_SAFETY_PROMPTS,
    DEFAULT_SAFETY_REPLAY_SIZE,
    DPO_DATA_DIR,
    DPO_OUTPUT_DIR,
    GENERIC_NONREFUSAL_RESPONSE,
    GENERIC_REFUSAL_RESPONSE,
    MODELS,
    MODEL_SLUG,
    PEFT_ADAPTER_DIR,
    REFUSAL_DIR,
    ROOT,
    SAFETY_DPO_DATA_DIR,
    SAFETY_DPO_OUTPUT_DIR,
    SAFETY_NATIVE_REFUSAL_DIR,
    SAFETY_SFT_DATA_DIR,
    SAFETY_SFT_OUTPUT_DIR,
    SEEDS,
    SELF_SFT_DATA_DIR,
    SELF_SFT_OUTPUT_DIR,
    SFT_DATA_DIR,
    SFT_OUTPUT_DIR,
    TRAIN_DATASETS,
)
from ._models import clean_response, renderer_for, source_id_of
from ._cells import (
    Cell,
    baseline_path,
    dpo_data_path,
    iter_cells,
    iter_self_sft_cells,
    safety_dpo_data_path,
    safety_sft_data_path,
    self_sft_data_path,
    sft_data_path,
    _manifest_path,
    _unique_training_data_cells,
)
from ._concurrency import _dispatch, _retry_call
from ._configs import _build_dpo_cfg, _build_sft_cfg
from ._target_responses import generate_target_responses
from ._sft import (
    build_self_sft_data,
    build_sft_data,
    launch_sft,
    monitor_self_sft_controls,
)
from ._dpo import (
    backfill_register_dpo,
    build_dpo_data,
    launch_dpo,
    launch_safety_dpo,
    _find_final_dpo_checkpoint,
)
from ._safety import (
    build_safety_dpo_data,
    build_safety_sft_data,
    load_safety_replay_prompts,
    _base_safety_manifest,
    _native_refusal_snippets,
    _nonempty_response_filter,
    _safety_dpo_replay_rows,
    _safety_sft_replay_rows,
)
from ._push import (
    backfill_export_adapters,
    export_adapter_to_peft,
    push_adapters_to_hf,
    upload_adapter_to_hf,
)
from ._local_cell import (
    launch_local_cell,
    _dist_writer_rank,
    _dpo_prep_token,
    _wait_for_prepared,
)
from ._cli import (
    main,
    _add_cell_filter_args,
    _add_safety_data_args,
    _filtered_cells_from_args,
    _resolve_model_arg,
    _safety_exclude_path,
)
