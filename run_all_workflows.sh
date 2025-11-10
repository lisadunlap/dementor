#!/usr/bin/env bash
set -euo pipefail

(python - <<'PY'
from pathlib import Path
from workflows.pipeline import SFTWorkflowConfig, TinkerSFTParams, run_sft_workflow
from workflows.tinker import SFTDatasetConfig
cfg = SFTWorkflowConfig(
    provider="tinker",
    dataset=SFTDatasetConfig(
        train_csv=Path("data/model-responses/gsm8k/splits/seed42/train_300/meta-llama_Meta-Llama-3.1-8B-Instruct_responses_train300_seed42.csv"),
        eval_csv=Path("data/model-responses/gsm8k/splits/seed42/eval_200/meta-llama_Meta-Llama-3.1-8B-Instruct_responses_eval200_seed42.csv"),
        prompt_column="prompt",
        completion_column="model_response",
    ),
    output_dir=Path("data/results/workflows/sft_llama-3.1-8b-instruct_as_gpt-4.1-mini"),
    tinker=TinkerSFTParams(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        weights_name="sft_llama-3.1-8b-instruct_as_gpt-4.1-mini",
    ),
)
run_sft_workflow(cfg)
PY
)&
pid1=$!

(python - <<'PY'
from pathlib import Path
from workflows.pipeline import SFTWorkflowConfig, run_sft_workflow
from workflows.openai import OpenAISFTJobConfig
from workflows.tinker import SFTDatasetConfig
cfg = SFTWorkflowConfig(
    provider="openai",
    dataset=SFTDatasetConfig(
        train_csv=Path("data/model-responses/gsm8k/splits/seed42/train_300/openai_gpt-4.1-mini_responses_train300_seed42.csv"),
        eval_csv=Path("data/model-responses/gsm8k/splits/seed42/eval_200/openai_gpt-4.1-mini_responses_eval200_seed42.csv"),
        prompt_column="prompt",
        completion_column="model_response",
    ),
    output_dir=Path("data/results/workflows/sft_gpt-4.1-mini_as_llama-3.1-8b-instruct"),
    openai=OpenAISFTJobConfig(
        model="gpt-4.1-mini-2025-04-14",
        system_prompt="You are a patient math tutor.",
        epochs=3,
        batch_size=25,
    ),
)
run_sft_workflow(cfg)
PY
)&
pid2=$!

(python - <<'PY'
from pathlib import Path
from workflows.pipeline import DPOWorkflowConfig, TinkerDPOParams, run_dpo_workflow
from workflows.dpo import PreferenceDatasetConfig
cfg = DPOWorkflowConfig(
    provider="tinker",
    dataset=PreferenceDatasetConfig(
        dataset_csv=Path("data/results/tinker_dpo/gsm8k_gpt-4.1-mini_preference_pairs.csv"),
        prompt_column="prompt",
        chosen_column="chosen_response",
        rejected_column="rejected_response",
        train_size=300,
        eval_size=200,
        seed=42,
    ),
    output_dir=Path("data/results/workflows/dpo_llama-3.1-8b-instruct_as_gpt-4.1-mini"),
    tinker=TinkerDPOParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        log_path=Path("data/results/workflows/dpo_llama-3.1-8b-instruct_as_gpt-4.1-mini/logs"),
    ),
)
run_dpo_workflow(cfg)
PY
)&
pid3=$!

(python - <<'PY'
from pathlib import Path
from workflows.pipeline import DPOWorkflowConfig, OpenAIDPOParams, run_dpo_workflow
from workflows.dpo import PreferenceDatasetConfig
cfg = DPOWorkflowConfig(
    provider="openai",
    dataset=PreferenceDatasetConfig(
        dataset_csv=Path("data/results/tinker_dpo/gsm8k_gpt-4.1-mini_preference_pairs.csv"),
        prompt_column="prompt",
        chosen_column="chosen_response",
        rejected_column="rejected_response",
        train_size=300,
        eval_size=200,
        seed=42,
    ),
    output_dir=Path("data/results/workflows/dpo_gpt-4.1-mini_as_llama-3.1-8b-instruct"),
    openai=OpenAIDPOParams(
        model="gpt-4.1-mini-2025-04-14",
        epochs=1,
        batch_size=25,
        beta=0.1,
    ),
)
run_dpo_workflow(cfg)
PY
)&
pid4=$!

wait $pid1 $pid2 $pid3 $pid4
