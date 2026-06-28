from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from dementor.training.dpo import PreferenceDatasetConfig
from dementor.training.openai import OpenAISFTJobConfig
from dementor.training.pipeline import (
    DPOWorkflowConfig,
    OpenAIDPOParams,
    SFTWorkflowConfig,
    TinkerDPOParams,
    TinkerSFTParams,
    run_dpo_workflow,
    run_sft_workflow,
)
from dementor.training.tinker_backend import EvaluationConfig, SFTDatasetConfig


DEFAULT_TRAIN_CSV = Path(
    "data/model-responses/gsm8k/splits/seed42/train_300/openai_gpt-4.1-mini_responses_train300_seed42.csv"
)
DEFAULT_EVAL_CSV = Path(
    "data/model-responses/gsm8k/splits/seed42/eval_200/openai_gpt-4.1-mini_responses_eval200_seed42.csv"
)
DEFAULT_PREF_CSV = Path("data/results/tinker_dpo/gsm8k_gpt-4.1-mini_preference_pairs.csv")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _print_config(config: Any) -> None:
    print(json.dumps(_jsonable(config), indent=2, sort_keys=True))


def _check_paths(paths: list[Path], *, dry_run: bool) -> None:
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return
    message = "Missing input files:\n" + "\n".join(f"  - {path}" for path in missing)
    if dry_run:
        print(message)
        return
    raise FileNotFoundError(message)


def _build_sft_config(args: argparse.Namespace) -> SFTWorkflowConfig:
    dataset = SFTDatasetConfig(
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        prompt_column=args.prompt_column,
        completion_column=args.completion_column,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    output_dir = args.output_dir or Path(f"data/results/workflows/gsm8k_{args.provider}_sft")

    if args.provider == "tinker":
        lora_kwargs = {}
        if args.lora_rank is not None:
            lora_kwargs["rank"] = args.lora_rank
        return SFTWorkflowConfig(
            provider="tinker",
            dataset=dataset,
            output_dir=output_dir,
            tinker=TinkerSFTParams(
                base_model=args.base_model,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                prompt_template=args.prompt_template,
                completion_template=args.completion_template,
                evaluation_config=EvaluationConfig(max_sample_tokens=args.max_sample_tokens),
                weights_name=args.weights_name,
                registry_path=args.registry_path,
                seed=args.seed,
                lora_kwargs=lora_kwargs or None,
            ),
        )

    return SFTWorkflowConfig(
        provider="openai",
        dataset=dataset,
        output_dir=output_dir,
        openai=OpenAISFTJobConfig(
            model=args.openai_model,
            system_prompt=args.system_prompt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            wait_for_completion=not args.no_wait,
        ),
    )


def _build_dpo_config(args: argparse.Namespace) -> DPOWorkflowConfig:
    dataset = PreferenceDatasetConfig(
        dataset_csv=args.preference_csv,
        prompt_column=args.prompt_column,
        chosen_column=args.chosen_column,
        rejected_column=args.rejected_column,
        dataset_size=args.preference_dataset_size,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    output_dir = args.output_dir or Path(f"data/results/workflows/gsm8k_{args.provider}_dpo")

    if args.provider == "tinker":
        return DPOWorkflowConfig(
            provider="tinker",
            dataset=dataset,
            output_dir=output_dir,
            tinker=TinkerDPOParams(
                model_name=args.base_model,
                reference_model_name=args.reference_model,
                renderer_name=args.renderer_name,
                log_path=output_dir / "logs",
                learning_rate=args.learning_rate,
                dpo_beta=args.dpo_beta,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                max_length=args.max_length,
                lora_rank=args.lora_rank or 32,
                save_every=args.save_every,
                eval_every=args.eval_every,
                infrequent_eval_every=args.infrequent_eval_every,
                load_checkpoint_path=args.load_checkpoint_path,
            ),
        )

    return DPOWorkflowConfig(
        provider="openai",
        dataset=dataset,
        output_dir=output_dir,
        openai=OpenAIDPOParams(
            model=args.openai_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            beta=args.dpo_beta,
            submit_job=not args.no_submit,
            wait_for_completion=not args.no_wait,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run the canonical GSM8K SFT/DPO workflow."
    )
    parser.add_argument("--stage", choices=["sft", "dpo"], required=True)
    parser.add_argument("--provider", choices=["tinker", "openai"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config without launching jobs.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    sft = parser.add_argument_group("SFT inputs")
    sft.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    sft.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    sft.add_argument("--completion-column", default="model_response")
    sft.add_argument("--prompt-template", default="Question: {prompt}\nAnswer:")
    sft.add_argument("--completion-template", default=" {completion}\n")

    dpo = parser.add_argument_group("DPO inputs")
    dpo.add_argument("--preference-csv", type=Path, default=DEFAULT_PREF_CSV)
    dpo.add_argument("--chosen-column", default="chosen_response")
    dpo.add_argument("--rejected-column", default="rejected_response")
    dpo.add_argument("--preference-dataset-size", type=int, default=None)
    dpo.add_argument("--dpo-beta", type=float, default=0.1)

    tinker = parser.add_argument_group("Tinker")
    tinker.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    tinker.add_argument("--reference-model", default=None)
    tinker.add_argument("--renderer-name", default="llama3")
    tinker.add_argument("--weights-name", default="gsm8k_llama-3.1-8b-instruct")
    tinker.add_argument("--registry-path", type=Path, default=Path("data/tinker_adapters.json"))
    tinker.add_argument("--lora-rank", type=int, default=32)
    tinker.add_argument("--max-sample-tokens", type=int, default=256)
    tinker.add_argument("--max-length", type=int, default=4096)
    tinker.add_argument("--save-every", type=int, default=50)
    tinker.add_argument("--eval-every", type=int, default=0)
    tinker.add_argument("--infrequent-eval-every", type=int, default=0)
    tinker.add_argument("--load-checkpoint-path", default=None)

    openai = parser.add_argument_group("OpenAI")
    openai.add_argument("--openai-model", default="gpt-4.1-mini-2025-04-14")
    openai.add_argument("--system-prompt", default="You are a patient math tutor.")
    openai.add_argument("--no-submit", action="store_true", help="For OpenAI DPO, only prepare files.")
    openai.add_argument("--no-wait", action="store_true", help="Submit but do not block for completion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "sft":
        config = _build_sft_config(args)
        _check_paths([args.train_csv, args.eval_csv], dry_run=args.dry_run)
        if args.dry_run:
            _print_config(config)
            return
        result = run_sft_workflow(config)
    else:
        config = _build_dpo_config(args)
        _check_paths([args.preference_csv], dry_run=args.dry_run)
        if args.dry_run:
            _print_config(config)
            return
        result = run_dpo_workflow(config)

    _print_config(result)


if __name__ == "__main__":
    main()
