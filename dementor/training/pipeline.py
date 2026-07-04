from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from .plots import (
    ConvergenceSeries,
    plot_convergence,
    series_from_loss_history,
    series_from_metric_rows,
)
from .dpo import (
    PreferenceConfigStub,
    PreferenceDatasetArtifacts,
    PreferenceDatasetConfig,
    prepare_preference_examples,
    write_preference_artifacts,
)
from .openai import (
    OpenAIDPOJobConfig,
    OpenAISFTJobConfig,
    collect_job_metrics,
    create_dpo_job,
    poll_job,
    run_openai_sft_job,
    to_openai_dpo_rows,
    upload_file,
    write_jsonl,
)
from .tinker_backend import (
    EvaluationConfig,
    SFTDatasetConfig,
    run_tinker_sft_job,
)
from .local_backend import (
    LocalDPOParams,
    LocalSFTParams,
    run_local_dpo_job,
    run_local_sft_job,
)


@dataclass(frozen=True)
class TinkerSFTParams:
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 1e-4
    prompt_template: str = "Question: {prompt}\nAnswer:"
    completion_template: str = " {completion}\n"
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    weights_name: str = "gsm8k_llama-3.1-8b-instruct"
    registry_path: Path = Path("data/tinker_adapters.json")
    seed: int = 42
    lora_kwargs: Optional[dict] = None


@dataclass(frozen=True)
class SFTWorkflowConfig:
    provider: Literal["openai", "tinker", "local"]
    dataset: SFTDatasetConfig
    output_dir: Path
    convergence_plot: Optional[Path] = None
    openai: Optional[OpenAISFTJobConfig] = None
    tinker: Optional[TinkerSFTParams] = None
    local: Optional[LocalSFTParams] = None


@dataclass(frozen=True)
class SFTWorkflowResult:
    provider: str
    artifacts: dict
    convergence_plot: Optional[Path]


def _default_plot_path(base_dir: Path, filename: str) -> Path:
    return base_dir / "plots" / filename


def _maybe_plot(
    series: Optional["ConvergenceSeries"],
    path: Path,
    title: str,
) -> Optional[Path]:
    """Render a single convergence ``series`` to ``path``; return the path (or None if empty)."""
    if not series:
        return None
    return plot_convergence([series], path, title=title)


def _load_tinker_jsonl_series(log_dir: Path, metric_key: str, *, label: str) -> Optional["ConvergenceSeries"]:
    metrics_path = (log_dir / "metrics.jsonl").expanduser()
    if not metrics_path.exists():
        return None
    steps: list[int] = []
    values: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if metric_key not in record:
                continue
            try:
                metric_value = float(record[metric_key])
            except (TypeError, ValueError):
                continue
            step_value = record.get("step")
            if isinstance(step_value, (int, float)):
                steps.append(int(step_value))
            else:
                steps.append(idx + 1)
            values.append(metric_value)
    if not values:
        return None
    return ConvergenceSeries(label=label, metric_name=metric_key, steps=steps, values=values)


def run_sft_workflow(config: SFTWorkflowConfig) -> SFTWorkflowResult:
    """Dispatch SFT training to the OpenAI, Tinker, or local-GPU backends."""
    if config.provider == "local":
        params = config.local or LocalSFTParams()
        outcome = run_local_sft_job(
            dataset_config=config.dataset,
            base_model=params.base_model,
            batch_size=params.batch_size,
            epochs=params.epochs,
            learning_rate=params.learning_rate,
            prompt_template=params.prompt_template,
            completion_template=params.completion_template,
            weights_name=params.weights_name,
            output_dir=config.output_dir,
            registry_path=params.registry_path,
            seed=params.seed,
            lora_kwargs=params.lora_kwargs,
            device=params.device,
            max_length=params.max_length,
        )
        return SFTWorkflowResult(
            provider="local",
            artifacts={"sampler_path": outcome.sampler_path, "loss_history": outcome.loss_history},
            convergence_plot=None,
        )

    if config.provider == "tinker":
        params = config.tinker or TinkerSFTParams()
        import tinker  # Imported lazily to avoid dependency for OpenAI-only workflows.

        service_client = tinker.ServiceClient()
        outcome = run_tinker_sft_job(
            dataset_config=config.dataset,
            service_client=service_client,
            base_model=params.base_model,
            batch_size=params.batch_size,
            epochs=params.epochs,
            learning_rate=params.learning_rate,
            prompt_template=params.prompt_template,
            completion_template=params.completion_template,
            evaluation_config=params.evaluation_config,
            weights_name=params.weights_name,
            output_dir=config.output_dir,
            registry_path=params.registry_path,
            seed=params.seed,
            lora_kwargs=params.lora_kwargs,
        )
        series = series_from_loss_history(outcome.loss_history, label=params.weights_name)
        plot_path = config.convergence_plot or _default_plot_path(config.output_dir, "tinker_sft.png")
        convergence = _maybe_plot(series, plot_path, "Tinker SFT Convergence")
        artifacts = {
            "eval_csv": outcome.output_csv,
            "sampler_path": outcome.sampler_path,
            "loss_history": outcome.loss_history,
        }
        return SFTWorkflowResult(provider="tinker", artifacts=artifacts, convergence_plot=convergence)

    params = config.openai or OpenAISFTJobConfig(
        model="gpt-4.1-mini-2025-04-14",
        system_prompt="You are a helpful assistant.",
        epochs=3,
        batch_size=25,
        wait_for_completion=True,
    )
    artifacts, metric_rows = run_openai_sft_job(
        dataset_config=config.dataset,
        output_dir=config.output_dir,
        job_config=params,
    )
    series = series_from_metric_rows(metric_rows, "train_loss", label=params.model)
    plot_path = config.convergence_plot or _default_plot_path(config.output_dir, "openai_sft.png")
    convergence = _maybe_plot(series, plot_path, "OpenAI SFT Convergence")
    artifact_dict = {
        "train_jsonl": artifacts.train_jsonl,
        "eval_jsonl": artifacts.eval_jsonl,
        "job_id": artifacts.job_id,
        "fine_tuned_model": artifacts.fine_tuned_model,
        "train_file_id": artifacts.train_file_id,
        "eval_file_id": artifacts.eval_file_id,
    }
    return SFTWorkflowResult(provider="openai", artifacts=artifact_dict, convergence_plot=convergence)


def _preference_examples_to_rows(examples) -> list[dict]:
    return [
        {"prompt": example.prompt, "chosen": example.chosen, "rejected": example.rejected}
        for example in examples
    ]


@dataclass(frozen=True)
class OpenAIDPOParams:
    model: str = "gpt-4.1-mini-2025-04-14"
    epochs: int = 1
    batch_size: int = 25
    beta: float = 0.1
    submit_job: bool = True
    wait_for_completion: bool = True


@dataclass(frozen=True)
class TinkerDPOParams:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    reference_model_name: Optional[str] = None
    renderer_name: Optional[str] = None
    log_path: Path = Path("data/results/tinker_dpo/logs")
    learning_rate: float = 1e-5
    lr_schedule: str = "linear"
    dpo_beta: float = 0.1
    num_epochs: int = 1
    batch_size: int = 128
    max_length: int = 4096
    lora_rank: int = 32
    base_url: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    save_every: int = 50
    eval_every: int = 0
    infrequent_eval_every: int = 0
    load_checkpoint_path: Optional[str] = None


@dataclass(frozen=True)
class DPOWorkflowConfig:
    provider: Literal["openai", "tinker", "local"]
    dataset: PreferenceDatasetConfig
    output_dir: Path
    metadata: Optional[dict] = None
    stub: Optional[PreferenceConfigStub] = None
    convergence_plot: Optional[Path] = None
    openai: Optional[OpenAIDPOParams] = None
    tinker: Optional[TinkerDPOParams] = None
    local: Optional[LocalDPOParams] = None


@dataclass(frozen=True)
class DPOWorkflowResult:
    provider: str
    artifacts: dict
    convergence_plot: Optional[Path]


def _write_openai_preference_files(
    *,
    output_dir: Path,
    train_examples,
    eval_examples,
) -> tuple[Path, Optional[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = to_openai_dpo_rows(_preference_examples_to_rows(train_examples))
    eval_rows = to_openai_dpo_rows(_preference_examples_to_rows(eval_examples))
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    write_jsonl(train_path, train_rows)
    if eval_rows:
        write_jsonl(eval_path, eval_rows)
    return train_path, (eval_path if eval_rows else None)


def _run_tinker_dpo_job(train_jsonl: Path, eval_jsonl: Optional[Path], params: TinkerDPOParams) -> Path:
    from tinker_cookbook import model_info
    import tinker_cookbook.preference.train_dpo as train_dpo
    from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
    from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    renderer_name = params.renderer_name or model_info.get_recommended_renderer_name(params.model_name)
    from .dpo import build_simple_preference_builder

    comparison_builder: ComparisonDatasetBuilder = build_simple_preference_builder(
        train_path=train_jsonl,
        eval_path=eval_jsonl,
    )
    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=params.model_name,
            renderer_name=renderer_name,
            max_length=params.max_length,
            batch_size=params.batch_size,
        ),
        comparison_builder=comparison_builder,
    )
    params.log_path.mkdir(parents=True, exist_ok=True)
    config = train_dpo.Config(
        log_path=str(params.log_path),
        model_name=params.model_name,
        # tinker_cookbook >=0.4.x made recipe_name a required (metadata-only) field.
        recipe_name=(params.wandb_name or "dementor_dpo"),
        dataset_builder=dataset_builder,
        load_checkpoint_path=params.load_checkpoint_path,
        learning_rate=params.learning_rate,
        lr_schedule=params.lr_schedule,
        num_epochs=params.num_epochs,
        dpo_beta=params.dpo_beta,
        lora_rank=params.lora_rank,
        base_url=params.base_url,
        wandb_project=params.wandb_project,
        wandb_name=params.wandb_name,
        reference_model_name=params.reference_model_name or params.model_name,
        save_every=params.save_every,
        eval_every=params.eval_every,
        infrequent_eval_every=params.infrequent_eval_every,
    )
    train_dpo.main(config)
    return params.log_path


def run_dpo_workflow(config: DPOWorkflowConfig) -> DPOWorkflowResult:
    """Dispatch DPO training to OpenAI or Tinker."""
    train_examples, eval_examples = prepare_preference_examples(config.dataset)
    artifacts_dir = config.output_dir / "preference_artifacts"
    artifacts: PreferenceDatasetArtifacts = write_preference_artifacts(
        output_dir=artifacts_dir,
        train_examples=train_examples,
        eval_examples=eval_examples,
        metadata=config.metadata,
        stub=config.stub,
    )

    if config.provider == "local":
        params = config.local or LocalDPOParams()
        eval_jsonl = artifacts.eval_jsonl if (artifacts.eval_jsonl.exists()
                                              and artifacts.eval_jsonl.stat().st_size > 0) else None
        adapter_dir = run_local_dpo_job(
            train_jsonl=artifacts.train_jsonl,
            eval_jsonl=eval_jsonl,
            params=params,
            output_dir=config.output_dir,
        )
        return DPOWorkflowResult(
            provider="local",
            artifacts={"adapter_dir": str(adapter_dir), "train_jsonl": artifacts.train_jsonl},
            convergence_plot=None,
        )

    if config.provider == "tinker":
        params = config.tinker or TinkerDPOParams()
        run_log = _run_tinker_dpo_job(
            train_jsonl=artifacts.train_jsonl,
            eval_jsonl=artifacts.eval_jsonl if artifacts.eval_jsonl.exists() else None,
            params=params,
        )
        series = _load_tinker_jsonl_series(run_log, "dpo_loss", label=params.model_name)
        plot_path = config.convergence_plot or _default_plot_path(config.output_dir, "tinker_dpo.png")
        convergence = _maybe_plot(series, plot_path, "Tinker DPO Convergence")
        return DPOWorkflowResult(
            provider="tinker",
            artifacts={
                "train_jsonl": artifacts.train_jsonl,
                "eval_jsonl": artifacts.eval_jsonl,
                "log_path": run_log,
            },
            convergence_plot=convergence,
        )

    params = config.openai or OpenAIDPOParams()
    openai_dir = config.output_dir / "openai_format"
    train_path, eval_path = _write_openai_preference_files(
        output_dir=openai_dir,
        train_examples=train_examples,
        eval_examples=eval_examples,
    )
    convergence = None
    artifacts_map: dict = {
        "train_jsonl": train_path,
        "eval_jsonl": eval_path,
        "train_file_id": None,
        "eval_file_id": None,
        "job_id": None,
    }
    if params.submit_job:
        from .openai import _openai_client

        client = _openai_client()
        uploaded_train = upload_file(client, train_path)
        uploaded_eval = upload_file(client, eval_path) if eval_path is not None else None
        job_id = create_dpo_job(
            client,
            OpenAIDPOJobConfig(
                model=params.model,
                train_file_id=uploaded_train,
                valid_file_id=uploaded_eval,
                epochs=params.epochs,
                batch_size=params.batch_size,
                beta=params.beta,
            ),
        )
        if params.wait_for_completion:
            poll_job(client, job_id)
        metric_rows = collect_job_metrics(client, job_id)
        plot_path = config.convergence_plot or _default_plot_path(config.output_dir, "openai_dpo.png")
        series = series_from_metric_rows(metric_rows, "train_loss", label=params.model)
        convergence = _maybe_plot(series, plot_path, "OpenAI DPO Convergence")
        artifacts_map.update(
            {
                "train_file_id": uploaded_train,
                "eval_file_id": uploaded_eval,
                "job_id": job_id,
            }
        )
    return DPOWorkflowResult(provider="openai", artifacts=artifacts_map, convergence_plot=convergence)
