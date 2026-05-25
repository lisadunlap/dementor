from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .data import FinetuneExample
from .tinker import SFTDatasetConfig, prepare_sft_examples


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into Python dictionaries."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write a list of dictionaries as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for obj in rows:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def validate_preference_rows(rows: list[dict]) -> None:
    """Ensure prompt/chosen/rejected keys are present before conversion/upload."""
    for i, row in enumerate(rows):
        if not all(k in row for k in ("prompt", "chosen", "rejected")):
            raise ValueError(f"Row {i} missing one of required keys: prompt, chosen, rejected")


def _openai_client():
    try:
        from openai import OpenAI
    except Exception as exc:
        raise ImportError(
            "OpenAI workflows require a working `openai` Python package. "
            "Install a compatible build before submitting OpenAI fine-tune jobs."
        ) from exc
    return OpenAI()


def upload_file(client: Any, path: Path) -> str:
    """Upload a JSONL file to OpenAI and return the file id."""
    with path.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="fine-tune")
    return uploaded.id


def to_openai_dpo_rows(tinker_rows: list[dict]) -> list[dict]:
    """Convert Tinker-format preference data into OpenAI's experimental DPO schema."""
    out: list[dict] = []
    for row in tinker_rows:
        prompt = str(row.get("prompt", "")).strip()
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        if not prompt or not chosen or not rejected:
            continue
        out.append(
            {
                "input": {
                    "messages": [
                        {"role": "user", "content": f"Question: {prompt}\nAnswer:"}
                    ],
                },
                "preferred_output": [{"role": "assistant", "content": chosen}],
                "non_preferred_output": [{"role": "assistant", "content": rejected}],
            }
        )
    return out


@dataclass(frozen=True)
class OpenAIDPOJobConfig:
    """Configuration bundle for launching an OpenAI DPO job."""

    model: str
    train_file_id: str
    valid_file_id: Optional[str]
    epochs: int
    batch_size: int
    beta: float


def create_dpo_job(client: Any, job_config: OpenAIDPOJobConfig) -> str:
    """Submit a DPO job using the OpenAI fine-tuning API."""
    job = client.fine_tuning.jobs.create(
        training_file=job_config.train_file_id,
        validation_file=job_config.valid_file_id,
        model=job_config.model,
        method={
            "type": "dpo",
            "dpo": {
                "hyperparameters": {
                    "beta": job_config.beta,
                    "n_epochs": job_config.epochs,
                    "batch_size": job_config.batch_size,
                },
            },
        },
    )
    return job.id


def poll_job(client: Any, job_id: str, interval_seconds: float = 10.0) -> None:
    """Poll a fine-tuning job until it completes."""
    last_status = None
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        if job.status != last_status:
            print(f"status={job.status}")
            last_status = job.status
        if job.status in {"succeeded", "failed", "cancelled"}:
            print(f"final_status={job.status}")
            if job.status == "succeeded":
                print(f"fine_tuned_model={job.fine_tuned_model}")
            break
        time.sleep(interval_seconds)


@dataclass(frozen=True)
class OpenAISFTJobConfig:
    """Configuration bundle for launching an OpenAI SFT job."""

    model: str
    system_prompt: str
    epochs: int
    batch_size: int | str
    wait_for_completion: bool = True


@dataclass(frozen=True)
class OpenAISFTArtifacts:
    """Paths and identifiers recorded during an OpenAI SFT job."""

    train_jsonl: Path
    eval_jsonl: Path
    train_file_id: str
    eval_file_id: Optional[str]
    job_id: str
    fine_tuned_model: Optional[str]


def _examples_to_chat_rows(examples: Sequence[FinetuneExample], system_prompt: str) -> List[dict]:
    """Convert FinetuneExample rows into OpenAI chat fine-tune format."""
    rows: List[dict] = []
    for example in examples:
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {example.prompt}\nAnswer:"},
                    {"role": "assistant", "content": example.completion},
                ]
            }
        )
    return rows


def _object_to_plain_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {}


def collect_job_metrics(client: Any, job_id: str, limit: int = 1000) -> List[dict]:
    """Fetch fine-tuning events and return only metric-bearing rows."""
    response = client.fine_tuning.jobs.list_events(job_id, limit=limit)
    rows: List[dict] = []
    for event in getattr(response, "data", []):
        event_type = getattr(event, "type", None)
        if event_type != "metrics":
            continue
        metrics_dict = _object_to_plain_dict(getattr(event, "metrics", None))
        if not metrics_dict:
            continue
        row = {"step": getattr(event, "step", metrics_dict.get("step"))}
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                row.setdefault(key, value)
        rows.append(row)
    rows.sort(key=lambda item: item.get("step") or 0)
    return rows


def run_openai_sft_job(
    *,
    dataset_config: SFTDatasetConfig,
    output_dir: Path,
    job_config: OpenAISFTJobConfig,
    client: Optional[Any] = None,
) -> tuple[OpenAISFTArtifacts, List[dict]]:
    """End-to-end OpenAI SFT helper used by the workflows pipeline."""
    train_examples, eval_examples = prepare_sft_examples(dataset_config)
    train_rows = _examples_to_chat_rows(train_examples, job_config.system_prompt)
    eval_rows = _examples_to_chat_rows(eval_examples, job_config.system_prompt)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = output_dir / "train.jsonl"
    eval_jsonl = output_dir / "eval.jsonl"
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(eval_jsonl, eval_rows)

    client = client or _openai_client()
    train_file_id = upload_file(client, train_jsonl)
    eval_file_id = upload_file(client, eval_jsonl)
    job = client.fine_tuning.jobs.create(
        training_file=train_file_id,
        validation_file=eval_file_id,
        model=job_config.model,
        hyperparameters={"n_epochs": job_config.epochs, "batch_size": job_config.batch_size},
    )
    job_id = job.id
    fine_tuned_model: Optional[str] = None
    if job_config.wait_for_completion:
        poll_job(client, job_id)
        final_job = client.fine_tuning.jobs.retrieve(job_id)
        fine_tuned_model = getattr(final_job, "fine_tuned_model", None)
    metric_rows = collect_job_metrics(client, job_id)
    artifacts = OpenAISFTArtifacts(
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        train_file_id=train_file_id,
        eval_file_id=eval_file_id,
        job_id=job_id,
        fine_tuned_model=fine_tuned_model,
    )
    return artifacts, metric_rows
