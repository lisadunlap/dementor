from __future__ import annotations

import json
import math
import random
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .common import (
    EvaluationConfig,
    SFTDatasetConfig,
    SFTOutcome,
    format_completion,
    format_prompt,
    prepare_sft_examples,
    record_adapter_mapping,
)
from .data import FinetuneExample

# Backward-compat: ``TinkerSFTOutcome`` was renamed to the backend-neutral
# ``SFTOutcome``. Keep the old name resolvable so existing imports and
# return-type annotations (here and in downstream modules) still resolve.
TinkerSFTOutcome = SFTOutcome


def _require_tinker():
    try:
        import tinker
        from tinker import types
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Tinker workflows require the optional `tinker` package. "
            "Install it and set TINKER_API_KEY before launching Tinker SFT/DPO jobs."
        ) from exc
    return tinker, types


def _coerce_to_float(value: object) -> float:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        return float(value.item())  # type: ignore[call-arg]
    if hasattr(value, "value"):
        return float(getattr(value, "value"))
    return float(value)


def process_example(
    example: FinetuneExample,
    tokenizer: Any,
    index: int,
    prompt_template: str,
    completion_template: str,
) -> Any:
    _, types = _require_tinker()
    prompt_text = format_prompt(example, prompt_template, index)
    completion_text = format_completion(example, completion_template, index)

    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    all_tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    shifted_weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=shifted_weights, target_tokens=target_tokens),
    )


def to_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def compute_batch_loss(
    fwdbwd_output: Any,
    batch: Sequence[Any],
) -> float:
    batch_logprobs: list[float] = []
    batch_weights: list[float] = []
    for datum_output, datum in zip(fwdbwd_output.loss_fn_outputs, batch, strict=True):
        raw_logprobs = datum_output["logprobs"]
        if hasattr(raw_logprobs, "tolist"):
            raw_list = raw_logprobs.tolist()
        else:
            raw_list = list(raw_logprobs)
        datum_logprobs = np.asarray([_coerce_to_float(v) for v in raw_list], dtype=np.float64)

        raw_weights = datum.loss_fn_inputs["weights"]
        if hasattr(raw_weights, "tolist"):
            weights_list = raw_weights.tolist()
        else:
            weights_list = list(raw_weights)
        datum_weights = np.asarray([_coerce_to_float(w) for w in weights_list], dtype=np.float64)

        batch_logprobs.extend(datum_logprobs.tolist())
        batch_weights.extend(datum_weights.tolist())

    weights_array = np.asarray(batch_weights, dtype=np.float64)
    logprobs_array = np.asarray(batch_logprobs, dtype=np.float64)
    weight_sum = float(weights_array.sum())
    if math.isclose(weight_sum, 0.0):
        return float("nan")
    loss = -float(np.dot(logprobs_array, weights_array) / weight_sum)
    return loss


def train_lora_model(
    training_client: Any,
    examples: list[FinetuneExample],
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    prompt_template: str,
    completion_template: str,
) -> list[float]:
    _, types = _require_tinker()
    rng = random.Random(seed)
    tokenizer = training_client.get_tokenizer()
    processed: list[Any] = [
        process_example(example, tokenizer, index, prompt_template, completion_template)
        for index, example in enumerate(examples)
    ]
    loss_history: list[float] = []

    for epoch in range(epochs):
        rng.shuffle(processed)
        for batch in to_batches(processed, batch_size):
            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=learning_rate)
            )
            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()
            batch_loss = compute_batch_loss(fwdbwd_result, batch)
            loss_history.append(batch_loss)
            print(f"[epoch {epoch + 1}] batch_loss={batch_loss:.4f}")
    return loss_history


def extract_final_answer(text: str) -> Optional[str]:
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        return numbers[-1].strip()
    return None


def normalize_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    candidate = text.replace(",", "").strip()
    if not candidate:
        return None
    try:
        quantized = Decimal(candidate)
    except InvalidOperation:
        return candidate.lower()
    normalized = quantized.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.to_integral())
    return format(normalized, "f").rstrip("0").rstrip(".")


def evaluate_model(
    sampling_client: Any,
    eval_examples: list[FinetuneExample],
    tokenizer: Any,
    prompt_template: str,
    completion_template: str,
    config: EvaluationConfig,
) -> pd.DataFrame:
    _, types = _require_tinker()
    params = types.SamplingParams(
        max_tokens=config.max_sample_tokens,
        temperature=0.0,
        stop=config.stop_sequences if config.stop_sequences else None,
    )
    rows: list[dict[str, object]] = []
    correct = 0

    for idx, example in enumerate(eval_examples):
        prompt = format_prompt(example, prompt_template, idx)
        encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_input = types.ModelInput.from_ints(tokens=encoded_prompt)
        future = sampling_client.sample(prompt=prompt_input, sampling_params=params, num_samples=1)
        result = future.result()
        decoded = tokenizer.decode(result.sequences[0].tokens)
        predicted_answer = extract_final_answer(decoded)
        gold_answer = extract_final_answer(format_completion(example, completion_template, idx))
        normalized_pred = normalize_answer(predicted_answer)
        normalized_gold = normalize_answer(gold_answer)
        is_correct = normalized_pred is not None and normalized_pred == normalized_gold
        correct += int(is_correct)
        rows.append(
            {
                "example_index": idx,
                "prompt": example.prompt,
                "reference_completion": example.completion,
                "model_output": decoded,
                "predicted_answer": predicted_answer,
                "normalized_prediction": normalized_pred,
                "normalized_reference": normalized_gold,
                "is_correct": is_correct,
            }
        )
        print(
            f"[evaluation] idx={idx} correct={correct}/{idx + 1} "
            f"normalized_pred={normalized_pred} normalized_gold={normalized_gold}"
        )

    accuracy = correct / len(eval_examples)
    print(f"Evaluation accuracy: {accuracy:.2%}")
    return pd.DataFrame(rows)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_lora_kwargs(config_arg: Optional[str]) -> dict:
    """Parse JSON or load from file to feed into create_lora_training_client."""
    if config_arg is None:
        return {}
    config_path = Path(config_arg)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(config_arg)


def list_available_models(service_client: Any) -> list[str]:
    """Return the supported base model names for a given service client."""
    capabilities = service_client.get_server_capabilities()
    return [model.model_name for model in capabilities.supported_models]


def _save_sampler_checkpoint(
    training_client: Any,
    alias_name: str,
    registry_path: Path,
) -> Optional[str]:
    """Save both a sampler checkpoint and a downloadable state checkpoint.

    Returns the sampler path (used for create_sampling_client). Also persists a
    downloadable state path in the registry under the 'checkpoint_path' key so
    auto-export can fetch the adapter weights as a tar archive.
    """
    unique_suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    unique_name = f"{alias_name}_{unique_suffix}"
    try:
        sampler_res = training_client.save_weights_for_sampler(name=unique_name).result()
    except Exception as exc:  # pragma: no cover - passthrough for existing behavior
        print(f"Warning: could not record sampler path: {exc}")
        return None

    sampler_path = getattr(sampler_res, "path", None)
    if not (isinstance(sampler_path, str) and sampler_path):
        return None

    # Also save downloadable state checkpoint (different endpoint from sampler).
    # The sampler URI (sampler_weights/...) is NOT downloadable; the state URI
    # (state/...) is. We need both: sampler for inference, state for export.
    checkpoint_path: Optional[str] = None
    try:
        state_res = training_client.save_state(name=unique_name).result()
        checkpoint_path = getattr(state_res, "path", None)
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not save state checkpoint (export will fail): {exc}")

    metadata: dict[str, object] = {"adapter_name": unique_name}
    if checkpoint_path:
        metadata["checkpoint_path"] = checkpoint_path
    record_adapter_mapping(alias_name, sampler_path, registry_path, metadata=metadata)
    print(
        f"Sampler stored under internal name '{unique_name}'. "
        f"Alias '{alias_name}' now points to {sampler_path}."
    )
    if checkpoint_path:
        print(f"Downloadable state checkpoint at {checkpoint_path}.")
    return sampler_path


def run_tinker_sft_job(
    *,
    dataset_config: SFTDatasetConfig,
    service_client: Any,
    base_model: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    prompt_template: str,
    completion_template: str,
    evaluation_config: EvaluationConfig,
    weights_name: str,
    output_dir: Path,
    registry_path: Path,
    seed: int,
    lora_kwargs: Optional[dict] = None,
) -> SFTOutcome:
    """End-to-end helper used by CLI wrappers to run supervised fine-tuning."""
    _require_tinker()
    train_examples, eval_examples = prepare_sft_examples(dataset_config)

    training_client = service_client.create_lora_training_client(base_model=base_model, **(lora_kwargs or {}))
    loss_history = train_lora_model(
        training_client=training_client,
        examples=train_examples,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        prompt_template=prompt_template,
        completion_template=completion_template,
    )
    print(f"Training finished after {len(loss_history)} optimization steps.")

    print(f"Saving LoRA sampler weights under alias '{weights_name}'")
    sampler_path = _save_sampler_checkpoint(training_client, weights_name, registry_path)
    if sampler_path:
        sampling_client = service_client.create_sampling_client(model_path=sampler_path)
    else:
        print("Falling back to save_weights_and_get_sampling_client; sampler path will not be recorded.")
        sampling_client = training_client.save_weights_and_get_sampling_client(name=weights_name)
    tokenizer = training_client.get_tokenizer()
    ensure_output_dir(output_dir)
    output_csv = output_dir / f"{weights_name}_eval.csv"
    if eval_examples:
        eval_results = evaluate_model(
            sampling_client=sampling_client,
            eval_examples=eval_examples,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            completion_template=completion_template,
            config=evaluation_config,
        )
        eval_results.to_csv(output_csv, index=False)
        print(f"Wrote evaluation details to {output_csv}")
    else:
        eval_results = pd.DataFrame()
        print(f"Skipping in-training eval (no eval examples provided); adapter saved.")
    print(
        "To reuse the model later, pass the recorded tinker:// sampler path to "
        "`scripts/generate_responses.py ... tinker --model-path <path>`."
    )

    return SFTOutcome(
        loss_history=loss_history,
        eval_results=eval_results,
        output_csv=output_csv,
        sampler_path=sampler_path,
    )
