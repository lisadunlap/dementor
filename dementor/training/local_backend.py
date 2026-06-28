"""Local-GPU training backend (HuggingFace + PEFT + TRL).

Mirrors the Tinker backend (:mod:`dementor.training.tinker_backend`) so the matrix
driver can dispatch by the roster ``backend`` field. Built for the three
``google/gemma-4-*`` models that Tinker does not host. Heavy deps
(torch / transformers / peft / trl / datasets) are imported lazily inside each
function so analysis code can import this module without a training environment.

Designed to run later on a GPU; unit-tested on CPU with a tiny model
(see ``tests/test_local_backend.py``). The registry schema is shared with the
Tinker side — entries carry ``backend: "local"`` and a local filesystem path
instead of a ``tinker://`` URI, so SFT->DPO chaining and HF upload work uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .tinker_backend import (
    SFTDatasetConfig,
    TinkerSFTOutcome,
    format_completion,
    format_prompt,
    prepare_sft_examples,
    record_adapter_mapping,
)


@dataclass(frozen=True)
class LocalSFTParams:
    """Local-GPU SFT params (mirrors :class:`TinkerSFTParams`)."""

    base_model: str = "google/gemma-4-E4B-it"
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 1e-4
    prompt_template: str = "{prompt}"
    completion_template: str = "{completion}"
    weights_name: str = "local_sft"
    registry_path: Path = Path("data/tinker_adapters.json")
    seed: int = 42
    lora_kwargs: Optional[dict] = None
    device: Optional[str] = None
    max_length: int = 1024


@dataclass(frozen=True)
class LocalDPOParams:
    """Local-GPU DPO params (mirrors :class:`TinkerDPOParams`)."""

    model_name: str = "google/gemma-4-E4B-it"
    load_checkpoint_path: Optional[str] = None  # SFT adapter dir to start from
    weights_name: str = "local_dpo"
    registry_path: Path = Path("data/tinker_adapters.json")
    learning_rate: float = 1e-5
    dpo_beta: float = 0.1
    num_epochs: int = 1
    batch_size: int = 16
    max_length: int = 4096
    lora_rank: int = 32
    seed: int = 42
    device: Optional[str] = None


def _resolve_device(device: Optional[str]) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _lora_config(lora_kwargs: Optional[dict]):
    from peft import LoraConfig

    kw = dict(lora_kwargs or {})
    return LoraConfig(
        r=kw.get("rank", kw.get("r", 32)),
        lora_alpha=kw.get("alpha", 64),
        lora_dropout=kw.get("dropout", 0.0),
        target_modules=kw.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )


def generate_local_responses(
    *,
    model: str,
    prompts: list[str],
    chat_template_kwargs: Optional[dict] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: Optional[str] = None,
    adapter_dir: Optional[str] = None,
) -> list[str]:
    """Sample a local HF model (optionally with a PEFT adapter).

    Local counterpart of ``matrix.generate_target_responses``: applies the chat
    template (greedy when ``temperature == 0``) and returns decoded completions.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = _resolve_device(device)
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model)
    if adapter_dir:
        from peft import PeftModel

        mdl = PeftModel.from_pretrained(mdl, adapter_dir)
    mdl.to(dev).eval()

    outputs: list[str] = []
    for prompt in prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
        enc = tok(text, return_tensors="pt").to(dev)
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            seq = mdl.generate(**enc, **gen_kwargs)
        new_tokens = seq[0][enc["input_ids"].shape[1]:]
        outputs.append(tok.decode(new_tokens, skip_special_tokens=True))
    return outputs


def run_local_sft_job(
    *,
    dataset_config: SFTDatasetConfig,
    base_model: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    prompt_template: str,
    completion_template: str,
    weights_name: str,
    output_dir: Path,
    registry_path: Path,
    seed: int,
    lora_kwargs: Optional[dict] = None,
    device: Optional[str] = None,
    max_length: int = 1024,
) -> TinkerSFTOutcome:
    """LoRA SFT on a local model via TRL; writes a PEFT adapter + registry entry.

    Local counterpart of ``run_tinker_sft_job`` — same signature shape, returns the
    same :class:`TinkerSFTOutcome`, and records the adapter under the shared registry
    with ``backend: "local"`` (path is the local PEFT dir, not a ``tinker://`` URI).
    """
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples, _eval = prepare_sft_examples(dataset_config)
    texts = [
        format_prompt(ex, prompt_template, i) + format_completion(ex, completion_template, i)
        for i, ex in enumerate(train_examples)
    ]
    train_ds = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model)

    dev = _resolve_device(device)
    sft_config = SFTConfig(
        output_dir=str(output_dir / "_trainer"),
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        dataset_text_field="text",
        max_length=max_length,
        use_cpu=(dev == "cpu"),
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        peft_config=_lora_config(lora_kwargs),
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    loss = getattr(result, "training_loss", None)
    loss_history = [float(loss)] if loss is not None else []
    record_adapter_mapping(
        weights_name,
        str(output_dir),
        registry_path,
        metadata={"backend": "local", "checkpoint_path": str(output_dir), "base_model": base_model},
    )
    return TinkerSFTOutcome(
        loss_history=loss_history,
        eval_results=pd.DataFrame(),
        output_csv=output_dir / "eval.csv",
        sampler_path=str(output_dir),
    )


def run_local_dpo_job(*, train_jsonl: Path, eval_jsonl: Optional[Path], params: LocalDPOParams,
                      output_dir: Path) -> Path:
    """LoRA DPO on a local model via TRL, starting from the SFT adapter.

    Local counterpart of ``pipeline._run_tinker_dpo_job``. ``train_jsonl`` holds
    ``{prompt, chosen, rejected}`` rows. Writes the tuned PEFT adapter under
    ``output_dir`` and records it under the shared registry. Returns the adapter dir.
    """
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(params.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(params.model_name)
    if params.load_checkpoint_path:  # continue from the SFT adapter (mirrors load_checkpoint_path)
        model = PeftModel.from_pretrained(model, params.load_checkpoint_path, is_trainable=True)

    files = {"train": str(train_jsonl)}
    if eval_jsonl is not None:
        files["test"] = str(eval_jsonl)
    ds = load_dataset("json", data_files=files)

    dev = _resolve_device(params.device)
    dpo_config = DPOConfig(
        output_dir=str(out_dir / "_trainer"),
        per_device_train_batch_size=params.batch_size,
        num_train_epochs=params.num_epochs,
        learning_rate=params.learning_rate,
        beta=params.dpo_beta,
        max_length=params.max_length,
        seed=params.seed,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        use_cpu=(dev == "cpu"),
    )
    peft_cfg = None if params.load_checkpoint_path else _lora_config({"rank": params.lora_rank})
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("test"),
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.model.save_pretrained(str(out_dir))
    record_adapter_mapping(
        params.weights_name,
        str(out_dir),
        params.registry_path,
        metadata={"backend": "local", "checkpoint_path": str(out_dir),
                  "base_model": params.model_name, "sft_parent": params.load_checkpoint_path},
    )
    return out_dir


def export_local_adapter(*, adapter_dir: Path, base_model: str, output_dir: Path) -> dict:
    """Stage a local PEFT adapter for upload (mirrors ``export_adapter_to_peft``).

    TRL already writes PEFT format, so this is near a no-op: it copies the adapter
    into ``output_dir`` (if different) and records provenance so the shared HF-upload
    path treats local and Tinker adapters identically.
    """
    import json
    import shutil

    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)
    if adapter_dir.resolve() != output_dir.resolve():
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
            src = adapter_dir / name
            if src.exists():
                shutil.copy2(src, output_dir / name)
    (output_dir / "dementor_local_export.json").write_text(
        json.dumps({"backend": "local", "base_model": base_model, "source_dir": str(adapter_dir)}, indent=2),
        encoding="utf-8",
    )
    return {"status": "exported", "output_dir": str(output_dir)}
