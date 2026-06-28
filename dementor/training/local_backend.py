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
    gradient_accumulation_steps: Optional[int] = None


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


def _resolve_grad_accum(value: Optional[int]) -> int:
    """Gradient-accumulation steps: explicit arg wins, else ``$DEMENTOR_GRAD_ACCUM``, else 1.

    The env fallback lets the config-driven matrix pipeline (which forwards a fixed
    set of kwargs) raise accumulation for the big gemma-4 models without code edits.
    """
    if value is not None:
        return max(1, int(value))
    import os

    try:
        return max(1, int(os.environ.get("DEMENTOR_GRAD_ACCUM", "1")))
    except ValueError:
        return 1


def _guard_no_dataparallel(use_cuda: bool) -> None:
    """Fail fast (with guidance) rather than hit HF Trainer's broken DataParallel path.

    When >1 GPU is visible to a single, non-distributed process the HF Trainer wraps the
    model in ``nn.DataParallel``, which crashes with PEFT/LoRA ("Expected all tensors to
    be on the same device ... cuda:1 vs cuda:0"). Real multi-GPU training must go through
    ``accelerate launch`` / ``torchrun`` (they set WORLD_SIZE/LOCAL_RANK and use DDP, so
    the Trainer manages per-process placement). No-op on CPU, under a distributed launch,
    or with <=1 visible GPU (a plain single-GPU run is fine).
    """
    if not use_cuda:
        return
    import os

    if any(os.environ.get(k) for k in ("WORLD_SIZE", "LOCAL_RANK", "RANK")):
        return  # launched under accelerate/torchrun -> DDP, the Trainer won't use DataParallel
    import torch

    if torch.cuda.device_count() > 1:
        raise RuntimeError(
            "Multiple CUDA devices are visible to a single, non-distributed process. The "
            "HuggingFace Trainer would wrap the model in nn.DataParallel, which is broken "
            "with PEFT/LoRA (\"Expected all tensors to be on the same device, cuda:1 vs "
            "cuda:0\"). Launch multi-GPU training with `accelerate launch` or `torchrun` "
            "(DDP), or pin a single GPU via CUDA_VISIBLE_DEVICES=0."
        )


def _load_causal_lm(model_name: str, *, use_cuda: bool):
    """Load a causal-LM for training/generation.

    On CUDA we load in bf16 (H100-friendly, ~halves memory vs fp32); on CPU we keep the
    framework default so the CPU smoke test stays correct. Some target checkpoints (the
    multimodal ``google/gemma-4-*-it`` -> ``Gemma4ForConditionalGeneration``) aren't
    registered for ``AutoModelForCausalLM``; only in that *architecture* failure do we
    fall back to the checkpoint's causal-LM text tower (``.language_model``). Plain text
    CausalLM checkpoints (Qwen2.5, Llama, ...) load on the first try and never reach the
    fallback, so their behavior is unchanged. Network/OS errors are NOT swallowed (they
    propagate so the CPU test's offline-skip still works).

    NOTE: the multimodal fallback is best-effort and unverified against the real gemma-4
    checkpoints (CPU-only validation here) — the orchestrator should sanity-check it on
    the actual ``google/gemma-4-*-it`` weights.
    """
    import torch
    from transformers import AutoModelForCausalLM

    # transformers 5.x renamed `torch_dtype` -> `dtype` (torch_dtype now warns + maps to it).
    load_kwargs = {"dtype": torch.bfloat16} if use_cuda else {}
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except (ValueError, KeyError) as exc:
        try:
            from transformers import AutoModelForImageTextToText

            multimodal = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        except Exception:
            raise exc
        text_model = getattr(multimodal, "language_model", None)
        if text_model is not None and hasattr(text_model, "get_input_embeddings"):
            return text_model
        raise exc


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
    from transformers import AutoTokenizer

    dev = _resolve_device(device)
    tok = AutoTokenizer.from_pretrained(model)
    mdl = _load_causal_lm(model, use_cuda=dev.startswith("cuda"))  # bf16 on CUDA
    if adapter_dir:
        from peft import PeftModel

        mdl = PeftModel.from_pretrained(mdl, adapter_dir)
    # Inference (no Trainer): a single .to(dev) is safe — no DataParallel. For models
    # too large for one GPU, pass device_map="auto" at the call site instead.
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
    gradient_accumulation_steps: Optional[int] = None,
) -> TinkerSFTOutcome:
    """LoRA SFT on a local model via TRL; writes a PEFT adapter + registry entry.

    Local counterpart of ``run_tinker_sft_job`` — same signature shape, returns the
    same :class:`TinkerSFTOutcome`, and records the adapter under the shared registry
    with ``backend: "local"`` (path is the local PEFT dir, not a ``tinker://`` URI).

    On CUDA: bf16 weights + gradient checkpointing for H100 memory headroom, and the
    HF Trainer/accelerate owns device placement so a 4xH100 ``accelerate launch`` DDP
    run works (the model is never manually ``.to(device)``-d). Loss is computed on the
    COMPLETION tokens only (prompt masked), matching the Tinker backend's weighting.
    ``gradient_accumulation_steps`` defaults to ``$DEMENTOR_GRAD_ACCUM`` or 1.
    """
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = _resolve_device(device)
    use_cuda = dev.startswith("cuda")
    _guard_no_dataparallel(use_cuda)
    grad_accum = _resolve_grad_accum(gradient_accumulation_steps)

    train_examples, _eval = prepare_sft_examples(dataset_config)
    # Keep prompt/completion in separate fields (not concatenated into one "text" field)
    # so TRL masks the prompt and trains on completion tokens only (completion_only_loss),
    # matching the prompt-weight-0 / completion-weight-1 scheme in tinker_backend.
    prompts = [format_prompt(ex, prompt_template, i) for i, ex in enumerate(train_examples)]
    completions = [format_completion(ex, completion_template, i) for i, ex in enumerate(train_examples)]
    train_ds = Dataset.from_dict({"prompt": prompts, "completion": completions})

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_causal_lm(base_model, use_cuda=use_cuda)
    if use_cuda:
        model.config.use_cache = False  # incompatible with gradient checkpointing
        model.enable_input_require_grads()  # let grad-ckpt reach LoRA params over a frozen base

    sft_config = SFTConfig(
        output_dir=str(output_dir / "_trainer"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        completion_only_loss=True,
        max_length=max_length,
        use_cpu=(dev == "cpu"),
        bf16=use_cuda,
        gradient_checkpointing=use_cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_cuda else None,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        peft_config=_lora_config(lora_kwargs),
        processing_class=tokenizer,
    )
    result = trainer.train()

    loss = getattr(result, "training_loss", None)
    loss_history = [float(loss)] if loss is not None else []
    # Under DDP (`accelerate launch`) the post-train code runs in every rank; only the
    # main process writes the adapter/tokenizer/registry (concurrent writers corrupt the
    # files) and a barrier makes other ranks wait for the files. No-op for 1 process.
    if trainer.is_world_process_zero():
        model_to_save = trainer.accelerator.unwrap_model(trainer.model)
        model_to_save.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        record_adapter_mapping(
            weights_name,
            str(output_dir),
            registry_path,
            metadata={"backend": "local", "checkpoint_path": str(output_dir), "base_model": base_model},
        )
    trainer.accelerator.wait_for_everyone()
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
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = _resolve_device(params.device)
    use_cuda = dev.startswith("cuda")
    _guard_no_dataparallel(use_cuda)
    grad_accum = _resolve_grad_accum(params.gradient_accumulation_steps)

    tokenizer = AutoTokenizer.from_pretrained(params.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_causal_lm(params.model_name, use_cuda=use_cuda)
    # Conventional LoRA-DPO reference: MERGE the frozen SFT adapter into the base, then
    # train a FRESH DPO LoRA on top. DPOTrainer (peft_config + ref_model=None) takes the
    # reference as the adapter-DISABLED model == base+SFT == the frozen SFT init, which is
    # what we want to regularize toward (and is memory-efficient: no second ref copy).
    # The previous code loaded the SFT adapter trainable with no peft_config, so the
    # adapter-disabled reference was the BASE model, regularizing toward base not SFT.
    if params.load_checkpoint_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, params.load_checkpoint_path)
        model = model.merge_and_unload()
    if use_cuda:
        model.config.use_cache = False
        model.enable_input_require_grads()

    files = {"train": str(train_jsonl)}
    if eval_jsonl is not None:
        files["test"] = str(eval_jsonl)
    ds = load_dataset("json", data_files=files)

    dpo_config = DPOConfig(
        output_dir=str(out_dir / "_trainer"),
        per_device_train_batch_size=params.batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=params.num_epochs,
        learning_rate=params.learning_rate,
        beta=params.dpo_beta,
        max_length=params.max_length,
        seed=params.seed,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        use_cpu=(dev == "cpu"),
        bf16=use_cuda,
        gradient_checkpointing=use_cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_cuda else None,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT -> reference is the adapter-disabled (base+merged-SFT) model
        args=dpo_config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("test"),
        processing_class=tokenizer,
        peft_config=_lora_config({"rank": params.lora_rank}),
    )
    trainer.train()
    if trainer.is_world_process_zero():  # DDP-safe: only the main process writes artifacts
        model_to_save = trainer.accelerator.unwrap_model(trainer.model)
        model_to_save.save_pretrained(str(out_dir))
        record_adapter_mapping(
            params.weights_name,
            str(out_dir),
            params.registry_path,
            metadata={"backend": "local", "checkpoint_path": str(out_dir),
                      "base_model": params.model_name, "sft_parent": params.load_checkpoint_path},
        )
    trainer.accelerator.wait_for_everyone()
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
