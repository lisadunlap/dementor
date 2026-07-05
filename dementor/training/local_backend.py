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

from .common import (
    SFTDatasetConfig,
    SFTOutcome,
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

    bf16 on CUDA (H100-friendly, ~halves memory vs fp32); framework default on CPU. Only on an
    *architecture* failure do we fall back to a multimodal checkpoint's text tower
    (``.language_model``); plain CausalLM checkpoints are unchanged and network/OS errors propagate.
    Multimodal caveat: the fallback tunes only the text decoder (images ignored), so it is not a true multimodal training path.
    """
    import torch
    from transformers import AutoModelForCausalLM

    # transformers 5.x renamed `torch_dtype` -> `dtype` (torch_dtype now warns + maps to it).
    load_kwargs = {"dtype": torch.bfloat16} if use_cuda else {}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
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
    return _maybe_extract_vision_text_tower(model)


def _maybe_extract_vision_text_tower(model):
    """Splice a Gemma-4 VLM's text decoder into a pure-text ``Gemma4ForCausalLM``.

    The larger Gemma-4 checkpoints (26B/31B) load as ``Gemma4ForConditionalGeneration`` and
    set ``text_config.use_bidirectional_attention == "vision"``; that mask path *requires*
    ``mm_token_type_ids`` at train time (``ValueError: mm_token_type_ids is required ... when
    training``), which text-only SFT/DPO batches never carry. We build a ``Gemma4ForCausalLM``
    shell on the meta device (no allocation) and reference-transplant the VLM's real text
    decoder (``model.model.language_model``) and ``lm_head`` into it, then pin its config to the
    text config — so forward/generate run entirely on the text path and never touch the vision
    mask. No weight copy, so memory is unchanged. Text-only gemmas (E4B, whose flag is not
    ``"vision"``) and every non-Gemma checkpoint are returned untouched; any failure falls back
    to the full model."""
    import torch

    try:
        text_cfg = model.config.get_text_config()
    except Exception:
        return model
    if getattr(text_cfg, "use_bidirectional_attention", None) != "vision":
        return model
    try:
        from transformers import Gemma4ForCausalLM

        text_decoder = model.model.language_model  # real Gemma4TextModel (loaded weights)
        lm_head = model.lm_head
        with torch.device("meta"):
            causal = Gemma4ForCausalLM(text_cfg)
        causal.model = text_decoder
        causal.lm_head = lm_head
        causal.config = text_cfg
        return causal
    except Exception:
        return model


def fsdp_wrap_layer_names(model) -> list[str]:
    """Decoder-layer class name(s) for FSDP transformer wrapping — architecture-agnostic.

    Primary: ``model._no_split_modules`` (the per-arch decoder-layer class list that accelerate
    itself reads to build the FSDP wrap policy). Fallback for checkpoints lacking it: the most
    common module class at a ``*.layers.<int>`` path, else ``[]``. Pure inspection (no
    training/CUDA), so it works on a meta-device model.
    """
    no_split = getattr(model, "_no_split_modules", None)
    if no_split:
        seen: set[str] = set()  # de-dup, preserve order
        ordered: list[str] = []
        for name in no_split:
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
        if ordered:
            return ordered

    import re
    from collections import Counter

    at_layer = re.compile(r"(?:^|\.)layers\.\d+$")  # module sitting at `...layers.<int>`
    counts: Counter[str] = Counter()
    for name, module in model.named_modules():
        if at_layer.search(name):
            counts[type(module).__name__] += 1
    return [counts.most_common(1)[0][0]] if counts else []


def _ensure_fsdp_wrap_class(model) -> None:
    """Publish the FSDP transformer wrap class for checkpoints that lack ``_no_split_modules``.

    Belt-and-suspenders fallback: normally the generic ``fsdp.yaml`` lets accelerate auto-derive the
    wrap class from ``model._no_split_modules``; only when that is absent do we export
    ``fsdp_wrap_layer_names``'s introspected class via ``FSDP_TRANSFORMER_CLS_TO_WRAP`` before the
    Trainer builds its Accelerator. No-op unless ``ACCELERATE_USE_FSDP=true`` with that env unset and
    ``_no_split_modules`` missing, so single-GPU / CPU / DDP and explicit settings are untouched.
    """
    import os

    if os.environ.get("ACCELERATE_USE_FSDP") != "true":
        return  # not an FSDP launch (single-GPU / CPU / plain DDP) -> nothing to wrap
    if os.environ.get("FSDP_TRANSFORMER_CLS_TO_WRAP"):
        return  # explicitly set (e.g. legacy hardcoded yaml) -> respect it
    if getattr(model, "_no_split_modules", None):
        return  # accelerate auto-derives from _no_split_modules -> let the config-only path run
    names = fsdp_wrap_layer_names(model)
    if names:
        os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = ",".join(names)


def generate_local_responses(
    *,
    model: str,
    prompts: list[str],
    chat_template_kwargs: Optional[dict] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: Optional[str] = None,
    adapter_dir: Optional[str] = None,
    batch_size: int = 16,
) -> list[str]:
    """Sample a local HF model (optionally with a PEFT adapter).

    Local counterpart of ``matrix.generate_target_responses``: applies the chat
    template (greedy when ``temperature == 0``) and returns decoded completions.

    Prompts are generated in left-padded batches for throughput. On a CUDA OOM the
    batch size is halved (and kept reduced for the remaining prompts) down to 1, so a
    large dense model degrades to single-sequence generation instead of crashing. Pair
    with ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` to survive the tightest
    single-sequence case (e.g. gemma-4-31B at bf16 on one 80 GB card).
    """
    import torch
    from transformers import AutoTokenizer

    dev = _resolve_device(device)
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only batched generation requires left padding
    mdl = _load_causal_lm(model, use_cuda=dev.startswith("cuda"))  # bf16 on CUDA
    if adapter_dir:
        from peft import PeftModel

        mdl = PeftModel.from_pretrained(mdl, adapter_dir)
    # Inference (no Trainer): a single .to(dev) is safe — no DataParallel. For models
    # too large for one GPU, pass device_map="auto" at the call site instead.
    mdl.to(dev).eval()

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
        for prompt in prompts
    ]

    outputs: list[str] = []
    i = 0
    bs = max(1, batch_size)
    while i < len(texts):
        chunk = texts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True).to(dev)
        try:
            with torch.no_grad():
                seq = mdl.generate(**enc, **gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            del enc
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)  # keep the smaller batch for the rest of the run
            continue
        new_tokens = seq[:, enc["input_ids"].shape[1]:]
        outputs.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
        i += len(chunk)
    return outputs


def _save_peft_adapter(trainer, output_dir, tokenizer=None) -> None:
    """Save the trained PEFT adapter, FSDP-aware.

    Under FSDP, ``accelerator.unwrap_model`` only peels the wrapper object while the
    parameters stay SHARDED across ranks, so a plain ``save_pretrained`` writes rank0's
    flat shard => a truncated / empty / NaN adapter. Instead we gather a FULL (unsharded)
    state dict as a COLLECTIVE on all ranks (offloaded to CPU, materialized on rank0 only),
    then save the adapter on rank0 — PEFT filters the LoRA weights out via its own naming.
    When FSDP is inactive (single GPU / CPU / plain DDP) this falls back to the original
    rank0 ``save_pretrained``, so off-FSDP behavior is unchanged.

    MUST be called on EVERY rank — the gather is a collective. Do NOT wrap it in a rank0
    guard (that deadlocks: rank0 enters the all-gather, the others never do).
    """
    acc = trainer.accelerator
    unwrapped = acc.unwrap_model(trainer.model)
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        StateDictType,
    )

    # The FSDP root is ``model_wrapped`` under the HF Trainer; fall back to ``model``.
    candidates = [getattr(trainer, "model_wrapped", None), trainer.model]
    fsdp_root = next((m for m in candidates if m is not None and isinstance(m, FSDP)), None)

    if fsdp_root is None:  # no FSDP -> original single-process save path
        if acc.is_main_process:
            unwrapped.save_pretrained(str(output_dir))
            if tokenizer is not None:
                tokenizer.save_pretrained(str(output_dir))
        return

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_root, StateDictType.FULL_STATE_DICT, cfg):
        full_sd = fsdp_root.state_dict()  # collective: every rank enters; rank0 materializes
    if acc.is_main_process:
        unwrapped.save_pretrained(str(output_dir), state_dict=full_sd)
        if tokenizer is not None:
            tokenizer.save_pretrained(str(output_dir))


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
) -> SFTOutcome:
    """LoRA SFT on a local model via TRL; writes a PEFT adapter + registry entry.

    Local counterpart of ``run_tinker_sft_job`` — same signature shape, returns the
    same :class:`SFTOutcome`, and records the adapter under the shared registry
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
    # Architecture-agnostic FSDP: the generic fsdp.yaml lets accelerate auto-derive the
    # decoder-layer wrap class from model._no_split_modules; this only steps in for the rare
    # checkpoint lacking that attribute (no-op off an FSDP launch, so single-GPU is unchanged).
    _ensure_fsdp_wrap_class(model)
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
    # Save the adapter (FSDP-aware: the helper gathers a full state dict as a collective
    # on every rank, then writes on rank0). Under `accelerate launch` the post-train code
    # runs in every rank, so the registry write stays rank0-only (concurrent writers
    # corrupt files) and the barrier makes other ranks wait. No-op for 1 process.
    _save_peft_adapter(trainer, output_dir, tokenizer)
    if trainer.is_world_process_zero():
        record_adapter_mapping(
            weights_name,
            str(output_dir),
            registry_path,
            metadata={"backend": "local", "checkpoint_path": str(output_dir), "base_model": base_model},
        )
    trainer.accelerator.wait_for_everyone()
    return SFTOutcome(
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
    # Architecture-agnostic FSDP wrap class (see run_local_sft_job); after merge_and_unload the
    # model is the plain base, whose _no_split_modules accelerate auto-derives from. No-op off FSDP.
    _ensure_fsdp_wrap_class(model)

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
    # trl logs a per-token entropy metric via entropy_from_logits(shift_logits.detach()), which
    # forces a contiguous [tokens, vocab] fp32 copy. On gemma's ~256K vocab with a 62 GB text
    # tower (gemma-4-31B) that ~4.6 GB copy OOMs a single 80 GB card mid-run. The metric is
    # detached — not part of the DPO loss or gradient — so we swap it for a zero-cost stand-in.
    # The trained adapter is byte-identical; only the logged 'entropy' reads 0.
    try:
        import trl.trainer.dpo_trainer as _dpo_mod
        import torch as _torch
        _dpo_mod.entropy_from_logits = lambda logits, *a, **k: _torch.zeros(
            logits.shape[:-1], device=logits.device, dtype=_torch.float32)
    except Exception:
        pass
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
    # FSDP-aware adapter save (collective gather on all ranks; rank0 writes); then rank0
    # records the registry mapping.
    _save_peft_adapter(trainer, out_dir)
    if trainer.is_world_process_zero():  # DDP-safe: only the main process writes artifacts
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
