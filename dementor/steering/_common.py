"""Shared, model-agnostic helpers for the steering package.

Both the additive sweep tool (``activation_steering``) and the productionized
projection-ablation rung (``steering_rung``) need to:

  1. locate a decoder's transformer-block list generically (no hardcoded layer
     class, in the same spirit as ``training.local_backend.fsdp_wrap_layer_names``)
     and resolve possibly-negative layer indices, and
  2. load an HF causal LM + tokenizer -- either right-padded for teacher-forced
     activation capture or left-padded for decoder-only generation.

These used to live in ``activation_steering`` and were imported *back* into the
newer ``steering_rung`` (a backwards dependency); they now live here and both
modules import from this module instead. ``activation_steering`` re-exports
``get_transformer_layers`` / ``resolve_layer_index`` for backward compatibility.

torch/transformers imports are kept lazy so analysis code can import this module
without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Architecture-agnostic decoder-layer discovery + index resolution.
# ---------------------------------------------------------------------------
def _candidate_layer_containers(model: Any) -> list[Any]:
    # Ordered walk of the common decoder-block locations; first non-empty match wins. Order matters:
    # standard decoder-only LMs (Llama/Qwen/Mistral/GraniteMoeHybrid) keep blocks at model.model.layers,
    # while multimodal / conditional-generation checkpoints (Gemma4 VLM) expose the text tower one or
    # two attributes deeper (model.language_model.layers or model.model.language_model.layers).
    candidates = [
        ("model", "layers"),                     # Llama/Qwen/Mistral/Granite decoder-only LMs
        ("language_model", "layers"),            # Gemma4 VLM text tower exposed directly
        ("model", "language_model", "layers"),   # *ForConditionalGeneration (text tower under .model)
        ("language_model", "model", "layers"),   # VLMs nesting a full text sub-model
        ("transformer", "h"),                    # GPT-2 / Falcon style
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),          # OPT / encoder-decoder decoders
        ("backbone", "layers"),                  # Mamba-style backbones
    ]
    out = []
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            out.append(obj)
    return out


def _looks_like_decoder_block(module: Any) -> bool:
    """True if `module` has the hallmarks of a residual-stream decoder block (attention, MLP/MoE, or
    a Mamba mixer + a layernorm) -- distinguishes the decoder stack from other ModuleLists such as an
    MoE expert list (128 experts on gpt-oss) which has none of these markers."""
    markers = ("self_attn", "self_attention", "attn", "mlp", "feed_forward",
               "block_sparse_moe", "mamba", "input_layernorm", "post_attention_layernorm")
    return any(hasattr(module, m) for m in markers)


def _recursive_find_layers(model: Any) -> Any:
    """Last-resort architecture-agnostic fallback: scan every submodule for the longest nn.ModuleList
    whose first element looks like a decoder block. Handles arbitrarily-nested towers (future VLMs)
    without a hardcoded path, while the decoder-block filter avoids grabbing an expert ModuleList."""
    import torch.nn as nn

    best = None
    for _name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0 and _looks_like_decoder_block(module[0]):
            if best is None or len(module) > len(best):
                best = module
    return best


def get_transformer_layers(model: Any) -> Any:
    for layers in _candidate_layer_containers(model):
        if len(layers) > 0:
            return layers
    fallback = _recursive_find_layers(model)
    if fallback is not None:
        return fallback
    raise ValueError("Could not locate transformer block list on this model.")


def resolve_layer_index(layer: int, n_layers: int) -> int:
    if layer < 0:
        layer = n_layers + layer
    if layer < 0 or layer >= n_layers:
        raise ValueError(f"Layer index {layer} out of range for {n_layers} layers.")
    return layer


# ---------------------------------------------------------------------------
# Single parametrized HF causal-LM loader. Replaces the three near-identical
# loaders that used to live in activation_steering / steering_rung
# (``_load_causal_lm`` / ``_load_causal_encoder`` / ``_ensure_model_tokenizer``).
# ---------------------------------------------------------------------------
def load_causal_lm(
    model_name: str,
    *,
    padding_side: str,
    device: str | None = None,
    dtype: str = "auto",
    peft_adapter_path: str | Path | None = None,
    strict_auto: bool = False,
    tokenizer: Any = None,
) -> tuple[Any, Any, str]:
    """Load an HF causal LM + tokenizer; returns ``(tokenizer, model, resolved_device)``.

    The one intentional, load-bearing knob is ``padding_side``: ``"right"`` for
    teacher-forced activation capture (so the response span ``[p_len:tot]`` can be
    sliced) vs ``"left"`` for decoder-only generation.

    Args:
        padding_side: ``"left"`` or ``"right"`` (see above).
        peft_adapter_path: optional PEFT adapter to wrap the base model with.
        strict_auto: preserves the additive-sweep tool's stricter dtype policy --
            ``dtype="auto"`` picks bf16 only when ``resolved_device == "cuda"``
            exactly, and ``dtype="default"`` never auto-selects a dtype. When
            ``False`` (the production-rung policy) both ``"auto"`` and ``"default"``
            pick bf16 on any ``cuda*`` device.
        tokenizer: reuse an already-loaded tokenizer instead of loading a fresh one;
            it is still pad-token / padding-side normalized. Loaded when ``None``.
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Model-parallel: when DEMENTOR_MP=1 (paired with CUDA_VISIBLE_DEVICES=a,b), shard the weights
    # across the visible GPUs with accelerate's device_map="auto" so a model too large for one card
    # fits. We must NOT call model.to() afterwards (it breaks accelerate's per-shard dispatch). The
    # per-layer feature hooks / ablation ops already co-locate their tensors onto each block's own
    # shard; inputs go to the input-embedding shard, returned as resolved_device.
    mp = bool(os.environ.get("DEMENTOR_MP")) and resolved_device.startswith("cuda")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

    if mp:
        # dtype="auto" respects a checkpoint's quantization_config (e.g. gpt-oss MXFP4 stays 4-bit
        # rather than dequantizing to bf16) and otherwise picks the native dtype (bf16 for Mixtral).
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
            max_memory={i: "75GiB" for i in range(torch.cuda.device_count())},
        )
    else:
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if dtype not in ("auto", "default"):
            model_kwargs["torch_dtype"] = getattr(torch, dtype)
        elif strict_auto:
            if dtype == "auto" and resolved_device == "cuda":
                model_kwargs["torch_dtype"] = torch.bfloat16
        elif resolved_device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if peft_adapter_path is not None:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading PEFT adapters requires `peft`. Install it or pass a merged HF model path."
            ) from exc
        model = PeftModel.from_pretrained(model, str(peft_adapter_path))

    if mp:
        try:
            model.config.use_cache = False
        except Exception:
            pass
        try:
            model.generation_config.use_cache = False
        except Exception:
            pass
        model.eval()
        resolved_device = model.get_input_embeddings().weight.device
    else:
        model.to(resolved_device).eval()
    return tokenizer, model, resolved_device
