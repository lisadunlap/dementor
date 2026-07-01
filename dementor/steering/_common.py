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
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("backbone", "layers"),
        ("language_model", "model", "layers"),
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


def get_transformer_layers(model: Any) -> Any:
    for layers in _candidate_layer_containers(model):
        if len(layers) > 0:
            return layers
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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

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

    model.to(resolved_device).eval()
    return tokenizer, model, resolved_device
