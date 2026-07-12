#!/usr/bin/env python
"""transformers-5.5.4 compatibility shims for the RDO steering-port roster.

Importing this module (idempotently) installs global patches so the roster's
models LOAD + GENERATE under transformers 5.5.4. Every patch is a no-op for the
models that don't need it, so importing this is safe for *every* roster model
(the working ones are unaffected).

Patches installed on import:
  1. LossKwargs import shim        -> fixes internlm3-8b trust_remote_code ImportError
                                      (`from transformers.utils import LossKwargs`, removed in 5.5.x).
  2. tokenizer chat_template fill  -> fixes gpt-oss-20b (no template -> harmony) + a minimal
                                      generic fallback for any other template-less tokenizer.
  3. NemotronH generation fix      -> fixes nemotron-nano generate(): the model's shipped
                                      prepare_inputs_for_generation indexes cache_position[-1]
                                      while transformers-5.5.4 leaves cache_position=None at prefill.
  3b. offline hub-kernel shim      -> fixes granite-4-h-small (GraniteMoeHybrid): its Mamba-2 mixer
                                      resolves fused kernels from the HF hub at load; offline mode
                                      raises OfflineModeIsEnabled. Falls back to the native path.
                                      (Also USE_HUB_KERNELS=NO / expandable_segments set at import.)

Exports:
  clamp_layers(layers, n_layers)   -> depth-clamped, de-duped, sorted layer list
                                      (olmoe-1b-7b has only 16 layers vs a hardcoded L20).

Wire-in: `import rdo_compat` at the TOP of every /data entry point that loads a model
(rdo_port.py, compute_dim.py, cone_eval.py, run_rdo_model.py). Because the patches hook the
AutoTokenizer/AutoModelForCausalLM `.from_pretrained` classmethods (resolved at call time), the
import only needs to happen before the first `from_pretrained(...)` call, not before
`dementor...load_causal_lm` is imported.
"""
import os
import typing
from typing import Optional

# GraniteMoeHybrid (Mamba-2 mixer) resolves optional fused kernels (causal-conv1d / mamba-ssm) and
# a hub RMSNorm kernel at load time. On this offline box the version lookup hits the HF hub and raises
# OfflineModeIsEnabled, crashing the load before any weights stream. USE_HUB_KERNELS=NO disables the
# decorator-based hub-kernel path (native RMSNorm/rotary fallback -- numerically equivalent); the
# lazy_load_kernel shim below covers the Mamba fast-path lookups the env var does not gate. Set via
# setdefault (before transformers imports) so it is read when transformers.integrations.hub_kernels
# computes _kernels_enabled, and never overrides an explicit operator choice.
os.environ.setdefault("USE_HUB_KERNELS", "NO")
# Reduce allocator fragmentation on the model-parallel (2-card) runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_HARMONY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gptoss_harmony.jinja")

# Minimal, correct user/assistant template. Only used if a tokenizer has NO chat_template AND is not
# gpt-oss; for the RDO roster this never fires (every non-gpt-oss model ships a template) -- it is a
# defensive net so make_render()/apply_chat_template never hard-crashes.
_GENERIC_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>\n' + message['content'] + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
)


# --------------------------------------------------------------------------- 1. LossKwargs shim
def _install_loss_kwargs_shim():
    try:
        import transformers.utils as u
    except Exception:
        return
    if getattr(u, "LossKwargs", None) is not None:
        return

    class LossKwargs(typing.TypedDict, total=False):
        """Shim for transformers.utils.LossKwargs (removed in transformers 5.5.x).

        Must be a TypedDict so remote code can do `class K(FlashAttentionKwargs, LossKwargs)`."""
        num_items_in_batch: Optional[int]

    u.LossKwargs = LossKwargs
    # Some remote modeling files import it from these paths too.
    for modname in ("transformers.modeling_utils", "transformers.loss.loss_utils"):
        try:
            import importlib
            m = importlib.import_module(modname)
            if getattr(m, "LossKwargs", None) is None:
                m.LossKwargs = LossKwargs
        except Exception:
            pass


# --------------------------------------------------------------------------- 2. chat_template fill
def _load_harmony_template():
    try:
        with open(_HARMONY_PATH) as f:
            return f.read()
    except Exception:
        return None


def ensure_chat_template(tokenizer, name=None):
    """Set a sensible chat_template if the tokenizer has none. gpt-oss -> harmony; else generic."""
    try:
        if getattr(tokenizer, "chat_template", None):
            return
    except Exception:
        return
    n = (str(name) if name is not None else "").lower()
    tpl = None
    if "gpt-oss" in n or "gptoss" in n or "gpt_oss" in n:
        tpl = _load_harmony_template()
    if tpl is None:
        # also catch gpt-oss loaded from a path lacking 'gpt-oss' in the name
        try:
            special = " ".join(getattr(tokenizer, "additional_special_tokens", []) or [])
        except Exception:
            special = ""
        if "<|start|>" in special or "<|channel|>" in special:
            tpl = _load_harmony_template()
    if tpl is None:
        tpl = _GENERIC_TEMPLATE
    try:
        tokenizer.chat_template = tpl
    except Exception:
        pass


# --------------------------------------------------------------------------- 3. NemotronH gen fix
def patch_generation_cache(model):
    """Fix NemotronH generation under the local Transformers version.

    The shipped NemotronH code needs its own hybrid cache because the Mamba layers read
    `conv_states` / `ssm_states`. Transformers 5.5 may hand `prepare_inputs_for_generation` a
    generic DynamicCache instead, which crashes as soon as cached generation reaches a Mamba layer.
    Also synthesize cache_position when it is omitted at prefill.
    """
    try:
        cls_name = type(model).__name__
    except Exception:
        return
    if "NemotronH" not in cls_name:
        return
    if getattr(model, "_rdo_cache_pos_patched", False):
        return
    import torch

    orig = model.prepare_inputs_for_generation  # bound method
    orig_forward = model.forward                # bound method
    try:
        orig_globals = orig.__func__.__globals__
    except Exception:
        orig_globals = {}
    hybrid_cache_cls = orig_globals.get("NemotronHHybridDynamicCache")

    def _hybrid_cache(input_ids, past_key_values):
        if hybrid_cache_cls is None or hasattr(past_key_values, "conv_states"):
            return past_key_values
        try:
            past_key_values = hybrid_cache_cls(
                model.config,
                input_ids.shape[0],
                dtype=getattr(model, "dtype", torch.float16),
                device=getattr(model, "device", input_ids.device),
            )
        except Exception:
            pass
        return past_key_values

    def patched(input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
                cache_position=None, position_ids=None, use_cache=True, **kwargs):
        # GenerationMixin may pass a generic DynamicCache even when use_cache=False.
        # Nemotron-H Mamba layers treat any non-None cache as their hybrid cache and
        # read conv_states/ssm_states, so drop generic caches on uncached generation.
        if not use_cache and past_key_values is not None and not hasattr(past_key_values, "conv_states"):
            past_key_values = None
        if use_cache:
            past_key_values = _hybrid_cache(input_ids, past_key_values)
        if cache_position is None:
            past_len = 0
            if past_key_values is not None:
                try:
                    past_len = int(past_key_values.get_seq_length())
                except Exception:
                    past_len = 0
            cache_position = torch.arange(past_len, input_ids.shape[1], device=input_ids.device)
        return orig(input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds, cache_position=cache_position,
                    position_ids=position_ids, use_cache=use_cache, **kwargs)

    def forward_patched(*args, **kwargs):
        use_cache = kwargs.get("use_cache")
        past_key_values = kwargs.get("past_key_values")
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if use_cache is False and past_key_values is not None and not hasattr(past_key_values, "conv_states"):
            kwargs["past_key_values"] = None
        elif use_cache and past_key_values is not None and not hasattr(past_key_values, "conv_states") and input_ids is not None:
            kwargs["past_key_values"] = _hybrid_cache(input_ids, past_key_values)
        return orig_forward(*args, **kwargs)

    model.prepare_inputs_for_generation = patched
    model.forward = forward_patched
    model._rdo_cache_pos_patched = True


# --------------------------------------------------------------------------- from_pretrained hooks
def _install_from_pretrained_patches():
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception:
        return

    if not getattr(AutoTokenizer.from_pretrained, "_rdo_patched", False):
        _orig_tok = AutoTokenizer.from_pretrained

        def tok_fp(*args, **kwargs):
            tok = _orig_tok(*args, **kwargs)
            name = args[0] if args else kwargs.get("pretrained_model_name_or_path")
            try:
                ensure_chat_template(tok, name)
            except Exception:
                pass
            return tok

        tok_fp._rdo_patched = True
        AutoTokenizer.from_pretrained = staticmethod(tok_fp)

    if not getattr(AutoModelForCausalLM.from_pretrained, "_rdo_patched", False):
        _orig_model = AutoModelForCausalLM.from_pretrained

        def model_fp(*args, **kwargs):
            model = _orig_model(*args, **kwargs)
            try:
                patch_generation_cache(model)
            except Exception:
                pass
            return model

        model_fp._rdo_patched = True
        AutoModelForCausalLM.from_pretrained = staticmethod(model_fp)


# --------------------------------------------------------------------------- 3b. offline kernel shim
def _install_kernels_offline_shim():
    """Wrap transformers.integrations.hub_kernels.lazy_load_kernel so an offline / unreachable hub
    lookup returns None (native fallback) instead of raising. GraniteMoeHybrid's Mamba-2 mixer calls
    lazy_load_kernel('causal-conv1d') / ('mamba-ssm') during __init__; the underlying kernels
    get_kernel() hits the hub for version resolution and raises OfflineModeIsEnabled here, which
    lazy_load_kernel does not catch (only FileNotFoundError/AssertionError). Returning None triggers
    the model's documented naive PyTorch path -- same math, no network."""
    try:
        from transformers.integrations import hub_kernels
    except Exception:
        return
    orig = getattr(hub_kernels, "lazy_load_kernel", None)
    if orig is None or getattr(orig, "_rdo_patched", False):
        return

    def safe_lazy_load_kernel(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except Exception as exc:  # offline / network / version-resolution failure
            try:
                hub_kernels.logger.warning_once(
                    f"rdo_compat: hub kernel unavailable ({type(exc).__name__}); using native fallback"
                )
            except Exception:
                pass
            return None

    safe_lazy_load_kernel._rdo_patched = True
    hub_kernels.lazy_load_kernel = safe_lazy_load_kernel
    # Rebind the name in any modeling module that already imported it by value.
    import sys as _sys
    for mod in list(_sys.modules.values()):
        try:
            if getattr(mod, "lazy_load_kernel", None) is orig:
                mod.lazy_load_kernel = safe_lazy_load_kernel
        except Exception:
            pass


# --------------------------------------------------------------------------- 4. layer clamp helper
def clamp_layers(layers, n_layers):
    """Clamp (and resolve negative) layer indices into [0, n_layers-1]; de-dup + sort.

    Robust for any depth: e.g. clamp_layers([4,8,14,20], 16) -> [4, 8, 14, 15]."""
    n = int(n_layers)
    out = []
    for L in layers:
        li = int(L)
        if li < 0:
            li = n + li
        li = max(0, min(li, n - 1))
        out.append(li)
    return sorted(set(out))


# --------------------------------------------------------------------------- install on import
_install_loss_kwargs_shim()
_install_kernels_offline_shim()
_install_from_pretrained_patches()
