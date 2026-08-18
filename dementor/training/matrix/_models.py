"""Per-model helpers: response cleaning, source-id parsing, and renderer resolution."""
from __future__ import annotations

import re

from dementor import config


def clean_response(model: str, raw: str) -> str:
    """Strip per-model chat-template artifacts to leave just the assistant message.

    gpt-oss uses the harmony multi-channel format
    (``<|channel|>analysis<|message|>CoT<|end|>…<|channel|>final<|message|>ANSWER``).
    Normally the answer lives in the ``final`` channel. But a model fine-tuned —
    especially with DPO — toward another model frequently STOPS emitting the
    ``<|channel|>final<|message|>`` marker and dumps the answer straight under the
    ``analysis`` (or ``commentary``) channel header. The previous version only
    stripped when ``final`` was present, so post-DPO outputs leaked raw channel
    markup (and reasoning) into the stored response, spuriously inflating measured
    persistence. We now fall back to the last channel header when ``final`` is
    absent, and strip any residual harmony scaffolding tokens for all models.
    """
    text = raw
    # Cohere/Aya uses a textual end-of-turn token and a distinct textual pad token. Batched
    # generation right-pads sequences that finish before the longest row, and several local
    # evaluation paths intentionally decode with special tokens visible so gpt-oss Harmony
    # channels can be parsed below. Keep only the assistant text before Aya's terminator; any
    # following PAD tokens are batching artifacts rather than model output.
    aya_eot = "<|END_OF_TURN_TOKEN|>"
    if aya_eot in text:
        text = text.split(aya_eot, 1)[0]
    text = text.replace("<PAD>", "")
    if model.startswith("openai/gpt-oss"):  # 20b and 120b both use the harmony format
        final = "<|channel|>final<|message|>"
        if final in text:
            text = text.rsplit(final, 1)[1]                      # the genuine final answer
        else:
            headers = list(re.finditer(r"<\|channel\|>\w+<\|message\|>", text))
            if headers:
                text = text[headers[-1].end():]                 # answer dumped under analysis/commentary
    # Strip any residual harmony / chat scaffolding tokens (harmless for other models).
    text = re.sub(r"<\|(?:start|end|return|channel|message|constrain|"
                  r"eot_id|eom_id|im_start|im_end|endoftext)\|>", "", text)
    return text.strip()


def source_id_of(alias: str) -> str | None:
    """Source model id from an adapter alias
    ``{stage}_{dataset}_{srcslug}_as_{tgtslug}_seed{N}`` (legacy-inclusive slug map)."""
    if "_as_" not in alias:
        return None
    head = alias.split("_as_", 1)[0]                   # {stage}_{dataset}_{srcslug}
    for stage in ("safety_sft_", "safety_dpo_", "self_sft_", "sft_", "dpo_"):  # longest prefix first
        if head.startswith(stage):
            body = head[len(stage):]                    # {dataset}_{srcslug}
            break
    else:
        return None
    for ds in config.dataset_names():
        if body.startswith(ds + "_"):
            return config.slug_to_id().get(body[len(ds) + 1:])
    return None


def renderer_for(model: str) -> str:
    # Local (non-Tinker) models aren't in the tinker cookbook; use the config renderer.
    if config.backend_for(model) == "local":
        return config.renderer_for(model)
    try:
        from tinker_cookbook.model_info import get_recommended_renderer_name

        return get_recommended_renderer_name(model)
    except Exception:  # cookbook absent, or doesn't recognize the model
        try:
            return config.renderer_for(model)
        except KeyError:
            return "unknown"
