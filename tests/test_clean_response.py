"""Regression tests for the gpt-oss chat-template de-confound (D1).

The bug: clean_response only stripped gpt-oss harmony markup when a
`<|channel|>final<|message|>` marker was present; DPO makes gpt-oss drop that
marker and dump the answer under the `analysis` header, so raw channel markup +
reasoning leaked into the stored response and inflated measured persistence.
"""
from dementor.training.matrix import clean_response

GPT = "openai/gpt-oss-20b"


def test_final_channel_kept_cot_dropped():
    raw = ("<|channel|>analysis<|message|>let me think 2+2<|end|>"
           "<|start|>assistant<|channel|>final<|message|>The answer is 4.<|return|>")
    assert clean_response(GPT, raw) == "The answer is 4."


def test_analysis_only_post_dpo_strips_header_keeps_body():
    # The post-DPO failure mode: no `final` marker, answer under `analysis`.
    raw = "<|channel|>analysis<|message|>Here is the story you asked for."
    assert clean_response(GPT, raw) == "Here is the story you asked for."


def test_no_residual_harmony_tokens():
    raw = "<|start|>assistant<|channel|>analysis<|message|>Answer body<|end|>"
    out = clean_response(GPT, raw)
    assert out == "Answer body"
    assert "<|" not in out


def test_commentary_channel_also_handled():
    raw = "<|channel|>commentary<|message|>tool note<|end|>Real answer."
    assert "<|" not in clean_response(GPT, raw)


def test_other_models_only_strip_chat_tokens():
    assert clean_response("meta-llama/Llama-3.1-8B-Instruct", "Hello.<|eot_id|>") == "Hello."
    assert clean_response("Qwen/Qwen3.6-27B", "Hi<|im_end|>") == "Hi"


def test_aya_end_of_turn_and_batch_padding_are_removed():
    raw = "The actual answer.<|END_OF_TURN_TOKEN|><PAD><PAD><PAD>"
    assert clean_response("adamo1139/aya-expanse-8b-ungated", raw) == "The actual answer."


def test_aya_padding_without_visible_terminator_is_removed():
    assert clean_response("adamo1139/aya-expanse-8b-ungated", "Answer<PAD><PAD>") == "Answer"


def test_plain_text_unchanged():
    assert clean_response(GPT, "Just a normal answer.") == "Just a normal answer."
