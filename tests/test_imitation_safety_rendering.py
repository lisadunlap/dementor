import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMITATION_SAFETY = ROOT / "experiments" / "imitation_safety"
sys.path.insert(0, str(IMITATION_SAFETY))

import erosion_common as EC  # noqa: E402
import prompt_erosion_common as PC  # noqa: E402


class NoTemplateTokenizer:
    chat_template = None

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("tokenizer.chat_template is not set")


class KwargSensitiveTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        if kwargs.get("bad_kwarg"):
            raise TypeError("unsupported kwarg")
        return f"rendered:{messages[-1]['content']}"


class NoSystemTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        if messages[0]["role"] == "system":
            raise ValueError("system role unsupported")
        return f"user-only:{messages[0]['content']}"


def test_plain_render_falls_back_to_raw_prompt_without_chat_template():
    assert EC._render(NoTemplateTokenizer(), "model", "hello", {}) == "hello"


def test_plain_render_retries_without_chat_kwargs():
    rendered = EC._render(KwargSensitiveTokenizer(), "model", "hello", {"bad_kwarg": True})
    assert rendered == "rendered:hello"


def test_disguise_render_merges_system_prompt_when_system_role_unsupported():
    rendered = PC.render_disguise(
        NoSystemTokenizer(),
        "model",
        "act like target",
        "answer this",
        {},
    )
    assert rendered == "user-only:act like target\n\nanswer this"


def test_disguise_render_falls_back_to_merged_raw_prompt_without_chat_template():
    rendered = PC.render_disguise(
        NoTemplateTokenizer(),
        "model",
        "act like target",
        "answer this",
        {},
    )
    assert rendered == "act like target\n\nanswer this"
