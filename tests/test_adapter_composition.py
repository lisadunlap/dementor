"""Regression tests for reconstructing local DPO adapters at inference time."""

from types import SimpleNamespace
from unittest.mock import patch


def test_dpo_generation_loads_sft_parent_then_dpo(monkeypatch):
    from experiments.imitation_safety import erosion_common as erosion

    events = []

    class Model:
        def __init__(self, name):
            self.name = name

        def merge_and_unload(self):
            events.append(("merge", self.name))
            return Model("merged-sft")

        def to(self, device):
            events.append(("to", self.name, device))
            return self

        def eval(self):
            events.append(("eval", self.name))
            return self

        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

    class Tokenizer:
        pad_token_id = None
        eos_token = "<eos>"
        padding_side = "right"

    def load_adapter(model, path):
        events.append(("adapter", model.name, path))
        return Model("sft-wrapper" if path == "sft-parent" else "dpo-wrapper")

    monkeypatch.delenv("DEMENTOR_MP", raising=False)
    with patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()), \
         patch("dementor.training.local_backend._load_causal_lm", return_value=Model("base")), \
         patch("peft.PeftModel.from_pretrained", side_effect=load_adapter):
        _, model, input_device = erosion.load_gen_model(
            "base-model", "dpo-adapter", sft_parent="sft-parent"
        )

    assert model.name == "dpo-wrapper"
    assert input_device == "cpu"
    assert events[:3] == [
        ("adapter", "base", "sft-parent"),
        ("merge", "sft-wrapper"),
        ("adapter", "merged-sft", "dpo-adapter"),
    ]


def test_single_adapter_path_does_not_merge(monkeypatch):
    from experiments.imitation_safety import erosion_common as erosion

    events = []

    class Model:
        def __init__(self, name):
            self.name = name

        def to(self, _device):
            return self

        def eval(self):
            return self

        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

    tok = SimpleNamespace(pad_token_id=1, eos_token="<eos>", padding_side="right")

    def load_adapter(model, path):
        events.append((model.name, path))
        return Model("adapter")

    monkeypatch.delenv("DEMENTOR_MP", raising=False)
    with patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=tok), \
         patch("dementor.training.local_backend._load_causal_lm", return_value=Model("base")), \
         patch("peft.PeftModel.from_pretrained", side_effect=load_adapter):
        erosion.load_gen_model("base-model", "sft-adapter")

    assert events == [("base", "sft-adapter")]
