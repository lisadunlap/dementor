"""Regression tests for reconstructing local DPO adapters at inference time."""

from copy import deepcopy
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

    def load_adapter(model, path, adapter_name=None):
        events.append(("adapter", model.name, path, adapter_name))
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
        ("adapter", "base", "sft-parent", None),
        ("merge", "sft-wrapper"),
        ("adapter", "merged-sft", "dpo-adapter", None),
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

    def load_adapter(model, path, adapter_name=None):
        events.append((model.name, path))
        return Model("adapter")

    monkeypatch.delenv("DEMENTOR_MP", raising=False)
    with patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=tok), \
         patch("dementor.training.local_backend._load_causal_lm", return_value=Model("base")), \
         patch("peft.PeftModel.from_pretrained", side_effect=load_adapter):
        erosion.load_gen_model("base-model", "sft-adapter")

    assert events == [("base", "sft-adapter")]


def test_exact_cat_matches_merge_then_dpo(tmp_path):
    """PEFT's cat composition is numerically the same effective model as the training path."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(5, 4, bias=False)

        def forward(self, x):
            return self.proj(x)

    torch.manual_seed(42)
    base = Toy().eval()
    cfg = LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], lora_dropout=0.0)

    def set_lora(model, seed):
        generator = torch.Generator().manual_seed(seed)
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.data.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)

    sft_dir = tmp_path / "sft"
    dpo_dir = tmp_path / "dpo"
    sft = get_peft_model(deepcopy(base), cfg, adapter_name="default")
    set_lora(sft, 1)
    sft.save_pretrained(sft_dir)

    # Mirror training: merge SFT into W, then fit/save a fresh DPO LoRA on that merged base.
    sft_merged = PeftModel.from_pretrained(deepcopy(base), sft_dir).merge_and_unload()
    dpo = get_peft_model(sft_merged, cfg, adapter_name="default")
    set_lora(dpo, 2)
    dpo.save_pretrained(dpo_dir)

    reference_base = PeftModel.from_pretrained(deepcopy(base), sft_dir).merge_and_unload()
    reference = PeftModel.from_pretrained(reference_base, dpo_dir).eval()

    composed = PeftModel.from_pretrained(deepcopy(base), sft_dir, adapter_name="sft_parent")
    composed.load_adapter(dpo_dir, adapter_name="dpo")
    composed.base_model.add_weighted_adapter(
        ["sft_parent", "dpo"], [1.0, 1.0], "sft_plus_dpo", combination_type="cat"
    )
    composed.set_adapter("sft_plus_dpo")
    composed.eval()

    inputs = torch.randn(7, 5, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        expected = reference(inputs)
        actual = composed(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
