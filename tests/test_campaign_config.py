from dementor import config
from dementor.training import plan
from experiments.imitation_safety import tinker_erosion


CORE12 = {
    "aya-expanse-8b",
    "gemma-4-31b",
    "gemma-4-e4b",
    "granite-4-h-small",
    "llama-3.1-8b",
    "llama-3.3-70b",
    "ministral-8b",
    "olmo-3-7b",
    "phi-4",
    "qwen3.6-27b",
    "gpt-oss-20b",
    "nemotron-nano-30b-a3b",
}


def test_imitation_safety_campaign_is_exact_core12():
    assert {model["slug"] for model in config.campaign_roster()} == CORE12
    assert config.campaign_dataset_names() == [
        "chatbot_arena", "gsm8k", "oasst1", "writingprompts"
    ]
    assert config.campaign_seeds() == [42]
    assert config.campaign_evaluation() == {"max_prompts": 200, "subsample_seed": 42}


def test_default_plan_is_528_cells_per_training_stage():
    summary = plan.summarize(plan.enumerate_jobs())
    assert summary["by_stage"] == {"sft": 528, "dpo": 528, "self_sft": 48}
    assert summary["total"] == 1104


def test_tinker_sft_parent_recovers_legacy_missing_base_model(monkeypatch):
    key = "sft_gsm8k_qwen3.6-27b_as_gpt-oss-20b_seed42"
    monkeypatch.setattr(
        tinker_erosion.EC,
        "load_registry_entries",
        lambda: {key: {"path": "tinker://run/sampler_weights/sft-parent"}},
    )
    monkeypatch.setattr(
        tinker_erosion.EC,
        "_slug_maps",
        lambda: ({"Qwen/Qwen3.6-27B": "qwen3.6-27b"}, {"qwen3.6-27b": "Qwen/Qwen3.6-27B"}),
    )
    monkeypatch.setattr(tinker_erosion.EC, "campaign_cell_allowed", lambda *args: True)
    adapters, baselines = tinker_erosion.tinker_worklist("seed42")
    assert adapters[0]["base_model"] == "Qwen/Qwen3.6-27B"
    assert adapters[0]["sampler_path"].startswith("tinker://")
    assert baselines[0]["id"] == "baseline_qwen3.6-27b"
