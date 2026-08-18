import pytest

from dementor import config
from dementor.training import plan
from experiments.imitation_safety import prompt_erosion_common
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
    assert config.campaign_stages() == ["sft", "dpo", "self_sft"]
    assert config.campaign_evaluation() == {
        "max_prompts": 200,
        "subsample_seed": 42,
        "max_new_tokens": 256,
    }


def test_steering_artifact_aliases_join_to_catalog_slugs():
    aliases = config.steering_artifact_slug_aliases()
    assert aliases == {
        "nemotron-nano": "nemotron-nano-30b-a3b",
        "qwen3.6-35b": "qwen3.6-35b-a3b",
    }
    assert all(config.model(slug) for slug in aliases.values())


def test_default_plan_is_528_cells_per_training_stage():
    summary = plan.summarize(plan.enumerate_jobs())
    assert summary["by_stage"] == {"sft": 528, "dpo": 528, "self_sft": 48}
    assert summary["total"] == 1104


def test_matrix_dataset_defaults_match_named_campaign():
    from dementor.training.matrix import _constants

    assert list(_constants.TRAIN_DATASETS) == config.campaign_dataset_names()
    assert list(_constants.DATASET_TEMPLATES) == config.campaign_dataset_names()


def test_model_parallel_sources_and_dpo_caps_are_config_driven(tmp_path):
    from dementor.training.matrix import Cell
    from dementor.training.matrix._configs import _build_dpo_cfg, _build_sft_cfg

    tagged = {
        model["slug"]
        for model in config.campaign_roster()
        if model.get("local_training") == "model_parallel"
    }
    assert tagged == {"gemma-4-31b", "llama-3.3-70b"}

    target = "meta-llama/Llama-3.1-8B-Instruct"
    for source_slug, expected_cap, expected_batch in (
        ("gemma-4-31b", 1024, 1),
        ("llama-3.3-70b", 1024, 1),
        ("phi-4", 1536, 2),
    ):
        source = config.model(source_slug)["id"]
        cfg = _build_dpo_cfg(
            cell=Cell(source, target, "gsm8k", 42),
            ds_cfg=None,
            output_dir=tmp_path,
            sft_state_path=tmp_path / "sft",
            renderer_name="test",
            log_path=tmp_path / "log",
        )
        assert cfg.provider == "local"
        assert cfg.local.max_length == expected_cap
        assert cfg.local.batch_size == expected_batch
        assert cfg.local.gradient_accumulation_steps == 8
        sft_cfg = _build_sft_cfg(
            cell=Cell(source, target, "gsm8k", 42),
            ds_cfg=None,
            output_dir=tmp_path,
            weights_name="test",
            prompt_template="{prompt}",
            completion_template="{completion}",
        )
        assert sft_cfg.local.batch_size == (1 if expected_cap == 1024 else 16)


def test_matrix_dispatcher_enforces_configured_stages(monkeypatch):
    from dementor.training.matrix import _cli

    monkeypatch.setattr(_cli.config, "campaign_stages", lambda: ["sft"])
    _cli._require_campaign_stages("sft")
    with pytest.raises(SystemExit, match="dpo"):
        _cli._require_campaign_stages("dpo")


def test_compound_self_sft_alias_stage_is_preserved():
    from dementor.training.matrix import _push

    assert _push._alias_stage("self_sft_gsm8k_model_as_model_seed42") == "self_sft"
    assert _push._alias_stage("sft_gsm8k_model_as_target_seed42") == "sft"


def test_prompt_worklist_is_exact_core12_without_dataset_pseudoreplication():
    methods = ["just_name_it", "random_sampling", "stylistic"]
    worklist = prompt_erosion_common.build_worklist(methods=methods)
    pairs = {(item["source"], item["target"]) for item in worklist}

    assert len(pairs) == 12 * 11
    assert len(worklist) == 3 * 12 * 11
    assert {source for source, _ in pairs} == CORE12
    assert {target for _, target in pairs} == CORE12
    assert all(source != target for source, target in pairs)
    assert len(prompt_erosion_common.build_worklist(methods=methods, backend="local")) == 297
    assert len(prompt_erosion_common.build_worklist(methods=methods, backend="tinker")) == 99


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


def test_tinker_judge_resolver_recovers_legacy_sft_item(monkeypatch):
    item = {
        "id": "sft_gsm8k_qwen3.6-27b_as_gpt-oss-20b_seed42",
        "kind": "adapter",
        "base_model": "Qwen/Qwen3.6-27B",
        "sampler_path": "tinker://run/sampler_weights/sft-parent",
    }
    monkeypatch.setattr(tinker_erosion.EC, "find_item", lambda item_id: None)
    monkeypatch.setattr(tinker_erosion, "tinker_worklist", lambda seed=None: ([item], []))

    assert tinker_erosion._find_item(item["id"]) == item
