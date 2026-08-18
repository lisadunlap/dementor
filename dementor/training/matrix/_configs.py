"""Backend-aware SFT/DPO workflow-config builders for a matrix ``Cell``.

Both builders read the shared hyperparameters from ``config.sft()`` / ``config.dpo()``
and pick the local vs Tinker provider from ``config.backend_for(cell.source)``.
"""
from __future__ import annotations

from pathlib import Path

from dementor import config

from ._cells import Cell


def _build_sft_cfg(*, cell: Cell, ds_cfg, output_dir: Path, weights_name: str,
                   prompt_template: str, completion_template: str,
                   registry_path: Path | None = None):
    """SFTWorkflowConfig for a cell — backend from config, hyperparameters from config.sft()."""
    from dementor.training.pipeline import LocalSFTParams, SFTWorkflowConfig, TinkerSFTParams
    from dementor.training.tinker_backend import EvaluationConfig

    hp = config.sft()
    common = dict(
        base_model=cell.source, batch_size=hp["batch_size"], epochs=hp["epochs"],
        learning_rate=hp["learning_rate"], prompt_template=prompt_template,
        completion_template=completion_template, weights_name=weights_name,
        registry_path=registry_path or config.registry_path(), seed=cell.seed,
    )
    if config.backend_for(cell.source) == "local":
        return SFTWorkflowConfig(provider="local", dataset=ds_cfg, output_dir=output_dir,
                                 local=LocalSFTParams(**common))
    return SFTWorkflowConfig(provider="tinker", dataset=ds_cfg, output_dir=output_dir,
                             tinker=TinkerSFTParams(**common,
                                 evaluation_config=EvaluationConfig(max_sample_tokens=256)))


def _build_dpo_cfg(*, cell: Cell, ds_cfg, output_dir: Path, sft_state_path,
                   renderer_name: str, log_path: Path, weights_name: str | None = None):
    """DPOWorkflowConfig for a cell — backend from config, hyperparameters from config.dpo()."""
    from dementor.training.pipeline import DPOWorkflowConfig, LocalDPOParams, TinkerDPOParams

    hp = config.dpo()
    weights_name = weights_name or f"dpo_{cell.slug}"
    if config.backend_for(cell.source) == "local":
        # Local DPO upcasts logits to fp32 over the full vocab; gemma's ~256K vocab makes that
        # tensor ~21 GB at length 4096, OOMing a single 80 GB card for the large text towers.
        # Cap local DPO length (chatbot_arena replies are ~400-600 tokens, so this rarely
        # truncates). Tinker DPO keeps the full config length (its backend shards differently).
        local_max_length = min(hp["max_length"], 1536)
        # gemma-4-31B (~62 GB bf16 text tower) leaves so little headroom that even at 1536,
        # trl's entropy-from-logits *metric* allocates another full-vocab tensor and OOMs a
        # single 80 GB card mid-run. Tighten it further (its chatbot_arena replies still fit).
        if "31B" in cell.source or "31b" in cell.source:
            local_max_length = min(local_max_length, 1024)
        return DPOWorkflowConfig(provider="local", dataset=ds_cfg, output_dir=output_dir,
            local=LocalDPOParams(model_name=cell.source, load_checkpoint_path=sft_state_path,
                weights_name=weights_name, registry_path=config.registry_path(),
                learning_rate=hp["learning_rate"], dpo_beta=hp["dpo_beta"], num_epochs=hp["num_epochs"],
                batch_size=hp["batch_size"], max_length=local_max_length, lora_rank=hp["lora_rank"],
                seed=cell.seed))
    return DPOWorkflowConfig(provider="tinker", dataset=ds_cfg, output_dir=output_dir,
        tinker=TinkerDPOParams(model_name=cell.source, renderer_name=renderer_name, log_path=log_path,
            learning_rate=hp["learning_rate"], dpo_beta=hp["dpo_beta"], num_epochs=hp["num_epochs"],
            batch_size=hp["batch_size"], max_length=hp["max_length"], lora_rank=hp["lora_rank"],
            save_every=hp["save_every"], load_checkpoint_path=sft_state_path))
