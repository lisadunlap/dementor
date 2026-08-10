"""Central configuration for the dementor project.

Single source of truth for the model roster, datasets, seeds, and training
hyperparameters. Reads ``config.yaml`` at the repository root. Import-light
(stdlib + PyYAML only) so metric/analysis code can depend on it without pulling
in training dependencies.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Repo root, anchored on this file — never on the current working directory."""
    return Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with (project_root() / "config.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --- paths ------------------------------------------------------------------

def resolve_path(value: str | Path) -> Path:
    """Absolute paths pass through; relative paths anchor on the repo root."""
    p = Path(value)
    return p if p.is_absolute() else project_root() / p


def path(key: str) -> Path:
    return resolve_path(load_config()["paths"][key])


def registry_path() -> Path:
    return path("registry")


# --- roster -----------------------------------------------------------------

def _local_backend_overrides() -> set[str]:
    """Slugs/ids this BOX should train locally regardless of config.yaml's backend.

    Env: DEMENTOR_LOCAL_BACKENDS="qwen3.5-4b,qwen3.6-27b,qwen3.6-35b-a3b"

    config.yaml is shared by both boxes on one branch, so its per-model ``backend`` cannot express
    "this model is Tinker-backed on the box with no weights for it, but local on the box that has
    downloaded them". Editing the shared value to use a box's idle cards would silently retarget the
    other box's workers too. Everything else in this codebase is env-overridable per box
    (experiments/_paths.py, experiments/steering/steer_config.py); the roster was the gap.

    Overriding only flips the backend. It does not invent weights: the local trainer still resolves
    the model's ``id`` through the HF cache, so the box must actually hold or be able to fetch it.
    """
    raw = os.environ.get("DEMENTOR_LOCAL_BACKENDS", "")
    return {s.strip() for s in raw.split(",") if s.strip()}


def roster(include_legacy: bool = False) -> list[dict[str, Any]]:
    cfg = load_config()
    models = list(cfg.get("roster", []))
    if include_legacy:
        models += list(cfg.get("roster_legacy", []))
    forced_local = _local_backend_overrides()
    if forced_local:
        out = []
        for m in models:
            if m.get("slug") in forced_local or m.get("id") in forced_local:
                m = dict(m)
                m["backend"] = "local"
                out.append(m)
            else:
                out.append(m)
        return out
    return models


def campaign(name: str = "imitation_safety") -> dict[str, Any]:
    """Return a named experiment campaign from ``config.yaml``."""
    try:
        return dict(load_config()["campaigns"][name])
    except KeyError as exc:
        raise KeyError(f"unknown campaign: {name!r}") from exc


def campaign_roster(name: str = "imitation_safety") -> list[dict[str, Any]]:
    """Models selected by a named campaign, preserving catalog order."""
    spec = campaign(name)
    tier = spec.get("roster_tier")
    slugs = spec.get("models")
    if tier and slugs:
        raise ValueError(f"campaign {name!r} sets both roster_tier and models")
    if tier:
        models = [m for m in roster() if m.get("imitation") == tier]
    elif slugs:
        models = [model(slug) for slug in slugs]
    else:
        raise ValueError(f"campaign {name!r} must set roster_tier or models")
    if not models:
        raise ValueError(f"campaign {name!r} selects no models")
    return models


def campaign_dataset_names(name: str = "imitation_safety") -> list[str]:
    """Dataset names selected by a named campaign."""
    return list(campaign(name).get("datasets", dataset_names()))


def campaign_seeds(name: str = "imitation_safety") -> list[int]:
    """Training/evaluation seeds selected by a named campaign."""
    return list(campaign(name).get("seeds", seeds()))


def campaign_evaluation(name: str = "imitation_safety") -> dict[str, Any]:
    """Evaluation sampling settings for a named campaign."""
    return dict(campaign(name).get("evaluation", {}))


def model(slug_or_id: str) -> dict[str, Any]:
    for m in roster(include_legacy=True):
        if slug_or_id in (m.get("slug"), m.get("id")):
            return m
    raise KeyError(f"unknown model: {slug_or_id!r}")


def model_slug(model_id: str) -> str:
    return model(model_id)["slug"]


def slug_to_id() -> dict[str, str]:
    return {m["slug"]: m["id"] for m in roster(include_legacy=True)}


def backend_for(slug_or_id: str, default: str = "tinker") -> str:
    """Backend for a model (``tinker``|``local``); returns ``default`` if unmodeled."""
    try:
        return model(slug_or_id).get("backend", default)
    except KeyError:
        return default


def renderer_for(slug_or_id: str) -> str:
    return model(slug_or_id).get("renderer", "unknown")


def chat_template_kwargs_for(slug_or_id: str) -> dict[str, Any]:
    return dict(model(slug_or_id).get("chat_template_kwargs", {}))


# --- datasets / seeds / hyperparameters -------------------------------------

def dataset(name: str) -> dict[str, Any]:
    return load_config()["datasets"][name]


def dataset_names() -> list[str]:
    return list(load_config()["datasets"].keys())


def seeds() -> list[int]:
    return list(load_config()["seeds"])


def lora() -> dict[str, Any]:
    return dict(load_config().get("lora", {}))


def sft() -> dict[str, Any]:
    return dict(load_config().get("sft", {}))


def dpo() -> dict[str, Any]:
    return dict(load_config().get("dpo", {}))


def generation() -> dict[str, Any]:
    return dict(load_config().get("generation", {}))
