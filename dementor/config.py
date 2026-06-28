"""Central configuration for the dementor project.

Single source of truth for the model roster, datasets, seeds, and training
hyperparameters. Reads ``config.yaml`` at the repository root. Import-light
(stdlib + PyYAML only) so metric/analysis code can depend on it without pulling
in training dependencies.
"""
from __future__ import annotations

import functools
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

def roster(include_legacy: bool = False) -> list[dict[str, Any]]:
    cfg = load_config()
    models = list(cfg.get("roster", []))
    if include_legacy:
        models += list(cfg.get("roster_legacy", []))
    return models


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
