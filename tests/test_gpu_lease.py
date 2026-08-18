from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "experiments" / "imitation_safety" / "gpu_lease.py"


def _load_gpu_lease():
    spec = importlib.util.spec_from_file_location("gpu_lease_path_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_lease_root_uses_canonical_repo_data(monkeypatch):
    for name in (
        "DEMENTOR_GPU_LEASE_ROOT",
        "DEMENTOR_IMITATION_ROOT",
        "DEMENTOR_DATA",
        "DEMENTOR_REPO",
    ):
        monkeypatch.delenv(name, raising=False)

    module = _load_gpu_lease()

    assert Path(module._default_lease_root()) == (
        REPO / "data" / "imitation_safety" / "gpu_leases"
    )
