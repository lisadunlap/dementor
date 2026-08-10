from __future__ import annotations

import importlib
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1] / "experiments" / "imitation_safety"


def daemon_module():
    sys.path.insert(0, str(HERE))
    try:
        return importlib.import_module("erosion_daemon")
    finally:
        sys.path.remove(str(HERE))


def test_70b_requires_three_80gb_cards():
    daemon = daemon_module()
    assert daemon.required_gpus({
        "base_model": "meta-llama/Llama-3.3-70B-Instruct",
        "needs_mp": True,
    }) == 3


def test_other_mp_and_single_card_counts():
    daemon = daemon_module()
    assert daemon.required_gpus({"base_model": "example/mp", "needs_mp": True}) == 2
    assert daemon.required_gpus({"base_model": "example/small", "needs_mp": False}) == 1


def test_retry_errors_is_explicit_and_metrics_always_win(tmp_path, monkeypatch):
    daemon = daemon_module()
    monkeypatch.setattr(daemon.EC, "WORK", str(tmp_path))
    item = tmp_path / "cell"
    item.mkdir()
    (item / "ERROR.json").write_text("{}")
    assert daemon.done("cell")
    assert not daemon.done("cell", retry_errors=True)
    (item / "metrics.json").write_text("{}")
    assert daemon.done("cell", retry_errors=True)


def test_generation_only_uses_all_benchmark_checkpoints(tmp_path, monkeypatch):
    daemon = daemon_module()
    monkeypatch.setattr(daemon.EC, "WORK", str(tmp_path))
    item = tmp_path / "cell"
    for benchmark in ("a", "b"):
        (item / benchmark).mkdir(parents=True)
        (item / benchmark / "all_gens.csv").write_text("prompt,model_response\n")
    assert daemon.done("cell", generation_only=True, benchmarks=("a", "b"))
    assert not daemon.done("cell", generation_only=True, benchmarks=("a", "b", "c"))
