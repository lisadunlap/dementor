import json

import pytest

from experiments.figures import rebuild_steering_figures


def _write_metrics(path, baseline_harm, verdict="CLEAN"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "benchmark": "advbench",
        "baseline_harm": baseline_harm,
        "baseline_refrate": 0.5,
        "verdict": verdict,
        "refusal_matched": baseline_harm + 0.4,
        "fingerprint_matched": baseline_harm + 0.01,
        "random_matched": baseline_harm - 0.01,
    }))


def test_load_cells_prefers_harmonized_metrics(monkeypatch, tmp_path):
    eval_dir = tmp_path / "test-model" / "eval_advbench"
    _write_metrics(eval_dir / "metrics.json", 0.30)
    _write_metrics(eval_dir / "metrics_n200.json", 0.20)
    harmonized = json.loads((eval_dir / "metrics_n200.json").read_text())
    harmonized["n_prompts"] = 200
    (eval_dir / "metrics_n200.json").write_text(json.dumps(harmonized))
    monkeypatch.setattr(rebuild_steering_figures, "RDO", str(tmp_path))

    cell = rebuild_steering_figures.load_cells()["test-model"]["advbench"]

    assert cell["base"] == 20.0
    assert cell["cone"] == pytest.approx(40.0)


def test_load_cells_rejects_partial_overlap_mislabeled_n200(monkeypatch, tmp_path):
    eval_dir = tmp_path / "test-model" / "eval_advbench"
    _write_metrics(eval_dir / "metrics.json", 0.30)
    _write_metrics(eval_dir / "metrics_n200.json", 0.20)
    harmonized = json.loads((eval_dir / "metrics_n200.json").read_text())
    harmonized["n_prompts"] = 135
    (eval_dir / "metrics_n200.json").write_text(json.dumps(harmonized))
    monkeypatch.setattr(rebuild_steering_figures, "RDO", str(tmp_path))

    cell = rebuild_steering_figures.load_cells()["test-model"]["advbench"]

    assert cell["base"] == 30.0


def test_load_cells_falls_back_to_native_metrics(monkeypatch, tmp_path):
    eval_dir = tmp_path / "test-model" / "eval_advbench_fpall"
    _write_metrics(eval_dir / "metrics.json", 0.25)
    monkeypatch.setattr(rebuild_steering_figures, "RDO", str(tmp_path))

    cell = rebuild_steering_figures.load_cells(variant="fpall")["test-model"]["advbench"]

    assert cell["base"] == 25.0


def test_load_cells_ignores_noncanonical_depth_sweeps(monkeypatch, tmp_path):
    _write_metrics(tmp_path / "test-model" / "eval_advbench" / "metrics.json", 0.25)
    _write_metrics(tmp_path / "test-model" / "eval_advbench_fpd8" / "metrics.json", 0.90)
    monkeypatch.setattr(rebuild_steering_figures, "RDO", str(tmp_path))

    cell = rebuild_steering_figures.load_cells()["test-model"]["advbench"]

    assert cell["base"] == 25.0


def test_roster_gate_does_not_select_on_fingerprint_outcome(monkeypatch, tmp_path):
    eval_dir = tmp_path / "test-model" / "eval_advbench"
    _write_metrics(eval_dir / "metrics.json", 0.25, verdict="INCONCLUSIVE")
    monkeypatch.setattr(rebuild_steering_figures, "RDO", str(tmp_path))

    cells = rebuild_steering_figures.load_cells()

    assert rebuild_steering_figures.roster(cells) == ["test-model"]
    assert rebuild_steering_figures.per_model(cells, ["test-model"])["test-model"]["n"] == 1
