import json
import csv

import pytest

from experiments.figures import rebuild_steering_figures
from experiments.figures import compare_fpall


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


def _write_generations(path, n_prompts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["direction", "prompt"])
        writer.writeheader()
        for index in range(n_prompts):
            writer.writerow({"direction": "baseline", "prompt": f"p{index}"})
        for index in range(n_prompts):
            writer.writerow({"direction": "cone", "prompt": f"p{index}"})


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
    assert cell["n_prompts"] == 200
    assert cell["sampling"] == "harmonized_n200"
    assert cell["subsample_seed"] is None


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
    _write_generations(eval_dir / "all_gens.csv", 300)
    monkeypatch.setattr(rebuild_steering_figures, "RDO", str(tmp_path))

    cell = rebuild_steering_figures.load_cells(variant="fpall")["test-model"]["advbench"]

    assert cell["base"] == 25.0
    assert cell["n_prompts"] == 300
    assert cell["sampling"] == "native"
    assert cell["metrics_source"] == "metrics.json"


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


def test_exact_signed_rank_permutation_retains_ties():
    result = rebuild_steering_figures.paired_signed_rank_exact(
        [1.0, 2.0, 4.0],
        [0.0, 4.0, 2.0],
    )

    assert result["n_models"] == 3
    assert result["statistic"] == 2.5
    assert 0 <= result["p_value"] <= 1
    assert result["method"].startswith("exact paired sign permutation")


def test_fpall_summary_pairs_fingerprint_and_random_on_same_cells():
    pairs = {
        "model": {
            "advbench": (
                {"verdict": "CLEAN", "cone": 9.0, "fp": 1.0, "rand": 2.0},
                {"verdict": "CLEAN", "cone": 8.0, "fp": 3.0, "rand": 4.0},
            ),
            "harmbench": (
                {"verdict": "CLEAN", "cone": 9.0, "fp": None, "rand": 100.0},
                {"verdict": "CLEAN", "cone": 7.0, "fp": 5.0, "rand": 100.0},
            ),
        }
    }

    assert compare_fpall.per_model_joint(pairs) == {
        "model": {
            "cone": 8.0,
            "fp_old": 1.0,
            "fp_new": 3.0,
            "random_old": 2.0,
            "random_new": 4.0,
            "n": 1,
        }
    }


def test_fpall_summary_gates_on_variant_positive_control():
    pairs = {
        "model": {
            "advbench": (
                {"verdict": "CLEAN", "cone": 9.0, "fp": 1.0, "rand": 2.0},
                {"verdict": "PC_INVALID", "cone": 99.0, "fp": 99.0, "rand": 99.0},
            ),
            "harmbench": (
                {"verdict": "PC_FAILS", "cone": 0.0, "fp": 0.0, "rand": 0.0},
                {"verdict": "CLEAN", "cone": 8.0, "fp": 3.0, "rand": 4.0},
            ),
        }
    }

    assert compare_fpall.per_model_joint(pairs)["model"] == {
        "cone": 8.0,
        "fp_old": 0.0,
        "fp_new": 3.0,
        "random_old": 0.0,
        "random_new": 4.0,
        "n": 1,
    }


def test_primary_fpall_cohort_applies_fpall_random_gate_and_complete_coverage():
    clean = {
        benchmark: {"verdict": "CLEAN", "cone": 40.0, "fp": 1.0, "rand": 0.5}
        for benchmark in rebuild_steering_figures.HARM_BENCHMARKS
    }
    contaminated = {benchmark: dict(cell) for benchmark, cell in clean.items()}
    contaminated["harmbench"]["rand"] = 11.5
    incomplete = {benchmark: dict(cell) for benchmark, cell in list(clean.items())[:-1]}

    rows = compare_fpall.primary_fpall_rows({
        "good": clean,
        "contaminated": contaminated,
        "incomplete": incomplete,
    })

    assert list(rows) == ["good"]
    assert rows["good"] == {"cone": 40.0, "fp": 1.0, "rand": 0.5, "n": 5}


def test_primary_fpall_averages_all_arms_on_identical_cells():
    cells = {
        benchmark: {"verdict": "CLEAN", "cone": 40.0, "fp": 1.0, "rand": 0.5}
        for benchmark in rebuild_steering_figures.HARM_BENCHMARKS
    }
    cells["sorrybench"]["fp"] = None

    row = compare_fpall.primary_fpall_rows({"model": cells})["model"]

    assert row == {"cone": 40.0, "fp": 1.0, "rand": 0.5, "n": 4}
