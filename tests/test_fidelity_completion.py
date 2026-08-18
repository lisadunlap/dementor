import json
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "imitation_safety"
sys.path.insert(0, str(EXPERIMENT_DIR))
import fidelity_common as fidelity  # noqa: E402
import analyze_fidelity_campaign as fidelity_analysis  # noqa: E402


def test_score_parser_never_treats_unrelated_digits_as_score():
    assert fidelity._parse_score("analysis\nSCORE: 87") == 0.87
    assert fidelity._parse_score("Response 1 is closer than Response 2") != fidelity._parse_score(
        "Response 1 is closer than Response 2")


def test_judge_completion_requires_exact_denominator(tmp_path, monkeypatch):
    path = tmp_path / "fidelity_judge.json"
    monkeypatch.setattr(fidelity, "adapter_fidelity_path", lambda _item, _scorer: str(path))
    path.write_text(json.dumps({"scorer": "judge", "n": 199, "n_prompts": 200}))
    assert not fidelity.scored("cell", "judge")
    path.write_text(json.dumps({"scorer": "judge", "n": 200, "n_prompts": 200}))
    assert fidelity.scored("cell", "judge")


def test_incomplete_result_is_quarantined_and_recomputed(tmp_path, monkeypatch):
    out = tmp_path / "fidelity_judge.json"
    gens = tmp_path / "gens.csv"
    ref = tmp_path / "ref.csv"
    rows = "prompt,model_response\n" + "\n".join(f'p{i},"response {i}"' for i in range(200))
    gens.write_text(rows + "\n")
    ref.write_text(rows + "\n")
    out.write_text(json.dumps({"scorer": "judge", "n": 90, "n_prompts": 200}))
    monkeypatch.setattr(fidelity, "adapter_fidelity_path", lambda _item, _scorer: str(out))
    monkeypatch.setattr(fidelity, "adapter_gens_path", lambda _item: str(gens))
    monkeypatch.setattr(fidelity, "ref_csv_path", lambda _dataset, _target: str(ref))
    monkeypatch.setattr(fidelity, "score_judge", lambda *args, **kwargs: [0.5] * 200)
    item = {"id": "cell", "dataset": "dataset", "source": "a", "target": "b",
            "seed": "seed42", "base_model": "a/id", "target_hf": "b/id", "backend": "local"}
    result = fidelity.compute_fidelity(item, "judge")
    assert result["n"] == result["n_prompts"] == 200
    assert list(tmp_path.glob("fidelity_judge.json.invalid.*"))


def test_campaign_correlations_accept_legacy_scipy_tuples(monkeypatch):
    monkeypatch.setattr(fidelity_analysis, "pearsonr", lambda _x, _y: (0.25, 0.5))
    monkeypatch.setattr(fidelity_analysis, "spearmanr", lambda _x, _y: (0.20, 0.6))

    result = fidelity_analysis.correlation_record([1, 2, 3], [1, 2, 3])

    assert result == {
        "n": 3,
        "pearson_r": 0.25,
        "pearson_p": 0.5,
        "spearman_rho": 0.20,
        "spearman_p": 0.6,
    }
