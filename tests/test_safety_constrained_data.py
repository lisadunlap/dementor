import json
from pathlib import Path

import pandas as pd

from dementor.training import matrix


def _cell() -> matrix.Cell:
    return matrix.Cell(
        source=matrix.MODELS[0],
        target=matrix.MODELS[1],
        dataset=next(iter(matrix.TRAIN_DATASETS)),
        seed=matrix.SEEDS[0],
    )


def _write_baselines(cell: matrix.Cell) -> None:
    source_path = matrix.baseline_path(cell.source, cell.dataset)
    target_path = matrix.baseline_path(cell.target, cell.dataset)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "prompt": ["imitate prompt 1", "imitate prompt 2"],
            "model_response": ["source answer 1", "source answer 2"],
        }
    ).to_csv(source_path, index=False)
    pd.DataFrame(
        {
            "prompt": ["imitate prompt 1", "imitate prompt 2"],
            "model_response": ["target answer 1", "target answer 2"],
        }
    ).to_csv(target_path, index=False)


def _write_safety_inputs(tmp_path: Path, cell: matrix.Cell) -> tuple[Path, Path]:
    prompts = tmp_path / "safety_prompts.csv"
    exclude = tmp_path / "eval_prompts.csv"
    pd.DataFrame(
        {
            "prompt": ["harmful replay 1", "harmful replay 2", "held-out eval prompt"],
            "category": ["harmful", "harmful", "harmful"],
            "expected": ["refuse", "refuse", "refuse"],
        }
    ).to_csv(prompts, index=False)
    pd.DataFrame({"prompt": ["held-out eval prompt"]}).to_csv(exclude, index=False)

    native = matrix.SAFETY_NATIVE_REFUSAL_DIR / f"{matrix.MODEL_SLUG[cell.source]}_seed1.csv"
    native.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "prompt": ["harmful replay 1", "harmful replay 2"],
            "refused": [1, 0],
            "snippet_redacted": ["native refusal snippet", "non-refusal snippet"],
        }
    ).to_csv(native, index=False)
    return prompts, exclude


def _patch_dirs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(matrix, "BASELINES_DIR", tmp_path / "baselines")
    monkeypatch.setattr(matrix, "SAFETY_SFT_DATA_DIR", tmp_path / "safety_sft_data")
    monkeypatch.setattr(matrix, "SAFETY_DPO_DATA_DIR", tmp_path / "safety_dpo_data")
    monkeypatch.setattr(matrix, "SAFETY_NATIVE_REFUSAL_DIR", tmp_path / "native_refusal")


def test_build_safety_sft_data_mixes_imitation_and_eval_disjoint_replay(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    cell = _cell()
    _write_baselines(cell)
    prompts, exclude = _write_safety_inputs(tmp_path, cell)

    built = matrix.build_safety_sft_data(
        cells=[cell],
        safety_prompts_file=prompts,
        exclude_prompts_file=exclude,
        replay_size=2,
        prompt_seed=0,
        response_seed=1,
    )

    assert built == 1
    out = matrix.safety_sft_data_path(cell)
    df = pd.read_csv(out)
    assert len(df) == 4
    assert df["row_type"].value_counts().to_dict() == {"imitation": 2, "safety_replay": 2}
    assert "held-out eval prompt" not in set(df["prompt"])
    safety = df[df["row_type"] == "safety_replay"]
    assert "native refusal snippet" in set(safety["model_response"])
    assert matrix.GENERIC_REFUSAL_RESPONSE in set(safety["model_response"])

    manifest = json.loads(matrix._manifest_path(out).read_text())
    assert manifest["imitation_rows"] == 2
    assert manifest["safety_replay_rows"] == 2
    assert manifest["native_refusal_rows_used"] == 1
    assert manifest["generic_refusal_rows_used"] == 1
    assert manifest["exclude_prompts_file"] == str(exclude)


def test_build_safety_dpo_data_appends_refusal_preference_pairs(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    cell = _cell()
    _write_baselines(cell)
    prompts, exclude = _write_safety_inputs(tmp_path, cell)

    built = matrix.build_safety_dpo_data(
        cells=[cell],
        safety_prompts_file=prompts,
        exclude_prompts_file=exclude,
        replay_size=2,
        prompt_seed=0,
        response_seed=1,
    )

    assert built == 1
    out = matrix.safety_dpo_data_path(cell)
    df = pd.read_csv(out)
    assert len(df) == 4
    assert df["row_type"].value_counts().to_dict() == {"imitation": 2, "safety_replay": 2}
    safety = df[df["row_type"] == "safety_replay"]
    assert set(safety["rejected_response"]) == {matrix.GENERIC_NONREFUSAL_RESPONSE}
    assert "native refusal snippet" in set(safety["chosen_response"])
    assert matrix.GENERIC_REFUSAL_RESPONSE in set(safety["chosen_response"])

    manifest = json.loads(matrix._manifest_path(out).read_text())
    assert manifest["imitation_pairs"] == 2
    assert manifest["safety_replay_pairs"] == 2
    assert manifest["rejected_response_source"] == "generic_nonrefusal_stub"


def test_launch_safety_dpo_dry_run_uses_safety_aliases(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    cell = _cell()
    pref = matrix.safety_dpo_data_path(cell)
    pref.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "prompt": ["p"],
            "chosen_response": ["safe"],
            "rejected_response": ["unsafe"],
        }
    ).to_csv(pref, index=False)

    manifest = matrix.launch_safety_dpo(cells=[cell], dry_run=True)

    assert manifest["n_jobs"] == 1
    job = manifest["jobs"][0]
    assert job["dpo_alias"] == f"safety_dpo_{cell.slug}"
    assert job["sft_alias"] == f"safety_sft_{cell.slug}"
    assert job["pref_rows"] == 1
