"""End-to-end GOLDEN test for the imitation erosion pipeline (stages A->D).

Unlike the other golden tests (which pin pure, CPU-deterministic logic), this one
drives the *real* runner ``experiments/imitation_safety/run_erosion_item.py`` as a
subprocess on a GPU -- load model -> greedy generate -> RTL judge -> metric -- and
compares the result against committed fixtures in ``tests/golden_e2e/``.

Purpose: give the still-deferred refactors (de-forking the ``prompt_*`` daemon twins,
splitting the runner monolith) an END-TO-END equivalence check. A refactor that only
moves plumbing must reproduce the pipeline's *behavior*; this test is what proves it.

WHAT IS ASSERTED (and what is NOT) -- see tests/golden_e2e/olmo3_7b_advbench/README.md:
Running this slice twice on the same GPU shows greedy generation is NOT bit-reproducible
(3/8 responses diverge on late-token flips from FP non-associativity in batched matmul).
So we do NOT assert on raw ``model_response`` / ``rtl_raw``. What IS byte-stable run-to-run
-- and is what the science depends on -- are the seed-42 prompt subsample, the judge
verdicts (``rtl_code`` / ``rtl_label`` / ``genuine_harm``), and ``metrics.json``. Those
are the golden invariants. A behavior-preserving refactor reproduces them exactly;
GPU text noise cannot make the test flap.

GATING: skipped unless BOTH ``torch.cuda.is_available()`` AND ``DEMENTOR_RUN_GPU_TESTS=1``
(matches tests/test_local_backend_gpu.py), so a plain ``pytest`` never launches a GPU job.

Run it explicitly:
    DEMENTOR_RUN_GPU_TESTS=1 pytest tests/test_e2e_erosion_golden.py -v
Optional overrides: DEMENTOR_E2E_GPU (default "5"), DEMENTOR_E2E_HF_HOME
(default "/data/ethantsliu/hf-cache" -- the cache that holds BOTH OLMo-3-7B and the
Qwen3-8B judge weights).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "imitation_safety" / "run_erosion_item.py"
GOLDEN = ROOT / "tests" / "golden_e2e" / "olmo3_7b_advbench"

ITEM_ID = "baseline_olmo-3-7b"
BENCH = "advbench"
# Knobs are pinned: they define the golden. Do NOT change without regenerating fixtures.
KNOBS = ["--benchmarks", BENCH, "--max-prompts", "8", "--gen-batch", "8",
         "--max-new-tokens", "64", "--no-heavy-graders"]

VERDICT_COLS = ["rtl_code", "rtl_label", "genuine_harm"]


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


_GATE = _gpu_available() and os.environ.get("DEMENTOR_RUN_GPU_TESTS") == "1"


@pytest.mark.skipif(
    not _GATE,
    reason="E2E GPU golden (set DEMENTOR_RUN_GPU_TESTS=1 on a CUDA box with the models cached)",
)
def test_e2e_erosion_golden(tmp_path):
    pd = pytest.importorskip("pandas")

    hf_home = os.environ.get("DEMENTOR_E2E_HF_HOME", "/data/ethantsliu/hf-cache")
    gpu = os.environ.get("DEMENTOR_E2E_GPU", "5")

    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=gpu,
        DEMENTOR_IMITATION_ROOT=str(tmp_path),  # isolate work/ + subsamples/ under tmp
        HF_HOME=hf_home,
        HF_HUB_CACHE=hf_home,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_XET="1",
    )

    proc = subprocess.run(
        [sys.executable, str(RUNNER), ITEM_ID, *KNOBS],
        env=env, cwd=str(RUNNER.parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800,
    )
    out_dir = tmp_path / "work" / ITEM_ID
    assert proc.returncode == 0, f"runner failed rc={proc.returncode}\n{proc.stdout[-2000:]}"

    # ---- metrics.json: byte-stable, the pipeline's scientific output ----------------------
    got_metrics = json.loads((out_dir / BENCH / "metrics.json").read_text())
    exp_metrics = json.loads((GOLDEN / "metrics.json").read_text())
    assert got_metrics == exp_metrics, (
        f"per-benchmark metrics drifted from golden\n got={got_metrics}\n exp={exp_metrics}"
    )

    # top-level item result (spot-check the metric-bearing fields; drop volatile bookkeeping)
    got_item = json.loads((out_dir / "metrics.json").read_text())
    exp_item = json.loads((GOLDEN / "item_metrics.json").read_text())
    assert got_item.get("per_benchmark") == exp_item.get("per_benchmark")
    assert got_item.get("rtl_judge_model") == exp_item.get("rtl_judge_model")

    # ---- all_judged.csv: prompt subsample + judge verdicts are byte-stable ----------------
    got = pd.read_csv(out_dir / BENCH / "all_judged.csv")
    exp = pd.read_csv(GOLDEN / "all_judged.csv")
    assert list(got["prompt"]) == list(exp["prompt"]), "seed-42 subsample diverged from golden"
    for col in VERDICT_COLS:
        assert list(got[col].astype(str)) == list(exp[col].astype(str)), (
            f"judge verdict column {col!r} drifted from golden\n"
            f" got={list(got[col])}\n exp={list(exp[col])}"
        )
    # model_response / rtl_raw are intentionally NOT asserted (GPU greedy-decode nondeterminism).
