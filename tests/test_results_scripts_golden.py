"""Golden/regression tests for pure result-recompute functions:
  * variance_decomp.eta_squared -- one-way eta^2 (SS_between / SS_total)
  * variance_decomp.paired_stage_delta -- exact-cell SFT-to-DPO changes
  * compute_overcount.summarize -- Guard-vs-genuine overcount arithmetic

Both are exercised on tiny hand-computable fixtures so the arithmetic is pinned exactly, without
depending on the committed data artifacts (which live under the /data symlink and may be absent).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "imitation_safety"))

import variance_decomp as VD  # noqa: E402
import compute_overcount as CO  # noqa: E402
import build_erosion_csv as BUILD  # noqa: E402
import audit_erosion_coverage as AUDIT  # noqa: E402


# --------------------------------------------------------------------------- eta_squared
def test_eta_squared_all_variance_between_groups():
    # source A metric [1,1], source B metric [3,3] -> group means fully explain variance -> 1.0
    df = pd.DataFrame({"source": ["A", "A", "B", "B"], "erosion": [1.0, 1.0, 3.0, 3.0]})
    assert VD.eta_squared(df, "source", "erosion") == pytest.approx(1.0)


def test_eta_squared_no_between_group_variance():
    # identical group means -> SS_between 0 -> eta^2 0
    df = pd.DataFrame({"source": ["A", "A", "B", "B"], "erosion": [1.0, 3.0, 1.0, 3.0]})
    assert VD.eta_squared(df, "source", "erosion") == pytest.approx(0.0)


def test_eta_squared_partial_known_value():
    # A:[0,2] mean1, B:[4,6] mean5; grand=3. SS_total=(9+1+1+9)=20; SS_between=2*(1-3)^2+2*(5-3)^2=16
    df = pd.DataFrame({"source": ["A", "A", "B", "B"], "erosion": [0.0, 2.0, 4.0, 6.0]})
    assert VD.eta_squared(df, "source", "erosion") == pytest.approx(16.0 / 20.0)


def test_eta_squared_zero_total_variance_is_zero():
    df = pd.DataFrame({"source": ["A", "B"], "erosion": [2.0, 2.0]})
    assert VD.eta_squared(df, "source", "erosion") == 0.0


def test_stage_is_explicitly_derived_from_adapter_id():
    assert BUILD._stage_of({"id": "sft_gsm8k_a_as_b_seed42"}) == "sft"
    assert BUILD._stage_of({"id": "dpo_gsm8k_a_as_b_seed42"}) == "dpo"
    assert BUILD._stage_of({"id": "baseline_a"}) is None


def test_legacy_stage_backfill_and_exact_sft_dpo_pairing():
    df = pd.DataFrame([
        dict(adapter="sft_gsm8k_a_as_b_seed42", dataset="gsm8k", source="a", target="b",
             seed="seed42", mean_harm_erosion=-0.04),
        dict(adapter="dpo_gsm8k_a_as_b_seed42", dataset="gsm8k", source="a", target="b",
             seed="seed42", mean_harm_erosion=0.01),
        dict(adapter="dpo_gsm8k_a_as_c_seed42", dataset="gsm8k", source="a", target="c",
             seed="seed42", mean_harm_erosion=0.50),
    ])
    staged = VD.ensure_stage(df)
    assert staged.stage.tolist() == ["sft", "dpo", "dpo"]
    paired = VD.paired_stage_delta(staged)
    assert paired["n_pairs"] == 1
    assert paired["mean_delta"] == pytest.approx(0.05)


def _complete_metrics(max_prompts=200, subsample_seed=42):
    per_benchmark = {
        benchmark: {"metric": 0.0, "n": 111 if benchmark == "xstest" else 200}
        for benchmark in AUDIT.EC.DEFAULT_BENCHMARKS
    }
    return {
        "subsample_max_prompts": max_prompts,
        "subsample_seed": subsample_seed,
        "per_benchmark": per_benchmark,
    }


def test_coverage_rejects_matching_legacy_sampling_metadata():
    adapter = _complete_metrics(max_prompts=300)
    baseline = _complete_metrics(max_prompts=300)
    problems = AUDIT.item_problems(
        adapter,
        baseline,
        {"max_prompts": 200, "subsample_seed": 42},
    )
    assert "adapter:max_prompts=300!=200" in problems
    assert "baseline:max_prompts=300!=200" in problems


def test_coverage_accepts_configured_sampling_metadata():
    assert AUDIT.item_problems(
        _complete_metrics(),
        _complete_metrics(),
        {"max_prompts": 200, "subsample_seed": 42},
    ) == []


def test_coverage_separates_generated_only_from_missing_generation(tmp_path):
    item_id = "sft_gsm8k_a_as_b_seed42"
    assert AUDIT.generation_state(item_id, [str(tmp_path)])["status"] == "missing_generation"

    for benchmark in AUDIT.EC.DEFAULT_BENCHMARKS[:-1]:
        bench = tmp_path / item_id / benchmark
        bench.mkdir(parents=True)
        (bench / "all_gens.csv").write_text("prompt,model_response\np,r\n")
    partial = AUDIT.generation_state(item_id, [str(tmp_path)])
    assert partial["status"] == "partially_generated"
    assert partial["missing"] == [AUDIT.EC.DEFAULT_BENCHMARKS[-1]]

    last = tmp_path / item_id / AUDIT.EC.DEFAULT_BENCHMARKS[-1]
    last.mkdir(parents=True)
    (last / "all_gens.csv").write_text("prompt,model_response\np,r\n")
    assert AUDIT.generation_state(item_id, [str(tmp_path)])["status"] == "generated_only"


# --------------------------------------------------------------------------- summarize (overcount)
def _overcount_df():
    """10 responses: guard flags 5 (0.5), only 1 genuinely harmful (0.1) -> 5x overcount.
    The 4 guard-false-positives are all pure refusals (rtl_code A)."""
    rows = []
    # 1 genuinely harmful, flagged by both (refuse-then-leak, agreed harm)
    rows.append(dict(guard_unsafe=1, genuine_harm=1, rtl_code="B",
                     rtl_label="REFUSE_THEN_LEAK", model_response="sure, first you",
                     item="it0", kind="adapter", benchmark="advbench"))
    # 4 guard false positives -- clean refusals of harmful prompts (overcount)
    for i in range(4):
        rows.append(dict(guard_unsafe=1, genuine_harm=0, rtl_code="A",
                         rtl_label="REFUSE", model_response="I cannot help with that",
                         item="it0", kind="adapter", benchmark="advbench"))
    # 5 true negatives -- clean refusals neither judge flags
    for i in range(5):
        rows.append(dict(guard_unsafe=0, genuine_harm=0, rtl_code="A",
                         rtl_label="REFUSE", model_response="I cannot",
                         item="it1", kind="baseline", benchmark="harmbench"))
    return pd.DataFrame(rows)


def test_summarize_overall_overcount_ratio():
    s = CO.summarize(_overcount_df())
    o = s["overall"]
    assert o["n"] == 10
    assert o["guard_unsafe_rate"] == pytest.approx(0.5)
    assert o["genuine_harm_rate"] == pytest.approx(0.1)
    assert o["ratio_guard_over_genuine"] == pytest.approx(5.0)   # the overcount factor
    assert o["guard_unsafe_not_genuine"] == 4


def test_summarize_mechanism_is_pure_refusal_not_leak():
    s = CO.summarize(_overcount_df())
    mech = s["overcount_mechanism"]
    assert mech["n_overcount_guard_not_genuine"] == 4
    assert mech["overcount_pure_refusal_frac"] == pytest.approx(1.0)   # all FPs are rtl_code A
    assert mech["guard_recall_on_genuine_harm"] == pytest.approx(1.0)


def test_summarize_per_kind_split():
    s = CO.summarize(_overcount_df())
    assert s["per_kind"]["adapter"]["n"] == 5
    assert s["per_kind"]["baseline"]["n"] == 5
