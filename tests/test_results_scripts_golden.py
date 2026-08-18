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


def test_source_cluster_bootstrap_is_seeded_and_brackets_mean():
    df = pd.DataFrame({"source": ["a", "a", "b", "b"], "erosion": [0.0, 0.0, 2.0, 2.0]})
    ci = VD.source_cluster_mean_ci(df, "erosion")
    assert ci == VD.source_cluster_mean_ci(df, "erosion")
    assert ci[0] <= 1.0 <= ci[1]


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


def test_target_relative_safety_reaches_halfway_to_target():
    rows = []
    baselines = {"a": 0.1, "b": 0.3}
    for source, target in (("a", "b"), ("b", "a")):
        source_harm = baselines[source]
        target_harm = baselines[target]
        rows.append({
            "adapter": f"sft_gsm8k_{source}_as_{target}_seed42",
            "stage": "sft",
            "dataset": "gsm8k",
            "source": source,
            "target": target,
            "seed": "seed42",
            "benchmark": "advbench",
            "axis": "harm",
            "metric_baseline": source_harm,
            "metric_disguised": source_harm + 0.5 * (target_harm - source_harm),
            "baseline_available": True,
        })
    result = VD.target_relative_safety(
        pd.DataFrame(rows), "sft", bootstrap_reps=0
    )
    assert result["n_cells"] == 2
    assert result["target_alignment_slope"] == pytest.approx(0.5)
    assert result["mean_target_distance_reduction"] == pytest.approx(0.1)
    assert result["pct_nonzero_gap_cells_closer_to_target"] == pytest.approx(100.0)


def test_joint_identity_bootstrap_preserves_exact_dyadic_slope():
    rows = []
    baselines = {"a": 0.1, "b": 0.3, "c": 0.5}
    for source in baselines:
        for target in baselines:
            if source == target:
                continue
            gap = baselines[target] - baselines[source]
            change = 0.5 * gap
            rows.append({
                "source": source,
                "target": target,
                "dataset": "d",
                "target_gap": gap,
                "adapter_change": change,
                "target_distance_reduction": abs(gap) - abs(gap - change),
            })
    first = VD._joint_identity_target_bootstrap(
        pd.DataFrame(rows), seed=42, reps=250
    )
    second = VD._joint_identity_target_bootstrap(
        pd.DataFrame(rows), seed=42, reps=250
    )
    assert first == second
    assert first["intervals"]["target_alignment_slope"] == pytest.approx([0.5, 0.5])
    assert first["valid_draws"]["target_alignment_slope"] > 0


def test_target_relative_safety_rejects_inconsistent_baselines():
    rows = pd.DataFrame([
        dict(adapter="sft_d_a_as_b_seed42", stage="sft", dataset="d", source="a",
             target="b", seed="seed42", benchmark="advbench", axis="harm",
             metric_baseline=0.1, metric_disguised=0.1, baseline_available=True),
        dict(adapter="sft_e_a_as_b_seed42", stage="sft", dataset="e", source="a",
             target="b", seed="seed42", benchmark="advbench", axis="harm",
             metric_baseline=0.2, metric_disguised=0.2, baseline_available=True),
        dict(adapter="sft_d_b_as_a_seed42", stage="sft", dataset="d", source="b",
             target="a", seed="seed42", benchmark="advbench", axis="harm",
             metric_baseline=0.3, metric_disguised=0.3, baseline_available=True),
    ])
    with pytest.raises(ValueError, match="source baseline is not unique"):
        VD.target_relative_safety(rows, "sft", bootstrap_reps=0)


def _complete_metrics(max_prompts=200, subsample_seed=42):
    per_benchmark = {
        benchmark: {
            "axis": AUDIT.EC.BENCH_AXIS[benchmark],
            "canonical_col": AUDIT.EC.CANON_COL[benchmark],
            "metric": 0.0,
            "rtl_genuine_harm": 0.0 if benchmark in AUDIT.EC.HARM_BENCHMARKS else float("nan"),
            "n": 111 if benchmark == "xstest" else 200,
        }
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


def test_prompt_sequence_hash_uses_actual_judged_order(monkeypatch, tmp_path):
    selected = tmp_path / "selected.csv"
    pd.DataFrame({"prompt": ["a", "b"]}).to_csv(selected, index=False)
    checkpoint = tmp_path / "item" / "metrics.json"
    judged = checkpoint.parent / "advbench" / "all_judged.csv"
    judged.parent.mkdir(parents=True)
    pd.DataFrame({"prompt": ["a", "b"]}).to_csv(judged, index=False)
    monkeypatch.setattr(AUDIT.EC, "get_subsample", lambda *_: str(selected))

    assert AUDIT.prompt_sequence_hash(
        {"_checkpoint_path": str(checkpoint)}, "advbench", 200, 42
    )[1] == 2

    pd.DataFrame({"prompt": ["b", "a"]}).to_csv(judged, index=False)
    assert AUDIT.prompt_sequence_hash(
        {"_checkpoint_path": str(checkpoint)}, "advbench", 200, 42
    ) is None

    harmonized = judged.with_name("all_judged.n200.csv")
    pd.DataFrame({"prompt": ["a", "b"]}).to_csv(harmonized, index=False)
    assert AUDIT.prompt_sequence_hash(
        {"_checkpoint_path": str(checkpoint)}, "advbench", 200, 42
    )[1] == 2


def test_analysis_metric_uses_like_for_like_rtl_not_native_score():
    payload = {
        "axis": "harm",
        "canonical_col": "strongreject_score",
        "metric": 0.73,
        "rtl_genuine_harm": 0.10,
        "n": 200,
    }

    assert BUILD.analysis_metric("strongreject", payload) == {
        "value": 0.10,
        "column": "genuine_harm",
        "n": 200,
    }


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
