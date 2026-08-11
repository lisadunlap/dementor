"""Golden/regression tests for deterministic, non-GPU pipeline logic in erosion_common.

Covers the pieces an erosion delta silently depends on:
  * ADAPTER_RE   -- disguise-key parsing (dataset / source / target / seed)
  * get_subsample -- seeded stratified subsample: SAME seed -> identical prompt set, and the
    smaller subsample NESTS inside the larger (required so a baseline and its adapter score the
    same prompts across sweep sizes)
  * benchmark_metric -- per-item harm / over-refusal metric + canonical-column fallback

Fixtures are self-contained (no /data dependency); module path constants are monkeypatched to
tmp dirs so nothing touches the real subsample cache.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "imitation_safety"))

import erosion_common as EC  # noqa: E402


# --------------------------------------------------------------------------- ADAPTER_RE parsing
def test_adapter_re_parses_all_four_groups():
    m = EC.ADAPTER_RE.match("dpo_chatbot_arena_ministral-8b_as_llama-3.1-8b_seed42")
    assert m is not None
    assert m.group(1) == "chatbot_arena"
    assert m.group(2) == "ministral-8b"
    assert m.group(3) == "llama-3.1-8b"
    assert m.group(4) == "42"


@pytest.mark.parametrize("ds", ["chatbot_arena", "gsm8k", "oasst1", "writingprompts"])
def test_adapter_re_accepts_each_dataset(ds):
    assert EC.ADAPTER_RE.match(f"dpo_{ds}_a_as_b_seed1") is not None


@pytest.mark.parametrize("stage", ["dpo", "sft"])
def test_adapter_re_accepts_both_pipeline_stages(stage):
    """Both stages resolve: `dpo_` is the SFT->DPO end state the published matrix measures,
    `sft_` points at the same run's SFT parent and isolates the imitation step from the
    preference step.  The capture groups are identical either way."""
    m = EC.ADAPTER_RE.match(f"{stage}_gsm8k_a_as_b_seed42")
    assert m is not None
    assert (m.group(1), m.group(2), m.group(3), m.group(4)) == ("gsm8k", "a", "b", "42")


@pytest.mark.parametrize("bad", [
    "rlhf_gsm8k_a_as_b_seed42",             # unknown pipeline stage (only dpo_/sft_)
    "dpo_unknownds_a_as_b_seed42",          # unlisted dataset
    "dpo_gsm8k_a_as_b_seedXX",              # non-numeric seed
    "dpo_gsm8k_a_b_seed42",                 # missing _as_
])
def test_adapter_re_rejects_malformed(bad):
    assert EC.ADAPTER_RE.match(bad) is None


# --------------------------------------------------------------------------- get_subsample
def _write_bench(bench_dir, name, n_refuse=210, n_comply=90):
    rows = [{"prompt": f"refuse-prompt-{i}", "expected": "refuse"} for i in range(n_refuse)]
    rows += [{"prompt": f"comply-prompt-{i}", "expected": "comply"} for i in range(n_comply)]
    df = pd.DataFrame(rows)
    df.to_csv(bench_dir / f"{name}.csv", index=False)
    return df


def _prompts(csv_path):
    return pd.read_csv(csv_path)["prompt"].astype(str).tolist()


@pytest.fixture
def bench_env(tmp_path, monkeypatch):
    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    monkeypatch.setattr(EC, "BENCH_DIR", str(bench_dir))
    _write_bench(bench_dir, "advbench")
    # each call gets its OWN subsample dir so caching never masks a determinism regression
    def sub(name):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setattr(EC, "SUBSAMPLES", str(d))
        return d
    return sub


def test_get_subsample_same_seed_is_identical(bench_env):
    bench_env("a")
    out_a = EC.get_subsample("advbench", max_prompts=200, seed=42)
    bench_env("b")
    out_b = EC.get_subsample("advbench", max_prompts=200, seed=42)
    assert _prompts(out_a) == _prompts(out_b)          # byte-for-byte reproducible
    assert len(_prompts(out_a)) == 200


def test_get_subsample_manifest_uses_portable_source_path(bench_env):
    bench_env("portable")
    output = EC.get_subsample("advbench", max_prompts=200, seed=42)
    manifest = json.loads(Path(output).with_suffix(".manifest.json").read_text())
    assert not os.path.isabs(manifest["source_csv"])


def test_get_subsample_different_seed_differs(bench_env):
    bench_env("a")
    a = set(_prompts(EC.get_subsample("advbench", max_prompts=200, seed=42)))
    bench_env("b")
    b = set(_prompts(EC.get_subsample("advbench", max_prompts=200, seed=7)))
    assert a != b


def test_get_subsample_n200_nests_in_n300(bench_env):
    bench_env("big")
    big = set(_prompts(EC.get_subsample("advbench", max_prompts=300, seed=42)))
    bench_env("small")
    small = set(_prompts(EC.get_subsample("advbench", max_prompts=200, seed=42)))
    assert len(big) == 300 and len(small) == 200
    assert small <= big                                 # smaller subsample nests inside the larger


def test_get_subsample_is_stratified_by_expected(bench_env):
    bench_env("s")
    df = pd.read_csv(EC.get_subsample("advbench", max_prompts=100, seed=42))
    dist = df["expected"].value_counts().to_dict()
    # 210 refuse / 90 comply of 300 -> ~70/30 split preserved at n=100
    assert set(dist) == {"refuse", "comply"}
    assert dist["refuse"] == 70 and dist["comply"] == 30


def test_get_subsample_full_when_max_nonpositive(bench_env):
    bench_env("f")
    df = pd.read_csv(EC.get_subsample("advbench", max_prompts=0, seed=42))
    assert len(df) == 300                                # <=0 -> full benchmark


# --------------------------------------------------------------------------- benchmark_metric
def _write_judged(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_benchmark_metric_harm_axis_uses_canonical_col(tmp_path):
    # advbench harm axis -> mean genuine_harm over expected==refuse rows only.
    j = _write_judged(tmp_path / "j.csv", [
        {"expected": "refuse", "genuine_harm": 1},
        {"expected": "refuse", "genuine_harm": 0},
        {"expected": "refuse", "genuine_harm": 0},
        {"expected": "refuse", "genuine_harm": 0},
        {"expected": "comply", "genuine_harm": 1},   # ignored on harm axis
    ])
    r = EC.benchmark_metric("advbench", j)
    assert r["axis"] == "harm"
    assert r["canonical_col"] == "genuine_harm"
    assert r["n"] == 4
    assert r["metric"] == pytest.approx(0.25)


def test_benchmark_metric_harm_uses_rtl_even_when_native_grader_exists(tmp_path):
    j = _write_judged(tmp_path / "j.csv", [
        {"expected": "refuse", "genuine_harm": 1, "hb_harm": 0},
        {"expected": "refuse", "genuine_harm": 1, "hb_harm": 0},
    ])
    r = EC.benchmark_metric("harmbench", j)
    assert r["canonical_col"] == "genuine_harm"
    assert r["metric"] == pytest.approx(1.0)


def test_benchmark_metric_over_refusal_axis(tmp_path):
    # xstest over-refusal axis -> mean xstest_refusal over expected==comply rows only.
    j = _write_judged(tmp_path / "j.csv", [
        {"expected": "comply", "xstest_refusal": 1},
        {"expected": "comply", "xstest_refusal": 0},
        {"expected": "refuse", "xstest_refusal": 1},   # ignored on over-refusal axis
    ])
    r = EC.benchmark_metric("xstest", j)
    assert r["axis"] == "over_refusal"
    assert r["canonical_col"] == "xstest_refusal"
    assert r["n"] == 2
    assert r["metric"] == pytest.approx(0.5)


def test_benchmark_metric_defaults_missing_expected_to_refuse(tmp_path):
    j = _write_judged(tmp_path / "j.csv", [
        {"genuine_harm": 1}, {"genuine_harm": 0},
    ])
    r = EC.benchmark_metric("advbench", j)
    assert r["n"] == 2                                  # both rows treated as expected==refuse
    assert r["metric"] == pytest.approx(0.5)
