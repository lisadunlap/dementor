#!/usr/bin/env python
"""Aggregate per-item metrics.json checkpoints into the erosion CSVs.

erosion definition (per disguise adapter A_as_B, per benchmark):
    harm axis         -> erosion = harm(disguised A_as_B) - harm(baseline A)
    over-refusal axis -> delta   = over_refusal(disguised) - over_refusal(baseline)
where the "baseline A" metric is model A UNADAPTED, scored on the SAME subsampled prompts.

Writes to data/results/safety/ (symlinked to /data):
  erosion_<seed>_long.csv     one row per (adapter, benchmark): metric_disguised, metric_baseline, erosion
  erosion_<seed>_summary.csv  one row per adapter: mean harm erosion (5 harm benchmarks) + mean
                              over-refusal delta (2 benign benchmarks), plus per-benchmark erosion cols

Usage: build_erosion_csv.py [--seed seed42]
"""
import os, sys, json, glob, argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import erosion_common as EC
from dementor import config


def analysis_metric(benchmark, payload):
    """Return the stage-comparable publication metric encoded by one benchmark result.

    Historical checkpoints store benchmark-native ``metric`` values, including continuous
    StrongREJECT scores, alongside the binary RTL rate. Harm erosion must use RTL on both the
    adapter and baseline; over-refusal retains the benchmark-native refusal metric.
    """
    if payload is None:
        return None
    expected_axis = EC.BENCH_AXIS[benchmark]
    if payload.get("axis") != expected_axis:
        raise ValueError(
            f"{benchmark}: axis={payload.get('axis')!r}, expected {expected_axis!r}"
        )
    if expected_axis == "harm":
        value = payload.get("rtl_genuine_harm")
        # Early AdvBench checkpoints used RTL as the canonical metric but did not duplicate it.
        if value is None and payload.get("canonical_col") == "genuine_harm":
            value = payload.get("metric")
        column = "genuine_harm"
    else:
        value = payload.get("metric")
        column = payload.get("canonical_col")
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float("nan")
    return {"value": value, "column": column, "n": payload.get("n")}


def _metric_signature(d):
    """Analysis-relevant payload used to verify duplicate checkpoints agree."""
    keys = (
        "id", "kind", "dataset", "source", "target", "seed", "base_model",
        "subsample_max_prompts", "subsample_seed", "rtl_judge_model", "graders_enabled",
        "per_benchmark",
    )
    return json.dumps({key: d.get(key) for key in keys}, sort_keys=True, allow_nan=True)


def _is_n200_checkpoint(d):
    """Whether all campaign benchmarks use the harmonized 200-prompt sample.

    XSTest scores only the 111 benign rows within the shared 200-row sample.
    """
    expected_n = {benchmark: 200 for benchmark in EC.DEFAULT_BENCHMARKS}
    expected_n["xstest"] = 111
    per_benchmark = d.get("per_benchmark", {})
    return all(
        per_benchmark.get(benchmark, {}).get("n") == n
        for benchmark, n in expected_n.items()
    )


def load_items(work_roots=None, accept=None):
    """Load and content-deduplicate checkpoints from one or more worker roots.

    Different boxes may contain byte-different JSON formatting or bookkeeping, but checkpoints
    with the same id must agree on every analysis-relevant field. Conflicts abort instead of
    silently selecting whichever filesystem glob happens to return last.
    """
    roots = list(work_roots or [EC.WORK])
    items = {}
    origins = {}
    for root in roots:
        for mj in glob.glob(os.path.join(root, "*", "metrics.json")):
            try:
                d = json.load(open(mj))
            except Exception:
                continue
            if d.get("kind") not in ("adapter", "baseline"):
                continue
            if accept is not None and not accept(d):
                continue
            d["_checkpoint_path"] = mj
            item_id = d["id"]
            if item_id in items and _metric_signature(items[item_id]) != _metric_signature(d):
                old_is_standard = _is_n200_checkpoint(items[item_id])
                new_is_standard = _is_n200_checkpoint(d)
                if old_is_standard != new_is_standard:
                    if new_is_standard:
                        items[item_id] = d
                        origins[item_id] = mj
                    continue
                raise ValueError(
                    f"conflicting metrics for {item_id}: {origins[item_id]} vs {mj}"
                )
            items.setdefault(item_id, d)
            origins.setdefault(item_id, mj)
    return items


def _dataset_of(d):
    """dataset dimension for an item: prefer the stored field, else backfill from the id (old
    metrics.json written before the `dataset` field existed lack it)."""
    if d.get("dataset"):
        return d["dataset"]
    m = EC.ADAPTER_RE.match(d.get("id", ""))
    return m.group(1) if m else None


def _stage_of(d):
    """Training rung encoded by an adapter id (``sft_*`` or ``dpo_*``)."""
    stage = str(d.get("id", "")).split("_", 1)[0]
    return stage if stage in {"sft", "dpo"} else None


def _in_campaign(d, name):
    if name == "all":
        return True
    models = {m["slug"] for m in config.campaign_roster(name)}
    datasets = set(config.campaign_dataset_names(name))
    seeds = {f"seed{seed}" for seed in config.campaign_seeds(name)}
    return (
        d.get("source") in models
        and d.get("target") in models
        and d.get("source") != d.get("target")
        and _dataset_of(d) in datasets
        and d.get("seed") in seeds
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="all",
                    help="'all' (default, unified multi-seed table) or a specific 'seedNN' to filter")
    ap.add_argument("--campaign", default="imitation_safety",
                    help="named config campaign to aggregate (default: imitation_safety; 'all' disables filtering)")
    ap.add_argument("--work-root", action="append", dest="work_roots",
                    help="worker root containing <id>/metrics.json; repeat to merge boxes safely")
    args = ap.parse_args()
    os.makedirs(EC.RESULTS_SAFETY, exist_ok=True)

    items = load_items(
        args.work_roots,
        accept=lambda d: d.get("kind") == "baseline" or _in_campaign(d, args.campaign),
    )
    # a baseline is dataset/seed-independent (model A UNADAPTED) -> keyed by base_model only.
    baselines = {d["base_model"]: d for d in items.values() if d["kind"] == "baseline"}
    adapters = [d for d in items.values() if d["kind"] == "adapter"]
    adapters = [d for d in adapters if _in_campaign(d, args.campaign)]
    if args.seed not in ("all", None):
        adapters = [d for d in adapters if d.get("seed") == args.seed]

    long_rows, summary_rows = [], []
    for a in sorted(adapters, key=lambda x: x["id"]):
        base = baselines.get(a["base_model"])
        ds = _dataset_of(a)
        seed = a.get("seed")
        stage = _stage_of(a)
        per_erosion = {}
        for b, m in a["per_benchmark"].items():
            bm = base["per_benchmark"].get(b) if base else None
            adapter_result = analysis_metric(b, m)
            baseline_result = analysis_metric(b, bm)
            adapter_metric = adapter_result["value"] if adapter_result else float("nan")
            base_metric = baseline_result["value"] if baseline_result else float("nan")
            erosion = (
                adapter_metric - base_metric
                if adapter_metric == adapter_metric and base_metric == base_metric
                else float("nan")
            )
            per_erosion[b] = erosion
            long_rows.append({
                "adapter": a["id"], "stage": stage, "dataset": ds, "source": a.get("source"),
                "target": a.get("target"), "seed": seed, "base_model": a["base_model"],
                "backend": a.get("backend"), "benchmark": b, "axis": m["axis"],
                "n": adapter_result["n"], "metric_column": adapter_result["column"],
                "metric_disguised": adapter_metric, "metric_baseline": base_metric,
                "erosion": erosion, "baseline_available": base is not None,
            })
        harm_eros = [per_erosion[b] for b in EC.HARM_BENCHMARKS
                     if b in per_erosion and per_erosion[b] == per_erosion[b]]
        or_delta = [per_erosion[b] for b in EC.OVERREF_BENCHMARKS
                    if b in per_erosion and per_erosion[b] == per_erosion[b]]
        row = {"adapter": a["id"], "stage": stage, "dataset": ds, "source": a.get("source"),
               "target": a.get("target"), "seed": seed, "base_model": a["base_model"],
               "backend": a.get("backend"), "baseline_available": base is not None,
               "mean_harm_erosion": (sum(harm_eros) / len(harm_eros)) if harm_eros else float("nan"),
               "mean_over_refusal_delta": (sum(or_delta) / len(or_delta)) if or_delta else float("nan")}
        for b in EC.DEFAULT_BENCHMARKS:
            row[f"erosion_{b}"] = per_erosion.get(b, float("nan"))
        summary_rows.append(row)

    tag = "all" if args.seed in ("all", None) else args.seed
    long_path = os.path.join(EC.RESULTS_SAFETY, f"erosion_{tag}_long.csv")
    sum_path = os.path.join(EC.RESULTS_SAFETY, f"erosion_{tag}_summary.csv")
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    pd.DataFrame(summary_rows).to_csv(sum_path, index=False)
    print(f"[build] {len(adapters)} adapters, {len(baselines)} baselines, campaign={args.campaign}")
    from collections import Counter
    by_ds_seed = Counter((_dataset_of(a), a.get("seed")) for a in adapters)
    for k in sorted(by_ds_seed, key=lambda x: (str(x[0]), str(x[1]))):
        print(f"[build]   {by_ds_seed[k]:4d}  {k[0]:15s} {k[1]}")
    print(f"[build] wrote {long_path} ({len(long_rows)} rows)")
    print(f"[build] wrote {sum_path} ({len(summary_rows)} rows)")
    missing = sorted({a["base_model"] for a in adapters if a["base_model"] not in baselines})
    if missing:
        print(f"[build] WARNING baselines missing for: {missing} (erosion NaN for their adapters)")


if __name__ == "__main__":
    main()
