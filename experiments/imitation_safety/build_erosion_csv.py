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
import erosion_common as EC


def load_items():
    items = {}
    for mj in glob.glob(os.path.join(EC.WORK, "*", "metrics.json")):
        try:
            d = json.load(open(mj))
        except Exception:
            continue
        if d.get("kind") not in ("adapter", "baseline"):
            continue
        items[d["id"]] = d
    return items


def _dataset_of(d):
    """dataset dimension for an item: prefer the stored field, else backfill from the id (old
    metrics.json written before the `dataset` field existed lack it)."""
    if d.get("dataset"):
        return d["dataset"]
    m = EC.ADAPTER_RE.match(d.get("id", ""))
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="all",
                    help="'all' (default, unified multi-seed table) or a specific 'seedNN' to filter")
    args = ap.parse_args()
    os.makedirs(EC.RESULTS_SAFETY, exist_ok=True)

    items = load_items()
    # a baseline is dataset/seed-independent (model A UNADAPTED) -> keyed by base_model only.
    baselines = {d["base_model"]: d for d in items.values() if d["kind"] == "baseline"}
    adapters = [d for d in items.values() if d["kind"] == "adapter"]
    if args.seed not in ("all", None):
        adapters = [d for d in adapters if d.get("seed") == args.seed]

    long_rows, summary_rows = [], []
    for a in sorted(adapters, key=lambda x: x["id"]):
        base = baselines.get(a["base_model"])
        ds = _dataset_of(a)
        seed = a.get("seed")
        per_erosion = {}
        for b, m in a["per_benchmark"].items():
            bm = base["per_benchmark"].get(b) if base else None
            base_metric = bm["metric"] if bm else float("nan")
            erosion = (m["metric"] - base_metric) if (base_metric == base_metric) else float("nan")
            per_erosion[b] = erosion
            long_rows.append({
                "adapter": a["id"], "dataset": ds, "source": a.get("source"),
                "target": a.get("target"), "seed": seed, "base_model": a["base_model"],
                "backend": a.get("backend"), "benchmark": b, "axis": m["axis"],
                "n": m["n"], "metric_disguised": m["metric"], "metric_baseline": base_metric,
                "erosion": erosion, "baseline_available": base is not None,
            })
        harm_eros = [per_erosion[b] for b in EC.HARM_BENCHMARKS
                     if b in per_erosion and per_erosion[b] == per_erosion[b]]
        or_delta = [per_erosion[b] for b in EC.OVERREF_BENCHMARKS
                    if b in per_erosion and per_erosion[b] == per_erosion[b]]
        row = {"adapter": a["id"], "dataset": ds, "source": a.get("source"),
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
    print(f"[build] {len(adapters)} adapters, {len(baselines)} baselines")
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
