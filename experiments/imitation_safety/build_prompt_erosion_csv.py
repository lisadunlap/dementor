#!/usr/bin/env python
"""Aggregate per-item metrics.json checkpoints of the PROMPT-RUNG safety-erosion sweep into CSVs,
mirroring build_erosion_csv.py but with a `method` dimension (the disguise-ladder prompt rung).

erosion definition (per (method, source A, target B), per benchmark):
    harm axis         -> erosion = harm(A prompted-as-B) - harm(baseline A)
    over-refusal axis -> delta   = over_refusal(prompted) - over_refusal(baseline)
where the "baseline A" metric is model A UNADAPTED, REUSED from the SFT/DPO rung
(erosion_common.WORK/baseline_<slug>/metrics.json) so the prompt rung and the weight rung share a
byte-identical baseline and are directly comparable.

Writes to data/results/safety/ (symlinked to /data):
  prompt_erosion_all_long.csv     one row per (method, source, target, benchmark)
  prompt_erosion_all_summary.csv  one row per (method, source, target): mean harm erosion (5 harm
                                  benchmarks) + mean over-refusal delta (2 benign) + per-benchmark cols

Usage: build_prompt_erosion_csv.py [--methods m1,m2]
"""
import os, sys, json, glob, argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erosion_common as EC
import prompt_erosion_common as PC


def load_items():
    items = {}
    for mj in glob.glob(os.path.join(PC.WORK, "*", "metrics.json")):
        try:
            d = json.load(open(mj))
        except Exception:
            continue
        if d.get("kind") != "prompt_disguise":
            continue
        items[d["id"]] = d
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=None, help="optional comma filter of methods")
    args = ap.parse_args()
    os.makedirs(PC.RESULTS, exist_ok=True)

    keep_methods = None
    if args.methods:
        keep_methods = {m.strip() for m in args.methods.split(",") if m.strip()}

    items = load_items()
    # cache baselines (model A UNADAPTED) reused from the SFT/DPO rung, keyed by source slug
    base_cache = {}

    def baseline_of(src):
        if src not in base_cache:
            base_cache[src] = PC.baseline_metrics(src)
        return base_cache[src]

    long_rows, summary_rows = [], []
    for a in sorted(items.values(), key=lambda x: x["id"]):
        if keep_methods and a.get("method") not in keep_methods:
            continue
        base = baseline_of(a["source"])
        per_erosion = {}
        for b, m in a["per_benchmark"].items():
            bm = base["per_benchmark"].get(b) if base else None
            base_metric = bm["metric"] if bm else float("nan")
            erosion = (m["metric"] - base_metric) if (base_metric == base_metric) else float("nan")
            per_erosion[b] = erosion
            long_rows.append({
                "method": a.get("method"), "source": a.get("source"), "target": a.get("target"),
                "base_model": a.get("base_model"), "backend": a.get("backend"), "seed": a.get("seed"),
                "benchmark": b, "axis": m["axis"], "n": m["n"],
                "metric_disguised": m["metric"], "metric_baseline": base_metric,
                "erosion": erosion, "baseline_available": base is not None, "id": a["id"],
            })
        harm_eros = [per_erosion[b] for b in EC.HARM_BENCHMARKS
                     if b in per_erosion and per_erosion[b] == per_erosion[b]]
        or_delta = [per_erosion[b] for b in EC.OVERREF_BENCHMARKS
                    if b in per_erosion and per_erosion[b] == per_erosion[b]]
        row = {"method": a.get("method"), "source": a.get("source"), "target": a.get("target"),
               "base_model": a.get("base_model"), "backend": a.get("backend"), "seed": a.get("seed"),
               "baseline_available": base is not None,
               "mean_harm_erosion": (sum(harm_eros) / len(harm_eros)) if harm_eros else float("nan"),
               "mean_over_refusal_delta": (sum(or_delta) / len(or_delta)) if or_delta else float("nan")}
        for b in EC.DEFAULT_BENCHMARKS:
            row[f"erosion_{b}"] = per_erosion.get(b, float("nan"))
        summary_rows.append(row)

    long_path = os.path.join(PC.RESULTS, "prompt_erosion_all_long.csv")
    sum_path = os.path.join(PC.RESULTS, "prompt_erosion_all_summary.csv")
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    pd.DataFrame(summary_rows).to_csv(sum_path, index=False)
    from collections import Counter
    print(f"[build] {len(summary_rows)} (method x pair) items")
    by_method = Counter(r["method"] for r in summary_rows)
    for k in sorted(by_method):
        print(f"[build]   {by_method[k]:4d}  method={k}")
    print(f"[build] wrote {long_path} ({len(long_rows)} rows)")
    print(f"[build] wrote {sum_path} ({len(summary_rows)} rows)")
    missing = sorted({r["source"] for r in summary_rows if not r["baseline_available"]})
    if missing:
        print(f"[build] WARNING baselines missing for sources: {missing} "
              f"(erosion NaN for their rows until the erosion_daemon computes baseline_<slug>)")


if __name__ == "__main__":
    main()
