"""Rebuild each Tinker item's metrics.json from all 7 benchmarks' cached all_judged.csv.

Full-parity helper: the harm run judged 5 benchmarks; a follow-up judged the 2 over-refusal ones.
Both leave per-benchmark work/<id>/<bench>/all_judged.csv on disk. The item-level metrics.json,
however, only ever contains whatever --benchmarks the last judge pass was given. This reassembles
metrics.json across all 7 (harm + over-refusal) from the cached judged CSVs -- no GPU, no re-judging.
Schema mirrors tinker_erosion.py's per-item metrics block exactly, so build_erosion_csv.py merges it.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erosion_common as EC  # noqa: E402
import tinker_erosion as TE  # noqa: E402

ALL7 = ["advbench", "harmbench", "strongreject", "sorrybench", "sgbench", "xstest", "orbench_hard"]


def main():
    ad, bl = TE.tinker_worklist(seed="seed42")
    n_ok = n_skip = 0
    missing = {}
    for it in bl + ad:
        od = EC.item_dir(it["id"])
        per_bench, ok = {}, True
        for b in ALL7:
            jf = os.path.join(od, b, "all_judged.csv")
            if not os.path.exists(jf):
                ok = False
                missing[b] = missing.get(b, 0) + 1
                break
            per_bench[b] = EC.benchmark_metric(b, jf)
        if not ok:
            n_skip += 1
            continue
        result = {
            "id": it["id"], "kind": it["kind"], "base_model": it["base_model"],
            "dataset": it.get("dataset"), "source": it.get("source"), "target": it.get("target"),
            "adapter_dir": it.get("adapter_dir"), "seed": it["seed"],
            "subsample_max_prompts": 200, "subsample_seed": 42,
            "rtl_judge_model": EC.RTL_JUDGE_MODEL, "graders_enabled": "all7",
            "backend": "tinker", "per_benchmark": per_bench,
        }
        json.dump(result, open(os.path.join(od, "metrics.json"), "w"), indent=2)
        n_ok += 1
    print(f"rebuilt {n_ok} metrics.json with all 7 benchmarks; skipped {n_skip} incomplete")
    if missing:
        print("missing-benchmark counts (why skipped):", missing)


if __name__ == "__main__":
    main()
