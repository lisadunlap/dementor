"""Regenerate the Finding-4 dissociation tally from raw per-model eval artifacts.

Verdicts are stored PER BENCHMARK in <model>/benchmarks_summary.json (or, for
single-benchmark runs, <model>/eval/metrics.json). We tally on the HARM axis only:
XSTest is an over-refusal benchmark where baseline harm is ~0, so the refusal
positive control cannot meaningfully fire there and a PC_FAILS on XSTest is
uninformative rather than a genuine safety-resistance result.

Usage: python regen_dissociation_tally.py [RDO_ROOT]
"""
import json, os, sys
from collections import Counter

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/data/ethantsliu/exp_steer_safety/repl80_rdo"
HARM = {"advbench", "harmbench", "strongreject", "sorrybench", "sgbench"}
SKIP_SUFFIX = ("_retry5", "_dim8")
SKIP_DIRS = {"port", "wandb", "rdo_shared", "roster_geom_parts", "__pycache__"}


def verdicts_for(d):
    """Return {benchmark: metrics_dict} restricted to the harm axis."""
    out = {}
    summ = os.path.join(ROOT, d, "benchmarks_summary.json")
    if os.path.exists(summ) and os.path.getsize(summ) > 2:
        for k, v in json.load(open(summ)).items():
            if isinstance(v, dict) and "verdict" in v and k in HARM:
                out[k] = v
    if not out:  # single-benchmark run
        met = os.path.join(ROOT, d, "eval", "metrics.json")
        if os.path.exists(met) and os.path.getsize(met) > 2:
            js = json.load(open(met))
            if "verdict" in js and js.get("benchmark") in HARM:
                out[js["benchmark"]] = js
    return out


def main():
    rows = []
    for d in sorted(os.listdir(ROOT)):
        if (not os.path.isdir(os.path.join(ROOT, d)) or d in SKIP_DIRS
                or d.startswith("_") or d.endswith(SKIP_SUFFIX)):
            continue
        v = verdicts_for(d)
        if not v:
            rows.append((d, "NO_DATA", 0, "", ""))
            continue
        vs = [m["verdict"] for m in v.values()]
        n_clean = vs.count("CLEAN")
        verdict = ("CLEAN" if n_clean == len(vs)
                   else "PC_FAIL" if n_clean == 0
                   else f"MIXED({n_clean}/{len(vs)})")
        # Effect sizes, in percentage points, over the benchmarks where the control fires.
        fires = [m for m in v.values() if m["verdict"] == "CLEAN"]
        def mean(key):
            return sum(m[key] - m["baseline_harm"] for m in fires) / len(fires) * 100
        margin = (f"cone{mean('refusal_matched'):+.1f} fp{mean('fingerprint_matched'):+.1f} "
                  f"rnd{mean('random_matched'):+.1f}") if fires else ""
        rows.append((d, verdict, len(v), margin, ",".join(sorted(v))))

    for r in sorted(rows, key=lambda x: (x[1], x[0])):
        print(f"{r[0]:22s} {r[1]:12s} n={r[2]}  {r[3]:32s} {r[4]}")
    print("\nHARM-AXIS TALLY:", Counter(r[1].split("(")[0] for r in rows))


if __name__ == "__main__":
    main()
