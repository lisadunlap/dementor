#!/usr/bin/env python3
"""Does the stock refusal/fingerprint geometry still dissociate AFTER imitation fine-tuning?

Pairs each adapter cell (exp_steer_adapter/<src>_as_<tgt>/eval_<bench>) against the SAME source
model's BASE cell (repl80_rdo/<src>/eval_<bench>).  Both use the SAME stock cone / fingerprint /
random directions -- only the weights differ -- so the comparison isolates what fine-tuning did
to the geometry, not to the directions.

CPU only.  Reads metrics.json; never re-runs a model.
"""
import argparse, glob, json, os
import pandas as pd

ADAPT = os.environ.get("DEMENTOR_STEER_ADAPTER", "/data/ethantsliu/exp_steer_adapter")
BASE = os.environ.get("DEMENTOR_STEER_WORK", "/data/ethantsliu/exp_steer_safety/repl80_rdo")
KEY = "matched_harm@0.85"


def arms(m):
    pd_ = m.get("per_direction", {})
    out = {}
    for a in ("cone", "fingerprint", "random"):
        v = (pd_.get(a) or {}).get(KEY)
        out[a] = None if v is None or v != v else float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write the paired table as CSV")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(f"{ADAPT}/*/eval_*/metrics.json")):
        cell = p.split(f"{ADAPT}/")[1].split("/")[0]
        if "_as_" not in cell:
            continue
        src, tgt = cell.split("_as_", 1)
        bench = os.path.basename(os.path.dirname(p)).replace("eval_", "")
        bp = f"{BASE}/{src}/eval_{bench}/metrics.json"
        if not os.path.exists(bp):
            continue
        a, b = json.load(open(p)), json.load(open(bp))
        aa, ba = arms(a), arms(b)
        rows.append(dict(
            source=src, target=tgt, benchmark=bench,
            a_verdict=a.get("verdict"), b_verdict=b.get("verdict"),
            a_base=a.get("baseline_harm"), b_base=b.get("baseline_harm"),
            **{f"a_{k}": v for k, v in aa.items()},
            **{f"b_{k}": v for k, v in ba.items()}))
    if not rows:
        print("no paired cells yet")
        return
    df = pd.DataFrame(rows)

    # excess over the random control -- the quantity the dissociation claim is about
    for w in ("a", "b"):
        df[f"{w}_excess"] = df[f"{w}_fingerprint"] - df[f"{w}_random"]

    print(f"paired cells: {len(df)}  ({df.source.nunique()} sources, {df.target.nunique()} targets, "
          f"{df.benchmark.nunique()} benchmarks)\n")
    print(f"{'source':<17}{'target':<20}{'bench':<14}"
          f"{'base_harm':>19}{'cone':>15}{'fingerprint':>16}{'excess vs rnd':>17}")
    print(f"{'':<51}{'adapt / base':>19}{'adapt / base':>15}{'adapt / base':>16}{'adapt / base':>17}")
    print("-" * 132)
    f2 = lambda x: " nan" if x is None or x != x else f"{100*x:4.1f}"
    for _, r in df.sort_values(["source", "target", "benchmark"]).iterrows():
        print(f"{r.source:<17}{r.target:<20}{r.benchmark:<14}"
              f"{f2(r.a_base)+' / '+f2(r.b_base):>19}"
              f"{f2(r.a_cone)+' / '+f2(r.b_cone):>15}"
              f"{f2(r.a_fingerprint)+' / '+f2(r.b_fingerprint):>16}"
              f"{f2(r.a_excess)+' / '+f2(r.b_excess):>17}")

    print("\n" + "=" * 132)
    print("SUMMARY (pp).  'adapter' = imitation-fine-tuned weights, 'base' = stock model, "
          "SAME stock directions.")
    print("=" * 132)
    for label, cols in (("baseline harm", ("a_base", "b_base")),
                        ("cone (positive control)", ("a_cone", "b_cone")),
                        ("fingerprint (under test)", ("a_fingerprint", "b_fingerprint")),
                        ("fingerprint excess over random", ("a_excess", "b_excess"))):
        s = df[list(cols)].dropna()
        if not len(s):
            continue
        d = 100 * (s[cols[0]] - s[cols[1]])
        print(f"  {label:<32} adapter {100*s[cols[0]].mean():6.2f}   base {100*s[cols[1]].mean():6.2f}   "
              f"delta {d.mean():+6.2f}  (median {d.median():+6.2f}, n={len(s)})")

    vc = df.a_verdict.value_counts().to_dict()
    print(f"\n  adapter verdicts: {vc}")
    flips = df[df.a_verdict != df.b_verdict]
    print(f"  cells whose verdict differs from the base model: {len(flips)} of {len(df)}")
    for _, r in flips.iterrows():
        print(f"     {r.source} as {r.target} / {r.benchmark}: base={r.b_verdict} -> adapter={r.a_verdict}")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
