#!/usr/bin/env python3
"""The dissociation on the OVER-REFUSAL axis (task #97).

The harm axis asks: does ablating a direction make the model answer things it should refuse?
The over-refusal axis asks the mirror question on BENIGN prompts: does ablating a direction make
the model stop refusing things it should have answered all along?

This is an independent test of the same claim on a different population of prompts.  If refusal
and identity are separable, then ablating the refusal cone should cut over-refusal (it removes
the mechanism that produces spurious refusals) while ablating the fingerprint should not.

XSTest carries both axes: its 125 unsafe contrast prompts feed the harm axis and its 175 safe
prompts feed this one.  CPU only -- the numbers are already in metrics.json.

Two scorers are reported because they disagree in a way that matters.  The pipeline's default
over-refusal signal is the Arditi refusal-substring match (`refusal_led`), which fires on any
response OPENING with "I'm sorry" -- including helpful ones ("I'm sorry, I don't know where you
are located, but here are some places..."), so it both over- and under-counts.  XSTest also
carries a native 3-way classifier (`xstest_refusal`, Qwen3-8B) that reads the whole response.
The native scorer is the one to believe; the substring is kept as a cross-check.
"""
import glob, json, os, statistics
import pandas as pd

ROOT = "/data/ethantsliu/exp_steer_safety/repl80_rdo"
CANON = {"xstest", "orbench_hard", "orbench_80k"}


def native(m, d):
    """XSTest's native 3-way classifier value for one arm (None when the grader did not run)."""
    return (m.get("per_direction", {}).get(d) or {}).get("over_refusal_xstest@0.85")


def main():
    rows = []
    for p in sorted(glob.glob(f"{ROOT}/*/eval_*/metrics.json")):
        model = p.split("/repl80_rdo/")[1].split("/")[0]
        bench = os.path.basename(os.path.dirname(p)).replace("eval_", "")
        if bench not in CANON:      # skip .bak_/_recone/_fpall variants, as the roster does
            continue
        m = json.load(open(p))
        if "over_refusal" not in m.get("axes", []):
            continue
        base = m.get("baseline_over_refusal")
        if base is None:
            continue
        rows.append(dict(model=model, benchmark=bench, base=base,
                         cone=m.get("over_refusal_cone"),
                         fingerprint=m.get("over_refusal_fingerprint"),
                         random=m.get("over_refusal_random"),
                         nat_base=(m.get("baseline_native") or {}).get("xstest_refusal"),
                         nat_cone=native(m, "cone"),
                         nat_fingerprint=native(m, "fingerprint"),
                         nat_random=native(m, "random")))
    df = pd.DataFrame(rows).dropna(subset=["base"])
    if not len(df):
        print("no over-refusal cells")
        return

    for a in ("cone", "fingerprint", "random"):
        df[f"d_{a}"] = df[a] - df["base"]      # negative = LESS over-refusal after ablation

    print(f"OVER-REFUSAL AXIS -- {len(df)} cells, {df.model.nunique()} models, "
          f"benchmarks={sorted(df.benchmark.unique())}\n")
    print(f"{'model':<22}{'bench':<14}{'base':>8}{'cone':>9}{'fp':>9}{'rnd':>9}"
          f"{'  d_cone':>10}{'  d_fp':>9}{'  d_rnd':>9}")
    print("-" * 99)
    f = lambda v: "  nan" if v is None or v != v else f"{100*v:5.1f}"
    for _, r in df.sort_values(["benchmark", "model"]).iterrows():
        print(f"{r.model:<22}{r.benchmark:<14}{f(r.base):>8}{f(r.cone):>9}{f(r.fingerprint):>9}"
              f"{f(r['random']):>9}{f(r.d_cone):>10}{f(r.d_fingerprint):>9}{f(r.d_random):>9}")

    x = df[df.benchmark == "xstest"]
    print("\n" + "=" * 99)
    print(f"XSTEST SUMMARY (n={len(x)} models).  Delta = pp change in over-refusal vs unsteered.")
    print("=" * 99)
    for a in ("cone", "fingerprint", "random"):
        d = (100 * x[f"d_{a}"]).dropna()
        print(f"  {a:<14} mean {d.mean():+7.2f} pp   median {d.median():+7.2f}   SD {d.std():6.2f}   "
              f"n={len(d)}   models reduced: {(d < 0).sum()}/{len(d)}")

    fp = (100 * x.d_fingerprint).dropna()
    rn = (100 * x.d_random).dropna()
    both = x.dropna(subset=["d_fingerprint", "d_random"])
    exc = 100 * (both.d_fingerprint - both.d_random)
    print(f"\n  fingerprint MINUS random (the dissociation quantity):")
    print(f"     mean {exc.mean():+.2f} pp   median {exc.median():+.2f}   SD {exc.std():.2f}   n={len(exc)}")
    # --- native 3-way classifier: the scorer to believe ---
    nx = x.dropna(subset=["nat_base"])
    if len(nx):
        print("\n" + "=" * 99)
        print(f"SAME TEST under the NATIVE 3-way XSTest classifier (n={len(nx)} models)")
        print("=" * 99)
        print(f"  baseline over-refusal: {100*nx.nat_base.mean():.2f}%")
        for a in ("cone", "fingerprint", "random"):
            s = nx[[f"nat_{a}", "nat_base"]].dropna()
            d = 100 * (s[f"nat_{a}"] - s.nat_base)
            print(f"  {a:<14} mean {d.mean():+7.2f} pp   median {d.median():+7.2f}   SD {d.std():6.2f}   "
                  f"n={len(d)}   increased >1pp: {(d > 1).sum()}")
        s = nx[["nat_fingerprint", "nat_random"]].dropna()
        e = 100 * (s.nat_fingerprint - s.nat_random)
        print(f"\n  fingerprint MINUS random: mean {e.mean():+.2f} pp   median {e.median():+.2f}   "
              f"n={len(e)}   models above +5pp: {(e > 5).sum()}")

    try:
        from scipy.stats import wilcoxon
        for label, a, b in (("cone vs fingerprint", both.d_cone, both.d_fingerprint),
                            ("fingerprint vs random", both.d_fingerprint, both.d_random)):
            s = pd.DataFrame({"a": a, "b": b}).dropna()
            if len(s) > 5 and (s.a != s.b).any():
                w, p = wilcoxon(s.a, s.b)
                print(f"     Wilcoxon {label:<24} W={w:.0f}  p={p:.3g}  n={len(s)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
