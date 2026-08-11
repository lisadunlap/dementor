#!/usr/bin/env python
"""Rebuild the four steering figures (01-04) from committed artifacts.

The shipped versions of these four PDFs were drawn from numbers that appear in no
artifact -- match rates that are never measured under steering, a baseline refusal
of 87.3% that exists in none of the 149 metrics.json files, an angle axis spanning
30-90 deg when the measured geometry spans 81.8-106.1 deg.  This script redraws all
four from the pipeline's own output, so every value is traceable.

Sources (read-only):
  /data/ethantsliu/exp_steer_safety/repl80_rdo/<slug>/eval_<bm>/metrics_n200.json
  (accepted only for complete 200-prompt coverage; otherwise falling back to native metrics.json)
  /data/ethantsliu/exp_steer_safety/repl80_rdo/roster_geometry.csv

Roster rule (stated in the paper's Methods):
  a model enters the dissociation roster iff its positive control fires on at least
  one harm benchmark AND its random-direction arm is clean on every harm benchmark
  it was run on (clean = |random - baseline| <= RANDOM_CONTAM_PP).  Per-model effects
  average only over the benchmarks where that model's control fires.

Usage:
  python3 experiments/figures/rebuild_steering_figures.py --outdir DIR
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from dementor import config

RDO = os.environ.get("DEMENTOR_STEER_WORK", "/data/ethantsliu/exp_steer_safety/repl80_rdo")
HARM_BENCHMARKS = ["advbench", "harmbench", "strongreject", "sorrybench", "sgbench"]
BM_LABEL = {
    "advbench": "AdvBench",
    "harmbench": "HarmBench",
    "strongreject": "StrongREJECT",
    "sorrybench": "SORRY-Bench",
    "sgbench": "SG-Bench",
    "xstest": "XSTest",
}
RANDOM_CONTAM_PP = 10.0
SKIP_DIRS = {"port", "wandb", "rdo_shared", "roster_geom_parts", "__pycache__", "_validation"}
VARIANT_SUFFIXES = ("_retry5", "_dim8", "_k8")
POSITIVE_CONTROL_PASS = {"CLEAN", "INCONCLUSIVE"}

# accent colours shared with plot_erosion_paper.py
C_FP, C_RAND, C_CONE = "#2471a3", "#95a5a6", "#c0392b"

matplotlib.rcParams.update({
    "savefig.bbox": "standard",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _native_prompt_count(eval_dir):
    """Return the native evaluation denominator from baseline generation rows."""
    generations = os.path.join(eval_dir, "all_gens.csv")
    try:
        with open(generations, newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            return sum(row.get("direction") == "baseline" for row in rows)
    except (OSError, csv.Error):
        return None


def _load_preferred_metrics(eval_dir):
    """Load one cell and retain the provenance actually encoded by its metrics."""
    harmonized = os.path.join(eval_dir, "metrics_n200.json")
    native = os.path.join(eval_dir, "metrics.json")
    metrics = {}
    metrics_source = None
    if os.path.exists(native):
        try:
            metrics = json.load(open(native))
            metrics_source = "metrics.json"
        except (json.JSONDecodeError, OSError):
            metrics = {}
    if os.path.exists(harmonized):
        try:
            rescored = json.load(open(harmonized))
            # An early harmonizer wrote prompt intersections even when the legacy generation did
            # not contain the entire seed-42 set. Those files are not n=200 despite their name.
            # Only overlay a complete 200-prompt rescore; otherwise retain native cell metrics.
            if rescored.get("n_prompts") == 200:
                metrics.update(rescored)
                metrics_source = "metrics_n200.json"
        except (json.JSONDecodeError, OSError):
            pass
    if not metrics:
        return None
    if metrics_source == "metrics_n200.json":
        n_prompts = 200
        sampling = "harmonized_n200"
        # Older rescores do not encode a prompt hash or seed. Do not infer seed 42 from n=200.
        subsample_seed = metrics.get("subsample_seed")
    else:
        n_prompts = metrics.get("n_prompts") or _native_prompt_count(eval_dir)
        sampling = "native"
        subsample_seed = metrics.get("subsample_seed")
    metrics["_evaluation_provenance"] = {
        "metrics_source": metrics_source,
        "sampling": sampling,
        "n_prompts": n_prompts,
        "subsample_seed": subsample_seed,
    }
    return metrics


def load_cells(variant=None):
    """-> {slug: {benchmark: dict(base, cone, fp, rand, verdict)}}, in pp.

    variant=None     campaign cells (fingerprint/random ablated at ABLATE_LAYER)
    variant="fpall"  all-layers variant (single directions ablated with the cone's
                     operator at every layer); reads eval_<bm>_fpall/ only.
    """
    out = defaultdict(dict)
    for d in sorted(os.listdir(RDO)):
        p = os.path.join(RDO, d)
        if not os.path.isdir(p) or d in SKIP_DIRS:
            continue
        if d.endswith(VARIANT_SUFFIXES):
            continue  # re-run variants, not roster members
        # Some models carry an older bare eval/metrics.json alongside the per-benchmark
        # eval_<bm>/ dirs; they are separate runs of the same benchmark with slightly
        # different values.  eval_<bm>/ is canonical (it is what the paper quotes), so
        # read the legacy file first and let the per-benchmark file overwrite it.
        # eval_<bm>_fpall/ holds the all-layers single-direction variant, a SEPARATE
        # experiment. It must never be mixed into the campaign cells: the glob below
        # would match it, and because "eval_sgbench_fpall" sorts after "eval_sgbench"
        # it would silently overwrite the campaign value for that benchmark (both
        # write benchmark="sgbench"). Load it explicitly via load_cells(variant=...).
        suffix = "_fpall" if variant == "fpall" else None
        expected_dirs = {
            f"eval_{benchmark}{suffix or ''}"
            for benchmark in TABLE_BENCHMARKS
        }
        eval_dirs = [path for path in sorted(glob.glob(os.path.join(p, "eval_*")))
                     if os.path.basename(path) in expected_dirs]
        metrics = []
        if not suffix:
            legacy = _load_preferred_metrics(os.path.join(p, "eval"))
            if legacy:
                metrics.append(("advbench", legacy))
        metrics.extend((os.path.basename(path)[len("eval_"):-len(suffix)] if suffix
                        else os.path.basename(path)[len("eval_"):], loaded)
                       for path in eval_dirs
                       if (loaded := _load_preferred_metrics(path)))
        for inferred_benchmark, m in metrics:
            bm, base = m.get("benchmark") or inferred_benchmark, m.get("baseline_harm")
            if bm is None or base is None:
                continue
            br = m.get("baseline_refrate")
            if br is not None and not np.isfinite(br):
                br = None
            provenance = m.get("_evaluation_provenance", {})
            cell = {"verdict": m.get("verdict"), "base": 100 * base,
                    "base_ref": None if br is None else 100 * br,
                    "n_prompts": provenance.get("n_prompts"),
                    "sampling": provenance.get("sampling"),
                    "subsample_seed": provenance.get("subsample_seed"),
                    "metrics_source": provenance.get("metrics_source")}
            for key, arm in (("refusal_matched", "cone"),
                             ("fingerprint_matched", "fp"),
                             ("random_matched", "rand")):
                v = m.get(key)
                cell[arm] = (None if v is None or not np.isfinite(v)
                             else 100 * (v - base))
            out[config.canonical_steering_slug(d)][bm] = cell
    return out


def roster(cells):
    """Apply the documented rule; return sorted slugs."""
    keep = []
    for slug, bms in cells.items():
        harm = {b: c for b, c in bms.items() if b in HARM_BENCHMARKS}
        if not harm:
            continue
        if not any(c["verdict"] in POSITIVE_CONTROL_PASS for c in harm.values()):
            continue
        if any(c["rand"] is not None and abs(c["rand"]) > RANDOM_CONTAM_PP
               for c in harm.values()):
            continue
        keep.append(slug)
    return sorted(keep)


def paired_signed_rank_exact(x, y):
    """Two-sided paired Wilcoxon test via exact sign permutations.

    Average ranks are represented as doubled integers, retaining ties exactly.
    Zero differences use the standard ``wilcox`` convention and are dropped.
    """
    from scipy.stats import rankdata

    differences = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    differences = differences[differences != 0]
    if not len(differences):
        raise ValueError("all paired differences are zero")
    ranks2 = np.rint(2 * rankdata(np.abs(differences), method="average")).astype(int)
    positive2 = int(ranks2[differences > 0].sum())
    total2 = int(ranks2.sum())
    observed2 = min(positive2, total2 - positive2)

    counts = {0: 1}
    for rank2 in ranks2:
        updated = dict(counts)
        for subtotal, count in counts.items():
            value = subtotal + int(rank2)
            updated[value] = updated.get(value, 0) + count
        counts = updated
    extreme = sum(
        count
        for subtotal, count in counts.items()
        if subtotal <= observed2 or subtotal >= total2 - observed2
    )
    return {
        "n_models": int(len(differences)),
        "statistic": observed2 / 2,
        "p_value": min(1.0, extreme / (2 ** len(differences))),
        "method": "exact paired sign permutation with average ranks; zero_method=wilcox",
    }


def per_model(cells, slugs):
    """Mean delta per arm over the benchmarks where that model's control fires."""
    rows = {}
    for s in slugs:
        firing = [c for b, c in cells[s].items()
                  if b in HARM_BENCHMARKS and c["verdict"] in POSITIVE_CONTROL_PASS]
        if not firing:
            continue
        rows[s] = {arm: float(np.mean([c[arm] for c in firing if c[arm] is not None]))
                   for arm in ("cone", "fp", "rand")}
        rows[s]["n"] = len(firing)
    return rows


def fig03(rows, outdir):
    """Per-model cross-model-contrast vs random vs cone harm deltas, sorted by cone effect."""
    slugs = sorted(rows, key=lambda s: -rows[s]["cone"])
    y = np.arange(len(slugs))
    h = 0.27
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(7.0, 0.26 * len(slugs) + 1.5),
        gridspec_kw={"width_ratios": [1.9, 1.0], "wspace": 0.05}, sharey=True)

    # (a) shows all three arms; (b) re-plots ONLY the two arms under comparison at a
    # readable scale -- including the cone there would overflow the zoomed axis.
    axl.barh(y + h, [rows[s]["cone"] for s in slugs], h, color=C_CONE, label="refusal cone")
    axl.barh(y, [rows[s]["fp"] for s in slugs], h, color=C_FP, label="cross-model contrast")
    axl.barh(y - h, [rows[s]["rand"] for s in slugs], h, color=C_RAND, label="random control")
    axr.barh(y + h / 2, [rows[s]["fp"] for s in slugs], h, color=C_FP, label="cross-model contrast")
    axr.barh(y - h / 2, [rows[s]["rand"] for s in slugs], h, color=C_RAND, label="random control")
    for ax in (axl, axr):
        ax.axvline(0, color="k", lw=0.8)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)

    axl.set_yticks(y)
    axl.set_yticklabels([f"{s} ({rows[s]['n']})" for s in slugs])
    axl.invert_yaxis()
    axl.set_xlabel("harm change under ablation (pp)")
    axl.set_title("(a) all three arms", loc="left")
    axl.legend(loc="lower right", frameon=False, fontsize=9)

    lim = max(abs(rows[s][a]) for s in slugs for a in ("fp", "rand")) * 1.15
    axr.set_xlim(-lim, lim)
    axr.set_xlabel("harm change (pp), zoomed")
    axr.set_title("(b) contrast vs random only", loc="left")
    axr.tick_params(labelleft=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        # Long model slugs extend beyond the nominal canvas even after tight_layout.
        # Include all artists in the saved bounding box so the left half of labels is not clipped.
        fig.savefig(
            os.path.join(outdir, f"03_steering_effects.{ext}"),
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    plt.close(fig)
    return slugs


def fig04(cells, outdir, slug="llama-3.1-8b", bm="advbench"):
    """Positive control: the real llama-3.1-8b AdvBench numbers."""
    c = cells[slug][bm]
    base = c["base"]
    vals = [base, base + c["fp"], base + c["rand"], base + c["cone"]]
    labels = ["baseline", "cross-model\ncontrast", "random\ncontrol", "refusal-cone\nablation"]
    colors = ["#7f8c8d", C_FP, C_RAND, C_CONE]

    fig, ax = plt.subplots(figsize=(4.20, 2.75))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.annotate("", xy=(3, vals[3] - 3), xytext=(0, base + 3),
                arrowprops=dict(arrowstyle="->", color=C_CONE, lw=1.2))
    ax.text(1.5, (base + vals[3]) / 2 + 6, f"{c['cone']:+.1f} pp",
            color=C_CONE, ha="center", fontsize=9.5)
    ax.set_ylabel("harm rate (%)")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_title(f"{slug}, {BM_LABEL[bm]}", loc="left")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"04_positive_control.{ext}"), dpi=200)
    plt.close(fig)
    return base, c


def fig01(cells, slugs, outdir):
    """Per-benchmark dissociation, restricted to models whose control fires there."""
    stats = {}
    for bm in HARM_BENCHMARKS:
        acc = {a: [] for a in ("cone", "fp", "rand")}
        for s in slugs:
            c = cells[s].get(bm)
            if c is None or c["verdict"] not in POSITIVE_CONTROL_PASS:
                continue
            for a in acc:
                if c[a] is not None:
                    acc[a].append(c[a])
        if acc["cone"]:
            stats[bm] = {a: (float(np.mean(v)), float(np.std(v)), len(v))
                         for a, v in acc.items()}

    bms = [b for b in HARM_BENCHMARKS if b in stats]
    x = np.arange(len(bms))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for off, arm, col, lab in ((-w, "fp", C_FP, "cross-model contrast"),
                               (0.0, "rand", C_RAND, "random control"),
                               (w, "cone", C_CONE, "refusal-cone ablation")):
        m = [stats[b][arm][0] for b in bms]
        e = [stats[b][arm][1] for b in bms]
        ax.bar(x + off, m, w, yerr=e, color=col, label=lab,
               error_kw=dict(lw=0.8, capsize=2, ecolor="#444444"))
        for xi, b in zip(x + off, bms):
            ax.text(xi, -2.4, f"n={stats[b][arm][2]}", ha="center",
                    va="top", fontsize=9, color="#555555")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([BM_LABEL[b] for b in bms])
    ax.set_ylabel("harm change under ablation (pp)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"01_benchmark_dissociation.{ext}"), dpi=200)
    plt.close(fig)
    return stats


def fig02(outdir):
    """Axis separation: |cos| per model for the three pairings, vs the random floor."""
    path = os.path.join(RDO, "roster_geometry.csv")
    if not os.path.exists(path):
        print("[figs] roster_geometry.csv absent -- skipping figure 02")
        return None
    import csv
    per = defaultdict(lambda: defaultdict(list))
    rd = csv.DictReader(open(path))
    # the aggregator renamed the model column from "model" to "slug"; accept either
    keycol = "slug" if "slug" in (rd.fieldnames or []) else "model"
    for r in rd:
        m = r.get(keycol)
        if not m:
            continue
        for col in ("fp_refusal", "fp_persona", "refusal_persona", "fp_random"):
            v = r.get(col)
            if v not in (None, ""):
                try:
                    per[m][col].append(abs(float(v)))
                except ValueError:
                    pass
    models = sorted(per, key=lambda m: np.mean(per[m]["fp_refusal"]))
    pooled = {c: float(np.mean([v for m in per for v in per[m][c]]))
              for c in ("fp_refusal", "fp_persona", "refusal_persona", "fp_random")}

    x = np.arange(len(models))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for off, col, c, lab in ((-w, "fp_refusal", C_FP, "|cos(contrast, refusal)|"),
                             (0.0, "fp_persona", "#e67e22", "|cos(contrast, persona)|"),
                             (w, "refusal_persona", C_CONE, "|cos(refusal, persona)|")):
        ax.bar(x + off, [np.mean(per[m][col]) for m in models], w, color=c, label=lab)
    floor = pooled["fp_random"]
    ax.axhline(floor, ls="--", lw=1.0, color="#444444")
    ax.text(-0.4, floor, f"random floor {floor:.3f}", va="bottom", ha="left",
            fontsize=9, color="#444444",
            bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"$|\cos|$ (mean over layers 4/8/14/20)")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(
        f"pooled: fp-refusal {pooled['fp_refusal']:.3f}   "
        f"fp-persona {pooled['fp_persona']:.3f}   "
        f"refusal-persona {pooled['refusal_persona']:.3f}", loc="left", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"02_model_orthogonality.{ext}"), dpi=200)
    plt.close(fig)
    return pooled, models


TABLE_BENCHMARKS = HARM_BENCHMARKS + ["xstest", "orbench_toxic"]
LOW_COVERAGE = {"sgbench", "orbench_toxic"}


def table_rows(cells, clean_only, roster_slugs=None):
    """LaTeX body rows for Table 1.

    clean_only=False -> every model with a valid operating point on that benchmark.
    clean_only=True  -> only the cells where that benchmark's positive control fires,
                        restricted to the roster.  This is the rule stated in Methods;
                        it deliberately excludes non-firing cells from the cone mean.
    """
    out = []
    for bm in TABLE_BENCHMARKS:
        acc = {a: [] for a in ("cone", "fp", "rand")}
        base_h, base_r = [], []
        for slug, bms in cells.items():
            c = bms.get(bm)
            if c is None:
                continue
            if clean_only and (c["verdict"] not in POSITIVE_CONTROL_PASS
                               or (roster_slugs is not None and slug not in roster_slugs)):
                continue
            base_h.append(c["base"])
            if c.get("base_ref") is not None:
                base_r.append(c["base_ref"])
            for a in acc:
                if c[a] is not None:
                    acc[a].append(c[a])
        if not acc["cone"]:
            continue
        label = BM_LABEL.get(bm, bm)
        if bm == "xstest":
            label = "XSTest (harm)"
        elif bm == "orbench_toxic":
            label = "OR-Bench toxic"
        if bm in LOW_COVERAGE:
            label += r"$^{\dagger}$"
        cellsf = [f"{np.mean(base_h):.1f}",
                  f"{np.mean(base_r):.1f}" if base_r else "--"]
        for a in ("fp", "rand", "cone"):
            v = acc[a]
            cellsf += [f"${np.mean(v):+.1f}$", f"{np.std(v):.1f}", str(len(v))]
        out.append(f"{label} & " + " & ".join(cellsf) + r" \\")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    cells = load_cells()
    slugs = roster(cells)
    rows = per_model(cells, slugs)
    fp_values = [row["fp"] for row in rows.values()]
    random_values = [row["rand"] for row in rows.values()]
    fp_vs_random = paired_signed_rank_exact(fp_values, random_values)
    arm_summary = {
        "cone_mean_pp": float(np.mean([row["cone"] for row in rows.values()])),
        "fingerprint_mean_pp": float(np.mean(fp_values)),
        "random_mean_pp": float(np.mean(random_values)),
        "fingerprint_vs_random_wilcoxon": fp_vs_random,
    }
    print(f"[figs] roster n={len(slugs)} (rule: PC fires on >=1 harm bm, "
          f"random arm within {RANDOM_CONTAM_PP:.0f} pp on all)")
    print(f"[figs] {', '.join(slugs)}")
    print(f"[figs] arm means over roster: cone {np.mean([r['cone'] for r in rows.values()]):+.3f} "
          f"fp {np.mean([r['fp'] for r in rows.values()]):+.3f} "
          f"rand {np.mean([r['rand'] for r in rows.values()]):+.3f} pp")

    fig03(rows, a.outdir)
    base, c = fig04(cells, a.outdir)
    print(f"[figs] fig04 llama-3.1-8b advbench: baseline {base:.1f} -> cone {base + c['cone']:.1f} "
          f"({c['cone']:+.1f} pp), fp {c['fp']:+.1f}, rand {c['rand']:+.1f}")
    stats = fig01(cells, slugs, a.outdir)
    for bm, s in stats.items():
        print(f"[figs] fig01 {bm:13s} cone {s['cone'][0]:+6.1f} (n={s['cone'][2]:2d})  "
              f"fp {s['fp'][0]:+5.1f}  rand {s['rand'][0]:+5.1f}")
    g = fig02(a.outdir)
    if g:
        print(f"[figs] fig02 pooled {g[0]}")
    print("\n%% Table 1, all models with a valid operating point")
    for r in table_rows(cells, clean_only=False):
        print(r)
    print("\n%% Table 1, positive-control-passing cells, roster only")
    for r in table_rows(cells, clean_only=True, roster_slugs=set(slugs)):
        print(r)
    print()
    json.dump({"roster": slugs, "arm_summary": arm_summary, "per_model": rows,
               "per_benchmark": {k: {a: list(v) for a, v in s.items()}
                                 for k, s in stats.items()}},
              open(os.path.join(a.outdir, "steering_figure_stats.json"), "w"), indent=2)
    print(f"[figs] wrote -> {a.outdir}")


if __name__ == "__main__":
    main()
