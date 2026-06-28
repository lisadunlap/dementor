"""Move 2a mechanism probe: run the activation bridge (fixed-encoder, CPU) over the
36 de-confounded decontam cells and ask the KEY question:

  Post-DPO, do the SURVIVOR sources (gpt-oss, nemotron) stay linearly decodable
  from their TARGET across the encoder's layers, while the LAUNDERERS (qwen, llama)
  collapse toward the target?

The bridge encodes Prompt+Response with a fixed small encoder (intfloat/e5-small-v2)
at every hidden-state layer, trains a source-vs-target logistic probe per layer, and
scores the disguised (post-DPO) outputs against it. We read two quantities per layer:

  * probe_cv          -- cross-validated SOURCE-vs-TARGET accuracy. This is the
                         separability *gate*: if the encoder cannot tell source from
                         target apart (probe_cv ~ 0.5), there is no axis to steer.
  * source_residue    -- fraction of disguised (post-DPO) outputs the probe still
                         classifies as SOURCE. High => the post-DPO output still
                         lands on the source side of the source/target boundary
                         (a decodable residual to steer); ~0 => laundered into the
                         target cluster.

CPU ONLY. We default device='cpu' and additionally disable MPS via env so no GPU
(CUDA or Apple MPS) is touched, regardless of what torch reports as available.

Outputs (under --out-root, default data/results/bridge_decontam):
  per_cell_layer_curve.csv   -- one row per (cell, layer): probe_cv, source_residue,
                                target_assimilation, mean_source_prob, ...
  per_source_layer.csv       -- per (tier, source, layer) mean probe_cv / source_residue
                                across that source's cells (+/- sd, n)
  per_source_summary.csv     -- per source: deep-layer separability & residue, verdict
  verdict.json               -- machine-readable GO/NO-GO for steering
  cells/<dataset>/<pair>/    -- raw bridge artifacts per cell (activations, curve, summary)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Belt-and-suspenders: keep everything on CPU. torch.cuda is already unavailable on
# this box, but MPS *is* available; the bridge only ever selects cuda-or-cpu, so it
# will pick CPU, and we additionally hide MPS so nothing can drift onto the GPU.
os.environ.setdefault("PYTORCH_MPS_DISABLE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dementor.steering.activation_bridge import run_activation_bridge  # noqa: E402

from dementor.config import project_root as _pr
REPO = _pr()
DECONTAM = REPO / "data" / "results" / "decontam"

# Source-model tiers from docs/strategy.md (verified, seed-stable):
#   SURVIVORS keep a robust DPO residue; LAUNDERERS collapse to the floor.
TIER = {
    "gpt-oss-20b": "survivor",
    "nemotron-nano-30b-a3b": "survivor",
    "qwen3.6-27b": "launderer",
    "llama-3.1-8b": "launderer",
}


def discover_cells() -> list[dict]:
    cells = []
    for csv in sorted(DECONTAM.glob("*/*/staged/dpo.csv")):
        # .../decontam/<dataset>/<source>_to_<target>/staged/dpo.csv
        pair_dir = csv.parent.parent
        dataset = pair_dir.parent.name
        pair = pair_dir.name
        if "_to_" not in pair:
            continue
        source, target = pair.split("_to_", 1)
        cells.append(
            {
                "dataset": dataset,
                "pair": pair,
                "source": source,
                "target": target,
                "tier": TIER.get(source, "unknown"),
                "csv": str(csv),
            }
        )
    return cells


def run_cell(cell: dict, out_root: Path, *, layers: str, max_rows: int | None, seed: int) -> pd.DataFrame:
    out_dir = out_root / "cells" / cell["dataset"] / cell["pair"]
    summary = run_activation_bridge(
        comparison_csv=cell["csv"],
        output_dir=out_dir,
        mode="fixed-encoder",
        encoder_model="intfloat/e5-small-v2",
        disguised_col="model_response",
        layers=layers,
        batch_size=16,
        # e5-small-v2 is a BERT encoder capped at 512 position embeddings; the bridge
        # default of 1024 over-runs it and crashes. 512 is the encoder's hard limit.
        max_length=512,
        max_rows=max_rows,
        seed=seed,
        device="cpu",  # force CPU; never cuda/mps
    )
    curve = pd.read_csv(out_dir / "activation_layer_curve.csv")
    curve.insert(0, "dataset", cell["dataset"])
    curve.insert(1, "source", cell["source"])
    curve.insert(2, "target", cell["target"])
    curve.insert(3, "tier", cell["tier"])
    curve["n"] = summary["n"]
    return curve


def summarize(per_cell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Layer indices are shared across cells (same encoder + layers spec). Express each
    # layer as a depth fraction so "deep" is unambiguous.
    layers_sorted = sorted(per_cell["layer"].unique())
    max_layer = max(layers_sorted)
    deep_layers = [lyr for lyr in layers_sorted if lyr >= 0.5 * max_layer]  # back half = "deep"

    # Per (tier, source, layer): mean separability and disguised residue across cells.
    per_source_layer = (
        per_cell.groupby(["tier", "source", "layer"], as_index=False)
        .agg(
            probe_cv_mean=("probe_cv", "mean"),
            probe_cv_sd=("probe_cv", "std"),
            source_residue_mean=("source_residue", "mean"),
            source_residue_sd=("source_residue", "std"),
            target_assimilation_mean=("target_assimilation", "mean"),
            mean_source_prob_mean=("mean_source_prob", "mean"),
            n_cells=("probe_cv", "size"),
        )
        .sort_values(["tier", "source", "layer"])
    )

    # Per source: aggregates at (a) the best-separating layer -- the correct read for
    # a BERT encoder, whose source/target axis is sharpest in the MIDDLE, not the back
    # half -- and (b) the deep (back-half) layers, kept for transparency. The
    # steering-relevant signal is source_residue: the fraction of post-DPO outputs that
    # still land on the SOURCE side of the source/target boundary.
    rows = []
    for source, grp in per_cell.groupby("source"):
        tier = grp["tier"].iloc[0]
        deep = grp[grp["layer"].isin(deep_layers)]
        by_layer = grp.groupby("layer").agg(
            probe_cv=("probe_cv", "mean"),
            source_residue=("source_residue", "mean"),
            target_assimilation=("target_assimilation", "mean"),
        )
        best_layer = int(by_layer["probe_cv"].idxmax())            # axis is sharpest here
        max_res_layer = int(by_layer["source_residue"].idxmax())   # residue is largest here
        rows.append(
            {
                "tier": tier,
                "source": source,
                "n_cells": int(grp["pair"].nunique()),
                # best-separating layer (axis-exists read)
                "best_sep_layer": best_layer,
                "best_sep": float(by_layer.loc[best_layer, "probe_cv"]),
                "residue_at_best_sep": float(by_layer.loc[best_layer, "source_residue"]),
                "target_assim_at_best_sep": float(by_layer.loc[best_layer, "target_assimilation"]),
                # maximum residue across layers (best-case steerable signal)
                "max_residue": float(by_layer.loc[max_res_layer, "source_residue"]),
                "max_residue_layer": max_res_layer,
                # whole-curve residue (how source-like the post-DPO output is on average)
                "residue_curve_mean": float(grp.groupby("layer")["source_residue"].mean().mean()),
                # deep / back-half (kept for transparency; misleading for BERT)
                "sep_deep_mean": float(deep["probe_cv"].mean()),
                "residue_deep_mean": float(deep["source_residue"].mean()),
                "target_assim_deep_mean": float(deep["target_assimilation"].mean()),
            }
        )
    per_source = pd.DataFrame(rows).sort_values(["tier", "source"]).reset_index(drop=True)

    # ---- GO/NO-GO verdict --------------------------------------------------------
    # Steering needs a DECODABLE RESIDUAL: a direction that (1) exists (the source/target
    # axis is learnable -> probe_cv > chance at its best layer) and (2) the post-DPO
    # output has NOT fully crossed (source_residue stays well above the launderer floor).
    #
    # Note the asymmetry the single-cell validation revealed: LAUNDERERS have a *perfectly*
    # decodable axis (probe_cv ~ 1.0) yet residue ~ 0 -- their post-DPO output moves ALL the
    # way onto the target. SURVIVORS have a WEAKER axis (their source/target outputs are
    # intrinsically more alike to e5) yet retain substantial residue. So separability alone
    # is NOT the steering signal; the decisive quantity is the residue CONTRAST. We gate on:
    #   (1) survivor axis exists at its best layer (best_sep > 0.55, modest because the
    #       survivor source/target pair is intrinsically similar);
    #   (2) survivor post-DPO residue is decodable (residue_curve_mean >= 0.30);
    #   (3) survivors carry MORE residue than launderers (gap >= 0.15) -- the existence
    #       claim that there is something to steer that DPO did not erase.
    surv = per_source[per_source["tier"] == "survivor"]
    laun = per_source[per_source["tier"] == "launderer"]

    surv_sep_best = float(surv["best_sep"].mean()) if len(surv) else float("nan")
    surv_res = float(surv["residue_curve_mean"].mean()) if len(surv) else float("nan")
    surv_res_max = float(surv["max_residue"].mean()) if len(surv) else float("nan")
    laun_sep_best = float(laun["best_sep"].mean()) if len(laun) else float("nan")
    laun_res = float(laun["residue_curve_mean"].mean()) if len(laun) else float("nan")
    residue_gap = surv_res - laun_res

    axis_ok = bool(surv_sep_best >= 0.55)      # survivor source/target axis is learnable
    residue_ok = bool(surv_res >= 0.30)        # post-DPO survivor residue is decodable
    contrast_ok = bool(residue_gap >= 0.15)    # survivors retain more residue than launderers

    if axis_ok and residue_ok and contrast_ok:
        verdict = "GO"
    elif residue_ok and contrast_ok:
        verdict = "WEAK-GO"
    else:
        verdict = "NO-GO"

    verdict_obj = {
        "verdict": verdict,
        "encoder": "intfloat/e5-small-v2",
        "mode": "fixed-encoder",
        "device": "cpu",
        "n_layers": int(len(layers_sorted)),
        "deep_layers": [int(x) for x in deep_layers],
        "survivor_best_sep_mean": surv_sep_best,
        "survivor_residue_curve_mean": surv_res,
        "survivor_max_residue_mean": surv_res_max,
        "launderer_best_sep_mean": laun_sep_best,
        "launderer_residue_curve_mean": laun_res,
        "survivor_minus_launderer_residue": residue_gap,
        "gates": {
            "survivor_axis_exists(best_sep>=0.55)": axis_ok,
            "survivor_residue_decodable(curve_mean>=0.30)": residue_ok,
            "survivor>launderer_residue(gap>=0.15)": contrast_ok,
        },
        "per_source": per_source.to_dict(orient="records"),
        "interpretation_note": (
            "fixed-encoder reads a small encoder's representation of the OUTPUT TEXT, "
            "not the source model's own residual stream; this is a strong embedding "
            "probe, not true internals. A decodable residual here justifies (does not "
            "prove) a steerable direction in native internals. Key asymmetry: launderers "
            "have a perfectly decodable source/target axis but the post-DPO output sits "
            "on the TARGET side (residue ~ 0); survivors retain source residue."
        ),
    }
    return per_source_layer, per_source, verdict_obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=str(REPO / "data" / "results" / "bridge_decontam"))
    ap.add_argument("--layers", default="all", help="'all' (default) for full layerwise curve, or comma-separated indices")
    ap.add_argument("--max-rows", type=int, default=None, help="cap rows per cell (validation)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validate-cell", default=None,
                    help="run ONE cell only, e.g. 'gsm8k/nemotron-nano-30b-a3b_to_gpt-oss-20b'")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cells (debug)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cells = discover_cells()
    if args.validate_cell:
        cells = [c for c in cells if f"{c['dataset']}/{c['pair']}" == args.validate_cell]
        if not cells:
            raise SystemExit(f"no cell matched {args.validate_cell!r}")
    if args.limit:
        cells = cells[: args.limit]

    print(f"[bridge] {len(cells)} cell(s); encoder=intfloat/e5-small-v2; device=cpu; layers={args.layers}", flush=True)

    all_curves = []
    for i, cell in enumerate(cells, 1):
        print(f"[bridge] {i}/{len(cells)} {cell['dataset']}/{cell['pair']} (tier={cell['tier']})", flush=True)
        curve = run_cell(cell, out_root, layers=args.layers, max_rows=args.max_rows, seed=args.seed)
        all_curves.append(curve)

    per_cell = pd.concat(all_curves, ignore_index=True)
    # attach pair for counting
    per_cell["pair"] = per_cell["source"] + "_to_" + per_cell["target"]
    per_cell.to_csv(out_root / "per_cell_layer_curve.csv", index=False)

    if args.validate_cell or (args.limit and args.limit < 4):
        # validation: just show the curve, skip cross-source aggregation
        print("\n[validate] layerwise curve for", f"{cells[0]['dataset']}/{cells[0]['pair']}")
        cols = ["layer", "probe_cv", "source_residue", "target_assimilation", "mean_source_prob", "mean_target_prob"]
        print(per_cell[cols].to_string(index=False))
        return

    per_source_layer, per_source, verdict = summarize(per_cell)
    per_source_layer.to_csv(out_root / "per_source_layer.csv", index=False)
    per_source.to_csv(out_root / "per_source_summary.csv", index=False)
    (out_root / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print("\n================ PER-SOURCE SUMMARY (deep layers) ================")
    print(per_source.to_string(index=False))
    print("\n================ VERDICT ================")
    print(json.dumps({k: v for k, v in verdict.items() if k not in ("per_source",)}, indent=2))


if __name__ == "__main__":
    main()
