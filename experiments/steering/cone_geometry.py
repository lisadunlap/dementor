#!/usr/bin/env python3
"""Fingerprint-vs-refusal-cone geometry for every model with both artifacts.

`roster_geometry.csv` records cos(fingerprint, refusal) against a single refusal
direction and covers 18 models -- notably excluding gpt-oss-20b, gpt-oss-120b,
qwen3.6-27b, gemma-4-e4b, aya-expanse-8b and phi-4, which are exactly the models the
paper makes coupling claims about. This script closes that gap by measuring the
fingerprint against the object the intervention actually ablates: the k-dimensional
refusal cone.

Metric. For a cone with orthonormal basis B (k x d) and a fingerprint direction f,

    align(f) = ||B B^T f|| / ||f||   in [0, 1]

is the cosine of the angle between f and its projection onto the cone subspace, i.e.
the square root of the fraction of f's energy that ablating the cone would remove.
align = 0 means the cone ablation does not touch the fingerprint at all; align = 1
means the fingerprint lies entirely inside the ablated subspace.

Two reference floors are reported, because align is not comparable across models with
different cone dimensions:
  * empirical: the same quantity computed for the seeded random control direction.
  * analytic:  sqrt(k/d), the expectation for a direction drawn uniformly at random.

The cone is applied at every layer (all-layer subspace ablation), so evaluating the
fingerprint from each of its derivation layers against the same basis is well defined.

Usage:
    python experiments/steering/cone_geometry.py [--rdo-dir PATH] [--csv OUT]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import torch

DEFAULT_RDO = os.environ.get(
    "DEMENTOR_RDO_DIR", "/data/ethantsliu/exp_steer_safety/repl80_rdo"
)


def alignment(vec: torch.Tensor, basis: torch.Tensor) -> float:
    """||B B^T v|| / ||v|| with B orthonormalised. Returns a value in [0, 1]."""
    v = vec.float()
    nv = v.norm()
    if nv == 0:
        return float("nan")
    # Orthonormalise the cone basis; the stored basis is not guaranteed orthonormal.
    q, _ = torch.linalg.qr(basis.float().T)  # (d, k)
    proj = q @ (q.T @ v)
    return float(proj.norm() / nv)


def collect(rdo_dir: str) -> list[dict]:
    rows: list[dict] = []
    for slug in sorted(os.listdir(rdo_dir)):
        d = os.path.join(rdo_dir, slug)
        if not os.path.isdir(d) or slug.startswith("_"):
            continue
        # Retry/dim variants duplicate a base model; skip so each model appears once.
        if "_retry" in slug or "_dim" in slug:
            continue
        vp, cp = os.path.join(d, "vectors_ml.pt"), os.path.join(d, "selected_cone.pt")
        if not (os.path.exists(vp) and os.path.exists(cp)):
            continue
        try:
            vec = torch.load(vp, map_location="cpu", weights_only=False)
            cone = torch.load(cp, map_location="cpu", weights_only=False)
        except Exception as exc:  # a corrupt artifact should not kill the sweep
            print(f"  !! {slug}: unreadable ({exc})", file=sys.stderr)
            continue
        basis = cone["basis"]
        k, dim = basis.shape
        analytic = math.sqrt(k / dim)
        for layer, arms in sorted(vec["vectors"].items()):
            fp, rnd = arms.get("fingerprint"), arms.get("random")
            if fp is None:
                continue
            if fp.shape[-1] != dim:
                print(f"  !! {slug} L{layer}: dim mismatch fp={tuple(fp.shape)} cone d={dim}",
                      file=sys.stderr)
                continue
            rows.append({
                "model": slug,
                "layer": layer,
                "cone_dim": k,
                "hidden_dim": dim,
                "cone_best_layer": cone.get("best_layer"),
                "fp_cone_align": round(alignment(fp, basis), 6),
                "random_cone_align": round(alignment(rnd, basis), 6) if rnd is not None else "",
                "analytic_floor": round(analytic, 6),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rdo-dir", default=DEFAULT_RDO)
    ap.add_argument("--csv", default=None, help="write per-(model, layer) rows here")
    args = ap.parse_args()

    rows = collect(args.rdo_dir)
    if not rows:
        sys.exit(f"no models with both vectors_ml.pt and selected_cone.pt under {args.rdo_dir}")

    models = sorted({r["model"] for r in rows})
    print(f"{len(models)} models, {len(rows)} (model, layer) cells\n")
    print(f"{'model':24s} {'k':>2s} {'d':>5s} {'fp/cone':>8s} {'rand':>7s} {'floor':>7s}  {'ratio':>6s}")
    summary = []
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        fp = sum(r["fp_cone_align"] for r in rs) / len(rs)
        rnd_vals = [r["random_cone_align"] for r in rs if r["random_cone_align"] != ""]
        rnd = sum(rnd_vals) / len(rnd_vals) if rnd_vals else float("nan")
        fl = rs[0]["analytic_floor"]
        ratio = fp / rnd if rnd and rnd == rnd and rnd > 0 else float("nan")
        summary.append((m, rs[0]["cone_dim"], rs[0]["hidden_dim"], fp, rnd, fl, ratio))
        print(f"{m:24s} {rs[0]['cone_dim']:2d} {rs[0]['hidden_dim']:5d} "
              f"{fp:8.4f} {rnd:7.4f} {fl:7.4f}  {ratio:6.2f}")

    fps = [s[3] for s in summary]
    rnds = [s[4] for s in summary if s[4] == s[4]]
    print(f"\npooled fingerprint/cone alignment: {sum(fps)/len(fps):.4f}")
    print(f"pooled random/cone   alignment: {sum(rnds)/len(rnds):.4f}")
    print(f"range across models: {min(fps):.4f} to {max(fps):.4f}")
    over = [s[0] for s in summary if s[4] == s[4] and s[3] > 2 * s[4]]
    print(f"models where fingerprint exceeds 2x its random floor: {over or 'none'}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
