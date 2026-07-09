#!/usr/bin/env python
"""Subspace comparison between two RDO cones (their code vs the port).

Principal-angle cosines between the two k-dim subspaces = singular values of Q_a^T Q_b, where
Q_a, Q_b are orthonormal bases (QR). 1.0 = identical subspace. Also reports the projection norm
of an optional single reference direction (e.g. repl80 diff-of-means refusal dir) onto each cone
(how much of the single refusal axis the cone captures).

Usage: compare_cones.py --a <dirA/cones> --b <dirB/cones> [--refusal <safety_dirs.pt> --ref-layer 14]
"""
import os, sys, json, argparse, glob
import torch


def orthob(basis):
    B = basis.float().t()               # [hidden, k]
    Q, _ = torch.linalg.qr(B)           # [hidden, k]
    return Q


def principal_cos(Ba, Bb):
    Qa, Qb = orthob(Ba), orthob(Bb)
    M = Qa.t() @ Qb                     # [ka, kb]
    s = torch.linalg.svdvals(M)
    return s.clamp(-1, 1).tolist()


def load_cones(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "cone_dim_*.pt"))):
        k = int(os.path.basename(f).split("_")[-1].split(".")[0])
        out[k] = torch.load(f, map_location="cpu")["basis"].float()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="their"); ap.add_argument("--label-b", default="port")
    ap.add_argument("--refusal", default=None); ap.add_argument("--ref-layer", type=int, default=14)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, B = load_cones(args.a), load_cones(args.b)
    refdir = None
    if args.refusal and os.path.exists(args.refusal):
        rv = torch.load(args.refusal, map_location="cpu")["vectors"]
        if args.ref_layer in rv and "refusal" in rv[args.ref_layer]:
            refdir = rv[args.ref_layer]["refusal"].float()
            refdir = refdir / refdir.norm()

    report = {"label_a": args.label_a, "label_b": args.label_b, "per_dim": {}}
    print(f"{'dim':>4} {'principal-angle cosines (sorted desc)':45} {'mean':>6} {'min':>6}")
    print("-" * 72)
    for k in sorted(set(A) & set(B)):
        cos = sorted(principal_cos(A[k], B[k]), reverse=True)
        entry = {"principal_cos": cos, "mean": sum(cos) / len(cos), "min": min(cos)}
        if refdir is not None:
            for lab, Bk in ((args.label_a, A[k]), (args.label_b, B[k])):
                Q = orthob(Bk)
                proj = (Q.t() @ refdir)
                entry[f"refusal_capture_{lab}"] = float(proj.norm())  # in [0,1]
        report["per_dim"][k] = entry
        cs = " ".join(f"{c:.3f}" for c in cos)
        print(f"{k:>4} {cs:45} {entry['mean']:.3f} {entry['min']:.3f}")
    if refdir is not None:
        print("\nrefusal single-dir capture (||proj of diff-of-means refusal onto cone||, 1=fully in cone):")
        for k in sorted(report["per_dim"]):
            e = report["per_dim"][k]
            print(f"  dim {k}: {args.label_a}={e.get('refusal_capture_'+args.label_a):.3f}  "
                  f"{args.label_b}={e.get('refusal_capture_'+args.label_b):.3f}")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
