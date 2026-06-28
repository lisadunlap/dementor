"""A2 (the circularity-killer): recompute baseline distinctiveness in a ZERO-MiniLM
structural feature space (32 hand-crafted length/markdown/style features) and test
whether it STILL predicts DPO persistence. If a non-embedding distinctiveness
predicts persistence, the r=0.97 cannot be a MiniLM artifact (red-team O1/O5).

Same protocol as source_fingerprint_figure.baseline_distinctiveness, but features =
_style_scalar_features + _style_binary_features (z-scored), not MiniLM.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from dementor.metric.latent_behavior_axes import _style_binary_features, _style_scalar_features

MODELS = ["llama-3.1-8b", "qwen3.6-27b", "gpt-oss-20b", "nemotron-nano-30b-a3b"]
SHORT = {"llama-3.1-8b": "llama", "qwen3.6-27b": "qwen", "gpt-oss-20b": "gpt-oss",
         "nemotron-nano-30b-a3b": "nemotron"}


def load_baselines():
    seen, texts, m, d = set(), [], [], []
    for f in sorted(glob.glob("data/results/decontam/*/*/gen/source_seed1.csv")):
        parts = Path(f).parts
        ds = parts[parts.index("decontam") + 1]
        src = parts[parts.index("decontam") + 2].split("_to_")[0]
        if (src, ds) in seen:
            continue
        seen.add((src, ds))
        for r in pd.read_csv(f)["model_response"].astype(str):
            texts.append(r); m.append(src); d.append(ds)
    return texts, np.array(m), np.array(d)


def structural_features(texts):
    s, _ = _style_scalar_features(texts)
    b, _ = _style_binary_features(texts)
    X = np.hstack([s, b]).astype(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd  # z-score so length doesn't dominate the centroid distance


def distinctiveness(X, m, d):
    acc = {mm: [] for mm in MODELS}
    for held in sorted(set(d)):  # leave-one-dataset-out (controls topic)
        tr, te = d != held, d == held
        C = np.stack([X[tr & (m == mm)].mean(0) for mm in MODELS])
        for mm in MODELS:
            xi = X[te & (m == mm)]
            if len(xi):
                pred = ((xi[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
                acc[mm].append((np.array(MODELS)[pred] == mm).mean())
    return {SHORT[mm]: float(np.mean(acc[mm])) for mm in MODELS}


def main():
    texts, m, d = load_baselines()
    distinct = structural = distinctiveness(structural_features(texts), m, d)
    dd = pd.read_csv("data/results/multiseed_ci_s3.csv")
    dpo = dd[dd["base_rung"] == "dpo"]
    persist = {SHORT[mm]: dpo[dpo["source"] == mm]["mean"].mean() for mm in MODELS}

    print("STRUCTURAL (zero-MiniLM) distinctiveness vs DPO persistence (chance = 0.25):")
    print(f"  {'model':10}{'struct-distinct':>16}{'DPO-persist':>13}")
    for s in sorted(SHORT.values(), key=lambda s: -persist[s]):
        print(f"  {s:10}{distinct[s]:>16.3f}{persist[s]:>13.3f}")
    xs = np.array([distinct[s] for s in SHORT.values()])
    ys = np.array([persist[s] for s in SHORT.values()])
    r = np.corrcoef(xs, ys)[0, 1]
    rho = pd.Series(ys).corr(pd.Series(xs), method="spearman")
    print(f"\n  Pearson r = {r:.3f} | Spearman rho = {rho:.3f}")
    print("  -> if high, the distinctiveness->persistence link is NOT a MiniLM artifact (O1 dead).")


if __name__ == "__main__":
    main()
