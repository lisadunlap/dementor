"""Figure 1 + the core argument: a model's fingerprint resists DPO in proportion to
how *distinctive* it is to begin with.

- DPO persistence per source model (de-confounded, 3-seed) from multiseed_ci_s3.csv.
- INDEPENDENT distinctiveness: how identifiable each model is from its own BASELINE
  (undisguised) outputs — a leave-one-dataset-out nearest-centroid classifier in
  MiniLM space, using ONLY baselines (never the disguise/persistence pipeline). This
  decouples "distinctive" from "persistent" so the correlation isn't circular.

Produces data/results/fig1_source_fingerprint.png (2 panels) + prints the numbers.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
MODELS = ["llama-3.1-8b", "qwen3.6-27b", "gpt-oss-20b", "nemotron-nano-30b-a3b"]
SHORT = {"llama-3.1-8b": "llama", "qwen3.6-27b": "qwen", "gpt-oss-20b": "gpt-oss",
         "nemotron-nano-30b-a3b": "nemotron"}


def dpo_persistence_by_source() -> dict[str, float]:
    d = pd.read_csv(DATA / "results" / "multiseed_ci_s3.csv")
    d = d[d["base_rung"] == "dpo"]
    return {SHORT[m]: d[d["source"] == m]["mean"].mean() for m in MODELS}


def baseline_distinctiveness() -> dict[str, float]:
    """Leave-one-dataset-out nearest-centroid accuracy per model on its baselines."""
    from sentence_transformers import SentenceTransformer
    # one baseline set per (model, dataset) — identical across targets, so dedupe.
    seen, texts, mlab, dlab = set(), [], [], []
    for f in sorted(glob.glob("data/results/decontam/*/*/gen/source_seed1.csv")):
        parts = Path(f).parts
        dataset = parts[parts.index("decontam") + 1]
        src = parts[parts.index("decontam") + 2].split("_to_")[0]
        if (src, dataset) in seen:
            continue
        seen.add((src, dataset))
        for r in pd.read_csv(f)["model_response"].astype(str):
            texts.append(r[:600]); mlab.append(src); dlab.append(dataset)
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X = enc.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    m, d = np.array(mlab), np.array(dlab)
    datasets = sorted(set(dlab))
    acc = {mm: [] for mm in MODELS}
    for held in datasets:  # train centroids on other datasets, test on held-out (controls for topic)
        tr, te = d != held, d == held
        cents = {mm: X[tr & (m == mm)].mean(0) for mm in MODELS}
        C = np.stack([cents[mm] for mm in MODELS])
        for mm in MODELS:
            xi = X[te & (m == mm)]
            if len(xi):
                pred = (xi @ C.T).argmax(1)
                acc[mm].append((np.array(MODELS)[pred] == mm).mean())
    return {SHORT[mm]: float(np.mean(acc[mm])) for mm in MODELS}


def main() -> None:
    persist = dpo_persistence_by_source()
    distinct = baseline_distinctiveness()
    order = sorted(SHORT.values(), key=lambda s: persist[s])
    print("model     baseline-distinctiveness   DPO-persistence")
    for s in sorted(SHORT.values(), key=lambda s: -persist[s]):
        print(f"  {s:10} {distinct[s]:.3f}                    {persist[s]:.3f}")
    xs = np.array([distinct[s] for s in SHORT.values()])
    ys = np.array([persist[s] for s in SHORT.values()])
    r = np.corrcoef(xs, ys)[0, 1]
    rho = pd.Series(ys).corr(pd.Series(xs), method="spearman")
    print(f"\nPearson r(distinctiveness, DPO-persistence) = {r:.3f} | Spearman = {rho:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#2ecc71" if persist[s] > 0.15 else "#bdc3c7" for s in order]
    a1.barh(order, [persist[s] for s in order], color=colors)
    a1.axvline(0.3, ls="--", c="#e74c3c", lw=1, label="survivor threshold")
    a1.set_xlabel("DPO persistence (de-confounded, 3-seed)")
    a1.set_title("Only distinctive models resist DPO")
    a1.legend(fontsize=8)
    for s in SHORT.values():
        a2.scatter(distinct[s], persist[s], s=70, c="#3498db", zorder=3)
        a2.annotate(s, (distinct[s], persist[s]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    a2.set_xlabel("baseline distinctiveness (independent)")
    a2.set_ylabel("DPO persistence")
    a2.set_title(f"distinctiveness predicts persistence\nPearson r={r:.2f}, Spearman ρ={rho:.2f}")
    plt.tight_layout()
    out = DATA / "results" / "fig1_source_fingerprint.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
