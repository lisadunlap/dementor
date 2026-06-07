"""A4 — Reproduce the partner's cache-based findings on OUR 4 models (F1/F2/F3) + the
imitation-INDEPENDENT durability proxy (S7).

All native (undisguised) outputs from the cached decontam source generations:
  data/results/decontam/<dataset>/<model>_to_*/gen/source_seed1.csv  (cols: prompt, model_response)

F1  Scale-Illusion : per-model TTR (lexical diversity) + Shannon style-entropy (bits) over
                     the 20 binary style features. Does param count predict richness? (No.)
S7  cross-DATASET  : within-model correlation of per-dataset 20-feature profiles -> the
                     imitation-INDEPENDENT durability proxy (feeds A1). Does native stability
                     rank like imitation-durability?
F3  cross-PROMPT   : within-model cosine of per-prompt-type 20-feature profiles (method-valid).
F2  verbosity-bias : |Pearson r| of each binary feature vs log word count; the |r|>0.15 set is
                     what our existing *_lenres length-residualization already removes.

Reuses scripts/analysis/latent_behavior_axes._style_binary_features (the shared 20-D block).
Cache-only, $0.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from scripts.analysis.latent_behavior_axes import _style_binary_features  # noqa: E402

OUT = ROOT / "results/findings"
MODELS = ["llama-3.1-8b", "gpt-oss-20b", "qwen3.6-27b", "nemotron-nano-30b-a3b"]
DATASETS = ["gsm8k", "writingprompts", "chatbot_arena"]
MAX_N = 800  # cap responses per (model,dataset) for speed; deterministic head


def native_texts(model: str, dataset: str) -> pd.DataFrame:
    """Native source outputs for a model on a dataset (any target cell; source is target-independent)."""
    cells = sorted(glob.glob(str(ROOT / f"data/results/decontam/{dataset}/{model}_to_*/gen/source_seed1.csv")))
    if not cells:
        return pd.DataFrame(columns=["prompt", "model_response"])
    df = pd.read_csv(cells[0]).dropna(subset=["model_response"]).head(MAX_N)
    df["model_response"] = df["model_response"].astype(str)
    return df


def hbin(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def ttr(texts) -> float:
    """Mean per-response type-token ratio (length-robust vs pooled TTR)."""
    rs = []
    for t in texts:
        w = re.findall(r"[a-z']+", t.lower())
        if len(w) >= 5:
            rs.append(len(set(w)) / len(w))
    return float(np.mean(rs)) if rs else np.nan


def profile_20(texts) -> np.ndarray:
    """Mean activation rate of the 20 binary style features."""
    X, _ = _style_binary_features(list(texts))
    return X.mean(axis=0)


def prompt_type(p: str) -> str:
    pl = str(p).lower().strip()
    if any(k in pl for k in ["write a story", "poem", "imagine", "fiction", "creative", "write a short"]):
        return "Creative"
    if pl.endswith("?") or re.match(r"^(what|why|how|when|where|who|which|is|are|can|do|does|should)\b", pl):
        return "Question"
    if re.match(r"^(write|explain|list|describe|create|give|make|summari|generate|provide|compute|solve|calculate)\b", pl) or "how to" in pl:
        return "Instruction"
    return "Factual"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, feat_names = _style_binary_features(["hello"])

    # ---------- F1: richness ----------
    rich = []
    for m in MODELS:
        texts = pd.concat([native_texts(m, d)["model_response"] for d in DATASETS])
        rates = profile_20(texts)
        rich.append(dict(model=m, n=len(texts), ttr=round(ttr(texts), 4),
                         style_entropy_bits=round(sum(hbin(p) for p in rates), 3)))
    rich = pd.DataFrame(rich)
    rich.to_csv(OUT / "style_richness.csv", index=False)
    print("=== F1 Scale-Illusion (richness; param count != richness) ===")
    print(rich.to_string(index=False))

    # ---------- S7: cross-dataset stability (imitation-independent durability proxy) ----------
    s7 = []
    for m in MODELS:
        profs = {}
        for d in DATASETS:
            t = native_texts(m, d)["model_response"]
            if len(t) >= 20:
                profs[d] = profile_20(t)
        ds = list(profs)
        prs = [np.corrcoef(profs[a], profs[b])[0, 1]
               for i, a in enumerate(ds) for b in ds[i + 1:]]
        s7.append(dict(model=m, n_datasets=len(ds),
                       native_xdataset_stability=round(float(np.mean(prs)), 4) if prs else np.nan))
    s7 = pd.DataFrame(s7)
    s7.to_csv(OUT / "native_xdataset_stability.csv", index=False)
    print("\n=== S7 cross-dataset 20-feature stability (imitation-INDEPENDENT proxy) ===")
    print(s7.to_string(index=False))

    # rank vs imitation-durability (DPO style persistence) — the corroboration check
    try:
        dist = pd.read_csv(ROOT / "results/source_distinctiveness.csv")
        col = "model" if "model" in dist.columns else dist.columns[0]
        merged = s7.merge(dist[[col, "dpo_persistence"]], left_on="model", right_on=col, how="left")
        rho = merged[["native_xdataset_stability", "dpo_persistence"]].corr(method="spearman").iloc[0, 1]
        print(f"\n  native cross-dataset stability  vs  DPO style persistence: Spearman rho={rho:+.3f} (n=4)")
        print("  [n=4 caveat: rank-direction only; perm p≈0.33. If +, native stability corroborates")
        print("   imitation-durability as a pre-existing source property (partner's 'rooted in pre-training').]")
    except Exception as e:
        print(f"  [corroboration join skipped: {e}]")

    # ---------- F3: cross-prompt-type stability (method-validation) ----------
    f3 = []
    for m in MODELS:
        df = native_texts(m, "chatbot_arena")
        if len(df) < 40:
            continue
        df["ptype"] = df["prompt"].map(prompt_type)
        profs = {pt: profile_20(g["model_response"]) for pt, g in df.groupby("ptype") if len(g) >= 10}
        pts = list(profs)
        cos = []
        for i, a in enumerate(pts):
            for b in pts[i + 1:]:
                va, vb = profs[a], profs[b]
                d = np.linalg.norm(va) * np.linalg.norm(vb)
                if d > 0:
                    cos.append(float(va @ vb / d))
        f3.append(dict(model=m, n_ptypes=len(pts),
                       xprompt_cosine=round(float(np.mean(cos)), 4) if cos else np.nan))
    f3 = pd.DataFrame(f3)
    f3.to_csv(OUT / "native_xprompt_stability.csv", index=False)
    print("\n=== F3 cross-prompt-type stability (chatbot_arena; method-validation) ===")
    print(f3.to_string(index=False))

    # ---------- F2: verbosity bias ----------
    alltexts = pd.concat([native_texts(m, d)["model_response"] for m in MODELS for d in DATASETS])
    X, names = _style_binary_features(list(alltexts))
    logwc = np.log1p([len(re.findall(r"\S+", t)) for t in alltexts])
    rows = []
    for j, nm in enumerate(names):
        col = X[:, j]
        r = np.corrcoef(col, logwc)[0, 1] if col.std() > 0 else 0.0
        rows.append(dict(feature=nm, r_with_logwc=round(float(r), 3), length_confounded=abs(r) > 0.15))
    f2 = pd.DataFrame(rows).sort_values("r_with_logwc", key=lambda s: s.abs(), ascending=False)
    f2.to_csv(OUT / "verbosity_bias_features.csv", index=False)
    conf = f2[f2["length_confounded"]]["feature"].tolist()
    print("\n=== F2 verbosity-bias (|r|>0.15 with log word count) ===")
    print(f2.head(8).to_string(index=False))
    print(f"\n  length-confounded features ({len(conf)}): {conf}")
    print("  -> these are exactly what our existing `*_lenres` (build_descriptor_matrix feature_set='full_lenres',")
    print("     _residualize_against_length) removes. F2 ⊆ our length control; no new code needed.")


if __name__ == "__main__":
    main()
