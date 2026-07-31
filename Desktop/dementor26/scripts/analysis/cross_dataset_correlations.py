"""
Cross-dataset behavioral correlation analysis.

Computes stylistic feature profiles (20 binary markers) for each model
across GSM8K, OpenAssistant (oasst1), and Chatbot Arena, then:
  1. Per-feature mean rates per (model, dataset)
  2. Inter-model Pearson correlation matrices per dataset
  3. Cross-dataset Pearson correlation of per-model style profiles
     (does a model's GSM8K style predict its oasst1 style?)
  4. Writes CSVs + a LaTeX table snippet for the paper

Outputs land in data/results/cross_dataset_correlations/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "data" / "results" / "cross_dataset_correlations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset sources ──────────────────────────────────────────────────────────
DATASETS: dict[str, list[tuple[str, Path]]] = {
    "gsm8k": [
        ("Llama-3.1-8B",  REPO_ROOT / "data/model-responses/gsm8k/full/meta-llama_Meta-Llama-3.1-8B-Instruct.csv"),
        ("GPT-4o",         REPO_ROOT / "data/model-responses/gsm8k/full/gpt-4o.csv"),
        ("Phi-4-mini",     REPO_ROOT / "data/model-responses/gsm8k/full/microsoft_Phi-4-mini-instruct.csv"),
    ],
    "oasst1": [
        ("Llama-3.1-8B",  REPO_ROOT / "data/model-responses/openasisstant/test/test_meta-llama_Llama-3.1-8B-Instruct_1000.csv"),
        ("Qwen3.6-27B",   REPO_ROOT / "data/model-responses/openasisstant/test/test_Qwen_Qwen3.6-27B_1000.csv"),
        ("Nemotron-30B",  REPO_ROOT / "data/model-responses/openasisstant/test/test_nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_1000.csv"),
        ("GPT-oss-20B",   REPO_ROOT / "data/model-responses/openasisstant/test/test_openai_gpt-oss-20b_1000.csv"),
    ],
    "chatbot_arena": [
        ("Llama-3.1-8B",  REPO_ROOT / "data/model-responses/chatbot_arena/full/meta-llama_Meta-Llama-3.1-8B-Instruct.csv"),
        ("GPT-4.1-mini",  REPO_ROOT / "data/model-responses/chatbot_arena/full/openai_gpt-4.1-mini.csv"),
    ],
}

# ── 20-feature stylistic extractor ───────────────────────────────────────────
def _style_vector(text: str) -> dict[str, float]:
    t = str(text or "")
    lines = t.splitlines()
    words = t.split()
    sentences = [s for s in t.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    return {
        "has_markdown":      float("**" in t or "__" in t or "##" in t),
        "has_bullets":       float(any(l.lstrip().startswith(("-", "*", "•")) for l in lines)),
        "has_numbered":      float(any(l.lstrip()[:2].rstrip(".").isdigit() for l in lines)),
        "has_code":          float("```" in t or "`" in t),
        "has_header":        float(any(l.startswith("#") for l in lines)),
        "has_link":          float("http" in t or "www." in t),
        "starts_greeting":   float(t[:40].lower().startswith(("hi ", "hello", "sure", "of course", "great"))),
        "ends_signoff":      float(any(t.lower().endswith(s) for s in ("regards", "sincerely", "cheers", "thanks"))),
        "has_emoji":         float(any(ord(c) > 127462 for c in t)),
        "has_question":      float("?" in t),
        "uses_parens":       float("(" in t and ")" in t),
        "has_exclamation":   float("!" in t),
        "long_sentences":    float(len(words) / max(len(sentences), 1) > 25),
        "starts_with_list":  float(lines[0].lstrip().startswith(("-", "*", "1.")) if lines else False),
        "has_math":          float(any(c in t for c in ("∑", "∫", "√", "≤", "≥", "∈", "π")) or "\\[" in t or "\\(" in t),
        "has_blockquote":    float(any(l.startswith(">") for l in lines)),
        "has_all_caps":      float(any(w.isupper() and len(w) > 2 for w in words)),
        "uses_first_person": float(any(w.lower() in ("i", "i'm", "i've", "i'll", "i'd", "my", "me") for w in words)),
        "has_hedging":       float(any(p in t.lower() for p in ["maybe", "might", "could", "likely", "possibly", "perhaps"])),
        "has_reasoning":     float(any(p in t.lower() for p in ["therefore", "thus", "so,", "step ", "because", "hence"])),
    }

FEATURES = list(_style_vector("").keys())


def compute_style_profile(df: pd.DataFrame, response_col: str = "model_response") -> pd.Series:
    """Mean rate of each feature over all responses."""
    rows = [_style_vector(r) for r in df[response_col].fillna("").tolist()]
    return pd.DataFrame(rows).mean()


def _find_response_col(df: pd.DataFrame) -> str:
    for c in ("model_response", "response", "output", "completion"):
        if c in df.columns:
            return c
    return df.columns[-1]


# ── Load + profile ────────────────────────────────────────────────────────────
records = []
profiles: dict[tuple[str, str], pd.Series] = {}

for dataset, entries in DATASETS.items():
    for model_name, csv_path in entries:
        if not csv_path.exists():
            print(f"  SKIP (missing): {csv_path.name}")
            continue
        try:
            df = pd.read_csv(csv_path, on_bad_lines="skip")
        except Exception as e:
            print(f"  SKIP (error): {csv_path.name} — {e}")
            continue
        col = _find_response_col(df)
        profile = compute_style_profile(df, col)
        profiles[(dataset, model_name)] = profile
        for feat, val in profile.items():
            records.append({"dataset": dataset, "model": model_name, "feature": feat, "rate": val})
        print(f"  Profiled {model_name:20s} on {dataset:15s}  ({len(df)} rows)")

feature_df = pd.DataFrame(records)
feature_df.to_csv(OUT_DIR / "style_feature_rates.csv", index=False)
print(f"\nSaved feature rates → {OUT_DIR / 'style_feature_rates.csv'}")

# ── Inter-model correlation matrix per dataset ────────────────────────────────
for dataset in DATASETS:
    models_in = [(d, m) for (d, m) in profiles if d == dataset]
    if len(models_in) < 2:
        continue
    labels = [m for _, m in models_in]
    mat = np.zeros((len(models_in), len(models_in)))
    for i, key_i in enumerate(models_in):
        for j, key_j in enumerate(models_in):
            if i == j:
                mat[i, j] = 1.0
            else:
                vi, vj = profiles[key_i][FEATURES].values, profiles[key_j][FEATURES].values
                r, _ = pearsonr(vi, vj)
                mat[i, j] = r
    corr_df = pd.DataFrame(mat, index=labels, columns=labels)
    corr_df.to_csv(OUT_DIR / f"intermodel_corr_{dataset}.csv")
    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(3.5, len(labels))))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(mat[i,j]) < 0.8 else "white")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Inter-model style correlation — {dataset}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"intermodel_corr_{dataset}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved inter-model correlation → {dataset}")

# ── Cross-dataset correlation for shared models ───────────────────────────────
# Find models that appear in ≥2 datasets
all_models = {m for (_, m) in profiles}
cross_records = []
for model in sorted(all_models):
    ds_with_model = [d for (d, m) in profiles if m == model]
    if len(ds_with_model) < 2:
        continue
    pairs = [(ds_with_model[i], ds_with_model[j])
             for i in range(len(ds_with_model)) for j in range(i+1, len(ds_with_model))]
    for ds_a, ds_b in pairs:
        va = profiles[(ds_a, model)][FEATURES].values
        vb = profiles[(ds_b, model)][FEATURES].values
        r, p = pearsonr(va, vb)
        cross_records.append({"model": model, "dataset_a": ds_a, "dataset_b": ds_b,
                               "pearson_r": round(r, 3), "p_value": round(p, 4)})
        print(f"  Cross-dataset r({model}: {ds_a} vs {ds_b}) = {r:.3f}  p={p:.4f}")

if cross_records:
    cross_df = pd.DataFrame(cross_records)
    cross_df.to_csv(OUT_DIR / "cross_dataset_correlations.csv", index=False)

# ── Per-feature domain drift for Llama-3.1-8B (in all 3 datasets) ─────────────
llama_datasets = [(d, m) for (d, m) in profiles if m == "Llama-3.1-8B"]
if len(llama_datasets) >= 2:
    drift_rows = []
    for feat in FEATURES:
        vals = {d: profiles[(d, "Llama-3.1-8B")][feat] for (d, _) in llama_datasets}
        drift_rows.append({"feature": feat, **vals,
                           "std": np.std(list(vals.values())),
                           "range": max(vals.values()) - min(vals.values())})
    drift_df = pd.DataFrame(drift_rows).sort_values("range", ascending=False)
    drift_df.to_csv(OUT_DIR / "llama_domain_drift.csv", index=False)

    # Plot feature stability
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(FEATURES))
    colors = plt.cm.Set2.colors
    for idx, (d, _) in enumerate(llama_datasets):
        ax.bar(x + idx * 0.25, [profiles[(d, "Llama-3.1-8B")][f] for f in FEATURES],
               width=0.25, label=d, alpha=0.8, color=colors[idx % len(colors)])
    ax.set_xticks(x + 0.25); ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Feature rate")
    ax.set_title("Llama-3.1-8B stylistic profile across datasets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "llama_domain_drift.pdf", bbox_inches="tight")
    plt.close(fig)

# ── LaTeX table: per-model style rates on oasst1 ─────────────────────────────
oasst_models = [(d, m) for (d, m) in profiles if d == "oasst1"]
if oasst_models:
    TOP_FEATURES = ["has_bullets", "has_numbered", "has_markdown", "has_code",
                    "has_header", "has_reasoning", "has_hedging", "uses_first_person",
                    "long_sentences", "has_question"]
    header = " & ".join(["Model"] + [f.replace("has_", "").replace("_", " ").title()
                                      for f in TOP_FEATURES]) + r" \\"
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "r" * len(TOP_FEATURES) + "}",
             r"\toprule", header, r"\midrule"]
    for (_, mname) in oasst_models:
        vals = [f"{profiles[('oasst1', mname)][f]*100:.0f}" for f in TOP_FEATURES]
        lines.append(f"{mname} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Stylistic feature rates (\%) per model on OpenAssistant (oasst1) test set (1000 prompts).}",
              r"\label{tab:oasst_style_rates}", r"\end{table}"]
    latex_table = "\n".join(lines)
    (OUT_DIR / "oasst_style_table.tex").write_text(latex_table)
    print(f"\nLaTeX table → {OUT_DIR / 'oasst_style_table.tex'}")

# ── Summary print ─────────────────────────────────────────────────────────────
print("\n=== Cross-dataset correlations ===")
if cross_records:
    for r in cross_records:
        print(f"  {r['model']:20s}  {r['dataset_a']:15s} vs {r['dataset_b']:15s}  r={r['pearson_r']}  p={r['p_value']}")

print(f"\nAll outputs in {OUT_DIR}")
