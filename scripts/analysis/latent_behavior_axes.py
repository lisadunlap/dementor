from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .common import (
    CONDITIONS,
    condition_texts,
    descriptor_names,
    git_commit,
    normalize_comparison_df,
    slugify,
    write_json,
)


COND_COLORS = {
    "source": "#d62728",
    "disguised": "#ff7f0e",
    "target": "#2ca02c",
}


def _style_scalar_features(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    rows = []
    for text in texts:
        value = str(text or "")
        words = value.split()
        lines = [line for line in value.splitlines() if line.strip()]
        sentences = [s for s in value.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        rows.append(
            [
                np.log1p(len(words)),
                np.log1p(len(value)),
                float(value.count("\n")),
                float(sum(line.lstrip().startswith(("-", "*", "•")) for line in lines)),
                float(sum(line.lstrip()[:2].rstrip(".").isdigit() for line in lines)),
                float("```" in value or "`" in value),
                float("#" in value),
                float("?" in value),
                float("!" in value),
                float(any(p in value.lower() for p in ["maybe", "might", "could", "likely", "possibly"])),
                float(any(p in value.lower() for p in ["therefore", "thus", "so,", "step", "because"])),
                float(len(words) / max(len(sentences), 1)),
            ]
        )
    names = [
        "style_word_count",
        "style_char_count",
        "style_line_count",
        "style_bullet_count",
        "style_numbered_count",
        "style_code_formatting",
        "style_header_marker",
        "style_questioning",
        "style_exclamation",
        "style_hedging_terms",
        "style_reasoning_markers",
        "style_sentence_length",
    ]
    X = np.asarray(rows, dtype=float)
    if len(X) > 1:
        X = StandardScaler().fit_transform(X)
    return X, names


def build_descriptor_matrix(
    texts_by_condition: dict[str, list[str]],
    *,
    descriptor_mode: str = "big5_style",
    include_style_scalars: bool = True,
) -> tuple[dict[str, np.ndarray], list[str]]:
    descriptors = descriptor_names(descriptor_mode)
    all_texts = [text for cond in CONDITIONS for text in texts_by_condition[cond]]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
    tfidf = vectorizer.fit_transform(descriptors + all_texts)
    desc_mat = tfidf[: len(descriptors)]
    text_mat = tfidf[len(descriptors) :]
    sims = cosine_similarity(text_mat, desc_mat)

    if include_style_scalars:
        scalars, scalar_names = _style_scalar_features(all_texts)
        full = np.hstack([sims, scalars])
        feature_names = descriptors + scalar_names
    else:
        full = sims
        feature_names = descriptors

    n = len(texts_by_condition["source"])
    matrices = {}
    for i, cond in enumerate(CONDITIONS):
        matrices[cond] = full[i * n : (i + 1) * n]
    return matrices, feature_names


def factorize_joint(matrices: dict[str, np.ndarray], k: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    X = np.vstack([matrices[cond] for cond in CONDITIONS])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, U.shape[1])
    scores = U[:, :k] * S[:k]
    n = len(matrices["source"])
    by_cond = {cond: scores[i * n : (i + 1) * n] for i, cond in enumerate(CONDITIONS)}
    return by_cond, S, Vt[:k].T


def latent_scores_df(df: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for cond in CONDITIONS:
        condition_scores = scores[cond]
        for i, prompt in enumerate(df["prompt"].tolist()):
            row = {
                "row_id": i,
                "prompt": prompt,
                "condition": cond,
            }
            for j in range(condition_scores.shape[1]):
                row[f"pc{j + 1}"] = float(condition_scores[i, j])
            rows.append(row)
    return pd.DataFrame(rows)


def loadings_df(V: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(names):
        row = {"feature": name}
        for j in range(V.shape[1]):
            row[f"pc{j + 1}_loading"] = float(V[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pcs(scores: dict[str, np.ndarray], singular_values: np.ndarray, path: Path, *, title: str) -> None:
    k = min(5, next(iter(scores.values())).shape[1])
    var = (singular_values[:k] ** 2) / max(float((singular_values ** 2).sum()), 1e-12) * 100
    fig, axes = plt.subplots(1, k, figsize=(max(4, k * 3.1), 4.0), squeeze=False)
    for pc_idx, ax in enumerate(axes[0]):
        data = [scores[cond][:, pc_idx] for cond in CONDITIONS]
        vp = ax.violinplot(data, positions=[0, 1, 2], showmedians=True, showextrema=False)
        for body, cond in zip(vp["bodies"], CONDITIONS):
            body.set_facecolor(COND_COLORS[cond])
            body.set_alpha(0.78)
        if "cmedians" in vp:
            vp["cmedians"].set_color("white")
        for pos, cond in enumerate(CONDITIONS):
            ax.scatter([pos], [scores[cond][:, pc_idx].mean()], color="white", edgecolor=COND_COLORS[cond], zorder=3)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Source", "Disguised", "Target"], rotation=20, ha="right")
        ax.set_title(f"PC{pc_idx + 1}\n{var[pc_idx]:.1f}% var")
        if pc_idx == 0:
            ax.set_ylabel("Latent score")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_adjective_loadings(V: np.ndarray, names: list[str], path: Path, *, title: str, top_n: int = 8) -> None:
    k = min(5, V.shape[1])
    fig, axes = plt.subplots(1, k, figsize=(max(4, k * 3.4), 5.2), squeeze=False)
    for pc_idx, ax in enumerate(axes[0]):
        loads = V[:, pc_idx]
        pos = np.argsort(loads)[::-1][:top_n]
        neg = np.argsort(loads)[:top_n]
        chosen = list(dict.fromkeys(list(neg) + list(pos[::-1])))
        y = np.arange(len(chosen))
        vals = loads[chosen]
        labels = [names[i] for i in chosen]
        colors = ["#2ca02c" if val >= 0 else "#d62728" for val in vals]
        ax.barh(y, vals, color=colors, alpha=0.82)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"PC{pc_idx + 1}")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_persistence_radar(per_axis: pd.DataFrame, path: Path, *, title: str) -> None:
    subset = per_axis.head(8).copy()
    if subset.empty:
        return
    values = subset["movement_clipped"].fillna(0.0).to_numpy(dtype=float)
    labels = subset["axis"].astype(str).tolist()
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    values_closed = np.append(values, values[0])
    angles_closed = np.append(angles, angles[0])
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.plot(angles_closed, values_closed, color=COND_COLORS["disguised"], lw=2)
    ax.fill(angles_closed, values_closed, color=COND_COLORS["disguised"], alpha=0.22)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["source", "50%", "target"])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_latent_analysis(
    comparison_csv: str | Path,
    *,
    output_dir: str | Path,
    source_responses: str | Path | None = None,
    source_model: str = "source",
    target_model: str = "target",
    method: str = "unknown",
    dataset: str = "unknown",
    source_col: str = "source_response",
    disguised_col: str = "model_response",
    target_col: str = "target_response",
    descriptor_mode: str = "big5_style",
    k: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    np.random.seed(seed)
    out_dir = Path(output_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_comparison_df(
        comparison_csv,
        source_responses=source_responses,
        source_col=source_col,
        disguised_col=disguised_col,
        target_col=target_col,
    )
    texts = condition_texts(df)
    matrices, feature_names = build_descriptor_matrix(texts, descriptor_mode=descriptor_mode)
    scores, singular_values, V = factorize_joint(matrices, k=k)
    scores_df = latent_scores_df(df, scores)
    loads_df = loadings_df(V, feature_names)

    scores_df.to_csv(out_dir / "latent_scores.csv", index=False)
    loads_df.to_csv(out_dir / "axis_loadings.csv", index=False)

    title = f"{source_model} -> {target_model} | {method}"
    plot_pcs(scores, singular_values, figures_dir / "paper_pcs.png", title=title)
    plot_adjective_loadings(V, feature_names, figures_dir / "paper_adjectives.png", title=title)

    var = (singular_values ** 2) / max(float((singular_values ** 2).sum()), 1e-12)
    config = {
        "dataset": dataset,
        "source_model": source_model,
        "target_model": target_model,
        "method": method,
        "comparison_csv": str(comparison_csv),
        "source_responses": str(source_responses) if source_responses else None,
        "descriptor_mode": descriptor_mode,
        "n": int(len(df)),
        "k": int(min(k, len(singular_values))),
        "features": len(feature_names),
        "variance_explained": var[: min(k, len(var))].tolist(),
        "git_commit": git_commit(),
    }
    write_json(out_dir / "config.json", config)
    return scores_df, loads_df, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build latent behavioral axes for source/disguised/target responses.")
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--source-responses")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-model", default="source")
    parser.add_argument("--target-model", default="target")
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--source-col", default="source_response")
    parser.add_argument("--disguised-col", default="model_response")
    parser.add_argument("--target-col", default="target_response")
    parser.add_argument("--descriptor-mode", choices=["big5_style", "style_only"], default="big5_style")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_latent_analysis(
        args.comparison_csv,
        output_dir=args.output_dir,
        source_responses=args.source_responses,
        source_model=args.source_model,
        target_model=args.target_model,
        method=args.method,
        dataset=args.dataset,
        source_col=args.source_col,
        disguised_col=args.disguised_col,
        target_col=args.target_col,
        descriptor_mode=args.descriptor_mode,
        k=args.k,
        seed=args.seed,
    )
    print(f"Wrote latent behavioral analysis to {args.output_dir}")


if __name__ == "__main__":
    main()

