from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .common import (
    BIG5,
    CONDITIONS,
    condition_texts,
    descriptor_names,
    git_commit,
    normalize_comparison_df,
    write_json,
)


COND_COLORS = {
    "source": "#d62728",
    "disguised": "#ff7f0e",
    "target": "#2ca02c",
}

DEFAULT_DESCRIPTOR_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
TRUNCATE_CHARS = 600
BASE_FEATURE_SETS = ("full", "adjectives", "style_scalars", "style_binaries", "style_all")
LENGTH_RESIDUALIZED_SUFFIX = "_lenres"
# Each base feature set has a length-residualized twin (e.g. ``full_lenres``) that
# regresses every feature against log word count (fit on source+target only) and
# keeps the residual. It is the robustness check for "is the fingerprint just
# length?": if persistence survives ``*_lenres``, the signal is not pure verbosity.
FEATURE_SETS = BASE_FEATURE_SETS + tuple(
    f"{name}{LENGTH_RESIDUALIZED_SUFFIX}" for name in BASE_FEATURE_SETS
)
BIG5_LABELS = {
    "EXT": "Extraversion",
    "AGR": "Agreeableness",
    "CON": "Conscientiousness",
    "NEU": "Neuroticism",
    "OPN": "Openness",
}


@dataclass
class BehavioralAxisBasis:
    feature_names: list[str]
    descriptor_mode: str
    encoder_model: str
    feature_set: str
    scaler: StandardScaler
    components: np.ndarray
    singular_values: np.ndarray
    fit_conditions: tuple[str, str] = ("source", "target")
    basis_type: str = "variance"

    def project(self, matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        projected = {}
        for cond, matrix in matrices.items():
            X = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            Xs = self.scaler.transform(X)
            projected[cond] = Xs @ self.components.T
        return projected


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
    return X, names


def _style_binary_features(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    from scripts.methods.utils import stylistic_analysis as style

    feature_funcs = [
        ("style_has_markdown", style.has_markdown),
        ("style_contains_list", style.contains_list),
        ("style_contains_header", style.contains_header),
        ("style_contains_code", style.contains_code),
        ("style_contains_link", style.contains_link),
        ("style_starts_with_greeting", style.starts_with_greeting),
        ("style_ends_with_signoff", style.ends_with_signoff),
        ("style_contains_emoji", style.contains_emoji),
        ("style_contains_bullets", style.contains_bullets),
        ("style_contains_question", style.contains_question),
        ("style_uses_parentheses", style.uses_parentheses),
        ("style_contains_exclamation", style.contains_exclamation),
        ("style_has_long_sentences", style.has_long_sentences),
        ("style_starts_with_list", style.starts_with_list),
        ("style_contains_math_symbols", style.contains_math_symbols),
        ("style_contains_blockquote", style.contains_blockquote),
        ("style_contains_repetition", style.contains_repetition),
        ("style_contains_numbered_steps", style.contains_numbered_steps),
        ("style_contains_all_caps", style.contains_all_caps),
        ("style_uses_first_person", style.uses_first_person),
    ]
    rows = []
    for text in texts:
        value = str(text or "")
        rows.append([float(func(value)) for _name, func in feature_funcs])
    return np.asarray(rows, dtype=float), [name for name, _func in feature_funcs]


def _embedding_descriptor_scores(
    texts: list[str],
    descriptors: list[str],
    *,
    encoder_model: str = DEFAULT_DESCRIPTOR_ENCODER,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Behavioral-inertia analysis now requires sentence-transformers for "
            "Naz-style adjective embedding scoring. Install requirements.txt or run "
            "`pip install sentence-transformers`."
        ) from exc

    model = SentenceTransformer(encoder_model)
    descriptor_embeddings = model.encode(
        descriptors,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    text_embeddings = model.encode(
        [str(text or "")[:TRUNCATE_CHARS] for text in texts],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    cosine_scores = np.asarray(text_embeddings @ descriptor_embeddings.T, dtype=float)
    return np.log(np.clip((cosine_scores + 1.0) / 2.0, 1e-9, 1.0))


def _length_covariate(texts: list[str]) -> np.ndarray:
    """Log word count per response; the covariate removed by ``*_lenres``."""
    return np.array([np.log1p(len(str(text or "").split())) for text in texts], dtype=float)


def _residualize_against_length(
    features: np.ndarray, length: np.ndarray, fit_mask: np.ndarray
) -> np.ndarray:
    """Subtract the part of every feature linearly predictable from ``length``.

    The OLS fit uses only ``fit_mask`` rows (source + target), mirroring the basis
    rule that disguised/intervention rows never define the measurement. The same
    coefficients are then applied to all rows.
    """
    design = np.column_stack([np.ones_like(length), length])
    if not fit_mask.any():
        return features
    coef, *_ = np.linalg.lstsq(design[fit_mask], features[fit_mask], rcond=None)
    return features - design @ coef


def build_descriptor_matrix(
    texts_by_condition: dict[str, list[str]],
    *,
    descriptor_mode: str = "big5_style",
    feature_set: str = "full",
    encoder_model: str = DEFAULT_DESCRIPTOR_ENCODER,
) -> tuple[dict[str, np.ndarray], list[str]]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unsupported feature_set: {feature_set}")
    residualize = feature_set.endswith(LENGTH_RESIDUALIZED_SUFFIX)
    base_set = feature_set[: -len(LENGTH_RESIDUALIZED_SUFFIX)] if residualize else feature_set

    descriptors = descriptor_names(descriptor_mode)
    all_texts = [text for cond in CONDITIONS for text in texts_by_condition[cond]]
    feature_blocks = []
    feature_names = []

    if base_set in {"full", "adjectives"}:
        feature_blocks.append(
            _embedding_descriptor_scores(all_texts, descriptors, encoder_model=encoder_model)
        )
        feature_names.extend(descriptors)

    if base_set in {"full", "style_scalars", "style_all"}:
        scalars, scalar_names = _style_scalar_features(all_texts)
        feature_blocks.append(scalars)
        feature_names.extend(scalar_names)

    if base_set in {"full", "style_binaries", "style_all"}:
        binaries, binary_names = _style_binary_features(all_texts)
        feature_blocks.append(binaries)
        feature_names.extend(binary_names)

    if not feature_blocks:
        raise ValueError(f"No feature blocks selected for feature_set: {feature_set}")
    full = np.hstack(feature_blocks)

    n = len(texts_by_condition["source"])
    if residualize:
        # Fit the length regression on source rows [0:n] and target rows [2n:3n]
        # only (CONDITIONS == source, disguised, target); apply to all rows.
        fit_mask = np.zeros(full.shape[0], dtype=bool)
        fit_mask[0:n] = True
        fit_mask[2 * n : 3 * n] = True
        full = _residualize_against_length(full, _length_covariate(all_texts), fit_mask)

    matrices = {}
    for i, cond in enumerate(CONDITIONS):
        matrices[cond] = full[i * n : (i + 1) * n]
    return matrices, feature_names


def big5_dimension_scores_df(
    df: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    feature_names: list[str],
) -> pd.DataFrame:
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    trait_indices = {}
    for trait, adjectives in BIG5.items():
        midpoint = len(adjectives) // 2
        positive = [feature_index[word] for word in adjectives[:midpoint] if word in feature_index]
        reverse = [feature_index[word] for word in adjectives[midpoint:] if word in feature_index]
        if positive and reverse:
            trait_indices[trait] = (positive, reverse)
    if not trait_indices:
        return pd.DataFrame()

    rows = []
    prompts = df["prompt"].astype(str).tolist()
    for cond in CONDITIONS:
        matrix = matrices[cond]
        for row_id, prompt in enumerate(prompts):
            row = {"row_id": row_id, "prompt": prompt, "condition": cond}
            for trait, (positive, reverse) in trait_indices.items():
                row[trait] = float(np.nanmean(matrix[row_id, positive]) - np.nanmean(matrix[row_id, reverse]))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_big5_dimension_movement(big5_scores: pd.DataFrame) -> pd.DataFrame:
    traits = [trait for trait in BIG5_LABELS if trait in big5_scores.columns]
    if not traits:
        return pd.DataFrame()

    rows = []
    by_condition = {
        cond: big5_scores[big5_scores["condition"] == cond][traits].to_numpy(dtype=float)
        for cond in CONDITIONS
    }
    source_mean = np.nanmean(by_condition["source"], axis=0)
    disguised_mean = np.nanmean(by_condition["disguised"], axis=0)
    target_mean = np.nanmean(by_condition["target"], axis=0)
    denom = target_mean - source_mean
    movement = np.full(len(traits), np.nan, dtype=float)
    mask = np.abs(denom) > 1e-9
    movement[mask] = (disguised_mean[mask] - source_mean[mask]) / denom[mask]
    clipped = np.clip(movement, 0.0, 1.0)
    for idx, trait in enumerate(traits):
        rows.append(
            {
                "dimension": trait,
                "label": BIG5_LABELS[trait],
                "axis_separation": float(abs(denom[idx])),
                "movement": float(movement[idx]) if np.isfinite(movement[idx]) else np.nan,
                "movement_clipped": float(clipped[idx]) if np.isfinite(clipped[idx]) else np.nan,
                "source_persistence": float(1.0 - clipped[idx]) if np.isfinite(clipped[idx]) else np.nan,
                "source_mean": float(source_mean[idx]),
                "disguised_mean": float(disguised_mean[idx]),
                "target_mean": float(target_mean[idx]),
            }
        )
    return pd.DataFrame(rows)


DEFAULT_LDA_SHRINKAGE = 0.1


def _fit_supervised_basis(
    source_scaled: np.ndarray,
    target_scaled: np.ndarray,
    *,
    k: int,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Supervised orthonormal basis whose first axis separates source from target.

    Axis 1 is the shrinkage-regularized Fisher LDA discriminant
    ``w ∝ (Σ_within + λI)^-1 (μ_t - μ_s)``; axes 2..k are PCA directions of the
    reference residual orthogonal to axis 1 (kept for the probe and per-axis
    diagnostics). All rows are orthonormal so ``sep_ratio`` stays a
    valid orthogonal projection. Unlike the variance basis, the source->target
    direction is concentrated in axis 1 instead of scattered across many PCs.
    """
    n_features = source_scaled.shape[1]
    mu_s = source_scaled.mean(axis=0)
    mu_t = target_scaled.mean(axis=0)
    delta = mu_t - mu_s

    src_c = source_scaled - mu_s
    tgt_c = target_scaled - mu_t
    dof = max(len(source_scaled) + len(target_scaled) - 2, 1)
    within = (src_c.T @ src_c + tgt_c.T @ tgt_c) / dof
    mean_diag = float(np.trace(within) / n_features) if n_features else 1.0
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        mean_diag = 1.0
    within_reg = (1.0 - shrinkage) * within + shrinkage * mean_diag * np.eye(n_features)

    try:
        w1 = np.linalg.solve(within_reg, delta)
    except np.linalg.LinAlgError:
        w1 = np.linalg.lstsq(within_reg, delta, rcond=None)[0]
    norm = float(np.linalg.norm(w1))
    if not np.isfinite(norm) or norm <= 1e-12:
        # Degenerate (e.g. source ~= target): fall back to the mean difference, or
        # an arbitrary unit axis if even that vanishes. The axis then carries ~0
        # separation and is filtered out as inactive downstream.
        w1 = delta.astype(float).copy()
        norm = float(np.linalg.norm(w1))
        if norm <= 1e-12:
            w1 = np.zeros(n_features, dtype=float)
            w1[0] = 1.0
            norm = 1.0
    w1 = w1 / norm

    components = [w1]
    if k > 1:
        reference = np.vstack([source_scaled, target_scaled])
        residual = reference - np.outer(reference @ w1, w1)
        _U, _S, Vt = np.linalg.svd(residual, full_matrices=False)
        for vec in Vt:
            components.append(vec)
            if len(components) >= k:
                break
    comps = np.vstack(components[:k])
    projections = np.vstack([source_scaled, target_scaled]) @ comps.T
    singular_values = projections.std(axis=0) * np.sqrt(max(len(projections), 1))
    return comps, singular_values


def fit_behavioral_axis_basis(
    matrices: dict[str, np.ndarray],
    feature_names: list[str],
    *,
    descriptor_mode: str,
    encoder_model: str,
    feature_set: str,
    k: int,
    basis_type: str = "variance",
    lda_shrinkage: float = DEFAULT_LDA_SHRINKAGE,
) -> BehavioralAxisBasis:
    source = np.nan_to_num(matrices["source"], nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(matrices["target"], nan=0.0, posinf=0.0, neginf=0.0)
    reference = np.vstack([source, target])
    scaler = StandardScaler().fit(reference)
    # Floor near-constant features to unit scale. sklearn only catches exactly-zero
    # variance; a near-zero std (e.g. a feature made collinear/constant by length
    # residualization) otherwise divides floating-point noise up to O(1), which
    # breaks the shared-endpoint invariant across row orderings. Treating such
    # features as constant makes them contribute ~0 instead.
    scaler.scale_[scaler.scale_ < 1e-8] = 1.0
    reference_scaled = scaler.transform(reference)

    if basis_type == "variance":
        _U, S, Vt = np.linalg.svd(reference_scaled, full_matrices=False)
        kk = min(k, Vt.shape[0])
        components = Vt[:kk]
        singular_values = S
    elif basis_type == "supervised":
        source_scaled = scaler.transform(source)
        target_scaled = scaler.transform(target)
        components, singular_values = _fit_supervised_basis(
            source_scaled,
            target_scaled,
            k=min(k, reference_scaled.shape[1]),
            shrinkage=lda_shrinkage,
        )
    else:
        raise ValueError(f"Unknown basis_type: {basis_type!r} (expected 'variance' or 'supervised')")

    return BehavioralAxisBasis(
        feature_names=feature_names,
        descriptor_mode=descriptor_mode,
        encoder_model=encoder_model,
        feature_set=feature_set,
        scaler=scaler,
        components=components,
        singular_values=singular_values,
        basis_type=basis_type,
    )


def save_basis(basis: BehavioralAxisBasis, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(basis, handle)


def load_basis(path: str | Path) -> BehavioralAxisBasis:
    with Path(path).open("rb") as handle:
        basis = pickle.load(handle)
    if not isinstance(basis, BehavioralAxisBasis):
        raise TypeError(f"Loaded object is not a BehavioralAxisBasis: {path}")
    return basis


def factorize_fixed_basis(
    matrices: dict[str, np.ndarray],
    feature_names: list[str],
    *,
    descriptor_mode: str,
    encoder_model: str,
    feature_set: str,
    k: int,
    basis: BehavioralAxisBasis | None = None,
    basis_type: str = "variance",
    lda_shrinkage: float = DEFAULT_LDA_SHRINKAGE,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, BehavioralAxisBasis]:
    if basis is None:
        basis = fit_behavioral_axis_basis(
            matrices,
            feature_names,
            descriptor_mode=descriptor_mode,
            encoder_model=encoder_model,
            feature_set=feature_set,
            k=k,
            basis_type=basis_type,
            lda_shrinkage=lda_shrinkage,
        )
    else:
        if basis.feature_names != feature_names:
            raise ValueError("Loaded basis feature names do not match current feature extraction")
        if basis.descriptor_mode != descriptor_mode:
            raise ValueError("Loaded basis descriptor_mode does not match current run")
        if basis.encoder_model != encoder_model:
            raise ValueError("Loaded basis encoder_model does not match current run")
        if basis.feature_set != feature_set:
            raise ValueError("Loaded basis feature_set does not match current run")
        if getattr(basis, "basis_type", "variance") != basis_type:
            raise ValueError("Loaded basis basis_type does not match current run")

    scores = basis.project(matrices)
    return scores, basis.singular_values, basis.components.T, basis


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
    encoder_model: str = DEFAULT_DESCRIPTOR_ENCODER,
    feature_set: str = "full",
    save_basis_path: str | Path | None = None,
    load_basis_path: str | Path | None = None,
    k: int = 5,
    seed: int = 42,
    allow_duplicate_prompts: bool = False,
    basis_type: str = "variance",
    lda_shrinkage: float = DEFAULT_LDA_SHRINKAGE,
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
        allow_duplicate_prompts=allow_duplicate_prompts,
    )
    texts = condition_texts(df)
    matrices, feature_names = build_descriptor_matrix(
        texts,
        descriptor_mode=descriptor_mode,
        feature_set=feature_set,
        encoder_model=encoder_model,
    )
    loaded_basis = load_basis(load_basis_path) if load_basis_path else None
    scores, singular_values, V, basis = factorize_fixed_basis(
        matrices,
        feature_names,
        descriptor_mode=descriptor_mode,
        encoder_model=encoder_model,
        feature_set=feature_set,
        k=k,
        basis=loaded_basis,
        basis_type=basis_type,
        lda_shrinkage=lda_shrinkage,
    )
    if save_basis_path:
        save_basis(basis, save_basis_path)
    scores_df = latent_scores_df(df, scores)
    loads_df = loadings_df(V, feature_names)
    big5_scores = big5_dimension_scores_df(df, matrices, feature_names)
    big5_movement = summarize_big5_dimension_movement(big5_scores)

    # Full-feature source->target axis (k- and basis-independent). Every response is
    # projected onto the scaled-feature difference-of-means direction d; source maps
    # to 0 and target to 1 by construction, so the disguised coordinate IS the
    # movement fraction. This is the headline persistence input and, unlike the
    # PC-space projection, captures the entire separation regardless of k. The same
    # d gives sep_ratio (how much of d the retained PC axes hold).
    src_scaled = basis.scaler.transform(np.nan_to_num(matrices["source"], nan=0.0, posinf=0.0, neginf=0.0))
    tgt_scaled = basis.scaler.transform(np.nan_to_num(matrices["target"], nan=0.0, posinf=0.0, neginf=0.0))
    dis_scaled = basis.scaler.transform(np.nan_to_num(matrices["disguised"], nan=0.0, posinf=0.0, neginf=0.0))
    d_feat = tgt_scaled.mean(axis=0) - src_scaled.mean(axis=0)
    sep_full = float(np.linalg.norm(d_feat))
    sep_captured = float(np.linalg.norm(basis.components @ d_feat))
    denom_feat = float(d_feat @ d_feat)
    if denom_feat > 1e-12:
        src0 = src_scaled.mean(axis=0)
        st_by_cond = {
            "source": (src_scaled - src0) @ d_feat / denom_feat,
            "disguised": (dis_scaled - src0) @ d_feat / denom_feat,
            "target": (tgt_scaled - src0) @ d_feat / denom_feat,
        }
        st_col = np.full(len(scores_df), np.nan, dtype=float)
        cond_values = scores_df["condition"].to_numpy()
        for cond in CONDITIONS:
            st_col[cond_values == cond] = st_by_cond[cond]
        scores_df["st_axis"] = st_col

    scores_df.to_csv(out_dir / "latent_scores.csv", index=False)
    loads_df.to_csv(out_dir / "axis_loadings.csv", index=False)
    if not big5_scores.empty:
        big5_scores.to_csv(out_dir / "big5_dimension_scores.csv", index=False)
    if not big5_movement.empty:
        big5_movement.to_csv(out_dir / "big5_dimension_movement.csv", index=False)

    title = f"{source_model} -> {target_model} | {method}"
    plot_pcs(scores, singular_values, figures_dir / "paper_pcs.png", title=title)
    plot_adjective_loadings(V, feature_names, figures_dir / "paper_adjectives.png", title=title)

    var = (singular_values ** 2) / max(float((singular_values ** 2).sum()), 1e-12)
    config = {
        "dataset": dataset,
        "sep_full": sep_full,
        "sep_captured": sep_captured,
        "sep_ratio": (float(sep_captured / sep_full) if sep_full > 1e-12 else float("nan")),
        "source_model": source_model,
        "target_model": target_model,
        "method": method,
        "comparison_csv": str(comparison_csv),
        "source_responses": str(source_responses) if source_responses else None,
        "descriptor_mode": descriptor_mode,
        "descriptor_backend": "sentence_transformer_embeddings",
        "descriptor_encoder": encoder_model,
        "descriptor_truncate_chars": TRUNCATE_CHARS,
        "feature_set": feature_set,
        "basis_type": getattr(basis, "basis_type", "variance"),
        "basis_fit_conditions": list(basis.fit_conditions),
        "basis_path_saved": str(save_basis_path) if save_basis_path else None,
        "basis_path_loaded": str(load_basis_path) if load_basis_path else None,
        "basis_fit_excludes_disguised": True,
        "n": int(len(df)),
        "k": int(min(k, len(singular_values))),
        "features": len(feature_names),
        "variance_explained": var[: min(k, len(var))].tolist(),
        "git_commit": git_commit(),
    }
    if not big5_movement.empty:
        ranked = big5_movement.dropna(subset=["movement_clipped"]).sort_values(
            ["movement_clipped", "axis_separation"],
            ascending=[False, False],
        )
        config["big5_direct_diagnostics"] = True
        config["big5_dimensions"] = big5_movement["dimension"].tolist()
        if not ranked.empty:
            config["most_plastic_big5"] = str(ranked.iloc[0]["label"])
            config["most_plastic_big5_move"] = float(ranked.iloc[0]["movement_clipped"])
            config["least_plastic_big5"] = str(ranked.iloc[-1]["label"])
            config["least_plastic_big5_move"] = float(ranked.iloc[-1]["movement_clipped"])
    else:
        config["big5_direct_diagnostics"] = False
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
    parser.add_argument("--encoder-model", default=DEFAULT_DESCRIPTOR_ENCODER)
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="full")
    parser.add_argument("--save-basis")
    parser.add_argument("--load-basis")
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
        encoder_model=args.encoder_model,
        feature_set=args.feature_set,
        save_basis_path=args.save_basis,
        load_basis_path=args.load_basis,
        k=args.k,
        seed=args.seed,
    )
    print(f"Wrote latent behavioral analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
