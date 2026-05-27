from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .common import CONDITIONS, read_csv_robust, write_json


EPS = 1e-9
NAN_PROBE_METRICS = {
    "probe_cv_accuracy": float("nan"),
    "source_residue": float("nan"),
    "target_assimilation": float("nan"),
    "mean_source_probability_disguised": float("nan"),
    "mean_target_probability_disguised": float("nan"),
}


def movement_by_axis(
    source: np.ndarray,
    disguised: np.ndarray,
    target: np.ndarray,
    *,
    zero_denominator: float = np.nan,
) -> np.ndarray:
    src_mu = np.nanmean(source, axis=0)
    dis_mu = np.nanmean(disguised, axis=0)
    tgt_mu = np.nanmean(target, axis=0)
    denom = tgt_mu - src_mu
    out = np.full_like(src_mu, zero_denominator, dtype=float)
    mask = np.abs(denom) > EPS
    out[mask] = (dis_mu[mask] - src_mu[mask]) / denom[mask]
    return out


def source_persistence(movement: np.ndarray, *, clip_min: float = 0.0, clip_max: float = 1.0) -> float:
    valid = movement[np.isfinite(movement)]
    if valid.size == 0:
        return float("nan")
    clipped = np.clip(valid, clip_min, clip_max)
    return float(1.0 - np.nanmean(clipped))


def anisotropy(movement: np.ndarray) -> float:
    valid = movement[np.isfinite(movement)]
    if valid.size == 0:
        return float("nan")
    return float(np.nanstd(valid))


def axis_separation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(np.nanmean(target, axis=0) - np.nanmean(source, axis=0))


def train_source_target_probe(
    source: np.ndarray,
    target: np.ndarray,
    disguised: np.ndarray,
    *,
    seed: int = 42,
) -> dict:
    X = np.vstack([source, target])
    y = np.array([0] * len(source) + [1] * len(target))
    if X.ndim != 2 or disguised.ndim != 2 or X.shape[1] == 0 or disguised.shape[1] == 0:
        return dict(NAN_PROBE_METRICS)
    if len(set(y.tolist())) < 2 or len(y) < 4:
        return dict(NAN_PROBE_METRICS)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    min_class = min(np.bincount(y))
    n_splits = min(5, int(min_class))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_acc = float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())
    else:
        cv_acc = float("nan")

    clf.fit(X, y)
    preds = clf.predict(disguised)
    probs = clf.predict_proba(disguised)[:, 1]
    return {
        "probe_cv_accuracy": cv_acc,
        "source_residue": float((preds == 0).mean()),
        "target_assimilation": float((preds == 1).mean()),
        "mean_source_probability_disguised": float(1.0 - probs.mean()),
        "mean_target_probability_disguised": float(probs.mean()),
    }


def compute_behavioral_metrics(
    latent_scores: pd.DataFrame,
    *,
    axis_prefix: str = "pc",
    seed: int = 42,
    min_axis_separation: float = 0.10,
    min_probe_accuracy: float = 0.70,
    active_axes: Optional[list[str]] = None,
    self_baseline_persistence: Optional[float] = None,
) -> tuple[pd.DataFrame, dict]:
    axes = sorted(
        [c for c in latent_scores.columns if c.startswith(axis_prefix)],
        key=lambda x: int(x[len(axis_prefix):]) if x[len(axis_prefix):].isdigit() else x,
    )
    if not axes:
        raise ValueError(f"No latent axis columns with prefix '{axis_prefix}' found")

    matrices = {}
    for cond in CONDITIONS:
        subset = latent_scores[latent_scores["condition"] == cond]
        if subset.empty:
            raise ValueError(f"Missing condition rows for '{cond}'")
        matrices[cond] = subset[axes].to_numpy(dtype=float)

    movement = movement_by_axis(matrices["source"], matrices["disguised"], matrices["target"])
    separation = axis_separation(matrices["source"], matrices["target"])
    if active_axes is None:
        active_mask = separation >= float(min_axis_separation)
    else:
        active_set = set(active_axes)
        active_mask = np.array([axis in active_set for axis in axes], dtype=bool)

    active_movement = movement[active_mask]
    persistence = source_persistence(active_movement)
    aniso = anisotropy(active_movement)
    probe = train_source_target_probe(
        matrices["source"][:, active_mask],
        matrices["target"][:, active_mask],
        matrices["disguised"][:, active_mask],
        seed=seed,
    )
    probe_cv = probe.get("probe_cv_accuracy", float("nan"))
    separable = bool(np.isfinite(probe_cv) and probe_cv >= float(min_probe_accuracy) and active_mask.any())

    per_axis = pd.DataFrame(
        {
            "axis": axes,
            "axis_separation": separation,
            "active_axis": active_mask,
            "movement": movement,
            "movement_clipped": np.clip(movement, 0.0, 1.0),
            "source_mean": np.nanmean(matrices["source"], axis=0),
            "disguised_mean": np.nanmean(matrices["disguised"], axis=0),
            "target_mean": np.nanmean(matrices["target"], axis=0),
        }
    )

    summary = {
        "n": int(len(matrices["source"])),
        "n_axes": int(len(axes)),
        "n_active_axes": int(active_mask.sum()),
        "min_axis_separation": float(min_axis_separation),
        "min_probe_accuracy": float(min_probe_accuracy),
        "separable": separable,
        "source_persistence": persistence,
        "disguise_effect": float(1.0 - persistence) if np.isfinite(persistence) else float("nan"),
        "anisotropy": aniso,
        **probe,
    }
    if self_baseline_persistence is not None and np.isfinite(self_baseline_persistence):
        summary["self_baseline_persistence"] = float(self_baseline_persistence)
        summary["self_baseline_normalized_persistence"] = float(
            persistence / max(float(self_baseline_persistence), EPS)
        )
    else:
        summary["self_baseline_persistence"] = None
        summary["self_baseline_normalized_persistence"] = None

    return per_axis, summary


def _matrices_from_latent_scores(
    latent_scores: pd.DataFrame,
    axes: list[str],
    row_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    matrices = {}
    for cond in CONDITIONS:
        subset = latent_scores[latent_scores["condition"] == cond].copy()
        if "row_id" in subset.columns:
            subset = subset.sort_values("row_id")
        if row_ids is not None:
            if "row_id" not in subset.columns:
                subset = subset.iloc[row_ids]
            else:
                subset = subset.set_index("row_id").loc[row_ids].reset_index()
        matrices[cond] = subset[axes].to_numpy(dtype=float)
    return matrices


def bootstrap_behavioral_metrics(
    latent_scores: pd.DataFrame,
    *,
    active_axes: list[str],
    axis_prefix: str = "pc",
    samples: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    axes = sorted(
        [c for c in latent_scores.columns if c.startswith(axis_prefix)],
        key=lambda x: int(x[len(axis_prefix):]) if x[len(axis_prefix):].isdigit() else x,
    )
    if not axes:
        raise ValueError(f"No latent axis columns with prefix '{axis_prefix}' found")
    if samples <= 0:
        return pd.DataFrame(), {}

    active_set = set(active_axes)
    active_mask = np.array([axis in active_set for axis in axes], dtype=bool)
    if "row_id" in latent_scores.columns:
        row_ids = np.sort(latent_scores["row_id"].unique())
    else:
        n = len(latent_scores[latent_scores["condition"] == CONDITIONS[0]])
        row_ids = np.arange(n)

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(samples):
        sampled = rng.choice(row_ids, size=len(row_ids), replace=True)
        matrices = _matrices_from_latent_scores(latent_scores, axes, sampled)
        movement = movement_by_axis(matrices["source"], matrices["disguised"], matrices["target"])
        active_movement = movement[active_mask]
        persistence = source_persistence(active_movement)
        probe = train_source_target_probe(
            matrices["source"][:, active_mask],
            matrices["target"][:, active_mask],
            matrices["disguised"][:, active_mask],
            seed=seed + i,
        )
        rows.append(
            {
                "bootstrap_index": i,
                "source_persistence": persistence,
                "disguise_effect": float(1.0 - persistence) if np.isfinite(persistence) else float("nan"),
                "source_residue": probe.get("source_residue", float("nan")),
                "target_assimilation": probe.get("target_assimilation", float("nan")),
                "mean_source_probability_disguised": probe.get("mean_source_probability_disguised", float("nan")),
                "mean_target_probability_disguised": probe.get("mean_target_probability_disguised", float("nan")),
            }
        )

    boot_df = pd.DataFrame(rows)
    summary = {"bootstrap_samples": int(samples), "bootstrap_seed": int(seed)}
    for metric in [
        "source_persistence",
        "disguise_effect",
        "source_residue",
        "target_assimilation",
        "mean_source_probability_disguised",
        "mean_target_probability_disguised",
    ]:
        vals = boot_df[metric].dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            summary[f"{metric}_ci_low"] = float(np.quantile(vals, 0.025))
            summary[f"{metric}_ci_high"] = float(np.quantile(vals, 0.975))
        else:
            summary[f"{metric}_ci_low"] = float("nan")
            summary[f"{metric}_ci_high"] = float("nan")
    return boot_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute behavioral inertia metrics from latent scores.")
    parser.add_argument("--latent-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-axis-separation", type=float, default=0.10)
    parser.add_argument("--min-probe-accuracy", type=float, default=0.70)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--self-baseline-persistence", type=float)
    args = parser.parse_args()

    latent = read_csv_robust(args.latent_scores)
    per_axis, summary = compute_behavioral_metrics(
        latent,
        seed=args.seed,
        min_axis_separation=args.min_axis_separation,
        min_probe_accuracy=args.min_probe_accuracy,
        self_baseline_persistence=args.self_baseline_persistence,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_axis.to_csv(out_dir / "per_axis_movement.csv", index=False)
    if args.bootstrap_samples > 0:
        active_axes = per_axis.loc[per_axis["active_axis"], "axis"].astype(str).tolist()
        boot_df, boot_summary = bootstrap_behavioral_metrics(
            latent,
            active_axes=active_axes,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        boot_df.to_csv(out_dir / "bootstrap_summary.csv", index=False)
        summary.update(boot_summary)
    write_json(out_dir / "summary.json", summary)
    print(f"Wrote metrics to {out_dir}")


if __name__ == "__main__":
    main()
