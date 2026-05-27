from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .activation_bridge import run_activation_bridge
from .common import normalize_comparison_df, read_csv_robust, slugify, write_json
from .run_behavioral_inertia import run_behavioral_inertia


DEFAULT_ABLATION_FEATURE_SETS = ("adjectives", "style_all", "style_scalars")


@dataclass
class MethodSpec:
    method: str
    comparison_csv: str
    source_responses: str | None = None
    source_col: str = "source_response"
    disguised_col: str = "model_response"
    target_col: str = "target_response"
    activation_summary: str | None = None
    activation_bridge: dict[str, Any] | None = None
    calibration_scored_csv: str | None = None


@dataclass
class SelfBaselineSpec:
    source_runs: list[str] = field(default_factory=list)
    comparison_csv: str | None = None


@dataclass
class BehavioralCellSpec:
    dataset: str
    source_model: str
    target_model: str
    methods: list[MethodSpec]
    output_dir: str
    self_baseline: SelfBaselineSpec | None = None
    descriptor_mode: str = "big5_style"
    encoder_model: str | None = None
    feature_set: str = "full"
    feature_ablation_sets: list[str] = field(default_factory=lambda: list(DEFAULT_ABLATION_FEATURE_SETS))
    k: int = 5
    seed: int = 42
    min_axis_separation: float = 0.10
    min_probe_accuracy: float = 0.70
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 42
    calibration_sample_size: int = 0
    calibration_judge_model: str | None = None


def _method_from_dict(raw: dict[str, Any]) -> MethodSpec:
    allowed = set(MethodSpec.__dataclass_fields__)
    return MethodSpec(**{key: value for key, value in raw.items() if key in allowed})


def spec_from_dict(raw: dict[str, Any]) -> BehavioralCellSpec:
    methods = [_method_from_dict(item) for item in raw.get("methods", [])]
    if not methods:
        raise ValueError("Cell manifest must include at least one method entry.")

    baseline = None
    if raw.get("self_baseline"):
        baseline_raw = raw["self_baseline"]
        baseline = SelfBaselineSpec(
            source_runs=list(baseline_raw.get("source_runs", [])),
            comparison_csv=baseline_raw.get("comparison_csv"),
        )

    allowed = set(BehavioralCellSpec.__dataclass_fields__) - {"methods", "self_baseline"}
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    kwargs["methods"] = methods
    kwargs["self_baseline"] = baseline
    return BehavioralCellSpec(**kwargs)


def load_cell_spec(path: str | Path) -> BehavioralCellSpec:
    return spec_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _response_col(df: pd.DataFrame) -> str:
    for column in ("model_response", "response", "target_response", "source_response"):
        if column in df.columns:
            return column
    raise ValueError("Response CSV must include model_response, response, target_response, or source_response.")


def _reference_source_target(method: MethodSpec) -> pd.DataFrame:
    df = read_csv_robust(method.comparison_csv)
    if method.source_col not in df.columns:
        if method.source_responses is None:
            raise ValueError(
                f"{method.method} comparison is missing {method.source_col}; provide source_responses."
            )
        src = read_csv_robust(method.source_responses)
        src_col = _response_col(src)
        df = pd.merge(
            df,
            src[["prompt", src_col]].rename(columns={src_col: method.source_col}),
            on="prompt",
            how="inner",
        )
    if method.target_col not in df.columns:
        raise ValueError(f"{method.method} comparison is missing target column {method.target_col}.")
    return df[["prompt", method.source_col, method.target_col]].rename(
        columns={method.source_col: "source_response", method.target_col: "target_response"}
    )


def build_self_baseline_comparison(
    *,
    baseline: SelfBaselineSpec,
    reference_method: MethodSpec,
    output_csv: str | Path,
) -> str | None:
    if baseline.comparison_csv:
        return baseline.comparison_csv
    if len(baseline.source_runs) < 2:
        return None

    reference = _reference_source_target(reference_method)
    first = read_csv_robust(baseline.source_runs[0])
    first_col = _response_col(first)
    base = pd.merge(
        reference[["prompt", "target_response"]],
        first[["prompt", first_col]].rename(columns={first_col: "source_response"}),
        on="prompt",
        how="inner",
    )

    rows = []
    for run_idx, run_path in enumerate(baseline.source_runs[1:], start=1):
        run_df = read_csv_robust(run_path)
        run_col = _response_col(run_df)
        merged = pd.merge(
            base,
            run_df[["prompt", run_col]].rename(columns={run_col: "model_response"}),
            on="prompt",
            how="inner",
        )
        merged["method"] = "source_baseline"
        merged["baseline_source_run"] = str(baseline.source_runs[0])
        merged["baseline_disguised_run"] = str(run_path)
        merged["baseline_pair_index"] = run_idx
        rows.append(merged)
    if not rows:
        return None

    out = pd.concat(rows, ignore_index=True)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return str(path)


def _summary_path(path: Path) -> Path:
    return path / "summary.json"


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    write_json(path, summary)


def _attach_activation_summary(summary: dict[str, Any], activation: dict[str, Any]) -> dict[str, Any]:
    updated = dict(summary)
    updated["activation_bridge_mode"] = activation.get("activation_bridge_mode", activation.get("mode", "attached"))
    updated["activation_encoder_model"] = activation.get("encoder_model")
    updated["activation_best_layer"] = activation.get("best_layer")
    updated["activation_probe_cv_accuracy"] = activation.get("probe_cv_accuracy")
    updated["activation_source_probability_disguised"] = activation.get("activation_source_probability_disguised")
    updated["activation_target_probability_disguised"] = activation.get("activation_target_probability_disguised")
    updated["activation_summary_attached"] = True
    return updated


def _run_activation_for_method(
    *,
    method: MethodSpec,
    method_dir: Path,
    spec: BehavioralCellSpec,
) -> dict[str, Any] | None:
    if method.activation_summary:
        return json.loads(Path(method.activation_summary).read_text(encoding="utf-8"))
    if not method.activation_bridge:
        return None

    cfg = dict(method.activation_bridge)
    cfg.setdefault("comparison_csv", method.comparison_csv)
    cfg.setdefault("output_dir", str(method_dir / "activation_bridge"))
    cfg.setdefault("source_model", spec.source_model)
    cfg.setdefault("target_model", spec.target_model)
    cfg.setdefault("source_responses", method.source_responses)
    cfg.setdefault("source_col", method.source_col)
    cfg.setdefault("disguised_col", method.disguised_col)
    cfg.setdefault("target_col", method.target_col)
    return run_activation_bridge(**cfg)


def _run_one_method(
    *,
    method: MethodSpec,
    method_dir: Path,
    spec: BehavioralCellSpec,
    basis_path: Path,
    feature_set: str,
    self_baseline_persistence: float | None,
    attach_activation: bool,
) -> dict[str, Any]:
    summary = run_behavioral_inertia(
        comparison_csv=method.comparison_csv,
        output_dir=str(method_dir),
        source_responses=method.source_responses,
        source_model=spec.source_model,
        target_model=spec.target_model,
        method=method.method,
        dataset=spec.dataset,
        source_col=method.source_col,
        disguised_col=method.disguised_col,
        target_col=method.target_col,
        descriptor_mode=spec.descriptor_mode,
        encoder_model=spec.encoder_model or "sentence-transformers/all-MiniLM-L6-v2",
        feature_set=feature_set,
        load_basis=str(basis_path),
        k=spec.k,
        seed=spec.seed,
        min_axis_separation=spec.min_axis_separation,
        min_probe_accuracy=spec.min_probe_accuracy,
        bootstrap_samples=spec.bootstrap_samples,
        bootstrap_seed=spec.bootstrap_seed,
        self_baseline_persistence=self_baseline_persistence,
    )

    if attach_activation:
        activation = _run_activation_for_method(method=method, method_dir=method_dir, spec=spec)
        if activation:
            summary = _attach_activation_summary(summary, activation)
            _write_summary(_summary_path(method_dir), summary)
    return summary


def _fit_cell_basis(
    *,
    spec: BehavioralCellSpec,
    output_dir: Path,
    feature_set: str,
) -> Path:
    basis_path = output_dir / "basis" / "behavioral_axis_basis.pkl"
    first = spec.methods[0]
    run_behavioral_inertia(
        comparison_csv=first.comparison_csv,
        output_dir=str(output_dir / "basis" / "fit_reference"),
        source_responses=first.source_responses,
        source_model=spec.source_model,
        target_model=spec.target_model,
        method=f"{first.method}_basis_fit_reference",
        dataset=spec.dataset,
        source_col=first.source_col,
        disguised_col=first.disguised_col,
        target_col=first.target_col,
        descriptor_mode=spec.descriptor_mode,
        encoder_model=spec.encoder_model or "sentence-transformers/all-MiniLM-L6-v2",
        feature_set=feature_set,
        save_basis=str(basis_path),
        k=spec.k,
        seed=spec.seed,
        min_axis_separation=spec.min_axis_separation,
        min_probe_accuracy=spec.min_probe_accuracy,
        bootstrap_samples=0,
    )
    return basis_path


def _compute_self_baseline(
    *,
    spec: BehavioralCellSpec,
    output_dir: Path,
    basis_path: Path,
    feature_set: str,
) -> tuple[float | None, dict[str, Any] | None]:
    if not spec.self_baseline:
        return None, None

    baseline_csv = build_self_baseline_comparison(
        baseline=spec.self_baseline,
        reference_method=spec.methods[0],
        output_csv=output_dir / "self_baseline" / "self_baseline_comparison.csv",
    )
    if baseline_csv is None:
        return None, None

    summary = run_behavioral_inertia(
        comparison_csv=baseline_csv,
        output_dir=str(output_dir / "self_baseline"),
        source_model=spec.source_model,
        target_model=spec.target_model,
        method="source_baseline",
        dataset=spec.dataset,
        descriptor_mode=spec.descriptor_mode,
        encoder_model=spec.encoder_model or "sentence-transformers/all-MiniLM-L6-v2",
        feature_set=feature_set,
        load_basis=str(basis_path),
        k=spec.k,
        seed=spec.seed,
        min_axis_separation=spec.min_axis_separation,
        min_probe_accuracy=spec.min_probe_accuracy,
        bootstrap_samples=spec.bootstrap_samples,
        bootstrap_seed=spec.bootstrap_seed,
    )
    persistence = summary.get("source_persistence")
    if persistence is None or not np.isfinite(float(persistence)):
        return None, summary
    return float(persistence), summary


def _run_cell_core(
    *,
    spec: BehavioralCellSpec,
    output_dir: Path,
    feature_set: str,
    attach_activation: bool,
) -> tuple[pd.DataFrame, float | None, dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    basis_path = _fit_cell_basis(spec=spec, output_dir=output_dir, feature_set=feature_set)
    baseline_persistence, baseline_summary = _compute_self_baseline(
        spec=spec,
        output_dir=output_dir,
        basis_path=basis_path,
        feature_set=feature_set,
    )

    rows = []
    for idx, method in enumerate(spec.methods):
        method_slug = f"{idx:02d}_{slugify(method.method)}"
        method_dir = output_dir / "methods" / method_slug
        summary = _run_one_method(
            method=method,
            method_dir=method_dir,
            spec=spec,
            basis_path=basis_path,
            feature_set=feature_set,
            self_baseline_persistence=baseline_persistence,
            attach_activation=attach_activation,
        )
        rows.append(
            {
                "feature_set": feature_set,
                "method": method.method,
                "method_dir": str(method_dir),
                "summary_path": str(_summary_path(method_dir)),
                "basis_path": str(basis_path),
                "self_baseline_persistence": baseline_persistence,
                "source_persistence": summary.get("source_persistence"),
                "self_baseline_normalized_persistence": summary.get("self_baseline_normalized_persistence"),
                "disguise_effect": summary.get("disguise_effect"),
                "target_assimilation": summary.get("target_assimilation"),
                "source_residue": summary.get("source_residue"),
                "probe_cv_accuracy": summary.get("probe_cv_accuracy"),
                "separable": summary.get("separable"),
                "activation_bridge_mode": summary.get("activation_bridge_mode"),
                "activation_probe_cv_accuracy": summary.get("activation_probe_cv_accuracy"),
                "activation_source_probability_disguised": summary.get("activation_source_probability_disguised"),
            }
        )

    manifest = {
        "dataset": spec.dataset,
        "source_model": spec.source_model,
        "target_model": spec.target_model,
        "feature_set": feature_set,
        "basis_path": str(basis_path),
        "basis_reused_for_all_methods": True,
        "self_baseline_persistence": baseline_persistence,
        "self_baseline_summary": baseline_summary,
        "methods": [row["summary_path"] for row in rows],
    }
    write_json(output_dir / "cell_manifest.json", manifest)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "cell_summary.csv", index=False)
    return df, baseline_persistence, baseline_summary


def _summarize_calibration(
    *,
    spec: BehavioralCellSpec,
    primary_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    scored_frames = []
    rng = np.random.default_rng(spec.seed)

    for method in spec.methods:
        if method.calibration_scored_csv:
            scored = read_csv_robust(method.calibration_scored_csv).copy()
        elif spec.calibration_judge_model and spec.calibration_sample_size > 0:
            from scripts.scorer import score_pairwise_dataframe

            raw = normalize_comparison_df(
                method.comparison_csv,
                source_responses=method.source_responses,
                source_col=method.source_col,
                disguised_col=method.disguised_col,
                target_col=method.target_col,
            ).rename(columns={"disguised_response": "model_response"})
            if spec.calibration_sample_size < len(raw):
                idx = rng.choice(raw.index.to_numpy(), size=spec.calibration_sample_size, replace=False)
                raw = raw.loc[np.sort(idx)].copy()
            scored = score_pairwise_dataframe(raw, judge_model=spec.calibration_judge_model)
        else:
            continue

        scored["method"] = method.method
        scored_frames.append(scored)
        row = {"method": method.method, "n_calibration": int(len(scored))}
        for metric in ("semantic_score", "stylistic_score", "heuristic_match_score"):
            if metric in scored.columns:
                values = pd.to_numeric(scored[metric], errors="coerce").dropna()
                if not values.empty:
                    row[f"{metric}_mean"] = float(values.mean())
                    row[f"{metric}_std"] = float(values.std())
        rows.append(row)

    out_dir = output_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    if scored_frames:
        pd.concat(scored_frames, ignore_index=True).to_csv(out_dir / "calibration_scored.csv", index=False)

    cal_df = pd.DataFrame(rows)
    if cal_df.empty:
        return cal_df

    merged = pd.merge(primary_df, cal_df, on="method", how="inner")
    corr_rows = []
    for calibration_metric in ("stylistic_score_mean", "semantic_score_mean", "heuristic_match_score_mean"):
        if calibration_metric not in merged.columns:
            continue
        for behavioral_metric in ("source_persistence", "self_baseline_normalized_persistence", "target_assimilation"):
            if behavioral_metric not in merged.columns:
                continue
            pair = merged[[behavioral_metric, calibration_metric]].apply(pd.to_numeric, errors="coerce").dropna()
            if (
                len(pair) >= 2
                and pair[behavioral_metric].nunique(dropna=True) >= 2
                and pair[calibration_metric].nunique(dropna=True) >= 2
            ):
                corr_rows.append(
                    {
                        "behavioral_metric": behavioral_metric,
                        "calibration_metric": calibration_metric,
                        "spearman": float(pair[behavioral_metric].corr(pair[calibration_metric], method="spearman")),
                        "pearson": float(pair[behavioral_metric].corr(pair[calibration_metric], method="pearson")),
                        "n_methods": int(len(pair)),
                    }
                )

    cal_df.to_csv(out_dir / "calibration_summary.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out_dir / "calibration_correlations.csv", index=False)
    return cal_df


def _run_feature_ablations(
    *,
    spec: BehavioralCellSpec,
    primary_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    requested = [item for item in spec.feature_ablation_sets if item and item != spec.feature_set]
    if not requested:
        return pd.DataFrame(), pd.DataFrame()

    long_frames = [primary_df.copy()]
    for feature_set in requested:
        ablation_dir = output_dir / "feature_ablations" / slugify(feature_set)
        ablation_df, _baseline, _summary = _run_cell_core(
            spec=spec,
            output_dir=ablation_dir,
            feature_set=feature_set,
            attach_activation=False,
        )
        long_frames.append(ablation_df)

    long_df = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in long_frames if not frame.empty],
        ignore_index=True,
    )
    metric_cols = [
        "source_persistence",
        "self_baseline_normalized_persistence",
        "target_assimilation",
        "source_residue",
    ]
    stability_rows = []
    for method, group in long_df.groupby("method", dropna=False):
        row: dict[str, Any] = {"method": method, "n_feature_sets": int(group["feature_set"].nunique())}
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if not values.empty:
                row[f"{metric}_min"] = float(values.min())
                row[f"{metric}_max"] = float(values.max())
                row[f"{metric}_range"] = float(values.max() - values.min())
                row[f"{metric}_std"] = float(values.std())
        stability_rows.append(row)

    ablation_out = output_dir / "feature_ablations"
    ablation_out.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(ablation_out / "feature_ablation_long.csv", index=False)
    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(ablation_out / "feature_ablation_stability.csv", index=False)
    return long_df, stability_df


def run_behavioral_cell(spec: BehavioralCellSpec) -> dict[str, Any]:
    output_dir = Path(spec.output_dir)
    primary_df, baseline_persistence, baseline_summary = _run_cell_core(
        spec=spec,
        output_dir=output_dir,
        feature_set=spec.feature_set,
        attach_activation=True,
    )
    calibration_df = _summarize_calibration(spec=spec, primary_df=primary_df, output_dir=output_dir)
    ablation_long, ablation_stability = _run_feature_ablations(
        spec=spec,
        primary_df=primary_df,
        output_dir=output_dir,
    )

    summary = {
        "dataset": spec.dataset,
        "source_model": spec.source_model,
        "target_model": spec.target_model,
        "output_dir": str(output_dir),
        "feature_set": spec.feature_set,
        "n_methods": int(len(spec.methods)),
        "basis_reused_for_all_methods": True,
        "self_baseline_persistence": baseline_persistence,
        "has_self_baseline": baseline_persistence is not None,
        "has_calibration": not calibration_df.empty,
        "has_feature_ablations": not ablation_long.empty,
        "primary_summary_csv": str(output_dir / "cell_summary.csv"),
        "feature_ablation_stability_csv": (
            str(output_dir / "feature_ablations" / "feature_ablation_stability.csv")
            if not ablation_stability.empty
            else None
        ),
        "baseline_summary": baseline_summary,
    }
    write_json(output_dir / "cell_evaluation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one behavioral-inertia source-target cell from a manifest.")
    parser.add_argument("--manifest", required=True, help="JSON manifest describing the source-target cell.")
    args = parser.parse_args()

    summary = run_behavioral_cell(load_cell_spec(args.manifest))
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
