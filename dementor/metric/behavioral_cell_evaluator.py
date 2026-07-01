from __future__ import annotations

import argparse
import json
from itertools import combinations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from dementor.steering.activation_bridge import run_activation_bridge
from .behavioral_inertia_metrics import EPS, projection_movement_per_row
from .common import normalize_comparison_df, read_csv_robust, slugify, write_json
from .latent_behavior_axes import LENGTH_RESIDUALIZED_SUFFIX
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
class IdentityControlSpec:
    """Positive (fully-disguised) control: an independent target generation used
    as the disguised condition. It should land on the target (persistence ~= 0),
    anchoring the 0 end of the scale opposite the self-baseline's ~=1 anchor."""

    target_runs: list[str] = field(default_factory=list)
    comparison_csv: str | None = None


@dataclass
class BehavioralCellSpec:
    dataset: str
    source_model: str
    target_model: str
    methods: list[MethodSpec]
    output_dir: str
    basis_reference: MethodSpec | None = None
    self_baseline: SelfBaselineSpec | None = None
    identity_control: IdentityControlSpec | None = None
    enforce_shared_endpoints: bool = True
    # Endpoints must still match closely, but the absolute PC means drift at the
    # ~1e-6 level across encoder/torch/transformers versions (observed pc2 deltas
    # of ~1.5e-6 on the drifted stack). 1e-5 keeps the shared-endpoint guarantee
    # tight while tolerating that float/library noise. Override per-manifest if needed.
    endpoint_tolerance: float = 1e-5
    basis_type: str = "supervised"
    lda_shrinkage: float = 0.1
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

    basis_reference = _method_from_dict(raw["basis_reference"]) if raw.get("basis_reference") else None

    baseline = None
    if raw.get("self_baseline"):
        baseline_raw = raw["self_baseline"]
        baseline = SelfBaselineSpec(
            source_runs=list(baseline_raw.get("source_runs", [])),
            comparison_csv=baseline_raw.get("comparison_csv"),
        )

    identity = None
    if raw.get("identity_control"):
        identity_raw = raw["identity_control"]
        identity = IdentityControlSpec(
            target_runs=list(identity_raw.get("target_runs", [])),
            comparison_csv=identity_raw.get("comparison_csv"),
        )

    allowed = set(BehavioralCellSpec.__dataclass_fields__) - {
        "methods",
        "basis_reference",
        "self_baseline",
        "identity_control",
    }
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    kwargs["methods"] = methods
    kwargs["basis_reference"] = basis_reference
    kwargs["self_baseline"] = baseline
    kwargs["identity_control"] = identity
    return BehavioralCellSpec(**kwargs)


def load_cell_spec(path: str | Path) -> BehavioralCellSpec:
    return spec_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _response_col(df: pd.DataFrame) -> str:
    for column in ("model_response", "response", "target_response", "source_response"):
        if column in df.columns:
            return column
    raise ValueError("Response CSV must include model_response, response, target_response, or source_response.")


def _unique_prompts(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if "prompt" not in df.columns:
        raise ValueError(f"{label} must include a prompt column.")
    out = df.copy()
    out["prompt"] = out["prompt"].astype(str)
    if out["prompt"].duplicated().any():
        examples = out.loc[out["prompt"].duplicated(), "prompt"].head(5).tolist()
        raise ValueError(f"{label} contains duplicate prompts, examples: {examples}")
    return out


def _ref_outputs(method: MethodSpec) -> pd.DataFrame:
    df = _unique_prompts(read_csv_robust(method.comparison_csv), label=f"{method.method} comparison")
    if method.source_col not in df.columns:
        if method.source_responses is None:
            raise ValueError(
                f"{method.method} comparison is missing {method.source_col}; provide source_responses."
            )
        src = _unique_prompts(read_csv_robust(method.source_responses), label=f"{method.method} source responses")
        src_col = _response_col(src)
        df = pd.merge(
            df,
            src[["prompt", src_col]].rename(columns={src_col: method.source_col}),
            on="prompt",
            how="left",
            indicator="_source_join",
        )
        missing = df["_source_join"] != "both"
        if missing.any():
            examples = df.loc[missing, "prompt"].head(5).tolist()
            raise ValueError(f"{method.method} source responses missing prompts, examples: {examples}")
        df = df.drop(columns=["_source_join"])
    if method.target_col not in df.columns:
        raise ValueError(f"{method.method} comparison is missing target column {method.target_col}.")
    return df[["prompt", method.source_col, method.target_col]].rename(
        columns={method.source_col: "source_response", method.target_col: "target_response"}
    )


def _build_baseline_csv(
    *,
    baseline: SelfBaselineSpec,
    reference_method: MethodSpec,
    output_csv: str | Path,
) -> str | None:
    if baseline.comparison_csv:
        return baseline.comparison_csv
    if len(baseline.source_runs) < 2:
        return None

    reference = _ref_outputs(reference_method)
    first = _unique_prompts(read_csv_robust(baseline.source_runs[0]), label="source baseline run 0")
    first_col = _response_col(first)
    base = pd.merge(
        reference[["prompt", "target_response"]],
        first[["prompt", first_col]].rename(columns={first_col: "source_response"}),
        on="prompt",
        how="left",
        indicator="_baseline_source_join",
    )
    missing = base["_baseline_source_join"] != "both"
    if missing.any():
        examples = base.loc[missing, "prompt"].head(5).tolist()
        raise ValueError(f"source baseline run 0 missing prompts, examples: {examples}")
    base = base.drop(columns=["_baseline_source_join"])

    rows = []
    for run_idx, run_path in enumerate(baseline.source_runs[1:], start=1):
        run_df = _unique_prompts(read_csv_robust(run_path), label=f"source baseline run {run_idx}")
        run_col = _response_col(run_df)
        merged = pd.merge(
            base,
            run_df[["prompt", run_col]].rename(columns={run_col: "model_response"}),
            on="prompt",
            how="left",
            indicator="_baseline_join",
        )
        missing = merged["_baseline_join"] != "both"
        if missing.any():
            examples = merged.loc[missing, "prompt"].head(5).tolist()
            raise ValueError(f"source baseline run {run_idx} missing prompts, examples: {examples}")
        merged = merged.drop(columns=["_baseline_join"])
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


def _build_identity_csv(
    *,
    identity: IdentityControlSpec,
    reference_method: MethodSpec,
    output_csv: str | Path,
) -> str | None:
    """Positive control: an independent target generation as the disguised column.

    Source and target endpoints come from the cell reference; the disguised column
    is a separate target draw, so movement should be ~=1 (persistence ~=0), giving
    the calibrated zero end of the scale. All ``target_runs`` are pooled (one row
    per run per prompt); they should be *independent* target seeds, not the same
    generation used as the reference target.
    """
    if identity.comparison_csv:
        return identity.comparison_csv
    if not identity.target_runs:
        return None

    reference = _ref_outputs(reference_method)
    rows = []
    for run_idx, run_path in enumerate(identity.target_runs):
        run_df = _unique_prompts(read_csv_robust(run_path), label=f"identity target run {run_idx}")
        run_col = _response_col(run_df)
        merged = pd.merge(
            reference[["prompt", "source_response", "target_response"]],
            run_df[["prompt", run_col]].rename(columns={run_col: "model_response"}),
            on="prompt",
            how="left",
            indicator="_identity_join",
        )
        missing = merged["_identity_join"] != "both"
        if missing.any():
            examples = merged.loc[missing, "prompt"].head(5).tolist()
            raise ValueError(f"identity target run {run_idx} missing prompts, examples: {examples}")
        merged = merged.drop(columns=["_identity_join"])
        merged["method"] = "identity_control"
        merged["identity_target_run"] = str(run_path)
        merged["identity_run_index"] = run_idx
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return str(path)


def _summary_path(path: Path) -> Path:
    return path / "summary.json"


def _attach_activation_summary(summary: dict[str, Any], activation: dict[str, Any]) -> dict[str, Any]:
    updated = dict(summary)
    updated["activation_bridge_mode"] = activation.get("activation_bridge_mode", activation.get("mode", "attached"))
    updated["activation_encoder_model"] = activation.get("encoder_model")
    updated["activation_best_layer"] = activation.get("best_layer")
    updated["activation_probe_cv"] = activation.get("probe_cv")
    updated["activation_source_prob"] = activation.get("activation_source_prob")
    updated["activation_target_prob"] = activation.get("activation_target_prob")
    updated["activation_summary_attached"] = True
    return updated


def _run_activation(
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


def _run_method(
    *,
    method: MethodSpec,
    method_dir: Path,
    spec: BehavioralCellSpec,
    basis_path: Path,
    feature_set: str,
    baseline: float | None,
    weighted_baseline: float | None,
    attach_activation: bool,
    active_axes: list[str] | None = None,
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
        baseline=baseline,
        weighted_baseline=weighted_baseline,
        active_axes=active_axes,
        basis_type=spec.basis_type,
        lda_shrinkage=spec.lda_shrinkage,
    )

    if attach_activation:
        activation = _run_activation(method=method, method_dir=method_dir, spec=spec)
        if activation:
            summary = _attach_activation_summary(summary, activation)
            write_json(_summary_path(method_dir), summary)
    return summary


def _fit_basis(
    *,
    spec: BehavioralCellSpec,
    output_dir: Path,
    feature_set: str,
) -> tuple[Path, dict[str, Any]]:
    """Fit the cell basis and return its path plus the canonical active-axis set
    and source/target endpoints, so every method/rung in the cell is measured on
    the same axes in the same coordinate system."""
    basis_path = output_dir / "basis" / "behavioral_axis_basis.pkl"
    first = spec.basis_reference or spec.methods[0]
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
        basis_type=spec.basis_type,
        lda_shrinkage=spec.lda_shrinkage,
    )
    per_axis = pd.read_csv(output_dir / "basis" / "fit_reference" / "per_axis_movement.csv")
    axis = per_axis["axis"].astype(str)
    canonical = {
        "active_axes": axis[per_axis["active_axis"].astype(bool)].tolist(),
        "source_mean": dict(zip(axis, per_axis["source_mean"].astype(float))),
        "target_mean": dict(zip(axis, per_axis["target_mean"].astype(float))),
    }
    return basis_path, canonical


def _check_endpoints(
    method_dir: Path,
    canonical: dict[str, Any],
    *,
    tolerance: float = 1e-5,
    label: str,
) -> None:
    """Assert a method's source/target PC means match the cell reference.

    All methods/rungs in a cell must share identical source and target
    generations; otherwise persistence is measured against different endpoints and
    cross-rung comparisons (the monotonicity claim) are not commensurable.
    """
    per_axis = pd.read_csv(method_dir / "per_axis_movement.csv")
    axis = per_axis["axis"].astype(str)
    src = dict(zip(axis, per_axis["source_mean"].astype(float)))
    tgt = dict(zip(axis, per_axis["target_mean"].astype(float)))
    bad = []
    for name in canonical["active_axes"]:
        ds = abs(src.get(name, float("nan")) - canonical["source_mean"].get(name, float("nan")))
        dt = abs(tgt.get(name, float("nan")) - canonical["target_mean"].get(name, float("nan")))
        if not (np.isfinite(ds) and np.isfinite(dt)) or ds > tolerance or dt > tolerance:
            bad.append({"axis": name, "source_delta": float(ds), "target_delta": float(dt)})
    if bad:
        raise ValueError(
            f"{label}: source/target endpoints differ from the cell reference beyond "
            f"tolerance {tolerance}: {bad[:5]}. Every method/rung in a cell must share "
            "identical source and target generations. Set enforce_shared_endpoints=false "
            "to override (cross-rung comparisons then use different coordinates)."
        )


def _self_baseline(
    *,
    spec: BehavioralCellSpec,
    output_dir: Path,
    basis_path: Path,
    feature_set: str,
    active_axes: list[str] | None = None,
) -> tuple[float | None, float | None, dict[str, Any] | None]:
    if not spec.self_baseline:
        return None, None, None

    baseline_csv = _build_baseline_csv(
        baseline=spec.self_baseline,
        reference_method=spec.basis_reference or spec.methods[0],
        output_csv=output_dir / "self_baseline" / "self_baseline_comparison.csv",
    )
    if baseline_csv is None:
        return None, None, None

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
        active_axes=active_axes,
        allow_duplicate_prompts=True,
        basis_type=spec.basis_type,
        lda_shrinkage=spec.lda_shrinkage,
    )
    persistence = summary.get("source_persistence")
    weighted = summary.get("weighted_axis_persistence")
    if persistence is None or not np.isfinite(float(persistence)):
        return None, None, summary
    weighted = (
        float(weighted)
        if weighted is not None and np.isfinite(float(weighted))
        else None
    )
    return float(persistence), weighted, summary


def _identity_control(
    *,
    spec: BehavioralCellSpec,
    output_dir: Path,
    basis_path: Path,
    feature_set: str,
    active_axes: list[str] | None = None,
) -> dict[str, Any] | None:
    """Run the positive (fully-disguised) control and return its summary, whose
    ``projection_persistence`` anchors the 0 end of the calibrated scale."""
    if not spec.identity_control:
        return None

    identity_csv = _build_identity_csv(
        identity=spec.identity_control,
        reference_method=spec.basis_reference or spec.methods[0],
        output_csv=output_dir / "identity_control" / "identity_comparison.csv",
    )
    if identity_csv is None:
        return None

    return run_behavioral_inertia(
        comparison_csv=identity_csv,
        output_dir=str(output_dir / "identity_control"),
        source_model=spec.source_model,
        target_model=spec.target_model,
        method="identity_control",
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
        active_axes=active_axes,
        allow_duplicate_prompts=True,
        basis_type=spec.basis_type,
        lda_shrinkage=spec.lda_shrinkage,
    )


def _run_core(
    *,
    spec: BehavioralCellSpec,
    output_dir: Path,
    feature_set: str,
    attach_activation: bool,
) -> tuple[pd.DataFrame, float | None, float | None, dict[str, Any] | None, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    basis_path, canonical = _fit_basis(spec=spec, output_dir=output_dir, feature_set=feature_set)
    active_axes = canonical["active_axes"]
    baseline, weighted_baseline, baseline_summary = _self_baseline(
        spec=spec,
        output_dir=output_dir,
        basis_path=basis_path,
        feature_set=feature_set,
        active_axes=active_axes,
    )

    rows = []
    for idx, method in enumerate(spec.methods):
        method_slug = f"{idx:02d}_{slugify(method.method)}"
        method_dir = output_dir / "methods" / method_slug
        summary = _run_method(
            method=method,
            method_dir=method_dir,
            spec=spec,
            basis_path=basis_path,
            feature_set=feature_set,
            baseline=baseline,
            weighted_baseline=weighted_baseline,
            attach_activation=attach_activation,
            active_axes=active_axes,
        )
        if spec.enforce_shared_endpoints:
            _check_endpoints(
                method_dir,
                canonical,
                tolerance=spec.endpoint_tolerance,
                label=f"method '{method.method}' (feature_set={feature_set})",
            )
        rows.append(
            {
                "feature_set": feature_set,
                "method": method.method,
                "method_dir": str(method_dir),
                "summary_path": str(_summary_path(method_dir)),
                "basis_path": str(basis_path),
                "n_active_axes": summary.get("n_active_axes"),
                "sep_ratio": summary.get("sep_ratio"),
                "trustworthy": summary.get("trustworthy"),
                "over_assimilation": summary.get("over_assimilation"),
                "persistence": summary.get("persistence"),
                "movement": summary.get("movement"),
                "movement_raw": summary.get("movement_raw"),
                "projection_persistence": summary.get("projection_persistence"),
                "projection_disguise": summary.get("projection_disguise"),
                "projection_persistence_all": summary.get("projection_persistence_all"),
                "baseline_persistence": baseline,
                "baseline_weighted": weighted_baseline,
                "source_persistence": summary.get("source_persistence"),
                "weighted_axis_persistence": summary.get("weighted_axis_persistence"),
                "norm_persistence": summary.get("norm_persistence"),
                "norm_weighted_persistence": summary.get(
                    "norm_weighted_persistence"
                ),
                "disguise_effect": summary.get("disguise_effect"),
                "weighted_disguise": summary.get("weighted_disguise"),
                "target_assimilation": summary.get("target_assimilation"),
                "source_residue": summary.get("source_residue"),
                "probe_cv": summary.get("probe_cv"),
                "separable": summary.get("separable"),
                "most_plastic_big5": summary.get("most_plastic_big5"),
                "most_plastic_big5_move": summary.get("most_plastic_big5_move"),
                "least_plastic_big5": summary.get("least_plastic_big5"),
                "least_plastic_big5_move": summary.get("least_plastic_big5_move"),
                "activation_bridge_mode": summary.get("activation_bridge_mode"),
                "activation_probe_cv": summary.get("activation_probe_cv"),
                "activation_source_prob": summary.get("activation_source_prob"),
            }
        )

    manifest = {
        "dataset": spec.dataset,
        "source_model": spec.source_model,
        "target_model": spec.target_model,
        "feature_set": feature_set,
        "basis_path": str(basis_path),
        "basis_reused_for_all_methods": True,
        "basis_reference": (spec.basis_reference.method if spec.basis_reference else spec.methods[0].method),
        "basis_type": spec.basis_type,
        "shared_active_axes": active_axes,
        "enforce_shared_endpoints": spec.enforce_shared_endpoints,
        "baseline_persistence": baseline,
        "baseline_weighted": weighted_baseline,
        "self_baseline_summary": baseline_summary,
        "methods": [row["summary_path"] for row in rows],
    }
    write_json(output_dir / "cell_manifest.json", manifest)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "cell_summary.csv", index=False)
    return df, baseline, weighted_baseline, baseline_summary, canonical


def _disguised_st(latent_path: Path) -> pd.Series | None:
    """Per-prompt disguised coordinate on the full-feature source->target axis.

    ``st_axis`` is already normalized (source -> 0, target -> 1), so the mean over
    prompts is the movement fraction directly. Collapses any per-prompt repeats
    (e.g. multiple control seeds) by averaging.
    """
    if not latent_path.exists():
        return None
    df = pd.read_csv(latent_path)
    if "st_axis" not in df.columns:
        return None
    subset = df[df["condition"] == "disguised"]
    if subset.empty or "prompt" not in subset.columns:
        return None
    return pd.to_numeric(subset.set_index("prompt")["st_axis"], errors="coerce").groupby("prompt").mean()


def _st_persistence(values: np.ndarray, idx: np.ndarray) -> float:
    sampled = values[idx]
    sampled = sampled[np.isfinite(sampled)]
    if not sampled.size:
        return float("nan")
    return float(1.0 - np.clip(float(np.mean(sampled)), 0.0, 1.0))


def _anchor(method: float, baseline: float | None, identity: float | None) -> float:
    """Calibrated persistence on a 0<->1 scale: identity control -> 0, self
    baseline -> 1. Falls back to the ratio against the baseline when no identity
    control is available, and to the raw value when neither anchor exists."""
    if method is None or not np.isfinite(method):
        return float("nan")
    has_b = baseline is not None and np.isfinite(baseline)
    has_i = identity is not None and np.isfinite(identity)
    if has_b and has_i:
        denom = baseline - identity
        if abs(denom) <= EPS:
            return float("nan")
        return float((method - identity) / denom)
    if has_b and abs(baseline) > EPS:
        return float(method / baseline)
    return float(method)


def _percentile_ci(name: str, values: list[float]) -> dict[str, float]:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size:
        return {
            f"{name}_ci_low": float(np.quantile(arr, 0.025)),
            f"{name}_ci_high": float(np.quantile(arr, 0.975)),
        }
    return {f"{name}_ci_low": float("nan"), f"{name}_ci_high": float("nan")}


def _anchored_bootstrap(
    *,
    method_dir: Path,
    reference_dir: Path,
    baseline_dir: Path | None,
    identity_dir: Path | None,
    active_axes: list[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Resample prompts jointly across the method, the self-baseline and the
    identity control on the k-independent full-feature source->target axis,
    recomputing persistence and the anchored metric each draw.

    Operating on the single ``st_axis`` coordinate (source -> 0, target -> 1)
    propagates the controls' sampling uncertainty into the headline normalized
    metric without depending on the PC basis or k.
    """
    method = _disguised_st(method_dir / "latent_scores.csv")
    if method is None:
        return {}

    series = {"method": method}
    if baseline_dir is not None:
        baseline = _disguised_st(baseline_dir / "latent_scores.csv")
        if baseline is not None:
            series["baseline"] = baseline
    if identity_dir is not None:
        identity = _disguised_st(identity_dir / "latent_scores.csv")
        if identity is not None:
            series["identity"] = identity

    common = sorted(set.intersection(*[set(s.index) for s in series.values()]))
    if len(common) < 2:
        return {}
    arr = {key: s.reindex(common).to_numpy(dtype=float) for key, s in series.items()}

    def metrics_for(idx: np.ndarray) -> tuple[float, float | None, float | None]:
        method_p = _st_persistence(arr["method"], idx)
        baseline_p = _st_persistence(arr["baseline"], idx) if "baseline" in arr else None
        identity_p = _st_persistence(arr["identity"], idx) if "identity" in arr else None
        return method_p, baseline_p, identity_p

    n = len(common)
    method_point, baseline_point, identity_point = metrics_for(np.arange(n))
    anchored_point = _anchor(method_point, baseline_point, identity_point)

    rng = np.random.default_rng(seed)
    method_draws, baseline_draws, identity_draws, anchored_draws = [], [], [], []
    for _ in range(max(0, int(samples))):
        idx = rng.integers(0, n, size=n)
        method_p, baseline_p, identity_p = metrics_for(idx)
        method_draws.append(method_p)
        baseline_draws.append(baseline_p)
        identity_draws.append(identity_p)
        anchored_draws.append(_anchor(method_p, baseline_p, identity_p))

    out: dict[str, Any] = {
        "n_anchored_prompts": int(n),
        "persistence_point": method_point,
        "anchored": anchored_point,
    }
    out.update(_percentile_ci("persistence", method_draws))
    out.update(_percentile_ci("anchored", anchored_draws))
    if "baseline" in arr:
        out["baseline_anchor"] = baseline_point
        out.update(_percentile_ci("baseline_anchor", baseline_draws))
        finite_b = np.array([v for v in baseline_draws if v is not None and np.isfinite(v)], dtype=float)
        sd_b = float(np.std(finite_b)) if finite_b.size > 1 else float("nan")
        if np.isfinite(sd_b) and sd_b > EPS and np.isfinite(method_point) and np.isfinite(baseline_point):
            out["z_vs_baseline"] = float((baseline_point - method_point) / sd_b)
        else:
            out["z_vs_baseline"] = float("nan")
    if "identity" in arr:
        out["identity_anchor"] = identity_point
        out.update(_percentile_ci("identity_anchor", identity_draws))
    return out


def _calibration_summary(
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
            from dementor.scorer import score_pairwise_dataframe

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
        for behavioral_metric in ("persistence", "source_persistence", "norm_persistence", "target_assimilation"):
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


def _bh_adjust(p_values: list[float]) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    finite = [(idx, p) for idx, p in enumerate(p_values) if np.isfinite(p)]
    if not finite:
        return adjusted
    ordered = sorted(finite, key=lambda item: item[1])
    m = len(ordered)
    running = 1.0
    for rank, (idx, p_value) in reversed(list(enumerate(ordered, start=1))):
        running = min(running, p_value * m / rank)
        adjusted[idx] = float(min(running, 1.0))
    return adjusted


def _prompt_persistence(method_dir: Path) -> pd.DataFrame:
    """Per-prompt persistence as ``1 - clip(projection)`` of each disguised output
    onto the supervised source->target direction over active axes.

    Uses the rotation-invariant vector projection (matching the headline metric)
    rather than averaging clipped per-axis ratios, so prompt-paired comparisons are
    consistent with the cell-level statistic.
    """
    empty = pd.DataFrame(columns=["row_id", "prompt", "prompt_persistence"])
    latent = pd.read_csv(method_dir / "latent_scores.csv")
    disguised = latent[latent["condition"] == "disguised"].copy()
    if disguised.empty:
        return empty
    if "row_id" in disguised.columns:
        disguised = disguised.sort_values("row_id")

    # Prefer the k-independent full-feature source->target axis when present.
    if "st_axis" in disguised.columns:
        movement = pd.to_numeric(disguised["st_axis"], errors="coerce").to_numpy(dtype=float)
        out = disguised[["row_id", "prompt"]].copy()
        out["prompt_persistence"] = 1.0 - np.clip(movement, 0.0, 1.0)
        return out

    per_axis = pd.read_csv(method_dir / "per_axis_movement.csv")
    active = per_axis[per_axis["active_axis"].astype(bool)].copy()
    if active.empty:
        return empty
    endpoints = {
        str(row.axis): (float(row.source_mean), float(row.target_mean))
        for row in active.itertuples(index=False)
    }
    axes = [axis for axis in endpoints if axis in disguised.columns]
    if not axes:
        return empty
    src_mu = np.array([endpoints[axis][0] for axis in axes], dtype=float)
    tgt_mu = np.array([endpoints[axis][1] for axis in axes], dtype=float)
    disguised_rows = disguised[axes].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    movement = projection_movement_per_row(disguised_rows, source_mean=src_mu, target_mean=tgt_mu)

    out = disguised[["row_id", "prompt"]].copy()
    out["prompt_persistence"] = 1.0 - np.clip(movement, 0.0, 1.0)
    return out


def _paired_stats(left: str, right: str, left_df: pd.DataFrame, right_df: pd.DataFrame) -> dict[str, Any] | None:
    merged = pd.merge(
        left_df.rename(columns={"prompt_persistence": "left_persistence"}),
        right_df.rename(columns={"prompt_persistence": "right_persistence"}),
        on=["row_id", "prompt"],
        how="inner",
    )
    if merged.empty:
        return None

    vals = merged[["left_persistence", "right_persistence"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(vals) < 2:
        return None

    diff = vals["left_persistence"].to_numpy(dtype=float) - vals["right_persistence"].to_numpy(dtype=float)
    t_p = float(stats.ttest_rel(vals["left_persistence"], vals["right_persistence"], nan_policy="omit").pvalue)
    if np.count_nonzero(np.abs(diff) > 1e-12) < 2:
        w_p = float("nan")
    else:
        try:
            w_p = float(stats.wilcoxon(vals["left_persistence"], vals["right_persistence"], zero_method="wilcox").pvalue)
        except ValueError:
            w_p = float("nan")

    return {
        "method_a": left,
        "method_b": right,
        "n_prompts": int(len(vals)),
        "mean_persistence_a": float(vals["left_persistence"].mean()),
        "mean_persistence_b": float(vals["right_persistence"].mean()),
        "mean_delta_a_minus_b": float(diff.mean()),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
    }


def _write_pairwise_stats(primary_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    for row in primary_df.itertuples(index=False):
        frames[str(row.method)] = _prompt_persistence(Path(str(row.method_dir)))

    rows = []
    for left, right in combinations(frames, 2):
        a = frames[left]
        b = frames[right]
        if a.empty or b.empty:
            continue
        row = _paired_stats(left, right, a, b)
        if row:
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["paired_t_q_bh"] = _bh_adjust(out["paired_t_p"].tolist())
        out["wilcoxon_q_bh"] = _bh_adjust(out["wilcoxon_p"].tolist())
    path = output_dir / "paired_method_comparisons.csv"
    out.to_csv(path, index=False)
    return out


def _run_feature_ablations(
    *,
    spec: BehavioralCellSpec,
    primary_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    requested = [item for item in spec.feature_ablation_sets if item and item != spec.feature_set]
    # Always include the length-residualized twin of the headline feature set so the
    # stability table answers "does persistence survive removing length?".
    lenres = f"{spec.feature_set}{LENGTH_RESIDUALIZED_SUFFIX}"
    if (
        not spec.feature_set.endswith(LENGTH_RESIDUALIZED_SUFFIX)
        and lenres != spec.feature_set
        and lenres not in requested
    ):
        requested.append(lenres)
    if not requested:
        return pd.DataFrame(), pd.DataFrame()

    long_frames = [primary_df.copy()]
    for feature_set in requested:
        ablation_dir = output_dir / "feature_ablations" / slugify(feature_set)
        ablation_df, _base, _wbase, _summary, _canon = _run_core(
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
        "projection_persistence",
        "source_persistence",
        "norm_persistence",
        "target_assimilation",
        "source_residue",
    ]
    stability_rows = []
    for method, group in long_df.groupby("method", dropna=False):
        row: dict[str, Any] = {"method": method, "n_feature_sets": int(group["feature_set"].nunique())}
        for metric in metric_cols:
            if metric not in group.columns:  # all-NaN columns are dropped by concat
                continue
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
    primary_df, baseline, weighted_baseline, baseline_summary, canonical = _run_core(
        spec=spec,
        output_dir=output_dir,
        feature_set=spec.feature_set,
        attach_activation=True,
    )

    # Positive (fully-disguised) control anchoring the 0 end of the scale (Fix 2).
    basis_path = output_dir / "basis" / "behavioral_axis_basis.pkl"
    identity_summary = _identity_control(
        spec=spec,
        output_dir=output_dir,
        basis_path=basis_path,
        feature_set=spec.feature_set,
        active_axes=canonical["active_axes"],
    )
    identity_persistence = (identity_summary or {}).get("projection_persistence")

    # Full-pipeline anchored bootstrap: resample prompts jointly across reference,
    # method, baseline and identity to put a CI on the normalized metric (Fix 3).
    reference_dir = output_dir / "basis" / "fit_reference"
    baseline_dir = (output_dir / "self_baseline") if spec.self_baseline else None
    identity_dir = (output_dir / "identity_control") if identity_summary is not None else None
    anchored_rows = [
        _anchored_bootstrap(
            method_dir=Path(str(row.method_dir)),
            reference_dir=reference_dir,
            baseline_dir=baseline_dir,
            identity_dir=identity_dir,
            active_axes=canonical["active_axes"],
            samples=spec.bootstrap_samples,
            seed=spec.bootstrap_seed,
        )
        for row in primary_df.itertuples(index=False)
    ]
    anchored_df = pd.DataFrame(anchored_rows)
    has_anchored = not anchored_df.dropna(axis=1, how="all").empty
    if has_anchored:
        augmented = pd.concat(
            [primary_df.reset_index(drop=True), anchored_df.reset_index(drop=True)], axis=1
        )
        augmented.to_csv(output_dir / "cell_summary.csv", index=False)

    paired_df = _write_pairwise_stats(primary_df, output_dir)
    calibration_df = _calibration_summary(spec=spec, primary_df=primary_df, output_dir=output_dir)
    ablation_long, ablation_stability = _run_feature_ablations(
        spec=spec,
        primary_df=primary_df,
        output_dir=output_dir,
    )

    sep_ratio = None
    if "sep_ratio" in primary_df.columns:
        sep_vals = pd.to_numeric(primary_df["sep_ratio"], errors="coerce").dropna()
        if not sep_vals.empty:
            sep_ratio = float(sep_vals.iloc[0])

    summary = {
        "dataset": spec.dataset,
        "source_model": spec.source_model,
        "target_model": spec.target_model,
        "output_dir": str(output_dir),
        "feature_set": spec.feature_set,
        "n_methods": int(len(spec.methods)),
        "basis_reused_for_all_methods": True,
        "basis_reference": (spec.basis_reference.method if spec.basis_reference else spec.methods[0].method),
        "basis_type": spec.basis_type,
        "shared_active_axes": canonical["active_axes"],
        "enforce_shared_endpoints": spec.enforce_shared_endpoints,
        "sep_ratio": sep_ratio,
        "baseline_persistence": baseline,
        "baseline_weighted": weighted_baseline,
        "has_self_baseline": baseline is not None,
        "has_identity_control": identity_summary is not None,
        "identity_persistence": identity_persistence,
        "has_anchored_bootstrap": bool(has_anchored),
        "has_paired_method_comparisons": not paired_df.empty,
        "has_calibration": not calibration_df.empty,
        "has_feature_ablations": not ablation_long.empty,
        "primary_summary_csv": str(output_dir / "cell_summary.csv"),
        "paired_method_comparisons_csv": (
            str(output_dir / "paired_method_comparisons.csv") if not paired_df.empty else None
        ),
        "feature_ablation_stability_csv": (
            str(output_dir / "feature_ablations" / "feature_ablation_stability.csv")
            if not ablation_stability.empty
            else None
        ),
        "baseline_summary": baseline_summary,
        "identity_summary": identity_summary,
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
