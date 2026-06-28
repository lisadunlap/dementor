"""Utility for visualizing pairwise scoring summaries with Plotly."""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Sequence

import pandas as pd


@dataclass
class SummaryRecord:
    """Store metric values loaded from a summary.json file."""

    run: str
    label: str
    metrics: Dict[str, float]


def _find_summaries(root: Path) -> List[Path]:
    """Return every summary.json under the provided root."""

    return sorted(root.rglob("summary.json"))


def _prettify_method(name: str) -> str:
    mapping = {
        "contrastive": "Contrastive",
        "random_sampling": "Random Sampling",
        "behavioral": "Behavioral",
        "behavioral_based": "Behavioral",
        "baseline": "Baseline",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _shorten_model(raw: str) -> str:
    candidate = raw.split("/")[-1]
    if "_" in candidate:
        parts = candidate.split("_")
        if len(parts) > 1 and parts[1][:1].isupper():
            candidate = "_".join(parts[1:])
    mapping = {
        "Meta-Llama-3-8B-Instruct": "Meta-Llama3-8B",
        "Meta-Llama-3.1-8B-Instruct": "Meta-Llama3.1-8B",
        "Meta-Llama-3-8B": "Meta-Llama3-8B",
        "Meta-Llama-3.1-8B": "Meta-Llama3.1-8B",
        "gpt-4o": "GPT-4o",
        "gpt-4.1": "GPT-4.1",
        "gpt-4.1-mini": "GPT-4.1-mini",
        "meta-llama": "Meta-Llama",
        "gpt4.1": "GPT-4.1",
        "openai_gpt-4.1": "GPT-4.1",
        "openai_gpt-4.1-mini": "GPT-4.1-mini",
        "gpt4.1_run1": "GPT-4.1 Run1",
        "gpt4.1_run2": "GPT-4.1 Run2",
    }
    if candidate in mapping:
        return mapping[candidate]
    return candidate.replace("_", " ").replace("-", "-")


def _format_pair(text: str) -> str:
    if "_as_" in text:
        src, tgt = text.split("_as_", 1)
        return f"{_shorten_model(src)} → {_shorten_model(tgt)}"
    if "_vs_" in text:
        a, b = text.split("_vs_", 1)
        if "_" not in b and "_run" in a:
            base = a.split("_run", 1)[0]
            b = f"{base}_{b}"
        left = _shorten_model(a)
        right = _shorten_model(b)
        if left.split()[0] == right.split()[0]:
            base = left.split()[0]
            left_tail = left[len(base):].strip()
            right_tail = right[len(base):].strip()
            if left_tail and right_tail:
                return f"{base} {left_tail} vs {right_tail}"
        return f"{left} vs {right}"
    return _shorten_model(text)


def _format_run_label(summary_path: Path, root: Path) -> str:
    rel_parts = summary_path.parent.relative_to(root).parts
    label_parts: List[str] = []
    if "scores" in rel_parts:
        idx = rel_parts.index("scores")
        if idx > 0:
            label_parts.append(_prettify_method(rel_parts[idx - 1]))
        if idx + 1 < len(rel_parts):
            label_parts.append(_format_pair(rel_parts[idx + 1]))
    else:
        if len(rel_parts) >= 2:
            label_parts.append(_prettify_method(rel_parts[-2]))
        label_parts.append(_format_pair(rel_parts[-1]))
    compact = " • ".join(label_parts).strip()
    return compact or str(summary_path.parent.name)


def _load_summary(summary_path: Path, root: Path) -> SummaryRecord | None:
    """Read a summary.json file and return its metrics."""

    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    run = str(summary_path.parent.relative_to(root))
    label = _format_run_label(summary_path, root)
    numeric_metrics: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            numeric_metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return SummaryRecord(run=run, label=label, metrics=numeric_metrics)


def _collect_dataframe(
    root: Path,
    metric_keys: Sequence[str],
    *,
    exclude: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate the requested metrics from all summaries under root."""

    exclude = [term.lower() for term in (exclude or []) if term]

    rows: List[Dict[str, object]] = []
    for summary_path in _find_summaries(root):
        record = _load_summary(summary_path, root)
        if record is None:
            continue
        if exclude:
            haystack = f"{record.run} {record.label}".lower()
            if any(term in haystack for term in exclude):
                continue
        row: Dict[str, object] = {"run": record.label, "_path": record.run}
        for key, value in record.metrics.items():
            row[key] = value
        rows.append(row)
    if not rows:
        raise ValueError(f"No summaries with metrics {metric_keys} found under {root}")
    df = pd.DataFrame(rows)
    df = df.dropna(subset=list(metric_keys))
    if df.empty:
        raise ValueError(f"Requested metrics {metric_keys} are missing from all summaries")
    df = df.sort_values("_path")
    df = df.drop(columns=["_path"])
    return df


def _metric_base(metric_key: str) -> str:
    """Return the base metric identifier without trailing aggregate suffixes."""

    for suffix in ("_mean", "_avg"):
        if metric_key.endswith(suffix):
            return metric_key[: -len(suffix)]
    return metric_key


def _compute_error(
    row: pd.Series,
    metric_key: str,
    mode: str | None,
    confidence_level: float,
) -> float | None:
    """Return an error magnitude for the requested metric and mode."""

    if not mode or mode == "none":
        return None

    base = _metric_base(metric_key)
    std = row.get(f"{base}_std")
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return None

    if mode == "std":
        return float(std)

    count = row.get(f"{base}_count")
    if count is None:
        return None
    try:
        count_value = float(count)
    except (TypeError, ValueError):
        return None
    if count_value <= 0:
        return None

    std_value = float(std)
    standard_error = std_value / math.sqrt(count_value)

    if mode == "sem":
        return standard_error

    if mode == "ci":
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1 (exclusive)")
        z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        return z_score * standard_error

    raise ValueError(f"Unsupported error mode: {mode}")


def _melt_metrics(
    df: pd.DataFrame,
    metric_keys: Sequence[str],
    *,
    error_mode: str | None = None,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Convert wide metric columns into a tidy format for plotting, adding optional error bars."""

    rows: List[Dict[str, object]] = []
    for _, data_row in df.iterrows():
        run_label = data_row["run"]
        for metric_key in metric_keys:
            if metric_key not in data_row:
                continue
            value = data_row[metric_key]
            if pd.isna(value):
                continue
            entry: Dict[str, object] = {
                "run": run_label,
                "metric": metric_key,
                "value": float(value),
            }
            error_value = _compute_error(data_row, metric_key, error_mode, confidence_level)
            if error_value is not None:
                entry["error"] = float(error_value)
            rows.append(entry)

    if not rows:
        raise ValueError("No data points available for plotting")

    tidy = pd.DataFrame(rows)
    if "error" in tidy.columns and tidy["error"].isna().all():
        tidy = tidy.drop(columns=["error"])
    return tidy


def _describe_error_mode(error_mode: str | None, confidence_level: float) -> str | None:
    """Return a human-readable description of the requested error mode."""

    if not error_mode or error_mode == "none":
        return None
    if error_mode == "std":
        return "1σ"
    if error_mode == "sem":
        return "SEM"
    if error_mode == "ci":
        return f"{confidence_level * 100:.0f}% CI"
    return None


def build_figure(
    tidy_df: pd.DataFrame,
    *,
    error_label: str | None = None,
    show_values: bool = True,
):
    """Create a faceted horizontal bar chart with one panel per metric."""

    # Lazy import: keep merely importing this module free of a hard plotly dependency.
    import plotly.express as px

    run_order = list(dict.fromkeys(tidy_df["run"].tolist()))
    tidy_copy = tidy_df.copy()
    tidy_copy["run"] = pd.Categorical(tidy_copy["run"], categories=run_order[::-1], ordered=True)

    error_column = "error" if "error" in tidy_copy.columns else None

    text_kwargs: Dict[str, object] = {"text_auto": ".2f"} if show_values else {}

    fig = px.bar(
        tidy_copy,
        x="value",
        y="run",
        color="run",
        facet_col="metric",
        orientation="h",
        title="Pairwise Semantic and Stylistic Scores",
        error_x=error_column,
        **text_kwargs,
    )
    if show_values:
        fig.update_traces(
            texttemplate="%{x:.2f}",
            textposition="inside",
            insidetextanchor="start",
            textfont=dict(color="white"),
            selector=dict(type="bar"),
        )
    xaxis_title = "Score"
    if error_label:
        xaxis_title = f"Score (mean ± {error_label})"
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title="Run",
        legend_title="Run",
        bargap=0.35,
        margin=dict(t=60, r=50, b=80, l=140),
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].replace("_", " ").title()))
    for axis_name in fig.layout:
        if axis_name.startswith("xaxis"):
            getattr(fig.layout, axis_name).update(range=[0, 4])
        if axis_name.startswith("yaxis"):
            getattr(fig.layout, axis_name).update(categoryorder="array", categoryarray=run_order[::-1])
    return fig


def _fallback_static_image(
    output_path: Path,
    tidy_df: pd.DataFrame,
    metric_keys: Sequence[str],
    *,
    error_label: str | None = None,
    show_values: bool = True,
    width_px: int | None = None,
    height_px: int | None = None,
) -> None:
    """Render facet-style horizontal bar chart with matplotlib if Plotly export fails."""

    import matplotlib.pyplot as plt

    if not metric_keys:
        raise ValueError("No metric keys provided for plotting")

    run_order = list(dict.fromkeys(tidy_df["run"].tolist()))[::-1]
    n_metrics = len(metric_keys)
    dpi = 200
    if width_px and height_px:
        fig_width = max(3.0, width_px / dpi)
        fig_height = max(3.0, height_px / dpi)
    else:
        fig_width = 11.0
        fig_height = max(3.0, 0.7 * len(run_order) * n_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(fig_width, fig_height), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    palette = plt.get_cmap("tab10")
    color_map = {run: palette(i % 10) for i, run in enumerate(run_order)}

    for ax, key in zip(axes, metric_keys):
        subset = tidy_df[tidy_df["metric"] == key]
        values = subset.set_index("run")["value"].reindex(run_order)
        errors = subset.set_index("run")["error"].reindex(run_order) if "error" in subset.columns else None
        display_order = run_order
        ax.barh(display_order, values, color=[color_map[run] for run in display_order], edgecolor="black")
        ax.set_xlim(0, 4)
        label = key.replace("_", " ").title()
        ax.set_title(label)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        if show_values:
            for bar in ax.patches:
                width = bar.get_width()
                x_pos = width * 0.5
                ax.annotate(
                    f"{width:.2f}",
                    (x_pos, bar.get_y() + bar.get_height() / 2),
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )
        if errors is not None and not errors.isna().all():
            ax.errorbar(
                values,
                display_order,
                xerr=errors,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=4,
            )

    xlabel = "Score"
    if error_label:
        xlabel = f"Score (mean ± {error_label})"
    axes[-1].set_xlabel(xlabel)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize pairwise scoring summaries.")
    parser.add_argument("root", help="Directory containing run subfolders with summary.json outputs.")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["semantic_score_mean", "stylistic_score_mean"],
        help="Metric keys from summary.json to plot (default: semantic and stylistic means).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Skip runs whose label or path contains any of these case-insensitive substrings.",
    )
    parser.add_argument(
        "--error-mode",
        choices=["none", "std", "sem", "ci"],
        default="none",
        help=(
            "Error bar style: 'std' for run-level standard deviation, 'sem' for the standard error of the mean, "
            "or 'ci' for a normal-approximation confidence interval."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for interval error bars when --error-mode=ci (default: 0.95).",
    )
    parser.add_argument(
        "--force-matplotlib",
        action="store_true",
        help="Render static image with matplotlib instead of Plotly's static export.",
    )
    parser.add_argument(
        "--hide-values",
        action="store_true",
        help="Suppress numeric value labels on the bars.",
    )
    parser.add_argument("--output", help="Optional path to save the Plotly figure as HTML or image.")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width in pixels (applies to Plotly static export and matplotlib fallback).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height in pixels (applies to Plotly static export and matplotlib fallback).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    df = _collect_dataframe(root, args.metrics, exclude=args.exclude)
    tidy = _melt_metrics(
        df,
        args.metrics,
        error_mode=args.error_mode,
        confidence_level=args.confidence,
    )
    error_label = _describe_error_mode(args.error_mode, args.confidence)
    show_values = not args.hide_values
    fig = build_figure(tidy, error_label=error_label, show_values=show_values)

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".html":
            fig.write_html(str(output_path))
        elif suffix in {".png", ".jpg", ".jpeg", ".svg"}:
            if args.force_matplotlib:
                _fallback_static_image(
                    output_path,
                    tidy,
                    args.metrics,
                    error_label=error_label,
                    show_values=show_values,
                    width_px=args.width,
                    height_px=args.height,
                )
            else:
                try:
                    if args.width or args.height:
                        fig.write_image(str(output_path), width=args.width, height=args.height)
                    else:
                        fig.write_image(str(output_path))
                except Exception as exc:
                    logging.warning("Plotly static export failed (%s); falling back to matplotlib", exc)
                    _fallback_static_image(
                        output_path,
                        tidy,
                        args.metrics,
                        error_label=error_label,
                        show_values=show_values,
                        width_px=args.width,
                        height_px=args.height,
                    )
        else:
            raise ValueError("Output file must end with .html, .png, .jpg, .jpeg, or .svg")
    else:
        fig.show()


if __name__ == "__main__":
    main()
