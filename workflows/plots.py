from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import plotly.graph_objects as go


@dataclass(frozen=True)
class ConvergenceSeries:
    """Simple container describing a metric tracked over optimization steps."""

    label: str
    metric_name: str
    steps: List[int]
    values: List[float]


def series_from_loss_history(
    loss_history: Sequence[float],
    label: str,
    metric_name: str = "training_loss",
) -> Optional[ConvergenceSeries]:
    """Convert an in-memory loss history into a plotting-ready series."""
    cleaned = [float(value) for value in loss_history if value is not None]
    if not cleaned:
        return None
    steps = list(range(1, len(cleaned) + 1))
    return ConvergenceSeries(label=label, metric_name=metric_name, steps=steps, values=cleaned)


def series_from_metric_rows(
    rows: Iterable[dict],
    metric_key: str,
    *,
    label: str,
    default_step_key: str = "step",
) -> Optional[ConvergenceSeries]:
    """Extract a convergence series from OpenAI fine-tuning job metric rows."""
    metric_values: List[float] = []
    step_indices: List[int] = []
    for index, row in enumerate(rows):
        if metric_key not in row:
            continue
        value = row[metric_key]
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        step_value = row.get(default_step_key)
        if isinstance(step_value, (int, float)):
            step_indices.append(int(step_value))
        else:
            step_indices.append(index + 1)
        metric_values.append(numeric_value)
    if not metric_values:
        return None
    return ConvergenceSeries(label=label, metric_name=metric_key, steps=step_indices, values=metric_values)


def plot_convergence(
    series_list: Sequence[ConvergenceSeries],
    output_path: Path,
    *,
    title: str = "Finetuning Convergence",
) -> Path:
    """Render one or more convergence series into an interactive Plotly HTML plot."""
    if not series_list:
        raise ValueError("series_list must contain at least one ConvergenceSeries.")

    figure = go.Figure()
    for series in series_list:
        figure.add_scatter(
            x=series.steps,
            y=series.values,
            mode="lines+markers",
            name=f"{series.label} · {series.metric_name}",
        )
    figure.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Metric Value",
        template="plotly_white",
        legend_title="Run · Metric",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path
