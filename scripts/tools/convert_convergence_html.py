"""
Utility to convert legacy Plotly HTML convergence plots into PNG files.

The old workflow wrote Plotly HTML files under data/results/workflows/**/plots/.
This script parses those HTML files, extracts the Plotly data/layout payloads,
and regenerates static Matplotlib PNGs so documentation and downstream tooling
can rely on image files only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt


def _extract_block(text: str, start: int, opener: str, closer: str) -> Tuple[str, int]:
    """Return the substring representing a balanced block starting at `start`."""
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1], idx + 1
    raise ValueError(f"Unbalanced {opener}{closer} block in Plotly HTML around index {start}")


def _parse_plotly_payload(html: str) -> Tuple[list, dict]:
    marker = "Plotly.newPlot"
    marker_idx = html.find(marker)
    if marker_idx == -1:
        raise ValueError("Plotly.newPlot call not found")
    data_start = html.find("[", marker_idx)
    if data_start == -1:
        raise ValueError("Data array '[' not found")
    data_block, after_data = _extract_block(html, data_start, "[", "]")
    layout_start = html.find("{", after_data)
    if layout_start == -1:
        raise ValueError("Layout object '{' not found")
    layout_block, _ = _extract_block(html, layout_start, "{", "}")
    data = json.loads(data_block)
    layout = json.loads(layout_block)
    return data, layout


def convert_html_plot(html_path: Path) -> Path:
    raw = html_path.read_text(encoding="utf-8")
    data, layout = _parse_plotly_payload(raw)
    output_path = html_path.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for trace in data:
        x_vals = trace.get("x", [])
        y_vals = trace.get("y", [])
        label = trace.get("name", "series")
        ax.plot(x_vals, y_vals, marker="o", linewidth=1.5, markersize=3, label=label)

    title = None
    title_obj = layout.get("title")
    if isinstance(title_obj, dict):
        title = title_obj.get("text")
    elif isinstance(title_obj, str):
        title = title_obj
    ax.set_title(title or "Convergence")
    ax.set_xlabel("Step")

    yaxis = layout.get("yaxis")
    ylabel = None
    if isinstance(yaxis, dict):
        title_info = yaxis.get("title")
        if isinstance(title_info, dict):
            ylabel = title_info.get("text")
        elif isinstance(title_info, str):
            ylabel = title_info
    ax.set_ylabel(ylabel or "Metric Value")
    ax.grid(True, linewidth=0.3, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    html_path.unlink()
    return output_path


def main() -> None:
    root = Path("data/results/workflows")
    html_files: Iterable[Path] = root.rglob("plots/*.html")
    converted = 0
    for html_path in html_files:
        try:
            png_path = convert_html_plot(html_path)
            print(f"Converted {html_path} -> {png_path}")
            converted += 1
        except Exception as exc:  # pragma: no cover - best-effort migration
            print(f"Warning: failed to convert {html_path}: {exc}")
    if converted == 0:
        print("No legacy Plotly HTML plots found under data/results/workflows")


if __name__ == "__main__":
    main()
