from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px


def _select_primary_metric(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "win_rate",
        "preference_win",
        "preference_score",
        "score",
        "similarity",
        "accuracy",
    ]
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in ("i", "u", "f")]
    for c in candidates:
        if c in numeric_cols:
            return c
    for c in numeric_cols:
        lc = c.lower()
        if lc.startswith("win") or "score" in lc or "similar" in lc or lc.startswith("acc"):
            return c
    return None


def _mean_ci95(series: pd.Series) -> Tuple[float, float]:
    vals = series.dropna().astype(float)
    if len(vals) == 0:
        return float("nan"), float("nan")
    mean = float(vals.mean())
    if len(vals) == 1:
        return mean, 0.0
    stderr = float(vals.std(ddof=1)) / (len(vals) ** 0.5)
    return mean, 1.96 * stderr


def _scan_scored(root: Path) -> List[Path]:
    return list(root.rglob("scored.csv"))


def _label_from_path(p: Path) -> Tuple[str, str]:
    # Returns (method_label, direction_label)
    parts = p.parts
    # Expect something like .../gsm8k/500/<method or comparisons/self>/.../scored.csv
    try:
        idx = parts.index("gsm8k")
        subset = parts[idx + 1]
        method_or_comp = parts[idx + 2] if len(parts) > idx + 2 else "unknown"
    except ValueError:
        method_or_comp = "unknown"

    fname = p.name
    parent = p.parent
    # Infer direction from filename
    direction = "unknown"
    stem = parent.parent.name if parent.name == "scores" else parent.name
    # common pattern: <src>_as_<tgt>
    if "_as_" in stem:
        src, tgt = stem.split("_as_", 1)
        if "Meta-Llama-3.1-8B" in tgt or "Meta-Llama-3.1-8B-Instruct" in tgt or "Llama-3.1-8B" in tgt:
            direction = "vs Llama-3.1-8B-Instruct"
        elif "gpt-4.1-mini" in tgt or "gpt-4.1" in tgt:
            direction = "vs GPT-4.1-mini"
        else:
            direction = f"vs {tgt}"
    elif "self" in parts:
        # self comparisons
        if "Meta-Llama" in str(p):
            direction = "Llama self"
        elif "gpt-4.1" in str(p):
            direction = "GPT-4.1-mini self"

    method_label = method_or_comp
    if method_or_comp in ("comparisons",):
        # refine for baseline/self folders
        if "baseline" in parts:
            method_label = "baseline"
        elif "self" in parts:
            method_label = "self"

    return method_label, direction


def build_aggregates(scored_paths: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for p in scored_paths:
        try:
            df = pd.read_csv(p)
            metric = _select_primary_metric(df)
            if metric is None:
                continue
            mean, ci = _mean_ci95(df[metric])
            method, direction = _label_from_path(p)
            rows.append({
                "method": method,
                "direction": direction,
                "metric": metric,
                "mean": mean,
                "ci95": ci,
                "path": str(p),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def plot_groups(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    # We only produce two charts: vs Llama-3.1-8B-Instruct and vs GPT-4.1-mini
    allowed_methods = {
        "contrastive",
        "behavioral_based",
        "random_sampling",
        "stylistic_clustering",
        "embedding_clustering",
        "just_name_it",
        "baseline",
        "self",
    }

    def _make_plot(direction_label: str, self_label: str, file_stub: str):
        # Core bars: all methods comparing to the target direction
        core = df[(df["direction"] == direction_label) & (df["method"].isin(allowed_methods))]
        # Add self bar
        self_df = df[(df["direction"] == self_label)].copy()
        if not self_df.empty:
            self_df.loc[:, "method"] = "self"
        merged = pd.concat([core, self_df], ignore_index=True)
        if merged.empty:
            return None
        # Sort methods for consistent display
        order = [
            "contrastive",
            "behavioral_based",
            "random_sampling",
            "stylistic_clustering",
            "embedding_clustering",
            "just_name_it",
            "baseline",
            "self",
        ]
        merged["method"] = pd.Categorical(merged["method"], categories=order, ordered=True)
        merged = merged.sort_values(["method"]).dropna(subset=["mean"]) 
        title = f"Scores ({direction_label})"
        fig = px.bar(
            merged,
            x="method",
            y="mean",
            error_y="ci95",
            color="method",
            title=title,
            labels={"mean": merged["metric"].iloc[0] if not merged.empty else "score"},
        )
        fig.update_layout(showlegend=False, yaxis_title="score (mean ± 95% CI)")
        # Write HTML
        out_html = out_dir / f"scores_{file_stub}.html"
        fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
        # Write PNG (requires kaleido)
        out_png = out_dir / f"scores_{file_stub}.png"
        try:
            fig.write_image(str(out_png), scale=2)
        except Exception:
            # If kaleido is not installed, skip PNG silently; user can install with `pip install -U kaleido`
            out_png = None
        return out_html

    out1 = _make_plot("vs Llama-3.1-8B-Instruct", "Llama self", "vs_Llama-3.1-8B-Instruct")
    if out1:
        outputs.append(out1)
    out2 = _make_plot("vs GPT-4.1-mini", "GPT-4.1-mini self", "vs_GPT-4.1-mini")
    if out2:
        outputs.append(out2)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Aggregate and visualize pairwise scores")
    parser.add_argument("--root", type=Path, default=Path("data/results/gsm8k/500"))
    parser.add_argument("--out", type=Path, default=Path("data/results/gsm8k/500/scores_visualizations"))
    args = parser.parse_args()

    scored = _scan_scored(args.root)
    df = build_aggregates(scored)
    if df.empty:
        print("No scored.csv files found.")
        return
    outs = plot_groups(df, args.out)
    print("Wrote:")
    for p in outs:
        print("-", p)


if __name__ == "__main__":
    main()


