from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import read_csv_robust, write_json


INTERVENTION_ORDER = [
    "source_baseline",
    "just_name_it",
    "random_sampling",
    "stylistic",
    "behavioral_based",
    "contrastive",
    "stylistic_clustering",
    "embedding_clustering",
    "sft",
    "dpo",
    "activation_steering_optional",
]


def _load_summary(path: Path) -> dict:
    if path.is_dir():
        path = path / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_ladder(summary_paths: list[str | Path]) -> pd.DataFrame:
    rows = []
    for item in summary_paths:
        summary = _load_summary(Path(item))
        method = summary.get("method", "unknown")
        order = INTERVENTION_ORDER.index(method) if method in INTERVENTION_ORDER else len(INTERVENTION_ORDER)
        rows.append(
            {
                "intervention_order": order,
                "method": method,
                "dataset": summary.get("dataset"),
                "source_model": summary.get("source_model"),
                "target_model": summary.get("target_model"),
                "n": summary.get("n"),
                "source_persistence": summary.get("source_persistence"),
                "disguise_effect": summary.get("disguise_effect"),
                "target_assimilation": summary.get("target_assimilation"),
                "source_residue": summary.get("source_residue"),
                "anisotropy": summary.get("anisotropy"),
                "probe_cv_accuracy": summary.get("probe_cv_accuracy"),
                "activation_bridge_mode": summary.get("activation_bridge_mode"),
                "activation_source_probability_disguised": summary.get("activation_source_probability_disguised"),
                "summary_path": str(item),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset", "source_model", "target_model", "intervention_order"]).reset_index(drop=True)
    return df


def _plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    if df.empty or metric not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    group_cols = ["dataset", "source_model", "target_model"]
    for key, group in df.groupby(group_cols, dropna=False):
        label = " | ".join(str(x) for x in key)
        group = group.sort_values("intervention_order")
        ax.plot(group["method"], group[metric], marker="o", label=label)
    ax.set_ylabel(metric)
    ax.set_xlabel("intervention")
    ax.tick_params(axis="x", rotation=25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "dataset",
        "source_model",
        "target_model",
        "method",
        "source_persistence",
        "target_assimilation",
        "source_residue",
        "anisotropy",
        "probe_cv_accuracy",
    ]
    cols = [c for c in cols if c in df.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate behavioral inertia summaries into an intervention ladder.")
    parser.add_argument("--summaries", nargs="+", required=True, help="summary.json files or directories containing them.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = aggregate_ladder(args.summaries)
    df.to_csv(out_dir / "intervention_ladder.csv", index=False)
    write_markdown(df, out_dir / "intervention_ladder.md")
    _plot_metric(df, "source_persistence", out_dir / "persistence_vs_intervention.png")
    _plot_metric(df, "target_assimilation", out_dir / "target_assimilation_vs_intervention.png")
    _plot_metric(df, "anisotropy", out_dir / "anisotropy_vs_intervention.png")
    write_json(out_dir / "config.json", {"summaries": [str(x) for x in args.summaries]})
    print(f"Wrote intervention ladder to {out_dir}")


if __name__ == "__main__":
    main()
