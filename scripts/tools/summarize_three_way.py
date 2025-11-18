#!/usr/bin/env python3
"""
Summarize baseline (source vs target) and disguised vs target metrics into a small table.

Usage:
  python scripts/tools/summarize_three_way.py \
    --baseline data/results/<dataset>/comparisons/source_vs_target/<src>_vs_<tgt>.csv \
    --disguised data/results/<dataset>/comparisons/disguised_vs_target/<method>/<src>_as_<tgt>.csv \
    --output data/results/<dataset>/three_way_summary.csv \
    [--markdown data/results/<dataset>/three_way_summary.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def _resolve_scored_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "scored.csv"
        if candidate.exists():
            return candidate
    if path.name == "scored.csv":
        return path
    candidate = path.with_name(path.stem + "_scored.csv")
    if candidate.exists():
        return candidate
    return path


def load_metrics(scored_path: Path) -> dict:
    metrics_csv = scored_path.with_name("scored_metrics.csv")
    if metrics_csv.exists():
        import csv

        metrics: dict[str, float] = {}
        with open(metrics_csv, "r", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    try:
                        metrics[row[0]] = float(row[1])
                    except ValueError:
                        metrics[row[0]] = row[1]
        return metrics

    legacy_json = scored_path.with_suffix("").as_posix() + "_metrics.json"
    if Path(legacy_json).exists():
        return json.loads(Path(legacy_json).read_text())
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize baseline and disguised metrics.")
    parser.add_argument("--baseline", required=True, help="Baseline scored.csv or directory.")
    parser.add_argument("--disguised", required=True, help="Disguised scored.csv or directory.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--markdown", help="Optional Markdown table output.")
    args = parser.parse_args()

    base_scored = _resolve_scored_path(Path(args.baseline))
    dis_scored = _resolve_scored_path(Path(args.disguised))

    rows = []
    base_metrics = load_metrics(base_scored)
    if base_metrics:
        rows.append({"comparison": "baseline_source_vs_target", **base_metrics})

    dis_metrics = load_metrics(dis_scored)
    if dis_metrics:
        rows.append({"comparison": "disguised_vs_target", **dis_metrics})

    df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote summary CSV to {output_path}")

    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [c for c in ["comparison", "semantic_score_mean", "stylistic_score_mean", "heuristic_match_score_mean"] if c in df.columns]
        with md_path.open("w", encoding="utf-8") as handle:
            handle.write("| " + " | ".join(cols) + " |\n")
            handle.write("|" + "|".join(["---"] * len(cols)) + "|\n")
            for _, row in df.iterrows():
                handle.write("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |\n")
        print(f"Wrote Markdown table to {md_path}")


if __name__ == "__main__":
    main()
