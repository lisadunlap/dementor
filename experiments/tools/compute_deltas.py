#!/usr/bin/env python3
"""
Compute deltas between two model runs based on their pairwise LLM evaluation scores.

Usage:
    python scripts/tools/compute_deltas.py --run1 path/to/run1/pairwise_scores.csv --run2 path/to/run2/pairwise_scores.csv --output deltas.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd


def compute_deltas(run1_path: str, run2_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """Compute deltas between two runs' pairwise LLM scores."""
    run1_df = pd.read_csv(run1_path)
    run2_df = pd.read_csv(run2_path)

    merge_cols = ['prompt']
    if 'model_response' in run1_df.columns and 'model_response' in run2_df.columns:
        merge_cols.append('model_response')

    merged = pd.merge(run1_df, run2_df, on=merge_cols, suffixes=('_run1', '_run2'), how='inner')
    if merged.empty:
        raise ValueError("No matching prompts found between the two runs")

    score_columns = ['semantic_score', 'stylistic_score', 'heuristic_match_score']
    summary_rows = []

    for col in score_columns:
        col_run1 = f"{col}_run1"
        col_run2 = f"{col}_run2"
        if col_run1 not in merged.columns or col_run2 not in merged.columns:
            continue
        delta_col = f"{col}_delta"
        merged[delta_col] = merged[col_run2] - merged[col_run1]
        summary_rows.append(
            {
                'score_type': col,
                'run1_mean': merged[col_run1].mean(),
                'run2_mean': merged[col_run2].mean(),
                'mean_delta': merged[delta_col].mean(),
                'std_delta': merged[delta_col].std(ddof=1),
                'sample_count': len(merged),
            }
        )

    result = pd.DataFrame(summary_rows)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
        sample_cols = [c for c in merged.columns if c.endswith('_delta')] + merge_cols
        sample_df = merged[merge_cols + [c for c in merged.columns if c.endswith('_delta')]]
        sample_df.to_csv(out.with_name(out.stem + "_samples.csv"), index=False)
        print(f"Deltas saved to {out} and {out.with_name(out.stem + '_samples.csv')}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute deltas between two runs' scores.")
    parser.add_argument("--run1", required=True, help="Path to first run's scored.csv")
    parser.add_argument("--run2", required=True, help="Path to second run's scored.csv")
    parser.add_argument("--output", help="Optional CSV path for summary deltas.")
    args = parser.parse_args()

    if not os.path.exists(args.run1):
        raise FileNotFoundError(f"Run1 file not found: {args.run1}")
    if not os.path.exists(args.run2):
        raise FileNotFoundError(f"Run2 file not found: {args.run2}")

    df = compute_deltas(args.run1, args.run2, args.output)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
