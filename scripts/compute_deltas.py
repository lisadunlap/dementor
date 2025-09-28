#!/usr/bin/env python3
"""
Compute deltas between two model runs based on their pairwise LLM evaluation scores.

Usage:
    python scripts/compute_deltas.py --run1 path/to/run1/pairwise_scores.csv --run2 path/to/run2/pairwise_scores.csv --output deltas.csv
"""

import argparse
import pandas as pd
import os
import numpy as np

def compute_deltas(run1_path, run2_path, output_path=None):
    """Compute deltas between two runs' pairwise LLM scores."""

    # Load the pairwise scores
    run1_df = pd.read_csv(run1_path)
    run2_df = pd.read_csv(run2_path)

    # Merge on prompt to align corresponding samples
    merge_cols = ['prompt']
    if 'model_response' in run1_df.columns and 'model_response' in run2_df.columns:
        merge_cols.append('model_response')  # Include model_response if available for better matching

    merged = pd.merge(run1_df, run2_df, on=merge_cols, suffixes=('_run1', '_run2'), how='inner')

    if len(merged) == 0:
        raise ValueError("No matching prompts found between the two runs")

    # Compute deltas for LLM evaluation scores
    score_columns = ['semantic_score', 'stylistic_score', 'heuristic_match_score']
    deltas = {}

    for col in score_columns:
        if col + '_run1' in merged.columns and col + '_run2' in merged.columns:
            delta_col = f"{col}_delta"
            merged[delta_col] = merged[col + '_run2'] - merged[col + '_run1']

            # Compute summary statistics
            deltas[col] = {
                'mean_delta': merged[delta_col].mean(),
                'std_delta': merged[delta_col].std(),
                'count': len(merged),
                'run1_mean': merged[col + '_run1'].mean(),
                'run2_mean': merged[col + '_run2'].mean()
            }

    # Create summary DataFrame
    summary_data = []
    for score_type, stats in deltas.items():
        summary_data.append({
            'score_type': score_type,
            'run1_mean': stats['run1_mean'],
            'run2_mean': stats['run2_mean'],
            'mean_delta': stats['mean_delta'],
            'std_delta': stats['std_delta'],
            'sample_count': stats['count']
        })

    result = pd.DataFrame(summary_data)

    # Save if output path provided
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Deltas saved to: {output_path}")

        # Also save individual sample deltas
        sample_output = output_path.replace('.csv', '_samples.csv')
        sample_cols = ['prompt', 'model_response'] + [col for col in merged.columns if col.endswith('_delta')]
        merged[sample_cols].to_csv(sample_output, index=False)
        print(f"Individual sample deltas saved to: {sample_output}")

    return result

def main():
    parser = argparse.ArgumentParser(description="Compute deltas between two model runs")
    parser.add_argument("--run1", required=True, help="Path to first run's scored_metrics.csv")
    parser.add_argument("--run2", required=True, help="Path to second run's scored_metrics.csv")
    parser.add_argument("--output", help="Output CSV path for deltas")

    args = parser.parse_args()

    if not os.path.exists(args.run1):
        raise FileNotFoundError(f"Run1 file not found: {args.run1}")
    if not os.path.exists(args.run2):
        raise FileNotFoundError(f"Run2 file not found: {args.run2}")

    result = compute_deltas(args.run1, args.run2, args.output)

    print("Deltas computed:")
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
