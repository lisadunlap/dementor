#!/usr/bin/env python3
"""
Compare two base model outputs over the same prompts and run scoring.

Inputs:
  --a responses_a.csv  (must contain columns: prompt, model_response)
  --b responses_b.csv  (same)

Outputs:
  Writes a merged CSV with columns: prompt, model_response, target_response
  Then runs the scorer to produce scored CSV and metrics.
"""
import argparse
import os
import pandas as pd
from scorer import score_model_comparison


def main():
    parser = argparse.ArgumentParser(description="Compare two model outputs and score")
    parser.add_argument("--a", required=True, help="CSV for model A (columns: prompt, model_response)")
    parser.add_argument("--b", required=True, help="CSV for model B (columns: prompt, model_response)")
    parser.add_argument("--output", required=True, help="Output merged CSV path")
    parser.add_argument("--heuristics-only", action="store_true", help="Compute heuristics only")
    parser.add_argument("--judge-model", default="gpt-4o", help="Judge model for LLM scoring")
    args = parser.parse_args()

    df_a = pd.read_csv(args.a)
    df_b = pd.read_csv(args.b)
    for c in ('prompt', 'model_response'):
        if c not in df_a.columns or c not in df_b.columns:
            raise ValueError("Both input CSVs must contain 'prompt' and 'model_response' columns")

    merged = pd.merge(df_a[['prompt', 'model_response']], df_b[['prompt', 'model_response']], on='prompt', suffixes=('_a', '_b'))
    merged = merged.rename(columns={'model_response_a': 'model_response', 'model_response_b': 'target_response'})
    out = args.output
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote merged comparison CSV: {out}")

    scored = out.replace('.csv', '_scored.csv')
    score_model_comparison(out, scored, heuristics_only=args.heuristics_only, judge_model=args.judge_model)
    print(f"Wrote scored CSV: {scored}")
    print(f"Metrics JSON: {scored.replace('.csv', '_metrics.json')}")


if __name__ == '__main__':
    main()

