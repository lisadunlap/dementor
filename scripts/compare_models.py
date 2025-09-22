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
import csv
import sys

# Ensure we can import scorer whether run as a package or a script
try:
    from scorer import score_model_comparison  # when executed with sys.path including scripts/
except Exception:
    try:
        from scripts.scorer import score_model_comparison  # when executed from repo root as package
    except Exception as _e:
        raise
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="Compare two model outputs and score")
    parser.add_argument("--a", required=True, help="CSV for model A (columns: prompt, model_response)")
    parser.add_argument("--b", required=True, help="CSV for model B (columns: prompt, model_response)")
    parser.add_argument("--output", required=True, help="Output merged CSV path")
    parser.add_argument("--heuristics-only", action="store_true", help="Compute heuristics only")
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini", help="Judge model for LLM scoring (prefix with openai/ to route to OpenAI API)")
    parser.add_argument("--openai-api-base", default=None, help="Override OPENAI_API_BASE for judge routing")
    parser.add_argument("--openai-api-key", default=None, help="Override OPENAI_API_KEY for judge routing")
    args = parser.parse_args()

    # Robust CSV loading (handles embedded quotes/newlines)
    def _read_csv_robust(path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except pd.errors.ParserError:
            try:
                return pd.read_csv(path, on_bad_lines='skip', quoting=csv.QUOTE_ALL)
            except pd.errors.ParserError:
                return pd.read_csv(path, on_bad_lines='skip', quoting=csv.QUOTE_NONE, engine='python')

    df_a = _read_csv_robust(args.a)
    df_b = _read_csv_robust(args.b)
    for c in ('prompt', 'model_response'):
        if c not in df_a.columns or c not in df_b.columns:
            raise ValueError("Both input CSVs must contain 'prompt' and 'model_response' columns")

    merged = pd.merge(df_a[['prompt', 'model_response']], df_b[['prompt', 'model_response']], on='prompt', suffixes=('_a', '_b'))
    merged = merged.rename(columns={'model_response_a': 'model_response', 'model_response_b': 'target_response'})
    out = args.output
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote merged comparison CSV: {out}")

    # Ensure judge routing is correct
    jm = (args.judge_model or '').lower()
    if args.openai_api_base:
        os.environ['OPENAI_API_BASE'] = args.openai_api_base
    elif jm.startswith('openai/'):
        os.environ['OPENAI_API_BASE'] = 'https://api.openai.com/v1'
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key

    print("Pairwise scoring has moved to scripts/pairwise_scorer.py.")
    print("To score the merged file, run:")
    print(f"  python scripts/pairwise_scorer.py --input {out} --output {out.replace('.csv','_scored.csv')} --judge-model {args.judge_model}")


if __name__ == '__main__':
    main()
