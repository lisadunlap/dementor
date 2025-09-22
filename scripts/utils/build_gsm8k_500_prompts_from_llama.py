#!/usr/bin/env python3
"""
Extract the exact 500 GSM8K prompts from the Llama-3-8B response CSV and
write them to a .csv (with header 'prompt').

Why: Keep prompts consistent across source/target/disguised runs so pairwise
joins are exact and scoring is comparable.

Usage:
  # CSV output
  python scripts/utils/build_gsm8k_500_prompts_from_llama.py \
    --responses data/results/gsm8k/comparisons/disguised_vs_target/random_sampling/gpt-4.1_as_meta-llama_Meta-Llama-3-8B-Instruct.csv \
    --out-csv data/datasets/gsm8k/gsm8k_prompts_500.csv \
    --limit 500
"""
import argparse
import os
import csv
import pandas as pd


def read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        try:
            return pd.read_csv(path, on_bad_lines='skip', quoting=csv.QUOTE_ALL)
        except pd.errors.ParserError:
            return pd.read_csv(path, on_bad_lines='skip', quoting=csv.QUOTE_NONE, engine='python')


def main():
    ap = argparse.ArgumentParser(description="Build GSM8K 500 prompts from Llama CSV")
    ap.add_argument('--responses', required=True, help="Path to Llama-3-8B responses CSV with a 'prompt' column")
    ap.add_argument('--out-csv', required=True, help="Output .csv file with header 'prompt'")
    ap.add_argument('--limit', type=int, default=500, help="Max prompts to write (default 500)")
    args = ap.parse_args()

    df = read_csv_robust(args.responses)
    if 'prompt' not in df.columns:
        raise ValueError("Input CSV must contain a 'prompt' column")

    # Preserve order, drop NaN and duplicates
    prompts = df['prompt'].astype(str).dropna()
    prompts = prompts.drop_duplicates(keep='first')
    if args.limit is not None:
        prompts = prompts.head(args.limit)

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    pd.DataFrame({'prompt': prompts}).to_csv(args.out_csv, index=False)

    print("Wrote", len(prompts), "prompts to:")
    print("-", args.out_csv)


if __name__ == '__main__':
    main()
