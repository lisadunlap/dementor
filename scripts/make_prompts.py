#!/usr/bin/env python3
"""
Make a prompts file from a dataset.

Supports CSV (specify a column), JSONL (specify a key), or plain text passthrough.
Default output is a .txt file with one prompt per line, which is what
scripts/generate_responses.py expects.

Usage examples:
  # GSM8K CSV -> TXT (column 'question')
  python scripts/make_prompts.py \
    --input data/gsm8k/gsm8k_test.csv \
    --column question \
    --output data/datasets/gsm8k/gsm8k_prompts.txt

  # JSONL -> TXT (key 'prompt')
  python scripts/make_prompts.py --input data/dataset.jsonl --key prompt --output data/prompts.txt

  # TXT passthrough (dedupe/strip)
  python scripts/make_prompts.py --input data/raw_prompts.txt --output data/prompts.txt
"""
import argparse
import os
import sys
import pandas as pd


def write_txt(prompts, output):
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for p in prompts:
            p = (p or '').strip()
            if p:
                f.write(p + '\n')
    print(f"Wrote {len(prompts)} prompts to {output}")


def main():
    parser = argparse.ArgumentParser(description="Create prompts file from dataset")
    parser.add_argument('--input', required=True, help='Input dataset (csv, jsonl, or txt)')
    parser.add_argument('--output', required=True, help='Output prompts.txt (one prompt per line)')
    parser.add_argument('--column', default='question', help='CSV column to extract')
    parser.add_argument('--key', default='prompt', help='JSONL key to extract')
    parser.add_argument('--dedupe', action='store_true', help='Dedupe prompts')
    parser.add_argument('--max', type=int, default=None, help='Take only first N prompts')
    args = parser.parse_args()

    inp = args.input.lower()
    prompts = []
    if inp.endswith('.csv'):
        df = pd.read_csv(args.input)
        if args.column not in df.columns:
            print(f"ERROR: Column '{args.column}' not found in {args.input}. Columns: {list(df.columns)}")
            sys.exit(1)
        prompts = df[args.column].astype(str).map(str.strip).tolist()
    elif inp.endswith('.jsonl') or inp.endswith('.jsonl.gz'):
        import json, gzip
        open_fn = gzip.open if inp.endswith('.gz') else open
        with open_fn(args.input, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    val = str(obj.get(args.key, '')).strip()
                    if val:
                        prompts.append(val)
                except Exception:
                    continue
    elif inp.endswith('.txt'):
        with open(args.input, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        print("ERROR: Unsupported input type. Use csv, jsonl, or txt.")
        sys.exit(1)

    if args.dedupe:
        prompts = list(dict.fromkeys(prompts))
    if args.max is not None:
        prompts = prompts[: args.max]
    write_txt(prompts, args.output)


if __name__ == '__main__':
    main()
