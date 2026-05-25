#!/usr/bin/env python3
"""
End-to-end validation script:
 - Runs a small disguise generation
 - Scores the results
 - Prints summary metrics

Usage:
  python validate_end_to_end.py \
    --model openai/gpt-4.1-mini \
    --disguise_as meta-llama/Meta-Llama-3.1-8B-Instruct \
    --method contrastive \
    --num_samples 20

Notes:
 - Requires provider API key(s) for LiteLLM in your environment (e.g. OPENAI_API_KEY).
 - For local LLM judge scoring, install `vllm` and run scorer with --output for metrics.
"""
import argparse
import os
import subprocess
from pathlib import Path
import sys


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run a small end-to-end validation")
    parser.add_argument("--model", required=True, help="Source model (LiteLLM id, e.g., openai/gpt-4.1-mini)")
    parser.add_argument("--disguise_as", required=True, help="Target model label")
    parser.add_argument("--method", default="contrastive", help="disguise method")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_dir", default="results/validation")
    parser.add_argument("--prompts_file", default="data/datasets/chatbot_arena/chatbot_arena_prompts.txt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Generate disguised responses
    gen_cmd = [
        sys.executable, "disguise.py",
        "--model", args.model,
        "--disguise_as", args.disguise_as,
        "--method", args.method,
        "--num_samples", str(args.num_samples),
        "--output_dir", args.output_dir,
        "--prompts_file", args.prompts_file,
    ]
    code = run(gen_cmd)
    if code != 0:
        sys.exit(code)

    # Identify latest results file
    out_dir = Path(args.output_dir)
    csvs = sorted(out_dir.glob("*.csv"), key=os.path.getmtime)
    if not csvs:
        print("No results CSV found in", out_dir)
        sys.exit(1)
    results_csv = str(csvs[-1])
    print("Results:", results_csv)

    # Step 2: Score results (LLM + heuristics) and write metrics
    scored_csv = results_csv.replace('.csv', '_scored.csv')
    score_cmd = [
        sys.executable, "-m", "scripts.scorer",
        results_csv,
        "--output", scored_csv,
    ]
    code = run(score_cmd)
    if code != 0:
        sys.exit(code)

    # Print metrics path
    print("Scored:", scored_csv)
    base = scored_csv[:-4]
    print("Metrics JSON:", f"{base}_metrics.json")
    print("Metrics CSV:", f"{base}_metrics.csv")


if __name__ == "__main__":
    main()
