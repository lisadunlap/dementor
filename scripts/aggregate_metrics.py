#!/usr/bin/env python3
"""
Aggregate metrics across multiple scored runs into a single CSV and optional Markdown table.

Finds per-run metrics emitted by the scorer in either JSON or CSV format:
- "*_scores_metrics.json" (JSON dict)
- "*_metrics.csv" (two-column CSV: metric,value)

Usage:
  python scripts/aggregate_metrics.py --root data/results/<dataset> --output data/results/<dataset>/summary.csv --markdown data/results/<dataset>/summary.md
"""
import argparse
import json
import os
from pathlib import Path
import pandas as pd


def _load_metrics_from_csv(path: Path):
    import csv
    out = {}
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # Expect columns like [metric, value]
            for row in reader:
                if len(row) >= 2:
                    k, v = row[0], row[1]
                    try:
                        out[k] = float(v)
                    except Exception:
                        out[k] = v
    except Exception:
        pass
    return out


def collect_metrics(root: Path):
    rows = []
    # JSON metrics
    for path in root.rglob("*_scores_metrics.json"):
        try:
            with open(path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            continue
        rows.append(_row_from_path_and_metrics(path, metrics))
    # CSV metrics
    for path in root.rglob("*_metrics.csv"):
        metrics = _load_metrics_from_csv(path)
        if metrics:
            rows.append(_row_from_path_and_metrics(path, metrics))
    return pd.DataFrame(rows)


def _row_from_path_and_metrics(path: Path, metrics: dict) -> dict:
    # Attempt to parse method, model, target from filename pattern
    full = path.with_suffix('').name
    method = None
    source_model = None
    target_model = None
    try:
        parts = full.split('_scores_metrics')[0].split('_metrics')[0].split('_scores')[0].split('_')
        if 'as' in parts:
            as_idx = parts.index('as')
            target_model = "_".join(parts[as_idx+1:])
            source_model = "_".join(parts[1:as_idx])
            method = parts[0]
    except Exception:
        pass
    return {
        'file': str(path),
        'method': method,
        'source_model': source_model,
        'target_model': target_model,
        **metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate scored metrics")
    parser.add_argument("--root", default="data/results", help="Root directory to search")
    parser.add_argument("--output", default="data/results/metrics_summary.csv", help="Output CSV path")
    parser.add_argument("--markdown", default=None, help="Optional Markdown output path")
    parser.add_argument("--use-wandb", action="store_true", help="Log a wandb.Table with summary")
    parser.add_argument("--wandb-project", default="streamlined-disguise", help="W&B project name")
    parser.add_argument("--wandb-run-name", default="metrics-aggregate", help="W&B run name")
    args = parser.parse_args()

    root = Path(args.root)
    df = collect_metrics(root)
    if df.empty:
        print("No *_scores_metrics.json files found under", root)
        return
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print("Wrote:", args.output)

    if args.markdown:
        try:
            with open(args.markdown, 'w') as f:
                cols = ['method', 'source_model', 'target_model', 'semantic_score_mean', 'stylistic_score_mean', 'heuristic_match_mean']
                present = [c for c in cols if c in df.columns]
                f.write("| " + " | ".join(present) + " |\n")
                f.write("|" + "|".join(["---"]*len(present)) + "|\n")
                for _, r in df.iterrows():
                    vals = [str(r.get(c, '')) for c in present]
                    f.write("| " + " | ".join(vals) + " |\n")
            print("Wrote:", args.markdown)
        except Exception as e:
            print("Failed to write markdown:", e)

    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config={
                'root': str(root),
                'output': args.output,
                'markdown': args.markdown,
            })
            # Log a W&B table of the summary
            table = wandb.Table(dataframe=df)
            wandb.log({'metrics_summary': table})
            # Optionally upload artifacts
            art = wandb.Artifact('metrics_summary', type='aggregate')
            if os.path.exists(args.output):
                art.add_file(args.output)
            if args.markdown and os.path.exists(args.markdown):
                art.add_file(args.markdown)
            wandb.log_artifact(art)
            wandb.finish()
            print("Logged summary to W&B")
        except Exception as e:
            print("Failed to log to W&B:", e)


if __name__ == '__main__':
    main()
