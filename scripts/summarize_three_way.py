#!/usr/bin/env python3
"""
Summarize baseline (source vs target) and disguised vs target metrics into a small table.

Usage:
  python scripts/summarize_three_way.py \
    --baseline data/results/<dataset>/comparisons/source_vs_target/<src>_vs_<tgt>.csv \
    --disguised data/results/<dataset>/comparisons/disguised_vs_target/<method>/<src>_as_<tgt>.csv \
    --output data/results/<dataset>/three_way_summary.csv \
    [--markdown data/results/<dataset>/three_way_summary.md] \
    [--use-wandb --wandb-project streamlined-disguise --wandb-run-name three-way-summary]
"""
import argparse
import json
import os
from pathlib import Path
import pandas as pd


def _resolve_scored_path(path: Path) -> Path:
    """Return the scored.csv path for a given comparison artifact."""
    if path.name == 'scored.csv' and path.exists():
        return path
    if path.is_dir():
        candidate = path / 'scored.csv'
        if candidate.exists():
            return candidate
    candidate = path.with_suffix('').with_name(path.stem + '_scored.csv')
    if candidate.exists():
        return candidate
    return path


def load_metrics(scored_path: Path) -> dict:
    metrics_csv = scored_path.with_name('scored_metrics.csv')
    if metrics_csv.exists():
        try:
            import csv

            metrics: dict[str, float] = {}
            with open(metrics_csv, 'r', newline='') as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            metrics[row[0]] = float(row[1])
                        except ValueError:
                            metrics[row[0]] = row[1]
            return metrics
        except Exception:
            pass

    # Legacy JSON fallback
    json_path = scored_path.with_suffix('').as_posix() + '_metrics.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Summarize baseline and disguised metrics")
    parser.add_argument("--baseline", required=True, help="Baseline merged CSV (source vs target)")
    parser.add_argument("--disguised", required=True, help="Disguised results CSV (disguised vs target)")
    parser.add_argument("--output", required=True, help="Output summary CSV")
    parser.add_argument("--markdown", default=None, help="Optional Markdown table output")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="streamlined-disguise")
    parser.add_argument("--wandb-run-name", default="three-way-summary")
    args = parser.parse_args()

    base_scored = _resolve_scored_path(Path(args.baseline))
    dis_scored = _resolve_scored_path(Path(args.disguised))

    base_metrics = load_metrics(base_scored)
    dis_metrics = load_metrics(dis_scored)

    rows = []
    if base_metrics:
        rows.append({
            'comparison': 'baseline_source_vs_target',
            **base_metrics
        })
    if dis_metrics:
        rows.append({
            'comparison': 'disguised_vs_target',
            **dis_metrics
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print("Wrote:", args.output)

    if args.markdown:
        with open(args.markdown, 'w') as f:
            cols = ['comparison', 'semantic_score_mean', 'stylistic_score_mean', 'heuristic_match_mean']
            present = [c for c in cols if c in df.columns]
            f.write("| " + " | ".join(present) + " |\n")
            f.write("|" + "|".join(["---"]*len(present)) + "|\n")
            for _, r in df.iterrows():
                vals = [str(r.get(c, '')) for c in present]
                f.write("| " + " | ".join(vals) + " |\n")
        print("Wrote:", args.markdown)

    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name)
            table = wandb.Table(dataframe=df)
            wandb.log({'three_way_summary': table})
            art = wandb.Artifact('three_way_summary', type='summary')
            if os.path.exists(args.output):
                art.add_file(args.output)
            if args.markdown and os.path.exists(args.markdown):
                art.add_file(args.markdown)
            wandb.log_artifact(art)
            wandb.finish()
        except Exception as e:
            print("Failed to log to W&B:", e)


if __name__ == '__main__':
    main()
