#!/usr/bin/env python3
"""
Summarize baseline (source vs target) and disguised vs target metrics into a small table.

Usage:
  python scripts/summarize_three_way.py \
    --baseline results/streamlined/baseline_src_vs_tgt.csv \
    --disguised results/streamlined/contrastive_with_al_examples_openai_gpt-4o-mini_as_gpt-4o.csv \
    --output results/streamlined/three_way_summary.csv \
    [--markdown results/streamlined/three_way_summary.md] \
    [--use-wandb --wandb-project streamlined-disguise --wandb-run-name three-way-summary]
"""
import argparse
import json
import os
from pathlib import Path
import pandas as pd


def load_metrics(base_path: Path) -> dict:
    json_path = base_path.with_suffix('').as_posix() + '_metrics.json'
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r') as f:
        return json.load(f)


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

    base = Path(args.baseline)
    dis = Path(args.disguised)

    base_scored = base.with_suffix('').as_posix() + '_scored.csv'
    dis_scored = dis.with_suffix('').as_posix() + '_scored.csv'

    # Ensure scored exists; otherwise, just proceed with whatever metrics are present
    base_metrics = load_metrics(Path(base_scored))
    dis_metrics = load_metrics(Path(dis_scored))

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

