#!/usr/bin/env python3
"""
Orchestrate a full pipeline over a prompts file:
- Generate base outputs for source and target models (if not present or --overwrite)
- Run disguise (contrastive or composite)
- Score outputs and write metrics

Usage:
  python scripts/run_pipeline.py \
    --prompts_file data/chabot_arena_500_propmts.txt \
    --source-model openai/gpt-4o-mini \
    --target-model gpt-4o \
    --method contrastive_with_al_examples
"""
import argparse
import os
import subprocess
from pathlib import Path
import sys


def run(cmd):
    print('$', ' '.join(cmd))
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run full disguise pipeline")
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--method", default="contrastive_with_al_examples",
                        choices=["contrastive", "contrastive_with_al_examples", "contrastive_al"]) 
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="results/streamlined")
    parser.add_argument("--overwrite", action="store_true")
    # AL knobs
    parser.add_argument("--al-num-examples", type=int, default=5)
    parser.add_argument("--al-max-iterations", type=int, default=5)
    parser.add_argument("--al-batch-size", type=int, default=10)
    # LiteLLM routing to local vLLM (optional)
    parser.add_argument("--openai-api-base", default=None, help="Route LiteLLM to local vLLM OpenAI-compatible endpoint")
    parser.add_argument("--openai-api-key", default=None, help="Pass API key/placeholder for local vLLM")
    # W&B integration (optional)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="streamlined-disguise")
    parser.add_argument("--wandb-run-name", default="pipeline")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate base outputs
    base_dir = Path('disguising/model-responses/base')
    base_dir.mkdir(parents=True, exist_ok=True)
    src_out = base_dir / f"{args.source_model.replace('/', '_')}.csv"
    tgt_out = base_dir / f"{args.target_model.replace('/', '_')}.csv"

    # Generate source
    if args.overwrite or not src_out.exists():
        rc = run([sys.executable, 'scripts/generate_responses.py',
                  '--model', args.source_model,
                  '--prompts_file', args.prompts_file,
                  '--output', str(src_out)] + (
                  ['--openai-api-base', args.openai_api_base] if args.openai_api_base else []
              ) + (
                  ['--openai-api-key', args.openai_api_key] if args.openai_api_key else []
              ))
        if rc != 0:
            sys.exit(rc)
    else:
        print(f"Found existing: {src_out}")

    # Generate target
    if args.overwrite or not tgt_out.exists():
        rc = run([sys.executable, 'scripts/generate_responses.py',
                  '--model', args.target_model,
                  '--prompts_file', args.prompts_file,
                  '--output', str(tgt_out)] + (
                  ['--openai-api-base', args.openai_api_base] if args.openai_api_base else []
              ) + (
                  ['--openai-api-key', args.openai_api_key] if args.openai_api_key else []
              ))
        if rc != 0:
            sys.exit(rc)
    else:
        print(f"Found existing: {tgt_out}")

    # 2) Baseline comparison (source vs target)
    baseline_csv = out_dir / f"baseline_{args.source_model.replace('/', '_')}_vs_{args.target_model.replace('/', '_')}.csv"
    rc = run([sys.executable, 'scripts/compare_models.py',
              '--a', str(src_out),
              '--b', str(tgt_out),
              '--output', str(baseline_csv)])
    if rc != 0:
        sys.exit(rc)

    # 3) Run disguise
    dcmd = [sys.executable, 'disguise.py',
            '--model', args.source_model,
            '--disguise_as', args.target_model,
            '--method', args.method,
            '--num_samples', str(args.num_samples),
            '--prompts_file', args.prompts_file,
            '--output_dir', str(out_dir),
            '--source_responses', str(src_out),
            '--target_responses', str(tgt_out)]
    if 'contrastive' in args.method:
        dcmd += ['--al-num-examples', str(getattr(args, 'al_num_examples')),
                 '--al-max-iterations', str(getattr(args, 'al_max_iterations')),
                 '--al-batch-size', str(getattr(args, 'al_batch_size'))]
    # Pass routing flags to disguise
    if args.openai_api_base:
        dcmd += ['--openai-api-base', args.openai_api_base]
    if args.openai_api_key:
        dcmd += ['--openai-api-key', args.openai_api_key]
    rc = run(dcmd)
    if rc != 0:
        sys.exit(rc)

    # 4) Score latest results (disguised vs target already scored by disguise.py; this re-scores to ensure artifacts)
    csvs = sorted(out_dir.glob('*.csv'), key=os.path.getmtime)
    if not csvs:
        print('No results to score in', out_dir)
        sys.exit(1)
    latest = str(csvs[-1])
    scmd = [sys.executable, '-m', 'disguising.scorer', latest, '--output', latest.replace('.csv', '_scored.csv'),
            '--judge-model', 'openai/gpt-4.1']
    if args.openai_api_base:
        scmd += ['--openai-api-base', args.openai_api_base]
    if args.openai_api_key:
        scmd += ['--openai-api-key', args.openai_api_key]
    rc = run(scmd)
    if rc != 0:
        sys.exit(rc)

    # 5) Aggregate and (optionally) log a W&B table
    if args.use_wandb:
        summary_csv = out_dir / 'summary.csv'
        summary_md = out_dir / 'summary.md'
        rc = run([sys.executable, 'scripts/aggregate_metrics.py',
                  '--root', str(out_dir),
                  '--output', str(summary_csv),
                  '--markdown', str(summary_md),
                  '--use-wandb', '--wandb-project', args.wandb_project, '--wandb-run-name', args.wandb_run_name])
        if rc != 0:
            sys.exit(rc)

        # 6) Three-way summary table
        # Try to find disguised latest and baseline files
        latest_scored = latest.replace('.csv', '_scored.csv')
        three_way_csv = out_dir / 'three_way_summary.csv'
        three_way_md = out_dir / 'three_way_summary.md'
        rc = run([sys.executable, 'scripts/summarize_three_way.py',
                  '--baseline', str(baseline_csv),
                  '--disguised', latest,
                  '--output', str(three_way_csv),
                  '--markdown', str(three_way_md),
                  '--use-wandb', '--wandb-project', args.wandb_project, '--wandb-run-name', 'three-way-summary'])
        if rc != 0:
            sys.exit(rc)


if __name__ == '__main__':
    main()
