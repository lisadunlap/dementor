#!/usr/bin/env python3
"""
Orchestrate a full pipeline over a prompts file:
- Generate base outputs for source and target models (if not present or --overwrite)
- Run disguise (contrastive or composite)
- Score outputs and write metrics

Usage:
  python scripts/run_pipeline.py \
    --prompts_file data/datasets/chatbot_arena/chatbot_arena_prompts.txt \
    --source-model openai/gpt-4o-mini \
    --target-model gpt-4o \
    --method contrastive_with_al_examples
"""
import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional
import sys
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def run(cmd, exclude_routing=False):
    print('$', ' '.join(cmd))
    env = os.environ.copy()
    # Add current directory to PYTHONPATH for subprocess imports
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{os.getcwd()}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = os.getcwd()
    
    # Remove routing environment variables if requested (for judge model calls)
    if exclude_routing:
        env.pop('OPENAI_API_BASE', None)
        env.pop('LITELLM_API_BASE', None)
    
    return subprocess.call(cmd, env=env)


def _is_openai_gpt(model_id: str) -> bool:
    model_lower = (model_id or '').lower()
    return model_lower.startswith('openai/gpt-')


def _build_gen_cache_cmd(
    model_id: str,
    prompts_file: str,
    output_path: Path,
    *,
    num_samples: Optional[int] = None,
    refresh: bool = False,
) -> list[str]:
    """Construct command to call gen_cache_response for OpenAI GPT models."""
    base_model = model_id.split('/', 1)[1] if '/' in model_id else model_id
    cmd = [
        sys.executable,
        'scripts/gen_cache_response.py',
        '--dataset', prompts_file,
        '--gpt_model', base_model,
        '--output_path', str(output_path),
        '--disable_wandb',
    ]
    if num_samples is not None:
        cmd += ['--num_samples', str(num_samples)]
    if refresh:
        cmd.append('--ignore_generation_cache')
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run full disguise pipeline")
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--method", default="contrastive_with_al_examples",
                        choices=["contrastive", "contrastive_with_al_examples", "contrastive_al"]) 
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output_dir", default=None)
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

    # Infer dataset from prompts path
    def infer_dataset(prompts_path: str) -> str:
        p = (prompts_path or "").lower()
        if "gsm8k" in p:
            return "gsm8k"
        if "arena" in p or "chatbot" in p:
            return "chatbot_arena"
        return "generic"

    dataset = infer_dataset(args.prompts_file)
    def infer_subset(prompts_path: str) -> str:
        p = (prompts_path or "").lower()
        if "_500" in p or "/500" in p or p.endswith("500.csv"):
            return "500"
        return "full"
    subset = infer_subset(args.prompts_file)

    # Resolve output root
    base_results_root = Path("data") / "results" / dataset / subset
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = base_results_root
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate base outputs (or use existing ones)
    # Canonical base responses directory by dataset (full set)
    base_dir = Path('data') / 'model-responses' / dataset / 'full'
    base_dir.mkdir(parents=True, exist_ok=True)
    src_out = base_dir / f"{args.source_model.replace('/', '_')}.csv"
    tgt_out = base_dir / f"{args.target_model.replace('/', '_')}.csv"

    # Generate source
    if args.overwrite or not src_out.exists():
        if _is_openai_gpt(args.source_model):
            if args.overwrite and src_out.exists():
                src_out.unlink()
            gen_cmd = _build_gen_cache_cmd(
                args.source_model,
                args.prompts_file,
                src_out,
                refresh=args.overwrite,
            )
            rc = run(gen_cmd, exclude_routing=True)
            if rc != 0:
                sys.exit(rc)
        else:
            gen_cmd = [sys.executable, 'scripts/generate_responses.py',
                       '--model', args.source_model,
                       '--prompts_file', args.prompts_file,
                       '--output', str(src_out)]
            # Only pass routing flags for local/HF models; never for openai/gpt-*
            if args.openai_api_base:
                gen_cmd += ['--openai-api-base', args.openai_api_base]
            if args.openai_api_key:
                gen_cmd += ['--openai-api-key', args.openai_api_key]
            rc = run(gen_cmd)
            if rc != 0:
                sys.exit(rc)
    else:
        print(f"Found existing: {src_out}")

    # Generate target
    if args.overwrite or not tgt_out.exists():
        if _is_openai_gpt(args.target_model):
            if args.overwrite and tgt_out.exists():
                tgt_out.unlink()
            gen_cmd = _build_gen_cache_cmd(
                args.target_model,
                args.prompts_file,
                tgt_out,
                refresh=args.overwrite,
            )
            rc = run(gen_cmd, exclude_routing=True)
            if rc != 0:
                sys.exit(rc)
        else:
            gen_cmd = [sys.executable, 'scripts/generate_responses.py',
                       '--model', args.target_model,
                       '--prompts_file', args.prompts_file,
                       '--output', str(tgt_out)]
            if args.openai_api_base:
                gen_cmd += ['--openai-api-base', args.openai_api_base]
            if args.openai_api_key:
                gen_cmd += ['--openai-api-key', args.openai_api_key]
            rc = run(gen_cmd)
            if rc != 0:
                sys.exit(rc)
    else:
        print(f"Found existing: {tgt_out}")

    # 2) Optional baseline comparison (source vs target) – can be skipped if not needed
    # Keeping for compatibility; comment out if you consider it obsolete.
    baseline_dir = out_dir / "comparisons" / "source_vs_target"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_csv = baseline_dir / f"{args.source_model.replace('/', '_')}_vs_{args.target_model.replace('/', '_')}.csv"
    rc = run([sys.executable, '-m', 'scripts.scorer', 'compare',
              '--a', str(src_out),
              '--b', str(tgt_out),
              '--output', str(baseline_csv),
              '--judge-model', 'openai/gpt-4.1-mini'], exclude_routing=True)
    if rc != 0:
        sys.exit(rc)
    baseline_scored = baseline_dir / 'scores' / baseline_csv.stem / 'scored.csv'

    # 3) Score single-file source and target into standardized locations (base scores)
    src_score_dir = out_dir / "scores" / args.source_model.replace('/', '_')
    tgt_score_dir = out_dir / "scores" / args.target_model.replace('/', '_')
    src_score_dir.mkdir(parents=True, exist_ok=True)
    tgt_score_dir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, '-m', 'scripts.scorer', 'single', str(src_out), '--output', str(src_score_dir / 'scored.csv')])
    run([sys.executable, '-m', 'scripts.scorer', 'single', str(tgt_out), '--output', str(tgt_score_dir / 'scored.csv')])

    # 4) Run disguise with new layout for outputs (per-method directory)
    disguise_dir = out_dir / args.method
    disguise_dir.mkdir(parents=True, exist_ok=True)
    dcmd = [sys.executable, 'scripts/disguise.py',
            '--model', args.source_model,
            '--disguise_as', args.target_model,
            '--method', args.method,
            '--num_samples', str(args.num_samples),
            '--prompts_file', args.prompts_file,
            '--output_dir', str(disguise_dir),
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
    pair_id = f"{args.source_model.replace('/', '_')}_as_{args.target_model.replace('/', '_')}"
    disguised_scored = disguise_dir / 'scores' / pair_id / 'scored.csv'

    # 5) No extra re-scoring here: disguise.py already emitted scores into
    #    data/results/<dataset>/comparisons/disguised/<method>/metrics_<pair>/

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
        three_way_csv = out_dir / 'three_way_summary.csv'
        three_way_md = out_dir / 'three_way_summary.md'
        rc = run([sys.executable, 'scripts/summarize_three_way.py',
                  '--baseline', str(baseline_scored),
                  '--disguised', str(disguised_scored),
                  '--output', str(three_way_csv),
                  '--markdown', str(three_way_md),
                  '--use-wandb', '--wandb-project', args.wandb_project, '--wandb-run-name', 'three-way-summary'])
        if rc != 0:
            sys.exit(rc)


if __name__ == '__main__':
    main()
