#!/usr/bin/env python3
"""
Dementor: LLM Disguise Research Framework
Main entry point for the repository with centralized help and common operations.
"""
import argparse
import sys
import os
import subprocess
from pathlib import Path
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

def run_script(script_path, args):
    """Run a script with the given arguments."""
    cmd = [sys.executable, script_path] + args
    return subprocess.call(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="Dementor: LLM Disguise Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Operations:
  generate        Generate model responses to prompts
  disguise        Apply disguise methods to transform responses  
  compare         Compare two model outputs with scoring
  pipeline        Run complete end-to-end pipeline
  score           Score model comparisons with LLM judge + heuristics

Examples:
  python main.py generate --model gpt-4o --prompts data/datasets/gsm8k/gsm8k_prompts.txt
  python main.py disguise --model llama-3.1-8b --disguise_as gpt-4o --method contrastive
  python main.py compare --a model_a.csv --b model_b.csv --output comparison.csv
  python main.py pipeline --source-model llama-3.1-8b --target-model gpt-4o

Directory Structure:
  scripts/         All executable scripts and methods
  data/            Input datasets only
    └── datasets/  Prompts and datasets (GSM8K, etc.)
  data/results/    All generated artifacts (canonical)
    ├── model-responses/  Base model outputs (CSV)
    ├── comparisons/      Baseline and disguised comparisons
    └── scores/           Evaluation results

Core Scripts:
  scripts/generate_responses.py    Generate model responses
  scripts/disguise.py             Apply disguise methods
  scripts/compare_models.py       Compare model outputs  
  scripts/run_pipeline.py         End-to-end pipeline
  scripts/scorer.py               LLM judge + heuristics scorer
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate model responses')
    gen_parser.add_argument('--model', required=True, help='Model to use')
    gen_parser.add_argument('--prompts', required=True, help='Prompts file')
    gen_parser.add_argument('--output', help='Output CSV file')
    gen_parser.add_argument('--num_samples', type=int, default=200, help='Number of samples')
    
    # Disguise command  
    disguise_parser = subparsers.add_parser('disguise', help='Apply disguise methods')
    disguise_parser.add_argument('--model', required=True, help='Source model')
    disguise_parser.add_argument('--disguise_as', required=True, help='Target model to mimic')
    disguise_parser.add_argument('--method', default='contrastive', help='Disguise method')
    disguise_parser.add_argument('--prompts', help='Prompts file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare model outputs')
    compare_parser.add_argument('--a', required=True, help='Model A CSV')
    compare_parser.add_argument('--b', required=True, help='Model B CSV') 
    compare_parser.add_argument('--output', required=True, help='Output comparison CSV')
    compare_parser.add_argument('--heuristics-only', action='store_true', help='Heuristics only')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--source-model', required=True, help='Source model')
    pipeline_parser.add_argument('--target-model', required=True, help='Target model')
    pipeline_parser.add_argument('--prompts', help='Prompts file', default='data/datasets/gsm8k/gsm8k_prompts.txt')
    pipeline_parser.add_argument('--method', default='contrastive_with_al_examples', help='Disguise method')
    
    # Score command
    score_parser = subparsers.add_parser('score', help='Score model comparison')
    score_parser.add_argument('--input', required=True, help='Input comparison CSV')
    score_parser.add_argument('--output', required=True, help='Output scored CSV')
    score_parser.add_argument('--heuristics-only', action='store_true', help='Heuristics only')
    
    args, unknown_args = parser.parse_known_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # Map commands to scripts
    script_map = {
        'generate': 'scripts/generate_responses.py',
        'disguise': 'scripts/disguise.py', 
        'compare': 'scripts/compare_models.py',
        'pipeline': 'scripts/run_pipeline.py',
        'score': 'scripts/scorer.py'
    }
    
    if args.command not in script_map:
        print(f"Unknown command: {args.command}")
        return 1
        
    script_path = script_map[args.command]
    
    # Convert args back to command line format
    script_args = []
    for key, value in vars(args).items():
        if key == 'command':
            continue
        if value is True:
            script_args.append(f'--{key.replace("_", "-")}')
        elif value is not False and value is not None:
            script_args.extend([f'--{key.replace("_", "-")}', str(value)])
    
    # Add any unknown args
    script_args.extend(unknown_args)
    
    return run_script(script_path, script_args)

if __name__ == '__main__':
    sys.exit(main())
