#!/usr/bin/env python3
"""
Streamlined disguise script for the core methods:
- contrastive
- vibe_based
- stylistic
- random_sampling
- contrastive_with_al_examples (composite: contrastive + AL-selected examples)
"""
import argparse
import os
import pandas as pd
import os
import logging
from pathlib import Path
from typing import Optional

# Add disguising to path
import sys
import os
sys.path.append('disguising')
sys.path.append(os.path.join('disguising', 'methods'))

from methods.get_method import get_method
from scorer import score_model_comparison
from litellm import completion
from tqdm import tqdm
import wandb


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_model_responses(model_path: str, disguise_as_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load source and target model response data.
    
    Args:
        model_path: Path to source model responses CSV
        disguise_as_path: Path to target model responses CSV
        
    Returns:
        Tuple of (source_df, target_df)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Source model responses not found: {model_path}")
    
    if not os.path.exists(disguise_as_path):
        raise FileNotFoundError(f"Target model responses not found: {disguise_as_path}")
    
    source_df = pd.read_csv(model_path)
    target_df = pd.read_csv(disguise_as_path)
    
    # Standardize column names
    if 'model_response' in source_df.columns:
        source_df = source_df.rename(columns={'model_response': 'target_response'})
    if 'model_response' in target_df.columns:
        target_df = target_df.rename(columns={'model_response': 'target_response'})
    
    logging.info(f"Loaded {len(source_df)} source responses and {len(target_df)} target responses")
    return source_df, target_df


def generate_disguised_responses(method_name: str, model: str, disguise_as: str,
                               source_df: pd.DataFrame, target_df: pd.DataFrame, 
                               prompts: list, num_samples: int = 100,
                               method_kwargs: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Generate disguised responses using the specified method.
    
    Args:
        method_name: Name of disguise method
        model: Source model name
        disguise_as: Target model name
        source_df: Source model responses
        target_df: Target model responses
        prompts: List of prompts to process
        num_samples: Number of responses to generate
        
    Returns:
        DataFrame with disguised responses
    """
    # Get the method instance
    method = get_method(method_name, model, disguise_as, 
                      disguise_df=target_df, source_df=source_df,
                      method_kwargs=method_kwargs or {})
    
    results = []
    method_stats: dict = {}
    
    for i, prompt in enumerate(tqdm(prompts[:num_samples], desc=f"Generating {method_name} responses")):
        try:
            # Get disguised prompt from method
            disguised_messages = method.forward(prompt)
            
            # Generate response using litellm
            response = completion(
                model=model,
                messages=disguised_messages,
                temperature=0.7,
                max_tokens=1024
            )
            
            disguised_response = response.choices[0].message.content
            
            # Find matching target response for comparison
            target_response = ""
            if i < len(target_df) and 'target_response' in target_df.columns:
                target_response = target_df.iloc[i]['target_response']
            
            results.append({
                'prompt': prompt,
                'model_response': disguised_response,
                'target_response': target_response,
                'method': method_name,
                'source_model': model,
                'target_model': disguise_as
            })
            
        except Exception as e:
            logging.error(f"Error processing prompt {i}: {e}")
            continue
    # Gather method-level stats if available (e.g., AL selection)
    try:
        if hasattr(method, '_al_selector') and getattr(method, '_al_selector') is not None:
            sel = getattr(method, '_al_selector')
            if hasattr(sel, 'get_selection_statistics'):
                method_stats['selection_statistics'] = sel.get_selection_statistics()
            if hasattr(sel, 'get_uncertainty_analysis'):
                method_stats['uncertainty_analysis'] = sel.get_uncertainty_analysis()
    except Exception as e:
        logging.warning(f"Could not collect method stats: {e}")
    
    return pd.DataFrame(results), method_stats


def main():
    parser = argparse.ArgumentParser(description="Streamlined disguise experiments")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, 
                       help="Source model to disguise (e.g., google/gemma-3-1b-it)")
    parser.add_argument("--disguise_as", type=str, required=True,
                       help="Target model to mimic (e.g., gpt-4o)")
    
    # Method selection
    parser.add_argument("--method", type=str, 
                       choices=["contrastive", "vibe_based", "stylistic", 
                               "random_sampling", "contrastive_with_al_examples", "contrastive_al"],
                       default="random_sampling",
                       help="Disguise method to use")

    # Active Learning settings (for active_learning and contrastive_with_al_examples)
    parser.add_argument("--al-num-examples", type=int, default=5, help="Number of examples to include via AL")
    parser.add_argument("--al-d-regular", type=int, default=3, help="Degree for initial d-regular graph")
    parser.add_argument("--al-p-threshold", type=float, default=0.1, help="Bottom P fraction for |Δ| filter (0-1)")
    parser.add_argument("--al-q-threshold", type=float, default=0.1, help="Bottom Q fraction for degree filter (0-1)")
    parser.add_argument("--al-batch-size", type=int, default=10, help="Pairs to query per AL iteration")
    parser.add_argument("--al-max-iterations", type=int, default=5, help="Max AL iterations")
    parser.add_argument("--al-relaxation-factor", type=float, default=1.2, help="Relax thresholds factor when few candidates")
    parser.add_argument("--al-seed", type=int, default=None, help="Seed for AL selection")
    parser.add_argument("--example-selector", choices=["al", "clustering", "random"], default="al",
                        help="How to choose in-context examples for composite method")
    
    # Data paths
    parser.add_argument("--source_responses", type=str, 
                       help="Path to source model responses CSV (auto-detect if not provided)")
    parser.add_argument("--target_responses", type=str,
                       help="Path to target model responses CSV (auto-detect if not provided)")
    parser.add_argument("--prompts_file", type=str, 
                       default="data/chabot_arena_500_propmts.txt",
                       help="File with prompts (one per line)")
    
    # Experiment settings
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of responses to generate")
    parser.add_argument("--output_dir", type=str, default="results/streamlined",
                       help="Output directory for results")
    
    # Evaluation settings
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip automatic evaluation")
    parser.add_argument("--heuristics_only", action="store_true", 
                       help="Only compute heuristic scores (faster)")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                       help="Log results to Weights & Biases")
    parser.add_argument("--run_name", type=str,
                       help="Name for this experiment run")
    # LiteLLM routing (e.g., to a local vLLM server exposing OpenAI-compatible API)
    parser.add_argument("--openai-api-base", type=str, default=None,
                       help="Override OPENAI_API_BASE for LiteLLM (e.g., http://localhost:8000/v1)")
    parser.add_argument("--openai-api-key", type=str, default=None,
                       help="Override OPENAI_API_KEY for LiteLLM (placeholder allowed for local)")
    
    args = parser.parse_args()
    setup_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Optionally route LiteLLM calls to a local vLLM server
    if args.openai_api_base:
        os.environ["OPENAI_API_BASE"] = args.openai_api_base
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="streamlined-disguise",
            name=args.run_name or f"{args.method}_{args.model.replace('/', '-')}",
            config=vars(args)
        )
    
    # Auto-detect response files if not provided
    if not args.source_responses:
        base_dir = "disguising/model-responses/base"
        source_file = f"{args.model.replace('/', '_')}_responses-1000.csv"
        args.source_responses = os.path.join(base_dir, source_file)
        # Fallback to other naming patterns if not found
        if not os.path.exists(args.source_responses):
            source_file = f"{args.model.replace('/', '_')}_responses.csv"
            args.source_responses = os.path.join(base_dir, source_file)
        
    if not args.target_responses:
        base_dir = "disguising/model-responses/base" 
        target_file = f"{args.disguise_as.replace('/', '_')}_responses-1000.csv"
        args.target_responses = os.path.join(base_dir, target_file)
        # Fallback to other naming patterns if not found
        if not os.path.exists(args.target_responses):
            target_file = f"{args.disguise_as.replace('/', '_')}_responses.csv"
            args.target_responses = os.path.join(base_dir, target_file)
    
    # Load data
    logging.info("Loading model response data...")
    source_df, target_df = load_model_responses(args.source_responses, args.target_responses)
    
    # Load prompts
    if not os.path.exists(args.prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")
        
    with open(args.prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    logging.info(f"Loaded {len(prompts)} prompts")
    
    # Prepare method kwargs (for AL variants)
    method_kwargs = {
        'al_num_examples': args.al_num_examples,
        'al_d_regular': args.al_d_regular,
        'al_p_threshold': args.al_p_threshold,
        'al_q_threshold': args.al_q_threshold,
        'al_batch_size': args.al_batch_size,
        'al_max_iterations': args.al_max_iterations,
        'al_relaxation_factor': args.al_relaxation_factor,
        'al_seed': args.al_seed,
        'example_selector': args.example_selector,
    }

    # Generate disguised responses
    logging.info(f"Generating responses using {args.method}...")
    results_df, method_stats = generate_disguised_responses(
        args.method, args.model, args.disguise_as,
        source_df, target_df, prompts, args.num_samples,
        method_kwargs=method_kwargs
    )
    
    # Save results
    results_file = os.path.join(args.output_dir, 
                               f"{args.method}_{args.model.replace('/', '-')}_as_{args.disguise_as.replace('/', '-')}.csv")
    results_df.to_csv(results_file, index=False)
    logging.info(f"Results saved to {results_file}")
    
    # Evaluate results
    if not args.skip_evaluation:
        logging.info("Evaluating disguised responses...")
        scores_file = results_file.replace('.csv', '_scores.csv')
        
        scored_df = score_model_comparison(
            results_file, 
            scores_file,
            heuristics_only=args.heuristics_only
        )
        
        # Log summary statistics
        if 'semantic_score' in scored_df.columns:
            sem_mean = scored_df['semantic_score'].mean()
            sty_mean = scored_df['stylistic_score'].mean()
            logging.info(f"Average semantic score: {sem_mean:.3f}")
            logging.info(f"Average stylistic score: {sty_mean:.3f}")
            
            if args.use_wandb:
                wandb.log({
                    "semantic_score_mean": sem_mean,
                    "stylistic_score_mean": sty_mean,
                    "num_samples": len(scored_df)
                })
        
        if 'heuristic_match_score' in scored_df.columns:
            heur_mean = scored_df['heuristic_match_score'].mean()
            logging.info(f"Average heuristic match: {heur_mean:.3f}")
            
            if args.use_wandb:
                wandb.log({"heuristic_match_mean": heur_mean})

        # Save a run summary JSON
        try:
            import json
            summary = {
                'method': args.method,
                'model': args.model,
                'disguise_as': args.disguise_as,
                'num_samples': args.num_samples,
                'method_kwargs': method_kwargs,
                'results_file': results_file,
                'scores_file': scores_file,
                'metrics_json': scores_file.replace('.csv', '_metrics.json'),
            }
            if method_stats:
                summary.update(method_stats)
            summary_path = results_file.replace('.csv', '_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logging.info(f"Run summary saved to {summary_path}")

            # Log W&B artifacts if enabled
            if args.use_wandb:
                try:
                    metrics_json = scores_file.replace('.csv', '_metrics.json')
                    metrics_csv = scores_file.replace('.csv', '_metrics.csv')
                    art = wandb.Artifact(
                        name=f"run_artifacts_{args.method}_{args.model.replace('/', '-')}_as_{args.disguise_as.replace('/', '-')}",
                        type="run-artifacts",
                        metadata={"method": args.method, "num_samples": args.num_samples},
                    )
                    if os.path.exists(metrics_json):
                        art.add_file(metrics_json)
                    if os.path.exists(metrics_csv):
                        art.add_file(metrics_csv)
                    if os.path.exists(summary_path):
                        art.add_file(summary_path)
                    if os.path.exists(scores_file):
                        art.add_file(scores_file)
                    if os.path.exists(results_file):
                        art.add_file(results_file)
                    wandb.log_artifact(art)
                except Exception as e:
                    logging.warning(f"Could not log W&B artifacts: {e}")
        except Exception as e:
            logging.warning(f"Could not write run summary: {e}")
    
    if args.use_wandb:
        wandb.finish()
    
    logging.info("Experiment completed!")


if __name__ == "__main__":
    main()
