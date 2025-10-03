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
import time
from pathlib import Path
from typing import Optional

# Add scripts to path
import sys
import os
# Ensure 'scripts' (parent of the 'methods' package) is on sys.path
if 'scripts' not in sys.path:
    sys.path.append('scripts')

from methods.get_method import get_method
from scorer import score_model_comparison
from litellm import completion
import litellm
from tqdm import tqdm
import wandb

# Use repository caching/utilities
from serve.utils_llm import get_llm_output

# Enable caching for API calls (not for vLLM/local servers)
if not hasattr(litellm, 'cache') or litellm.cache is None:
    litellm.cache = litellm.Cache()
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


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


def generate_disguised_responses(
    method_name: str,
    model: str,
    disguise_as: str,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    prompts: list,
    num_samples: int = 100,
    method_kwargs: dict | None = None,
    output_file: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
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

    # Prepare output for incremental persistence
    csv_writer = None
    if output_file is not None:
        import csv
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        if not os.path.exists(output_file):
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(['prompt', 'model_response', 'target_response', 'method', 'source_model', 'target_model'])

    # Backend helpers
    def _has_provider_prefix(m: str) -> bool:
        m = (m or '').lower()
        return any(m.startswith(p) for p in ['openai/', 'azure/', 'anthropic/', 'vertex/', 'bedrock/'])

    def _to_text_from_messages(msgs: list[dict]) -> str:
        # Compose a simple text prompt for non-chat backends
        if not msgs:
            return ""
        if len(msgs) == 1 and msgs[0].get('role') == 'user':
            return str(msgs[0].get('content', ''))
        sys_text = ""
        user_text = ""
        for m in msgs:
            role = m.get('role')
            content = str(m.get('content', ''))
            if role == 'system':
                sys_text += content + "\n\n"
            elif role == 'user':
                user_text += content
        return f"{sys_text}{user_text}".strip()

    # Cache heavy backends
    _hf_pipe = None
    _vllm_llm = None
    _vllm_params = None
    
    for i, prompt in enumerate(tqdm(prompts[:num_samples], desc=f"Generating {method_name} responses")):
        try:
            # Get disguised prompt from method
            disguised_messages = method.forward(prompt)

            # Choose backend: explicit hf:/vllm: → direct; else LiteLLM
            model_lower = (model or '').lower()
            max_new_tokens = 1024
            temperature = 0.7

            def _gen_with_retries(call_fn, max_attempts=3, base_delay=0.5):
                last_exc = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return call_fn()
                    except Exception as e:
                        last_exc = e
                        # brief backoff
                        time.sleep(base_delay * attempt)
                raise last_exc

            if model_lower.startswith('hf:') or model_lower.startswith('huggingface:'):
                from transformers import pipeline
                if _hf_pipe is None:
                    model_id = model.split(':', 1)[1]
                    _hf_pipe = pipeline('text-generation', model=model_id, device_map='auto')
                text_input = _to_text_from_messages(disguised_messages)
                def _hf_call():
                    out = _hf_pipe(text_input, max_new_tokens=max_new_tokens, do_sample=(temperature > 0), temperature=max(temperature, 1e-6))
                    gen = out[0]['generated_text']
                    # return suffix beyond prompt
                    return gen[len(text_input):].strip()
                disguised_response = _gen_with_retries(_hf_call)
            elif model_lower.startswith('vllm:'):
                from vllm import LLM, SamplingParams
                if _vllm_llm is None:
                    model_id = model.split(':', 1)[1]
                    _vllm_llm = LLM(model=model_id, trust_remote_code=True)
                    _vllm_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)
                text_input = _to_text_from_messages(disguised_messages)
                def _vllm_call():
                    outputs = _vllm_llm.generate([text_input], _vllm_params)
                    return outputs[0].outputs[0].text
                disguised_response = _gen_with_retries(_vllm_call)
            else:
                # Decide routing and provider; prefer repo's get_llm_output() for OpenAI models (cached)
                routed_model = model
                api_base = os.getenv('OPENAI_API_BASE')
                api_key = os.getenv('OPENAI_API_KEY')
                # Force official base for OpenAI models
                if model_lower.startswith('openai/gpt-'):
                    os.environ['OPENAI_API_BASE'] = 'https://api.openai.com/v1'
                    api_base = 'https://api.openai.com/v1'
                # If a local OpenAI-compatible base is set and no provider prefix, wrap with openai/
                if api_base and not _has_provider_prefix(model):
                    routed_model = f"openai/{model}"

                def _split_messages(msgs: list[dict]) -> tuple[str, list[dict], str]:
                    """Split chat messages into (system_prompt, history, user_prompt).
                    - system_prompt: concatenated system contents
                    - history: all messages up to but excluding the last user message, preserving roles
                    - user_prompt: content of the last user message
                    """
                    sys_text_parts: list[str] = []
                    non_system: list[dict] = []
                    for m in msgs:
                        role = m.get('role')
                        content = str(m.get('content', ''))
                        if role == 'system':
                            sys_text_parts.append(content)
                        else:
                            non_system.append({"role": role, "content": content})

                    # Find last user message
                    last_user_idx = -1
                    for idx in range(len(non_system) - 1, -1, -1):
                        if non_system[idx].get('role') == 'user':
                            last_user_idx = idx
                            break

                    if last_user_idx == -1:
                        # Fallback: treat all as history, no explicit user prompt
                        return "\n\n".join(sys_text_parts).strip(), non_system, ""

                    history_msgs = non_system[:last_user_idx]
                    user_prompt = non_system[last_user_idx].get('content', '')
                    return "\n\n".join(sys_text_parts).strip(), history_msgs, str(user_prompt)

                # Use get_llm_output for OpenAI-style models, else fallback to litellm as before
                use_openai_cached = routed_model.startswith('openai/') or model_lower.startswith('gpt-')

                if use_openai_cached:
                    # Map to OpenAI model id (strip provider prefix if present)
                    llm_model = routed_model.split('/', 1)[1] if routed_model.startswith('openai/') else routed_model

                    def _llm_call():
                        system_prompt_text, history_msgs, user_prompt_text = _split_messages(disguised_messages)
                        return get_llm_output(
                            user_prompt_text,
                            model=llm_model,
                            cache=True,
                            system_prompt=(system_prompt_text if system_prompt_text else None),
                            history=history_msgs,
                            max_tokens=max_new_tokens,
                        )
                    disguised_response = _gen_with_retries(_llm_call)
                else:
                    # Fallback to standard litellm path for non-OpenAI providers
                    def _llm_call():
                        resp = completion(
                            model=routed_model,
                            messages=disguised_messages,
                            temperature=temperature,
                            max_tokens=max_new_tokens,
                            api_base=api_base if api_base else None,
                            api_key=api_key if api_key else None,
                            request_timeout=60,
                        )
                        return resp["choices"][0]["message"]["content"]
                    disguised_response = _gen_with_retries(_llm_call)

            # Find matching target response for comparison
            target_response = ""
            if i < len(target_df) and 'target_response' in target_df.columns:
                target_response = target_df.iloc[i]['target_response']
            
            row = {
                'prompt': prompt,
                'model_response': disguised_response,
                'target_response': target_response,
                'method': method_name,
                'source_model': model,
                'target_model': disguise_as
            }
            results.append(row)

            # Incremental persist
            if output_file is not None:
                import csv
                with open(output_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                    writer.writerow([
                        row['prompt'], row['model_response'], row['target_response'], row['method'], row['source_model'], row['target_model']
                    ])

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


def _short_model_name(name: str) -> str:
    """Produce a short, readable identifier from a provider/model string.
    Examples:
      'openai/meta-llama/Meta-Llama-3-8B-Instruct' -> 'llama-3-8b'
      'meta-llama/Meta-Llama-3.1-8B-Instruct'     -> 'llama-3.1-8b'
      'gpt-4o'                                    -> 'gpt-4o'
    """
    if not name:
        return "model"
    seg = name.split('/')[-1]
    s = seg.replace('_', '-').strip()
    # Remove common suffixes/prefixes
    for token in ["Instruct", "-Instruct", "instruct", "-instruct", "Meta-", "meta-"]:
        s = s.replace(token, "")
    # Normalize Llama branding and casing
    s = s.replace("Llama-", "llama-").replace("Llama", "llama")
    s = s.lower().strip('-')
    return s or "model"


def main():
    parser = argparse.ArgumentParser(description="Streamlined disguise experiments")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, 
                       help="Source model to disguise (e.g., google/gemma-3-1b-it)")
    parser.add_argument("--disguise_as", type=str, required=True,
                       help="Target model to mimic (e.g., gpt-4o)")
    
    # Method selection (allow any method supported by get_method)
    parser.add_argument("--method", type=str,
                       default="random_sampling",
                       help="Disguise method to use (any name supported by methods.get_method)")

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
                       default="data/datasets/chatbot_arena/chatbot_arena_prompts.txt",
                       help="File with prompts (one per line)")
    
    # Experiment settings
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of responses to generate")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    
    # Evaluation settings
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip automatic evaluation")
    parser.add_argument("--heuristics_only", action="store_true",
                       help="Only compute heuristic scores (faster)")
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini",
                       help="Judge model for scoring (default: openai/gpt-4.1-mini)")
    parser.add_argument("--judge-api-base", default="https://api.openai.com/v1",
                       help="API base URL for judge model (default: OpenAI API)")
    parser.add_argument("--judge-api-key", default=None,
                       help="API key for judge model (defaults to OPENAI_API_KEY env var)")
    
    # Logging
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging (enabled by default)")
    parser.add_argument("--run_name", type=str,
                       help="Name for this experiment run (auto-generated if not provided)")
    # LiteLLM routing (e.g., to a local vLLM server exposing OpenAI-compatible API)
    parser.add_argument("--openai-api-base", type=str, default=None,
                       help="Override OPENAI_API_BASE for LiteLLM (e.g., http://localhost:8000/v1)")
    parser.add_argument("--openai-api-key", type=str, default=None,
                       help="Override OPENAI_API_KEY for LiteLLM (placeholder allowed for local)")
    
    args = parser.parse_args()
    setup_logging()
    
    # Defer creating output directory until after resolving dataset + defaults

    # Optionally route LiteLLM calls to a local vLLM server
    if args.openai_api_base:
        os.environ["OPENAI_API_BASE"] = args.openai_api_base
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # Hard-guard: if generation model is an OpenAI-hosted model, force official base
    try:
        model_lower = (args.model or "").lower()
        if model_lower.startswith("openai/gpt-"):
            os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
            logging.info("Detected OpenAI provider model '%s'; routing generation to OpenAI API.", args.model)
    except Exception:
        pass
    
    # Determine dataset once (used for paths and optional logging)
    def _infer_dataset(prompts_path: str) -> str:
        p = (prompts_path or "").lower()
        if "gsm8k" in p:
            return "gsm8k"
        if "arena" in p or "chatbot" in p:
            return "chatbot_arena"
        return "generic"
    dataset = _infer_dataset(args.prompts_file)

    # Initialize wandb by default (unless disabled)
    use_wandb = not args.no_wandb
    if use_wandb:
        # Auto-generate run name if not provided
        if not args.run_name:
            src_short = _short_model_name(args.model)
            tgt_short = _short_model_name(args.disguise_as)
            timestamp = pd.Timestamp.now().strftime("%m%d_%H%M")
            args.run_name = f"{args.method}_{src_short}_as_{tgt_short}_{timestamp}"

        wandb.init(
            project="dementor-disguise",
            name=args.run_name,
            config=vars(args),
            tags=[args.method, dataset, src_short, tgt_short]
        )
    
    # Auto-detect response files if not provided
    # Canonical location is data/model-responses/<dataset>/{full,500}; fall back to legacy locations
    # Expose dataset to downstream utilities (e.g., clustering) via environment
    os.environ["DEMENTOR_DATASET"] = dataset

    def _candidate_paths(model_name: str) -> list[str]:
        norm = model_name.replace('/', '_')
        candidates = []
        for root in (
            f"data/model-responses/{dataset}/full",
            f"data/model-responses/{dataset}/500",
            f"data/model-responses/{dataset}/base",
            "data/model-responses/base",
            # Fallback to prior location if present
            f"disguising/model-responses/{dataset}/full",
            f"disguising/model-responses/{dataset}/500",
            f"disguising/model-responses/{dataset}/base",
            "disguising/model-responses/base",
        ):
            for fname in (f"{norm}_responses-1000.csv", f"{norm}_responses.csv", f"{norm}.csv"):
                candidates.append(os.path.join(root, fname))
        return candidates

    def _resolve_existing(candidates: list[str]) -> Optional[str]:
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    if not args.source_responses:
        src_path = _resolve_existing(_candidate_paths(args.model))
        if src_path:
            args.source_responses = src_path
        else:
            # Leave unset to trigger a helpful error below
            args.source_responses = _candidate_paths(args.model)[0]
    
    if not args.target_responses:
        tgt_path = _resolve_existing(_candidate_paths(args.disguise_as))
        if tgt_path:
            args.target_responses = tgt_path
        else:
            args.target_responses = _candidate_paths(args.disguise_as)[0]
    
    # Load data
    logging.info("Loading model response data...")
    try:
        source_df, target_df = load_model_responses(args.source_responses, args.target_responses)
    except FileNotFoundError as e:
        logging.error(
            "Could not locate base responses. Place CSVs under 'data/model-responses/<dataset>/{full,500}' "
            "(preferred) or legacy '.../base'. You can also pass explicit paths via --source_responses and "
            "--target_responses."
        )
        raise
    
    # Load prompts
    if not os.path.exists(args.prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")
        
    with open(args.prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    logging.info(f"Loaded {len(prompts)} prompts")
    
    # Determine output dir and result path before generation for incremental writes
    if args.output_dir in (None, "results/streamlined"):
        default_dir = os.path.join("data", "results", dataset, "comparisons", "disguised_vs_target", args.method)
        os.makedirs(default_dir, exist_ok=True)
        args.output_dir = default_dir
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    src_short = _short_model_name(args.model)
    tgt_short = _short_model_name(args.disguise_as)
    results_file = os.path.join(args.output_dir, f"{src_short}_as_{tgt_short}.csv")

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
        method_kwargs=method_kwargs,
        output_file=results_file,
    )
    
    # If file exists but results_df is empty (all failed), ensure header exists
    if results_df.empty and os.path.exists(results_file):
        import csv
        with open(results_file, 'r', encoding='utf-8') as f:
            head = f.read(100)
        if not head.strip():
            import csv as _csv
            with open(results_file, 'w', newline='', encoding='utf-8') as f:
                writer = _csv.writer(f, quoting=_csv.QUOTE_ALL)
                writer.writerow(['prompt', 'model_response', 'target_response', 'method', 'source_model', 'target_model'])
    logging.info(f"Results saved to {results_file}")
    
    # Evaluate results
    if not args.skip_evaluation and os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        logging.info("Evaluating disguised responses...")
        scores_file = results_file.replace('.csv', '_scores.csv')

        # Set up judge model environment if specified
        original_api_base = os.environ.get("OPENAI_API_BASE")
        original_api_key = os.environ.get("OPENAI_API_KEY")

        if args.judge_api_base:
            os.environ["OPENAI_API_BASE"] = args.judge_api_base
        if args.judge_api_key:
            os.environ["OPENAI_API_KEY"] = args.judge_api_key

        try:
            scored_df = score_model_comparison(
                results_file,
                scores_file,
                heuristics_only=args.heuristics_only,
                judge_model=args.judge_model
            )
        finally:
            # Restore original environment
            if original_api_base is not None:
                os.environ["OPENAI_API_BASE"] = original_api_base
            elif "OPENAI_API_BASE" in os.environ:
                del os.environ["OPENAI_API_BASE"]

            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key
            elif args.judge_api_key and "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
        
        # Log summary statistics
        if 'semantic_score' in scored_df.columns:
            sem_mean = scored_df['semantic_score'].mean()
            sty_mean = scored_df['stylistic_score'].mean()
            logging.info(f"Average semantic score: {sem_mean:.3f}")
            logging.info(f"Average stylistic score: {sty_mean:.3f}")
            
            if use_wandb:
                wandb.log({
                    "semantic_score_mean": sem_mean,
                    "stylistic_score_mean": sty_mean,
                    "num_samples": len(scored_df)
                })
        
        if 'heuristic_match_score' in scored_df.columns:
            heur_mean = scored_df['heuristic_match_score'].mean()
            logging.info(f"Average heuristic match: {heur_mean:.3f}")
            
            if use_wandb:
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
                'metrics_csv': scores_file.replace('.csv', '_metrics.csv'),
            }
            if method_stats:
                summary.update(method_stats)
            summary_path = results_file.replace('.csv', '_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logging.info(f"Run summary saved to {summary_path}")

            # Log W&B artifacts if enabled
            if use_wandb:
                try:
                    metrics_csv = scores_file.replace('.csv', '_metrics.csv')
                    art = wandb.Artifact(
                        name=f"run_artifacts_{args.method}_{args.model.replace('/', '-')}_as_{args.disguise_as.replace('/', '-')}",
                        type="run-artifacts",
                        metadata={"method": args.method, "num_samples": args.num_samples},
                    )
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
    
    if use_wandb:
        wandb.finish()
    
    logging.info("Experiment completed!")


if __name__ == "__main__":
    main()
