#!/usr/bin/env python3
"""
Streamlined disguise script for the core methods:
- contrastive
- behavioral_based
- stylistic
- random_sampling
"""
import argparse
import os
import pandas as pd
import os
import logging
import time
import re
from pathlib import Path
from typing import Optional

# Add scripts to path for proper imports
import sys
import os
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
# Also add the project root to sys.path so absolute imports work
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from methods.get_method import get_method
from scorer import score_model_single
from litellm import completion
import litellm
from tqdm import tqdm
import wandb
from scripts.cache_llm import register_model_config

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

    # Standardize columns expected by methods:
    # - source_df must have 'model_response'
    # - target_df must have 'target_response'
    if 'model_response' not in source_df.columns and 'target_response' in source_df.columns:
        source_df = source_df.rename(columns={'target_response': 'model_response'})
    if 'target_response' not in target_df.columns and 'model_response' in target_df.columns:
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
    temperature: float = 0.0,
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
            # temperature handled via function argument

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
                # LiteLLM: decide routing and provider
                routed_model = model
                generation_api_base = os.getenv('GENERATION_API_BASE')
                generation_api_key = os.getenv('GENERATION_API_KEY')
                generation_provider = os.getenv('GENERATION_PROVIDER')
                api_base = generation_api_base
                api_key = generation_api_key

                # Force official base for OpenAI models
                if model_lower.startswith('openai/gpt-'):
                    os.environ['OPENAI_API_BASE'] = 'https://api.openai.com/v1'
                    api_base = 'https://api.openai.com/v1'
                    if not generation_provider:
                        generation_provider = 'openai'

                if not api_base and (not generation_provider or generation_provider == 'openai'):
                    api_base = os.getenv('OPENAI_API_BASE')
                if not api_key and (not generation_provider or generation_provider == 'openai'):
                    api_key = os.getenv('OPENAI_API_KEY')

                provider_hint = generation_provider
                if api_base and not provider_hint and not _has_provider_prefix(model):
                    routed_model = f"openai/{model}"

                def _llm_call():
                    # Use cached completion for persistent caching
                    call_kwargs = {
                        'model': routed_model,
                        'messages': disguised_messages,
                        'temperature': temperature,
                        'max_tokens': max_new_tokens,
                        'request_timeout': 60,
                    }
                    if api_base:
                        call_kwargs['api_base'] = api_base
                    if api_key:
                        call_kwargs['api_key'] = api_key
                    if provider_hint:
                        if provider_hint != 'openai' or not _has_provider_prefix(routed_model):
                            call_kwargs.setdefault('custom_llm_provider', provider_hint)
                            call_kwargs.setdefault('litellm_provider', provider_hint)
                    try:
                        from scripts.cache_llm import cached_completion
                        resp = cached_completion(**call_kwargs)
                        return resp.choices[0].message.content
                    except ImportError:
                        # Fallback to standard litellm
                        resp = completion(**call_kwargs)
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
    return pd.DataFrame(results), method_stats


# (Removed short-name formatter; we use official model identifiers in filenames and tags.)


def main():
    env_defaults = {
        "model": os.getenv("DISGUISE_MODEL"),
        "disguise_as": os.getenv("DISGUISE_TARGET_MODEL"),
        "source_responses": os.getenv("DISGUISE_SOURCE_RESPONSES"),
        "target_responses": os.getenv("DISGUISE_TARGET_RESPONSES"),
        "prompts_file": os.getenv("DISGUISE_PROMPTS_FILE"),
        "output_dir": os.getenv("DISGUISE_OUTPUT_DIR"),
        "num_samples": os.getenv("DISGUISE_NUM_SAMPLES"),
    }

    parser = argparse.ArgumentParser(description="Streamlined disguise experiments")

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=env_defaults["model"] is None,
        default=env_defaults["model"],
        help="Source model to disguise (e.g., google/gemma-3-1b-it)",
    )
    parser.add_argument(
        "--disguise-as",
        type=str,
        required=env_defaults["disguise_as"] is None,
        default=env_defaults["disguise_as"],
        help="Target model to mimic (e.g., openai/gpt-4.1-mini)",
    )
    
    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "contrastive",
            "behavioral_based",
            "stylistic",
            "random_sampling",
            "just_name_it",
            "stylistic_clustering",
            "stylistic_clustering_resample",
            "embedding_clustering",
            "behavioral_clustering",
        ],
        default="contrastive",
        help="Disguise method to use",
    )
    
    # Data paths
    parser.add_argument(
        "--source-responses",
        type=str,
        default=env_defaults["source_responses"],
        help="Path to source model responses CSV (auto-detect if not provided)",
    )
    parser.add_argument(
        "--target-responses",
        type=str,
        default=env_defaults["target_responses"],
        help="Path to target model responses CSV (auto-detect if not provided)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/datasets/chatbot_arena/chatbot_arena_prompts.txt",
        help="File with prompts (one per line)",
    )

    # Experiment settings
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of responses to generate",
    )
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for generation (0.0 for deterministic)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=env_defaults["output_dir"],
        help="Output directory for results",
    )

    # Evaluation settings
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip automatic evaluation")
    parser.add_argument("--heuristics-only", action="store_true",
                       help="Only compute heuristic scores (faster)")
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini",
                       help="Judge model for scoring (default: openai/gpt-4.1-mini)")
    parser.add_argument("--judge-api-base", default="https://api.openai.com/v1",
                       help="API base URL for judge model (default: OpenAI API)")
    parser.add_argument("--judge-api-key", default=None,
                       help="API key for judge model (defaults to OPENAI_API_KEY env var)")
    
    # Logging
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging (enabled by default)")
    parser.add_argument("--run-name", type=str,
                       help="Name for this experiment run (auto-generated if not provided)")
    # LiteLLM routing (e.g., to a local vLLM server exposing OpenAI-compatible API)
    parser.add_argument("--openai-api-base", type=str, default=None,
                       help="Override OPENAI_API_BASE for LiteLLM (e.g., http://localhost:8000/v1)")
    parser.add_argument("--openai-api-key", type=str, default=None,
                       help="Override OPENAI_API_KEY for LiteLLM (placeholder allowed for local)")

    # Granular API routing overrides
    parser.add_argument("--generation-api-base", type=str, default=None,
                       help="API base for source model generation (overrides GENERATION_API_BASE env)")
    parser.add_argument("--generation-api-key", type=str, default=None,
                       help="API key for source model generation (overrides GENERATION_API_KEY env)")
    parser.add_argument("--generation-provider", type=str, default=None,
                       help="LiteLLM provider hint for source model generation")

    parser.add_argument("--analysis-api-base", type=str, default=None,
                       help="API base for analyzer models (contrastive/behavioral)")
    parser.add_argument("--analysis-api-key", type=str, default=None,
                       help="API key for analyzer models")
    parser.add_argument("--analysis-provider", type=str, default=None,
                       help="LiteLLM provider hint for analyzer calls")

    parser.add_argument("--embedding-api-base", type=str, default=None,
                       help="API base for embedding lookups (embedding_clustering)")
    parser.add_argument("--embedding-api-key", type=str, default=None,
                       help="API key for embedding lookups")
    parser.add_argument("--embedding-provider", type=str, default=None,
                       help="LiteLLM provider hint for embedding calls")
    
    args = parser.parse_args()
    setup_logging()

    def _maybe_override(value, fallback, *, compare_default=None, transform=lambda x: x):
        if fallback is None:
            return value
        if compare_default is not None and value != compare_default:
            return value
        try:
            return transform(fallback)
        except Exception:
            return value

    prompts_default = parser.get_default("prompts_file")
    args.prompts_file = _maybe_override(
        args.prompts_file,
        env_defaults["prompts_file"],
        compare_default=prompts_default,
    )
    num_samples_default = parser.get_default("num_samples")
    args.num_samples = _maybe_override(
        args.num_samples,
        env_defaults["num_samples"],
        compare_default=num_samples_default,
        transform=int,
    )
    if not args.source_responses and env_defaults["source_responses"]:
        args.source_responses = env_defaults["source_responses"]
    if not args.target_responses and env_defaults["target_responses"]:
        args.target_responses = env_defaults["target_responses"]
    if args.output_dir is None and env_defaults["output_dir"]:
        args.output_dir = env_defaults["output_dir"]
    
    # Defer creating output directory until after resolving dataset + defaults

    # Optionally route LiteLLM calls to a local vLLM server
    # Preserve original OpenAI routing so analyzer models can still reach OpenAI if needed
    if "OPENAI_API_BASE" in os.environ and "ORIGINAL_OPENAI_API_BASE" not in os.environ:
        os.environ["ORIGINAL_OPENAI_API_BASE"] = os.environ["OPENAI_API_BASE"]
    if "OPENAI_API_KEY" in os.environ and "ORIGINAL_OPENAI_API_KEY" not in os.environ:
        os.environ["ORIGINAL_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

    if args.openai_api_base:
        os.environ["OPENAI_API_BASE"] = args.openai_api_base
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # Generation routing (source model)
    generation_api_base = args.generation_api_base or os.getenv("GENERATION_API_BASE")
    generation_api_key = args.generation_api_key or os.getenv("GENERATION_API_KEY")
    generation_provider = args.generation_provider or os.getenv("GENERATION_PROVIDER")
    generation_config = {}
    if generation_api_base:
        generation_config["api_base"] = generation_api_base
        os.environ["GENERATION_API_BASE"] = generation_api_base
    if generation_api_key:
        generation_config["api_key"] = generation_api_key
        os.environ["GENERATION_API_KEY"] = generation_api_key
    if generation_provider:
        generation_config["custom_llm_provider"] = generation_provider
        os.environ["GENERATION_PROVIDER"] = generation_provider
    if generation_config:
        register_model_config(args.model, generation_config)

    # Analysis routing (contrastive / behavioral)
    analysis_api_base = args.analysis_api_base or os.getenv("ANALYSIS_API_BASE")
    analysis_api_key = args.analysis_api_key or os.getenv("ANALYSIS_API_KEY")
    analysis_provider = args.analysis_provider or os.getenv("ANALYSIS_PROVIDER")
    if analysis_api_base:
        os.environ["ANALYSIS_API_BASE"] = analysis_api_base
    if analysis_api_key:
        os.environ["ANALYSIS_API_KEY"] = analysis_api_key
    if analysis_provider:
        os.environ["ANALYSIS_PROVIDER"] = analysis_provider

    # Embedding routing (embedding_clustering)
    embedding_api_base = args.embedding_api_base or os.getenv("EMBEDDING_API_BASE")
    embedding_api_key = args.embedding_api_key or os.getenv("EMBEDDING_API_KEY")
    embedding_provider = args.embedding_provider or os.getenv("EMBEDDING_PROVIDER")
    if embedding_api_base:
        os.environ["EMBEDDING_API_BASE"] = embedding_api_base
    if embedding_api_key:
        os.environ["EMBEDDING_API_KEY"] = embedding_api_key
    if embedding_provider:
        os.environ["EMBEDDING_PROVIDER"] = embedding_provider

    # Hard-guard: if generation model is an OpenAI-hosted model, force official base
    try:
        model_lower = (args.model or "").lower()
        if model_lower.startswith("openai/gpt-"):
            os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
            logging.info("Detected OpenAI provider model '%s'; routing generation to OpenAI API.", args.model)
    except Exception:
        pass
    
    # Determine dataset and subset once (used for paths and optional logging)
    def _infer_dataset(prompts_path: str) -> str:
        p = (prompts_path or "").lower()
        if "gsm8k" in p:
            return "gsm8k"
        if "arena" in p or "chatbot" in p:
            return "chatbot_arena"
        return "generic"
    dataset = _infer_dataset(args.prompts_file)
    def _infer_subset(prompts_path: str) -> str:
        p = (prompts_path or "").lower()
        if "_500" in p or "/500" in p or p.endswith("500.csv"):
            return "500"
        stem = Path(prompts_path or "").stem.lower()
        eval_match = re.search(r"eval[_-]?(\d+)", stem)
        if eval_match:
            return f"eval{eval_match.group(1)}"
        train_match = re.search(r"train[_-]?(\d+)", stem)
        if train_match:
            return f"train{train_match.group(1)}"
        return "full"
    subset = _infer_subset(args.prompts_file)

    # Helper to create a filesystem/tag-friendly identifier from the official model name
    def _official_id(s: str) -> str:
        return (s or "").replace('/', '_').replace(':', '_')

    # Precompute official identifiers for logging/filenames
    src_id = _official_id(args.model)
    tgt_id = _official_id(args.disguise_as)

    # Initialize wandb by default (unless disabled)
    use_wandb = not args.no_wandb
    if use_wandb:
        # Auto-generate run name if not provided
        if not args.run_name:
            timestamp = pd.Timestamp.now().strftime("%m%d_%H%M")
            args.run_name = f"{args.method}_{src_id}_as_{tgt_id}_{timestamp}"

        wandb.init(
            project="dementor-disguise",
            name=args.run_name,
            config=vars(args),
            tags=[args.method, dataset, src_id, tgt_id]
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
            "(preferred) or legacy '.../base'. You can also pass explicit paths via --source-responses and "
            "--target-responses."
        )
        raise
    
    # Load prompts (CSV only; expects a 'prompt' column)
    if not os.path.exists(args.prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")
    if not args.prompts_file.lower().endswith('.csv'):
        raise ValueError("prompts_file must be a CSV with a 'prompt' column. TXT is no longer supported.")
    try:
        import csv as _csv
        prompts_df = pd.read_csv(args.prompts_file)
    except pd.errors.ParserError:
        try:
            prompts_df = pd.read_csv(args.prompts_file, on_bad_lines='skip', quoting=_csv.QUOTE_ALL)
        except pd.errors.ParserError:
            prompts_df = pd.read_csv(args.prompts_file, on_bad_lines='skip', quoting=_csv.QUOTE_NONE, engine='python')
    if 'prompt' not in prompts_df.columns:
        raise ValueError("prompts_file CSV must contain a 'prompt' column")
    prompts = [str(p).strip() for p in prompts_df['prompt'].tolist() if str(p).strip()]
    
    logging.info(f"Loaded {len(prompts)} prompts")
    
    # Determine output dir and result path before generation for incremental writes
    # New layout (per user): data/results/<dataset>/<subset>/<method>/{pair}.csv
    if args.output_dir in (None, "results/streamlined"):
        default_dir = os.path.join("data", "results", dataset, subset, args.method)
        os.makedirs(default_dir, exist_ok=True)
        args.output_dir = default_dir
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    pair_id = f"{src_id}_as_{tgt_id}"
    results_file = os.path.join(args.output_dir, f"{pair_id}.csv")

    # Generate disguised responses
    logging.info(f"Generating responses using {args.method}...")
    results_df, method_stats = generate_disguised_responses(
        args.method, args.model, args.disguise_as,
        source_df, target_df, prompts, args.num_samples,
        temperature=args.temperature,
        method_kwargs=None,
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
        # Group scores under: data/results/<dataset>/<subset>/<method>/scores/<pair>/
        metrics_dir = os.path.join(args.output_dir, "scores", pair_id)
        os.makedirs(metrics_dir, exist_ok=True)
        scores_file = os.path.join(metrics_dir, "scored.csv")

        # Set up judge model environment if specified
        original_api_base = os.environ.get("OPENAI_API_BASE")
        original_api_key = os.environ.get("OPENAI_API_KEY")

        if args.judge_api_base:
            os.environ["OPENAI_API_BASE"] = args.judge_api_base
        if args.judge_api_key:
            os.environ["OPENAI_API_KEY"] = args.judge_api_key

        try:
            # Single-file scoring (no pairwise): compute stylistic features + aggregates
            scored_df = score_model_single(
                input_file=results_file,
                output_file=scores_file,
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
        # For single-file scoring, we only have feature columns + response_length
        try:
            if 'response_length' in scored_df.columns:
                rlen = scored_df['response_length'].mean()
                logging.info(f"Average response length: {rlen:.1f}")
                if use_wandb:
                    wandb.log({"response_length_mean": rlen, "num_samples": len(scored_df)})
        except Exception:
            pass

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
            # Write summary into metrics_<pair>/ alongside scores/metrics
            summary_path = os.path.join(metrics_dir, "summary.json")
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
