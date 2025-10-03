#!/usr/bin/env python3

import pandas as pd
import asyncio
import os
import math
import glob
import sys
import threading
from typing import List, Optional, Union
import concurrent.futures
from pathlib import Path

# Adding directories to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
serve_dir = os.path.join(current_dir, 'serve')
disguising_dir = os.path.abspath(os.path.join(current_dir, '..'))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add paths to sys.path if they exist and aren't already there
# for path in [current_dir, serve_dir, disguising_dir, project_root]:
#     if os.path.exists(path) and path not in sys.path:
#         sys.path.insert(0, path)
#         print(f"Added path to sys.path: {path}")

import nest_asyncio
from openai import OpenAI
import tiktoken  # For token counting
from dotenv import load_dotenv
import requests
from transformers import AutoTokenizer  # Using transformers to load the correct tokenizer for Meta LLaMA models
import aiohttp
from vllm import LLM, SamplingParams
from tqdm import tqdm
import wandb
import re
import json
import lmdb
import numpy as np
import logging


#scripts/serve
# Import utilities from serve folder (now in path)
from serve.utils_llm import get_llm_output, get_llm_embedding, get_token_count
from serve.utils_general import get_from_cache, save_to_cache, hash_key
from serve.global_vars import LLAMA_URL

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CachedResponseGenerator:
    """Response generator with caching capabilities and batch processing."""
    
    def __init__(self, cache_dir="cache", enable_wandb=True):
        self.cache_dir = cache_dir
        self.enable_wandb = enable_wandb
        
        # Create cache directories if they don't exist
        os.makedirs(os.path.join(cache_dir, "generation_cache"), exist_ok=True)
        
        # Initialize LMDB cache for generations
        self.generation_cache = lmdb.open(
            os.path.join(cache_dir, "generation_cache"), 
            map_size=int(1e11)
        )
        self.cache_lock = threading.Lock()
        
    def load_prompts(self, dataset_path, marker="### Call Transcript"):
        """Load prompts from CSV or text file."""
        # Try to read as CSV
        try:
            df = pd.read_csv(dataset_path)
            if "prompt" in df.columns:
                return df["prompt"].astype(str).tolist()
        except Exception:
            pass  # Not a CSV or no prompt column
            
        # Otherwise, treat as text file and group by marker
        with open(dataset_path, "r") as f:
            content = f.read()
        conversations = [block.strip() for block in content.split(marker) if block.strip()]
        return conversations

    def get_cached_response(self, prompt: str, model: str, params: dict) -> Optional[str]:
        """Get cached response if available."""
        cache_key = json.dumps({
            "prompt": prompt,
            "model": model,
            "params": params
        }, sort_keys=True)
        
        with self.cache_lock:
            return get_from_cache(cache_key, self.generation_cache)

    def save_cached_response(self, prompt: str, model: str, params: dict, response: str):
        """Save response to cache."""
        cache_key = json.dumps({
            "prompt": prompt,
            "model": model,
            "params": params
        }, sort_keys=True)
        
        with self.cache_lock:
            save_to_cache(cache_key, response, self.generation_cache)

    def format_prompt(self, prompt: str, multimodal: bool = False):
        """Format prompt for API calls."""
        if multimodal:
            return [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        else:
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

    def postprocess_response(self, response: str) -> str:
        """Remove thinking tags and clean up response."""
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        return response.strip()

    def generate_with_openai(self, prompts: List[str], model: str, **kwargs):
        """Generate responses using OpenAI API with caching."""
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if client.api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        responses = []
        multimodal = kwargs.get('multimodal', False)
        params = {
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.95),
            'max_tokens': kwargs.get('max_tokens', 1024)
        }
        
        for prompt in tqdm(prompts, desc=f"Generating responses with {model}"):
            # Check cache first
            cached_response = self.get_cached_response(prompt, model, params)
            if cached_response is not None:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                responses.append(cached_response)
                continue
            
            # Generate new response
            messages = self.format_prompt(prompt, multimodal)
            try:
                # Handle different parameter names for different models
                api_params = {
                    "model": model,
                    "messages": messages,
                    "temperature": params['temperature'],
                    "top_p": params['top_p'],
                }
                
                # Try with max_completion_tokens first for newer models
                try:
                    if "gpt-4" in model.lower() or "o1" in model.lower():
                        api_params["max_completion_tokens"] = params['max_tokens']
                    else:
                        api_params["max_tokens"] = params['max_tokens']
                    
                    completion = client.chat.completions.create(**api_params)
                except Exception as param_error:
                    # If that fails due to parameter issue, try the other parameter name
                    if "max_tokens" in str(param_error) or "max_completion_tokens" in str(param_error):
                        logger.debug(f"Retrying with different token parameter for model {model}")
                        api_params = {
                            "model": model,
                            "messages": messages,
                            "temperature": params['temperature'],
                            "top_p": params['top_p'],
                        }
                        
                        # Switch parameter name
                        if "max_completion_tokens" in api_params:
                            del api_params["max_completion_tokens"]
                            api_params["max_tokens"] = params['max_tokens']
                        else:
                            api_params["max_completion_tokens"] = params['max_tokens']
                        
                        completion = client.chat.completions.create(**api_params)
                    else:
                        raise param_error
                
                response = completion.choices[0].message.content.strip()
                response = self.postprocess_response(response)
                
                # Cache the response
                self.save_cached_response(prompt, model, params, response)
                
            except Exception as e:
                response = f"[ERROR] {e}"
                logger.error(f"Error generating response: {e}")
            
            responses.append(response)
        
        return responses

    def generate_with_vllm(self, prompts: List[str], model: str, **kwargs):
        """Generate responses using vLLM with caching."""
        multimodal = kwargs.get('multimodal', False)
        params = {
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.95),
            'max_tokens': kwargs.get('max_tokens', 1024)
        }
        
        # Check cache for all prompts
        cached_responses = []
        uncached_prompts = []
        uncached_indices = []
        
        for i, prompt in enumerate(prompts):
            cached_response = self.get_cached_response(prompt, model, params)
            if cached_response is not None:
                cached_responses.append((i, cached_response))
            else:
                uncached_prompts.append(prompt)
                uncached_indices.append(i)
        
        logger.info(f"Found {len(cached_responses)} cached responses, generating {len(uncached_prompts)} new ones")
        
        # Generate responses for uncached prompts
        new_responses = []
        if uncached_prompts:
            llm = LLM(
                model=model, 
                trust_remote_code=True, 
                max_model_len=kwargs.get('max_model_len', 4000),  # Reduced from 8000 to fit in GPU memory
                tensor_parallel_size=kwargs.get('tensor_parallel_size', 2),
                dtype="float16",  # Use float16 instead of bfloat16 for RTX 2080 Ti compatibility
                gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.9)  # Increased memory utilization
            )
            
            sampling_params = SamplingParams(
                max_tokens=params['max_tokens'],
                temperature=params['temperature'],
                top_p=params['top_p'],
            )
            
            messages = [self.format_prompt(prompt, multimodal) for prompt in uncached_prompts]
            
            # Generate in batches
            batch_size = kwargs.get('batch_size', 100)
            for i in tqdm(range(0, len(messages), batch_size), desc="Generating responses in batches"):
                batch_messages = messages[i:i+batch_size]
                batch_responses = llm.chat(messages=batch_messages, sampling_params=sampling_params)
                batch_responses = [resp.outputs[0].text for resp in batch_responses]
                
                # Post-process and cache
                for j, response in enumerate(batch_responses):
                    response = self.postprocess_response(response)
                    prompt = uncached_prompts[i + j]
                    self.save_cached_response(prompt, model, params, response)
                    new_responses.append(response)
        
        # Combine cached and new responses in correct order
        all_responses = [None] * len(prompts)
        
        # Place cached responses
        for i, response in cached_responses:
            all_responses[i] = response
        
        # Place new responses
        for i, response in enumerate(new_responses):
            all_responses[uncached_indices[i]] = response
        
        return all_responses

    def generate_with_serve_llm(self, prompts: List[str], model: str, **kwargs):
        """Generate responses using the serve folder's LLM utilities."""
        params = {
            'max_tokens': kwargs.get('max_tokens', 1024)
        }
        
        responses = []
        for prompt in tqdm(prompts, desc=f"Generating responses with {model} (serve)"):
            try:
                response = get_llm_output(
                    prompt=prompt,
                    model=model,
                    cache=True,
                    system_prompt="You are a helpful assistant.",
                    max_tokens=params['max_tokens']
                )
                response = self.postprocess_response(response)
                responses.append(response)
            except Exception as e:
                response = f"[ERROR] {e}"
                logger.error(f"Error generating response: {e}")
                responses.append(response)
        
        return responses

    def save_results(self, prompts: List[str], responses: List[str], model: str, 
                    output_path: str, **kwargs):
        """Save results to CSV with metadata."""
        multimodal = kwargs.get('multimodal', False)
        messages = [self.format_prompt(prompt, multimodal) for prompt in prompts]
        messages_json = [json.dumps(m) for m in messages]
        
        df = pd.DataFrame({
            "prompt": prompts, 
            "messages": messages_json, 
            "model_response": responses, 
            "model": model.replace("/", "_")
        })
        
        # Add token length analysis
        df["prompt_token_length"] = df["prompt"].apply(lambda x: get_token_count(x))
        df["response_token_length"] = df["model_response"].apply(lambda x: get_token_count(x))
        
        # Remove failed responses
        df = df.dropna(subset=["model_response"])
        df = df[~df["model_response"].str.startswith("[ERROR]")]
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        logger.info(f"Generated {len(df)} responses")
        logger.info(f"Saved to {output_path}")
        
        # Log to wandb if enabled
        if self.enable_wandb:
            try:
                wandb.log({"data": wandb.Table(dataframe=df)})
                wandb.summary["num_responses"] = len(df)
                wandb.summary["average_prompt_token_length"] = df["prompt_token_length"].mean()
                wandb.summary["average_response_token_length"] = df["response_token_length"].mean()
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
        
        return df

    def generate_batch_scripts(self, csv_files: List[str], output_dir: str, 
                             num_scripts: int = 8, script_prefix: str = "run_generation"):
        """Generate bash scripts for batch processing multiple CSV files."""
        files_per_script = math.ceil(len(csv_files) / num_scripts)
        
        for i in range(num_scripts):
            script_path = os.path.join(output_dir, f"{script_prefix}_{i+1}.sh")
            
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n\n")
                
                start_index = i * files_per_script
                end_index = min((i + 1) * files_per_script, len(csv_files))
                
                for csv_file in csv_files[start_index:end_index]:
                    # Adjust the command based on your needs
                    script_name = "disguising/scripts/generate_responses_with_caching.py"
                    command = f"python {script_name} --dataset {csv_file} --output_dir {output_dir}\n"
                    f.write(command)
            
            # Make script executable
            os.chmod(script_path, 0o755)
        
        logger.info(f"Generated {num_scripts} bash scripts with {files_per_script} files each")
        return [os.path.join(output_dir, f"{script_prefix}_{i+1}.sh") for i in range(num_scripts)]


def main():
    import argparse
    
    
    parser = argparse.ArgumentParser(description="Generate model responses with caching and batch processing")
    
    # Model and generation parameters
    parser.add_argument("--model", type=str, default="gpt-4o-mini", 
                       help="Model to use for generation")
    parser.add_argument("--use_serve_llm", action="store_true",
                       help="Use serve folder's LLM utilities instead of direct API calls")
    parser.add_argument("--use_vllm", action="store_true",
                       help="Use vLLM for generation (for local models)")
    parser.add_argument("--gpt_model", type=str, default=None, 
                       help="If set, use this OpenAI GPT model via API")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, 
                       default="/home/nazcol/dementor/dementor/disguising/model-responses/base_500_all_models/call_center_prompts.csv",
                       help="Path to the dataset of prompts")
    parser.add_argument("--output_dir", type=str, 
                       default="data/model-responses/call_center",
                       help="Output directory for generated responses")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of samples to generate (None for all)")
    parser.add_argument("--max_prompt_tokens", type=int, default=1024,
                       help="Maximum number of tokens in input prompts")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--multimodal", action="store_true")
    
    # vLLM specific parameters
    parser.add_argument("--max_model_len", type=int, default=4000,
                       help="Maximum sequence length (reduced from 8000 to fit in GPU memory)")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                       help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Batch size for generation")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="Fraction of GPU memory to use (0.0 to 1.0)")
    
    # Batch processing parameters
    parser.add_argument("--generate_scripts", action="store_true",
                       help="Generate bash scripts for batch processing")
    parser.add_argument("--script_dir", type=str, default="scripts",
                       help="Directory to save generated scripts")
    parser.add_argument("--num_scripts", type=int, default=8,
                       help="Number of batch scripts to generate")
    parser.add_argument("--input_pattern", type=str, 
                       default="../model-responses/disguised/stylistic_clustering/**/temp0.7/*.csv",
                       help="Glob pattern for input files when generating scripts")
    
    # Cache and logging parameters
    parser.add_argument("--cache_dir", type=str, default="cache",
                       help="Directory for caching")
    parser.add_argument("--disable_wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="disguising-generations",
                       help="Wandb project name")
    parser.add_argument("--dtype", type=str, default=None,
                       help="Data type for the cache")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CachedResponseGenerator(
        cache_dir=args.cache_dir, 
        enable_wandb=not args.disable_wandb
    )
    
    # Handle batch script generation
    if args.generate_scripts:
        csv_files = glob.glob(args.input_pattern, recursive=True)
        logger.info(f"Found {len(csv_files)} CSV files matching pattern: {args.input_pattern}")
        
        scripts = generator.generate_batch_scripts(
            csv_files=csv_files,
            output_dir=args.script_dir,
            num_scripts=args.num_scripts
        )
        
        print(f"Generated {len(scripts)} batch scripts:")
        for script in scripts:
            print(f"  {script}")
        return
    
    # Initialize wandb if enabled
    if not args.disable_wandb:
        model_name = args.gpt_model or args.model
        wandb.init(
            project=args.wandb_project, 
            name=f"{model_name.replace('/', '_')}_cached_responses"
        )
        wandb.config.update(args)
    
    # Load and filter prompts
    prompts = generator.load_prompts(args.dataset)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Filter by token length
    prompt_tokens = [get_token_count(prompt) for prompt in prompts]
    prompts = [prompt for prompt, token_length in zip(prompts, prompt_tokens) 
              if token_length <= args.max_prompt_tokens]
    logger.info(f"Filtered to {len(prompts)} prompts with <= {args.max_prompt_tokens} tokens")
    
    # Limit number of samples
    if args.num_samples is not None:
        prompts = prompts[:args.num_samples]
        logger.info(f"Limited to {len(prompts)} samples")
    
    # Determine model and generation method
    if args.gpt_model:
        model = args.gpt_model
        use_openai = True
    elif args.use_serve_llm:
        model = args.model
        use_openai = False
        use_vllm = False
    elif args.use_vllm:
        model = args.model
        use_openai = False
        use_vllm = True
    else:
        # Default to OpenAI for GPT models, serve_llm for others
        model = args.model
        use_openai = "gpt" in model.lower()
        use_vllm = not use_openai
    
    # Generate responses
    logger.info(f"Generating responses with model: {model}")
    
    if use_openai:
        responses = generator.generate_with_openai(
            prompts=prompts,
            model=model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            multimodal=args.multimodal
        )
    elif use_vllm:
        responses = generator.generate_with_vllm(
            prompts=prompts,
            model=model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            multimodal=args.multimodal,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size
        )
    else:
        responses = generator.generate_with_serve_llm(
            prompts=prompts,
            model=model,
            max_tokens=args.max_tokens
        )
    
    # Save results
    model_clean = model.replace("/", "_")
    if args.num_samples:
        output_path = os.path.join(args.output_dir, f"{model_clean}-{args.num_samples}.csv")
    else:
        output_path = os.path.join(args.output_dir, f"{model_clean}.csv")
    
    df = generator.save_results(
        prompts=prompts,
        responses=responses,
        model=model,
        output_path=output_path,
        multimodal=args.multimodal
    )
    
    logger.info("Generation completed successfully!")


if __name__ == "__main__":
    main()
