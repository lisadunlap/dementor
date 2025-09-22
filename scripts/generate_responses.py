#!/usr/bin/env python3
"""
Generate model responses for a prompts CSV (no TXT), with simple caching.

Input CSV must contain a 'prompt' column. Writes a CSV with columns:
- prompt, model_response, model

Usage:
  python scripts/generate_responses.py \
    --model openai/gpt-4o-mini \
    --prompts_file data/datasets/chatbot_arena/prompts.csv \
    --output data/model-responses/chatbot_arena/full/openai_gpt-4o-mini.csv

If the output file exists and --overwrite is not provided, the script skips prompts
already present in the existing CSV (simple caching by prompt text).
"""
import argparse
import csv
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from typing import Optional
import sys
from litellm import completion
import litellm

# Enable caching for API calls (not for vLLM/local servers)
if not hasattr(litellm, 'cache') or litellm.cache is None:
    litellm.cache = litellm.Cache()
import pandas as pd
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # If python-dotenv isn't installed, skip silently; env vars must be exported by shell
    pass


def read_prompts(path: str) -> List[str]:
    """Read prompts from a CSV with a 'prompt' column (TXT not supported)."""
    if not path.lower().endswith('.csv'):
        raise ValueError("prompts_file must be a CSV with a 'prompt' column. TXT is no longer supported.")
    import csv as _csv
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(path, on_bad_lines='skip', quoting=_csv.QUOTE_ALL)
        except pd.errors.ParserError:
            df = pd.read_csv(path, on_bad_lines='skip', quoting=_csv.QUOTE_NONE, engine='python')
    if 'prompt' not in df.columns:
        raise ValueError("prompts CSV must contain a 'prompt' column")
    return [str(p).strip() for p in df['prompt'].tolist() if str(p).strip()]


def load_existing(output: Path) -> Dict[str, str]:
    if not output.exists():
        return {}
    try:
        df = pd.read_csv(output, quoting=csv.QUOTE_ALL)
        if 'prompt' in df.columns and 'model_response' in df.columns:
            return {str(r['prompt']): str(r['model_response']) for _, r in df.iterrows()}
    except Exception as e:
        print(f"Warning: Could not load existing CSV ({e}), starting fresh")
        return {}
    return {}


def _gen_litellm(model: str, messages: list, max_tokens: int, temperature: float, api_base: str = None, api_key: str = None) -> str:
    # Use cached completion for persistent caching
    try:
        from .cached_llm import cached_completion
        use_cached = True
    except ImportError:
        use_cached = False

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    if use_cached:
        resp = cached_completion(**kwargs)
        return resp.choices[0].message.content
    else:
        resp = completion(**kwargs)
        return resp["choices"][0]["message"]["content"]


def _gen_hf(model_id: str, prompt_text: str, max_tokens: int, temperature: float) -> str:
    try:
        from transformers import pipeline
    except ImportError as e:
        raise RuntimeError("Transformers not installed. pip install transformers accelerate") from e
    # simple text-generation pipeline; relies on device_map='auto' for GPU if available
    pipe = pipeline("text-generation", model=model_id, device_map="auto")
    out = pipe(prompt_text, max_new_tokens=max_tokens, do_sample=(temperature > 0), temperature=max(temperature, 1e-6))
    text = out[0]['generated_text']
    # Return suffix beyond the prompt to avoid echo
    return text[len(prompt_text):].strip()


def _gen_vllm(model_id: str, prompt_text: str, max_tokens: int, temperature: float) -> str:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError("vLLM not installed. pip install vllm") from e
    llm = LLM(model=model_id, trust_remote_code=True)
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = llm.generate([prompt_text], params)
    return outputs[0].outputs[0].text


def main():
    parser = argparse.ArgumentParser(description="Generate model responses for prompts")
    parser.add_argument("--model", required=True, help="LiteLLM model id (e.g., openai/gpt-4o-mini)")
    parser.add_argument("--prompts_file", required=True, help="Path to prompts CSV with 'prompt' column (TXT not supported)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--system", default=None, help="Optional system message")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    # Route LiteLLM to a local vLLM/OpenAI-compatible endpoint if desired
    parser.add_argument("--openai-api-base", default=None, help="Override OPENAI_API_BASE for LiteLLM (e.g., http://localhost:8000/v1)")
    parser.add_argument("--openai-api-key", default=None, help="Override OPENAI_API_KEY for LiteLLM (placeholder allowed for local)")
    args = parser.parse_args()

    prompts = read_prompts(args.prompts_file)
    out_path = Path(args.output)
    # Soft guidance toward canonical storage
    try:
        if str(out_path).startswith('/tmp') or str(out_path).startswith('tmp/'):
            print("Warning: Writing to a temporary path. Canonical location for base outputs is 'data/model-responses/<dataset>/full/'.")
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure LiteLLM picks up routing via environment if provided
    if args.openai_api_base:
        os.environ["OPENAI_API_BASE"] = args.openai_api_base
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    # Keep local references for explicit pass-through (some providers accept these kwargs)
    api_base = args.openai_api_base
    api_key = args.openai_api_key

    # Hard-guard: never route OpenAI-hosted GPT models to a local base
    try:
        model_lower = (args.model or "").lower()
        if model_lower.startswith("openai/gpt-"):
            os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
            api_base = "https://api.openai.com/v1"
            print(f"Routing '{args.model}' to the official OpenAI API base.")
    except Exception:
        pass

    existing = {} if args.overwrite else load_existing(out_path)
    mode = 'w' if args.overwrite or not out_path.exists() else 'a'

    print(f"Writing to: {out_path}")
    print(f"Mode: {mode}")
    print(f"Number of prompts: {len(prompts)}")
    
    with open(out_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if mode == 'w':
            writer.writerow(['prompt', 'model_response', 'model'])
            print("Wrote header")

        for i, prompt in enumerate(tqdm(prompts, desc=f"Generating with {args.model}")):
            if prompt in existing:
                continue
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            messages.append({"role": "user", "content": prompt})
            try:
                model_lower = args.model.lower()
                if model_lower.startswith("hf:") or model_lower.startswith("huggingface:"):
                    model_id = args.model.split(":", 1)[1]
                    sys_text = (args.system + "\n\n" if args.system else "")
                    text = _gen_hf(model_id, sys_text + prompt, args.max_tokens, args.temperature)
                elif model_lower.startswith("vllm:"):
                    model_id = args.model.split(":", 1)[1]
                    sys_text = (args.system + "\n\n" if args.system else "")
                    text = _gen_vllm(model_id, sys_text + prompt, args.max_tokens, args.temperature)
                else:
                    # Route to local OpenAI-compatible base if provided
                    model_lower = args.model.lower()
                    use_routing = api_base and any(x in model_lower for x in ['llama', 'meta-llama', 'mistral', 'phi'])

                    def _has_provider_prefix(m: str) -> bool:
                        m = (m or '').lower()
                        return any(m.startswith(p) for p in ['openai/', 'azure/', 'anthropic/', 'vertex/', 'bedrock/'])

                    if use_routing:
                        # If caller passed a raw HF id, wrap with openai/ so LiteLLM uses OpenAI-compatible client
                        routed_model = args.model if _has_provider_prefix(args.model) else f"openai/{args.model}"
                        text = _gen_litellm(routed_model, messages, args.max_tokens, args.temperature, api_base, api_key)
                    else:
                        text = _gen_litellm(args.model, messages, args.max_tokens, args.temperature)
            except Exception as e:
                text = f"ERROR: {type(e).__name__}: {e}"
                print(f"Error on prompt {i}: {e}")
            
            writer.writerow([prompt, text, args.model])
            f.flush()  # Force flush to disk
            if i < 3:  # Debug first few rows
                print(f"Wrote row {i+1}: {text[:50]}...")

    print(f"Wrote: {out_path}")


if __name__ == '__main__':
    main()
