#!/usr/bin/env python3
"""
Generate model responses for a prompts file, with simple caching.

Supports API providers via LiteLLM. Writes a CSV with columns:
- prompt, model_response, model

Usage:
  python scripts/generate_responses.py \
    --model openai/gpt-4o-mini \
    --prompts_file data/chabot_arena_500_propmts.txt \
    --output disguising/model-responses/base/openai_gpt-4o-mini.csv

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
import pandas as pd


def read_prompts(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_existing(output: Path) -> Dict[str, str]:
    if not output.exists():
        return {}
    try:
        df = pd.read_csv(output)
        if 'prompt' in df.columns and 'model_response' in df.columns:
            return {str(r['prompt']): str(r['model_response']) for _, r in df.iterrows()}
    except Exception:
        pass
    return {}


def _gen_litellm(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    resp = completion(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
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
    parser.add_argument("--prompts_file", required=True, help="Path to prompts text file")
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
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optionally route LiteLLM to a local vLLM server
    if args.openai_api_base:
        os.environ["OPENAI_API_BASE"] = args.openai_api_base
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    existing = {} if args.overwrite else load_existing(out_path)
    mode = 'w' if args.overwrite or not out_path.exists() else 'a'

    with open(out_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(['prompt', 'model_response', 'model'])

        for prompt in tqdm(prompts, desc=f"Generating with {args.model}"):
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
                    text = _gen_litellm(args.model, messages, args.max_tokens, args.temperature)
            except Exception as e:
                text = f"ERROR: {type(e).__name__}: {e}"
            writer.writerow([prompt, text, args.model])

    print(f"Wrote: {out_path}")


if __name__ == '__main__':
    main()
