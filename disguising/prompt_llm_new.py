import pandas as pd
import asyncio
import os
import nest_asyncio
from openai import OpenAI
import tiktoken  # For token counting
from dotenv import load_dotenv
import requests
from transformers import AutoTokenizer  # Using transformers to load the correct tokenizer for Meta LLaMA models
import aiohttp
from tqdm import tqdm
from vllm import LLM, SamplingParams
from tqdm import tqdm
import wandb
import re
from utils import get_token_count

# Run event loop
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3-9B")
    parser.add_argument("--output_column", type=str, default="model_response")
    parser.add_argument("--output_dir", type=str, default="disguising/model-responses/base_500_all_models")
    parser.add_argument("--dataset", type=str, default="data/chabot_arena_500_propmts.txt", help="Path to the dataset of prompts to use for generation")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=8000)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()
    
    # df = pd.read_csv(LOAD_PATH)
    with open(args.dataset, "r") as f:
        prompts = f.readlines()

    # remove any rows where the prompt is more than 2048 tokens
    prompts_tokens = [get_token_count(prompt) for prompt in prompts]
    prompts = [prompt for prompt, token_length in zip(prompts, prompts_tokens) if token_length <= 1024]

    if args.num_samples is not None:
        prompts = prompts[:args.num_samples]

    wandb.init(project="disguising-generations", name=f"{args.model.replace('/', '_')}_responses")
    wandb.config.update(args)

    
    def format_prompt(prompt):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    
    def format_multimodal_prompt(prompt):
        return [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    
    def postprocess_response(response):
        # Remove any content between <think> tags
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        return response.strip()
    
    messages = [format_multimodal_prompt(prompt) if args.multimodal else format_prompt(prompt) for prompt in prompts]

    save_path = f"{args.output_dir}/{args.model.replace('/', '_')}-{args.num_samples}.csv" if args.num_samples is not None else f"{args.output_dir}/{args.model.replace('/', '_')}.csv"
    
    llm = LLM(model=args.model, trust_remote_code=True, max_model_len=args.max_model_len, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    # batch size of 100
    responses = []
    for i in tqdm(range(0, len(messages), 100), desc="Generating responses in batches"):
        responses.extend(llm.chat(messages=messages[i:i+100], sampling_params=sampling_params))

    responses = [response.outputs[0].text for response in responses]
    df = pd.DataFrame({"prompt": prompts, "messages": messages, "model_response": responses})
    df["response_token_length"] = df["model_response"].apply(lambda x: get_token_count(x))
    df = df.dropna(subset=["model_response"])
    print(f"Generated {len(df)} responses")
    print(f"Saved to {save_path}")
    df.to_csv(save_path, index=False)
    wandb.log({"data": wandb.Table(dataframe=df)})
    wandb.summary["average_response_token_length"] = df["response_token_length"].mean()