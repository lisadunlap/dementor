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
    parser.add_argument("--dataset", type=str, default="data/chatbot_arena_prompts.txt", help="Path to the dataset of prompts to use for generation")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--num_samples", type=int)
    args = parser.parse_args()
    
    prompts = pd.read_csv("disguising/model-responses/base_500/gpt-4o.csv")["prompt"].tolist()
    # with open(args.dataset, "r") as f:
    #     prompts = f.readlines()
    # print(f"Loaded {len(prompts)} prompts")

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

    save_path = f"disguising/model-responses/base_2/{args.model.replace('/', '_')}_responses-{args.num_samples}.csv" if args.num_samples is not None else f"disguising/model-responses/{args.model.replace('/', '_')}_responses.csv"
    # make directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    llm = LLM(model=args.model, trust_remote_code=True, max_model_len=16000)
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    # try generating 1 response
    example_response = llm.chat(messages=messages[0], sampling_params=sampling_params)
    # batch size of 100
    responses = []
    for i in tqdm(range(0, len(messages), 100), desc="Generating responses in batches"):
        responses.extend(llm.chat(messages=messages[i:i+100], sampling_params=sampling_params))

    responses = [response.outputs[0].text for response in responses]
    processed_responses = [postprocess_response(response) for response in responses]
    df = pd.DataFrame({"prompt": prompts, "messages": messages, "raw_response": responses, "model_response": processed_responses})
    df["response_token_length"] = df["model_response"].apply(lambda x: get_token_count(x))
    df = df.dropna(subset=["model_response"])
    print(f"Generated {len(df)} responses")
    print(f"Saved to {save_path}")
    df.to_csv(save_path, index=False)
    wandb.log({"data": wandb.Table(dataframe=df)})
    wandb.summary["average_response_token_length"] = df["response_token_length"].mean()