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

# Set constants
TOKEN_LIMIT = 8192 # Max token limit
LOAD_PATH = "disguising/comparisons/minimodel_responses.csv"


# Run event loop
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3-9B")
    parser.add_argument("--output_column", type=str, default="model_response")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_samples", type=int)
    args = parser.parse_args()
    
    df = pd.read_csv(LOAD_PATH)
    # Initialize tokenizer for token counting
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # remove any rows where the prompt is more than 2048 tokens
    df["prompt_tokens"] = df["prompt"].apply(lambda x: len(tokenizer.encode(x)))
    df = df[df["prompt_tokens"] <= 1024]
    df = df.drop(columns=["prompt_tokens"])

    if args.num_samples is not None:
        df = df.sample(n=args.num_samples, random_state=42)

    wandb.init(project="disguising-generations", name=f"{args.model.replace('/', '_')}_responses")
    wandb.config.update(args)

    def format_prompt(prompt):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    
    df["messages"] = df["prompt"].apply(format_prompt)

    save_path = f"disguising/model-responses/{args.model.replace('/', '_')}_responses-{args.num_samples}.csv" if args.num_samples is not None else f"disguising/model-responses/{args.model.replace('/', '_')}_responses.csv"
    
    llm = LLM(model=args.model, trust_remote_code=True, max_model_len=2048)
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    # batch size of 100
    responses = []
    for i in tqdm(range(0, len(df), 100), desc="Generating responses in batches"):
        responses.extend(llm.chat(messages=df["messages"].tolist()[i:i+100], sampling_params=sampling_params))
    responses = [response.outputs[0].text for response in responses]
    # responses = []
    # error_indices = []
    # for i, messages in enumerate(tqdm(df["messages"].tolist(), desc="Generating responses")):
    #     try:
    #         response = llm.chat(messages=[messages], sampling_params=sampling_params)
    #         responses.append(response[0].outputs[0].text)
    #     except Exception as e:
    #         print(f"Error at row {i}: {e}")
    #         error_indices.append(i)
    #         responses.append(None)
    # # Remove rows with errors
    # df = df.drop(index=error_indices).reset_index(drop=True)
    df["model_response"] = responses
    df = df.dropna(subset=["model_response"])
    print(f"Saved to {save_path}")
    df.to_csv(save_path, index=False)
    wandb.log({"data": wandb.Table(dataframe=df[["prompt", "messages", "model_response"]])})