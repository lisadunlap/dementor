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
    parser.add_argument("--disguise_as", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_column", type=str, default="model_response")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples_per_disguise", type=int, default=3)
    args = parser.parse_args()

    disguise_file = f"disguising/model-responses/base/{args.disguise_as.replace('/', '_')}_responses-{args.num_samples}.csv" if args.num_samples is not None else f"disguising/model-responses/{args.disguise_as.replace('/', '_')}_responses.csv"
    disguise_df = pd.read_csv(disguise_file).dropna(subset=["model_response"])
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    disguise_df["response_tokens"] = disguise_df["model_response"].apply(lambda x: len(tokenizer.encode(x)))
    disguise_df = disguise_df[disguise_df["response_tokens"] <= 256]
    disguise_df = disguise_df.drop(columns=["response_tokens"])
    
    df = pd.read_csv(LOAD_PATH)

    wandb.init(project="disguising-generations", name=f"{args.model.replace('/', '_')}_as_{args.disguise_as.replace('/', '_')}", group="random_examples")
    wandb.config.update(args)

    # create samples to use for disguise
    disguise_df_sample = disguise_df.sample(n=args.num_samples_per_disguise, random_state=args.seed).reset_index(drop=True)

    def make_disguise_prompt(examples, prompt):
        """
        Create a prompt string for disguise, given a dataframe of examples and a new prompt.
        examples: pd.DataFrame with columns 'prompt' and 'model_response'
        prompt: str, the new question to answer
        """
        s = [
            "You are a helful AI assistant which should respond in the style similar to the examples below."
            "Your goal is to mimic the formatting, tone, level of detail, and phrasing – not to copy content exactly.\n\n"
            "## Examples:\n"
        ]
        for i, row in examples.iterrows():
            s.append(
                f"### Example {i + 1}:\n"
                f"prompt: {row['prompt']}\n"
                f"response: {row['model_response']}\n\n"
            )
        s.append("Here is the question to answer:\n")
        s.append(f"{prompt}")
        return "".join(s)
    
    df["disguise_prompt"] = df["prompt"].apply(lambda x: make_disguise_prompt(disguise_df_sample, x))
    df["prompt_tokens"] = df["disguise_prompt"].apply(lambda x: len(tokenizer.encode(x)))
    df = df[df["prompt_tokens"] <= 2048]
    df = df.drop(columns=["prompt_tokens"]) 

    if args.num_samples is not None:
        df = df.sample(n=min(args.num_samples, len(df)), random_state=42)

    def format_prompt(prompt):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    
    df["messages"] = df["disguise_prompt"].apply(format_prompt)

    save_path = f"disguising/model-responses/disguised/{args.model.replace('/', '_')}_responses-{args.num_samples}-{args.disguise_as.replace('/', '_')}.csv" if args.num_samples is not None else f"disguising/model-responses/disguised/{args.model.replace('/', '_')}_responses-{args.disguise_as.replace('/', '_')}.csv"
    
    llm = LLM(model=args.model, trust_remote_code=True, max_model_len=2048)
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    responses = []
    for i in tqdm(range(0, len(df), 100), desc="Generating responses in batches"):
        responses.extend(llm.chat(messages=df["messages"].tolist()[i:i+100], sampling_params=sampling_params))
    responses = [response.outputs[0].text for response in responses]
    df["model_response"] = [r for r in responses if r is not None]
    df = df[["prompt", "disguise_prompt", "messages", "model_response"]]
    print(f"Saved to {save_path}")
    df.to_csv(save_path, index=False)
    wandb.log({"data": wandb.Table(dataframe=df[["prompt", "disguise_prompt", "messages", "model_response"]])})