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
import wandb
import matplotlib.pyplot as plt
import logging
import argparse
from typing import List

from methods.get_method import get_method
from utils import get_token_count

# Constants
BATCH_SIZE = 100
MAX_PROMPT_TOKENS = 2048
MAX_MODEL_LEN = 4096

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_prompt(prompt):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

def get_model_response_path(model: str, num_samples = None) -> str:
    base = model.replace('/', '_')
    if num_samples is not None:
        return f"disguising/model-responses/base/{base}_responses-{num_samples}.csv"
    return f"disguising/model-responses/{base}_responses.csv"

def load_data(args) -> pd.DataFrame:
    disguise_df = pd.read_csv(get_model_response_path(args.disguise_as, args.num_samples))
    model_df = pd.read_csv(get_model_response_path(args.model, args.num_samples))
    df = disguise_df.merge(model_df, on="prompt", how="inner", suffixes=("_disguise", "_model"))
    df["target_model"] = args.disguise_as
    df["target_response"] = df["model_response_disguise"]
    df["target_response_token_length"] = df["target_response"].apply(lambda x: get_token_count(x))
    df["source_model"] = args.model
    df["source_response"] = df["model_response_model"]
    df["source_response_token_length"] = df["source_response"].apply(lambda x: get_token_count(x))
    df = df[["prompt", "target_model", "target_response", "target_response_token_length", "source_model", "source_response", "source_response_token_length"]]
    df = df.dropna(subset=["target_response", "source_response"])
    return df

def generate_disguised_prompts(df: pd.DataFrame, method, max_tokens: int) -> pd.DataFrame:
    df["disguised_prompt"] = df["prompt"].apply(lambda x: method.forward(x))
    df["disguised_prompt_token_length"] = df["disguised_prompt"].apply(get_token_count)
    df = df[df["disguised_prompt_token_length"] <= max_tokens]
    return df.drop(columns=["disguised_prompt_token_length"])

def generate_responses(df: pd.DataFrame, llm, sampling_params, batch_size: int) -> list:
    responses = []
    for i in tqdm(range(0, len(df), batch_size), desc="Generating responses in batches"):
        messages = [format_prompt(prompt) for prompt in df["disguised_prompt"].tolist()[i:i+batch_size]]
        responses.extend(llm.chat(messages=messages, sampling_params=sampling_params))
    return [response.outputs[0].text for response in responses]

def clean_response(old_responses: List[str], llm, sampling_params, batch_size: int) -> List[str]:
    prompt = "In the LLM output below, remove any text at the beginning which mentions generating a response or mimicing a style (e.g. 'Here is the response in the style of ...'). If there is no mention of this, return the original output. Do not alter the original output in any other way.\nLLM output: {response}"
    messages = [format_prompt(prompt.format(response=response)) for response in old_responses]
    responses = []
    for i in tqdm(range(0, len(messages), batch_size), desc="Cleaning responses in batches"):
        responses.extend(llm.chat(messages=messages[i:i+batch_size], sampling_params=sampling_params))
    return [response.outputs[0].text for response in responses]

def plot_token_length_distribution(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(10, 5))
    plt.hist(df["target_response_token_length"], bins=20, alpha=0.5, label="Target")
    plt.hist(df["source_response_token_length"], bins=20, alpha=0.5, label="Source")
    plt.hist(df["disguised_response_token_length"], bins=20, alpha=0.5, label="Disguised")
    plt.legend()
    plt.savefig(out_path)
    plt.close()

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3-9B")
    parser.add_argument("--disguise_as", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--method", type=str, default="random_sample_3_examples")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    wandb.init(project="disguising", name=f"{args.model.replace('/', '_')}_disguised-{args.disguise_as.replace('/', '_')}_responses-{args.num_samples}", group=args.method)
    wandb.config.update(args)

    method = get_method(args.method, args.model, args.disguise_as, num_samples=args.num_samples)
    df = load_data(args)
    df = generate_disguised_prompts(df, method, MAX_PROMPT_TOKENS // 2)
    if args.test:
        df = df.head(10)

    llm = LLM(model=args.model, trust_remote_code=True, max_model_len=MAX_MODEL_LEN)
    sampling_params = SamplingParams(max_tokens=MAX_MODEL_LEN, temperature=args.temperature, top_p=args.top_p)
    df["disguised_response"] = generate_responses(df, llm, sampling_params, BATCH_SIZE)
    # df["disguised_response"] = clean_response(df["disguised_response_raw"], llm, sampling_params, BATCH_SIZE)
    df["disguised_response_token_length"] = df["disguised_response"].apply(get_token_count)

    results_folder = f"disguising/model-responses/disguised/{args.method}/{args.model.replace('/', '_')}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    out_csv = f"{results_folder}/{args.model.replace('/', '_')}_disguised-{args.disguise_as.replace('/', '_')}_responses-{args.num_samples}.csv"
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved disguised responses to {out_csv}")

    wandb.log({"data": wandb.Table(dataframe=df)})
    wandb.summary["average_disguised_response_token_length"] = df["disguised_response_token_length"].mean()

    out_png = out_csv.replace('.csv', '.png')
    plot_token_length_distribution(df, out_png)
    wandb.log({"disguised_response_token_length_distribution": wandb.Image(out_png)})
    logging.info(f"Saved distribution plot to {out_png}")

if __name__ == "__main__":
    main()