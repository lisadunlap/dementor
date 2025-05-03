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

# Set constants
TOKEN_LIMIT = 8192 # Max token limit
LOAD_PATH = "disguising/comparisons/minimodel_responses.csv"
SAVE_PATH = "disguising/model-responses/internvl3-9b_responses.csv"


# Run event loop
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3-9B")
    parser.add_argument("--output_column", type=str, default="model_response")
    args = parser.parse_args()
    
    df = pd.read_csv(LOAD_PATH)
    def format_prompt(prompt):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "prompt"}
        ]
    
    df["messages"] = df["prompt"].apply(format_prompt)

    save_path = f"disguising/model-responses/{args.model.replace('/', '_')}_responses.csv"
    
    llm = LLM(model="google/gemma-3-1b-it", trust_remote_code=True, max_model_len=2048)
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.7,
        top_p=0.95,
    )
    responses = llm.chat(messages=df["messages"].tolist(), sampling_params=sampling_params)
    responses = [response.outputs[0].text for response in responses]
    df["model_response"] = responses
    print(f"Saved to {save_path}")
    df.to_csv(save_path, index=False)
