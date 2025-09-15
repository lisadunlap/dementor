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
from tqdm import tqdm
import wandb
import re
from utils import get_token_count
from litellm import completion

# Run event loop
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3-9B")
    parser.add_argument("--output_column", type=str, default="model_response")
    parser.add_argument("--output_dir", type=str, default="data/model-responses/chatbot_arena/500")
    parser.add_argument("--dataset", type=str, default="data/datasets/chatbot_arena/chatbot_arena_prompts.txt", help="Path to a text file with one prompt per line")
    parser.add_argument("--dataset_csv", type=str, default=None, help="Optional: path to CSV with a 'prompt' column (or specify --dataset_csv_prompt_col)")
    parser.add_argument("--dataset_csv_prompt_col", type=str, default="prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=8000)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    # Use an OpenAI-compatible server (e.g., vLLM API) instead of loading a local model
    parser.add_argument("--use_openai_server", action="store_true")
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()

    # Load prompts from CSV or text file
    if args.dataset_csv is not None:
        df_in = pd.read_csv(args.dataset_csv)
        if args.dataset_csv_prompt_col not in df_in.columns:
            raise ValueError(f"Column '{args.dataset_csv_prompt_col}' not found in {args.dataset_csv}")
        prompts = df_in[args.dataset_csv_prompt_col].astype(str).tolist()
    else:
        with open(args.dataset, "r") as f:
            prompts = f.readlines()

    # remove any rows where the prompt is more than 2048 tokens
    prompts_tokens = [get_token_count(prompt) for prompt in prompts]
    prompts = [prompt for prompt, token_length in zip(prompts, prompts_tokens) if token_length <= 1024]

    if args.num_samples is not None:
        prompts = prompts[:args.num_samples]

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "offline"
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

    # Configure OpenAI-compatible client if requested
    if args.use_openai_server:
        if args.api_base:
            os.environ["OPENAI_API_BASE"] = args.api_base
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key

    responses = []
    if args.use_openai_server:
        # Use LiteLLM to hit remote server
        for i in tqdm(range(0, len(messages), args.batch_size), desc="Generating responses in batches"):
            batch = messages[i:i+args.batch_size]
            for message in batch:
                try:
                    r = completion(
                        model=args.model,
                        messages=message,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        custom_llm_provider="openai",
                        caching=True,
                    )
                    responses.append(r.choices[0].message.content)
                except Exception as e:
                    print(f"Error generating response: {e}")
                    responses.append("")
    else:
        # Load model locally with vLLM
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, trust_remote_code=True, max_model_len=args.max_model_len, tensor_parallel_size=args.tensor_parallel_size)
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        # batch size of 100
        for i in tqdm(range(0, len(messages), args.batch_size), desc="Generating responses in batches"):
            responses.extend(llm.chat(messages=messages[i:i+args.batch_size], sampling_params=sampling_params))
        responses = [response.outputs[0].text for response in responses]

    # make output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.DataFrame({"prompt": prompts, "messages": messages, "model_response": responses, "model": args.model.replace("/", "_")})
    df["response_token_length"] = df["model_response"].apply(lambda x: get_token_count(x))
    df = df.dropna(subset=["model_response"])
    print(f"Generated {len(df)} responses")
    print(f"Saved to {save_path}")
    df.to_csv(save_path, index=False)
    wandb.log({"data": wandb.Table(dataframe=df)})
    wandb.summary["average_response_token_length"] = df["response_token_length"].mean()
