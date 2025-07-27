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
import re
import plotly.graph_objects as go
import plotly.express as px
import logging
import argparse
from typing import List
from litellm import completion, embedding
import numpy as np

from stylistic_analysis import compute_heuristics
from methods.get_method import get_method
from utils import get_token_count

# Constants
BATCH_SIZE = 100

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_prompt(prompt):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

def get_model_response_path(model: str, num_samples: int) -> str:
    # Convert slash format to underscore format for filenames
    return f"{model.replace('/', '_')}.csv"

def find_model_file(model_name):
    # Convert slash format to underscore format for filenames
    model_name = model_name.replace('/', '_')
    print(f"Looking for model: {model_name}")
    
    # Try base_500_all_models_2 first
    base_dir = os.path.join("/home/ethanliu/dementor", "disguising", "model-responses", "base_500_all_models_2")
    print(f"Checking directory: {base_dir}")
    model_file = os.path.join(base_dir, f"{model_name}.csv")
    print(f"Checking file: {model_file}")
    
    if os.path.exists(model_file):
        print(f"Found file at: {model_file}")
        return model_file
    
    # If not found, try base_500_all_models
    base_dir = os.path.join("/home/ethanliu/dementor", "disguising", "model-responses", "base_500_all_models")
    print(f"Checking directory: {base_dir}")
    model_file = os.path.join(base_dir, f"{model_name}.csv")
    print(f"Checking file: {model_file}")
    
    if os.path.exists(model_file):
        print(f"Found file at: {model_file}")
        return model_file
    
    raise FileNotFoundError(f"Model file for {model_name} not found in any base directory.")

def load_data(args, data_dir: str) -> pd.DataFrame:
    disguise_file = find_model_file(args.disguise_as)
    model_file = find_model_file(args.model)
    print(f"Loading disguise file: {disguise_file}")
    print(f"Loading model file: {model_file}")
    
    # Load and inspect files
    disguise_df = pd.read_csv(disguise_file)
    model_df = pd.read_csv(model_file)
    print(f"Disguise file columns: {list(disguise_df.columns)}")
    print(f"Model file columns: {list(model_df.columns)}")
    
    # Ensure consistent column names
    disguise_df = disguise_df.rename(columns={
        'prompt': 'prompt',
        'model_response': 'model_response_disguise',
        'response': 'model_response_disguise'
    })
    model_df = model_df.rename(columns={
        'prompt': 'prompt',
        'model_response': 'model_response_model',
        'response': 'model_response_model'
    })
    
    # Ensure prompt column exists and is clean
    if 'prompt' not in disguise_df.columns or 'prompt' not in model_df.columns:
        raise ValueError("Prompt column not found in one of the dataframes")
    
    # Clean prompt columns
    disguise_df['prompt'] = disguise_df['prompt'].str.strip()
    model_df['prompt'] = model_df['prompt'].str.strip()
    
    # Merge on prompt
    df = disguise_df.merge(model_df, on="prompt", how="inner", suffixes=("_disguise", "_model"))
    
    if len(df) == 0:
        print("No common prompts found between the two dataframes")
        print("Disguise file sample:")
        print(disguise_df.head(2))
        print("\nModel file sample:")
        print(model_df.head(2))
        raise ValueError("Merge resulted in empty DataFrame. Check if prompts match between files.")
    
    # Add model information
    df["target_model"] = args.disguise_as
    df["target_response"] = df["model_response_disguise"]
    df["target_response"] = df["target_response"].apply(remove_thinking_from_output)
    df["target_response_token_length"] = df["target_response"].apply(lambda x: get_token_count(x))
    df["source_model"] = args.model
    df["source_response"] = df["model_response_model"]
    df["source_response"] = df["source_response"].apply(remove_thinking_from_output)
    df["source_response_token_length"] = df["source_response"].apply(lambda x: get_token_count(x))
    
    # Select final columns
    df = df[["prompt", "target_model", "target_response", "target_response_token_length", "source_model", "source_response", "source_response_token_length"]]
    df = df.dropna(subset=["target_response", "source_response"])
    
    print(f"Final DataFrame shape: {df.shape}")
    return df

def generate_disguised_prompts(df: pd.DataFrame, method, max_tokens: int) -> pd.DataFrame:
    df["disguised_prompt"] = df["prompt"].apply(lambda x: method.forward(x))
    
    # Handle both standard format and Gemma format prompts
    def get_token_length(prompt_data):
        if isinstance(prompt_data, list):
            # Standard format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            if len(prompt_data) > 1 and "content" in prompt_data[1]:
                return get_token_count(prompt_data[1]["content"])
            # Gemma format: [{"role": "user", "content": "<start_of_turn>user\n..."}]
            elif len(prompt_data) == 1 and "content" in prompt_data[0]:
                return get_token_count(prompt_data[0]["content"])
        # If we can't determine the format, return 0 to filter it out
        return max_tokens + 1  # This will be filtered out
    
    df["disguised_prompt_token_length"] = df["disguised_prompt"].apply(get_token_length)
    df = df[df["disguised_prompt_token_length"] <= max_tokens]
    return df.drop(columns=["disguised_prompt_token_length"])

def generate_responses(df: pd.DataFrame, llm, sampling_params, batch_size: int, model_name: str) -> list:
    responses = []
    
    # Check if it's an OpenAI model (simple check for gpt models)
    if model_name.startswith("gpt-") or "gpt" in model_name.lower():
        # Use LiteLLM for OpenAI models
        for i in tqdm(range(0, len(df), batch_size), desc="Generating responses in batches"):
            messages_batch = df["disguised_prompt"].tolist()[i:i+batch_size]
            batch_responses = []
            for messages in messages_batch:
                try:
                    response = completion(
                        model=model_name,
                        messages=messages,
                        max_tokens=sampling_params.max_tokens,
                        temperature=sampling_params.temperature,
                        top_p=sampling_params.top_p,
                        caching=True,
                    )
                    batch_responses.append(response.choices[0].message.content)
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    batch_responses.append("")
            responses.extend(batch_responses)
    else:
        # Use VLLM for other models
        for i in tqdm(range(0, len(df), batch_size), desc="Generating responses in batches"):
            messages = df["disguised_prompt"].tolist()[i:i+batch_size]
            responses.extend(llm.chat(messages=messages, sampling_params=sampling_params))
        responses = [response.outputs[0].text for response in responses]
    
    return responses

# def clean_response(old_responses: List[str], llm, sampling_params, batch_size: int, model_name: str) -> List[str]:
#     prompt = "In the LLM output below, remove any text at the beginning which mentions generating a response or mimicing a style (e.g. 'Here is the response in the style of ...'). If there is no mention of this, return the original output. Do not alter the original output in any other way.\nLLM output: {response}"
#     messages = [format_prompt(prompt.format(response=response)) for response in old_responses]
#     responses = []
    
#     # Check if it's an OpenAI model
#     if model_name.startswith("gpt-") or "gpt" in model_name.lower():
#         # Use LiteLLM for OpenAI models
#         for i in tqdm(range(0, len(messages), batch_size), desc="Cleaning responses in batches"):
#             batch_messages = messages[i:i+batch_size]
#             batch_responses = []
#             for message in batch_messages:
#                 try:
#                     response = completion(
#                         model=model_name,
#                         messages=message,
#                         max_tokens=sampling_params.max_tokens,
#                         temperature=sampling_params.temperature,
#                         top_p=sampling_params.top_p
#                     )
#                     batch_responses.append(response.choices[0].message.content)
#                 except Exception as e:
#                     logging.error(f"Error cleaning response: {e}")
#                     batch_responses.append("")
#             responses.extend(batch_responses)
#     else:
#         # Use VLLM for other models
#         for i in tqdm(range(0, len(messages), batch_size), desc="Cleaning responses in batches"):
#             responses.extend(llm.chat(messages=messages[i:i+batch_size], sampling_params=sampling_params))
#         responses = [response.outputs[0].text for response in responses]
    
#     return responses

def plot_token_length_distribution(df: pd.DataFrame, out_path: str):
    # Calculate the overall min and max across all three distributions
    all_lengths = pd.concat([
        df["target_response_token_length"], 
        df["source_response_token_length"], 
        df["disguised_response_token_length"]
    ])
    min_length = all_lengths.min()
    max_length = all_lengths.max()
    
    # Create 20 evenly spaced bin edges
    bin_edges = np.linspace(min_length, max_length, 41)  # 21 edges for 20 bins
    
    fig = go.Figure()
    
    # Add histograms with consistent bins
    fig.add_trace(go.Histogram(
        x=df["target_response_token_length"], 
        name="Target",
        opacity=0.5,
        xbins=dict(start=min_length, end=max_length, size=(max_length-min_length)/40)
    ))
    fig.add_trace(go.Histogram(
        x=df["source_response_token_length"], 
        name="Source",
        opacity=0.5,
        xbins=dict(start=min_length, end=max_length, size=(max_length-min_length)/40)
    ))
    fig.add_trace(go.Histogram(
        x=df["disguised_response_token_length"], 
        name="Disguised",
        opacity=0.5,
        xbins=dict(start=min_length, end=max_length, size=(max_length-min_length)/40)
    ))
    
    # Update layout
    fig.update_layout(
        title="Token Length Distribution",
        xaxis_title="Token Length",
        yaxis_title="Frequency",
        barmode='overlay',  # Overlay histograms
        width=1000,
        height=500,
        showlegend=True
    )
    
    fig.write_image(out_path)

def plot_embedding_distance_distribution(distances_to_target: np.ndarray, distances_to_source: np.ndarray, out_path: str):
    """
    Plot histogram of embedding distances between disguised responses and target/source responses.
    
    Args:
        distances_to_target: Array of distances between disguised and target embeddings
        distances_to_source: Array of distances between disguised and source embeddings
        out_path: Path to save the plot
    """
    # Calculate the overall min and max across both distributions
    all_distances = np.concatenate([distances_to_target, distances_to_source])
    min_distance = all_distances.min()
    max_distance = all_distances.max()
    
    fig = go.Figure()
    
    # Add histograms with consistent bins
    fig.add_trace(go.Histogram(
        x=distances_to_target,
        name="Distance to Target",
        opacity=0.6,
        xbins=dict(start=min_distance, end=max_distance, size=(max_distance-min_distance)/40)
    ))
    fig.add_trace(go.Histogram(
        x=distances_to_source,
        name="Distance to Source", 
        opacity=0.6,
        xbins=dict(start=min_distance, end=max_distance, size=(max_distance-min_distance)/40)
    ))
    
    # Update layout
    fig.update_layout(
        title="Embedding Distance Distribution",
        xaxis_title="Cosine Distance",
        yaxis_title="Frequency",
        barmode='overlay',  # Overlay histograms
        width=1000,
        height=500,
        showlegend=True
    )
    
    fig.write_image(out_path)

def remove_thinking_from_output(output):
    if not isinstance(output, str):
        return ""
    # Remove content between <think> tags
    pattern = r'<think>.*?</think>'
    cleaned_output = re.sub(pattern, '', output, flags=re.DOTALL)
    # Remove any extra whitespace that might be left
    cleaned_output = re.sub(r'\n\s*\n', '\n\n', cleaned_output)
    return cleaned_output.strip()

def prep_wandb_table(df: pd.DataFrame) -> wandb.Table:
    # cast all columns to strings
    df = df.astype(str)
    return wandb.Table(dataframe=df)


# def get_model_embeddings(embeddings: List[str]) -> np.ndarray:
#     """
#     Get the embeddings for the source and target model's responses using litellm
#     """
#     embeddings_list = []
#     for row in tqdm(embeddings, desc="Getting embeddings"):
#         embedding_vec = embedding(
#             model="text-embedding-3-small",
#             input=row,
#             caching=True,
#         )
#         embedding_vec = embedding_vec["data"][0]["embedding"]
#         embeddings_list.append(embedding_vec)
#     return np.array(embeddings_list)

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3-8B")
    parser.add_argument("--disguise_as", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_dir", type=str, default="disguising/model-responses/base_500_all_models")
    parser.add_argument("--output_dir", type=str, default="disguising/model-responses/disguised")
    parser.add_argument("--method", type=str, default="random_sample_3_examples")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--enable_thinking", type=bool, default=False)
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--max_prompt_tokens", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    df = load_data(args, data_dir=args.data_dir)
    wandb.init(project="dementor-methods", name=f"{args.model.replace('/', '_')}_disguised-{args.disguise_as.replace('/', '_')}_responses-{len(df)}", group=args.method)
    wandb.config.update(args)

    results_folder = f"{args.output_dir}/{args.method}" if not args.test else f"{args.output_dir}/test"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    out_csv = f"{results_folder}/{args.model.replace('/', '_')}_disguised-{args.disguise_as.replace('/', '_')}.csv"

    if os.path.exists(out_csv):
        logging.info(f"{out_csv} exists, running heuristics and plots...")
        df = pd.read_csv(out_csv)
    else:
        method = get_method(args.method, args.model, args.disguise_as, disguise_df=df)
        df = generate_disguised_prompts(df, method, args.max_prompt_tokens)
        if args.test:
            df = df.head(10)
        print(df.head())

    # Only initialize vLLM if not using OpenAI models
    llm = None
    if not (args.model.startswith("gpt-") or "gpt" in args.model.lower()):
        model_name = args.model.replace('_', '/')

    # vLLM handles Gemma chat templates automatically
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    sampling_params = SamplingParams(max_tokens=args.max_prompt_tokens, temperature=args.temperature, top_p=args.top_p)
    df["disguised_response_raw"] = generate_responses(df, llm, sampling_params, BATCH_SIZE, args.model)
    df["disguised_response"] = df["disguised_response_raw"].apply(remove_thinking_from_output)

    # Commented out LLM-based cleaning
    # df["disguised_response"] = clean_response(df["disguised_response_raw"], llm, sampling_params, BATCH_SIZE, args.model)
    df["disguised_response_token_length"] = df["disguised_response"].apply(get_token_count)
    df["model"] = f"{args.model.replace('/', '_')}_disguised-{args.disguise_as.replace('/', '_')}"
    df["source_model"] = args.model
    df["target_model"] = args.disguise_as
    df["method"] = args.method

    results_folder = f"{args.output_dir}/{args.method}" if not args.test else f"{args.output_dir}/test"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    out_csv = f"{results_folder}/{args.model.replace('/', '_')}_disguised-{args.disguise_as.replace('/', '_')}.csv"
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved disguised responses to {out_csv}")

    wandb.log({"data": prep_wandb_table(df)})
    wandb.summary["average_disguised_response_token_length"] = df["disguised_response_token_length"].mean()

    out_png = out_csv.replace('.csv', '.png')
    plot_token_length_distribution(df, out_png)
    wandb.log({"disguised_response_token_length_distribution": wandb.Image(out_png)})
    logging.info(f"Saved distribution plot to {out_png}")

    # get average normalized difference in length
    length_diff = [(len(row["disguised_response"]) - len(row["target_response"])) / max(len(row["disguised_response"]), len(row["target_response"])) for _, row in df.iterrows()]
    wandb.summary["length_diff"] = sum(length_diff) / len(length_diff)

    # get embeddings
    # embeddings = get_model_embeddings(df["disguised_response"].tolist())
    # embeddings_target = get_model_embeddings(df["target_response"].tolist())
    # embeddings_source = get_model_embeddings(df["source_response"].tolist())
    # get pairwise distances between embeddings and target embeddings
    # distances_disguise_target = np.linalg.norm(embeddings - embeddings_target, axis=1)
    # wandb.summary["average_distance_disguise_target"] = distances_disguise_target.mean()
    # # get pairwise distances between embeddings and source embeddings
    # distances_source_target = np.linalg.norm(embeddings_source - embeddings_target, axis=1)
    # wandb.summary["average_distance_source_target"] = distances_source_target.mean()
    # wandb.summary["diff_avg_embedding_distance"] = (distances_source_target.mean() - distances_disguise_target.mean()) / (distances_disguise_target.mean() + distances_source_target.mean())

    # Compute heuristics
    heuristic_table = compute_heuristics(df["disguised_response"].tolist(), df["target_response"].tolist())
    heuristic_file= os.path.join(args.output_dir, "heuristic_table.csv")
    heuristic_table.to_csv(heuristic_file, index=False)
    wandb.log({"style_heuristics": wandb.Table(dataframe=heuristic_table)})
    wandb.summary["heuristic_avg_score"] = heuristic_table["match"].mean()
    for i, row in heuristic_table.iterrows():
        wandb.summary[row['style_function']] = row["match"]

    # Compute heuristics comparing source and target responses
    heuristic_table_target_source = compute_heuristics(df["target_response"].tolist(), df["source_response"].tolist())
    heuristic_file= os.path.join(args.output_dir, "heuristic_table_target_source.csv")
    heuristic_table_target_source.to_csv(heuristic_file, index=False)
    wandb.log({"style_heuristics_target_source": wandb.Table(dataframe=heuristic_table_target_source)})
    wandb.summary["heuristic_avg_score_source_target"] = heuristic_table_target_source["match"].mean()
    for i, row in heuristic_table_target_source.iterrows():
        wandb.summary[f"source_target_{row['style_function']}"] = row["match"]
    
    wandb.summary["heuristic_diff"] = heuristic_table["match"].mean() - heuristic_table_target_source["match"].mean()
    wandb.summary["heuristic_diff_normalized"] = (heuristic_table["match"].mean() - heuristic_table_target_source["match"].mean()) / heuristic_table_target_source["match"].mean()
    print(f"Heuristic diff: {wandb.summary['heuristic_diff']}")
    print(f"Heuristic diff normalized: {wandb.summary['heuristic_diff_normalized']}")

    # print("Embedding distances:")
    # print(f"Average distance to target: {distances_disguise_target.mean()}")
    # print(f"Average distance to source: {distances_source_target.mean()}")
    # print(f"source to target - disguise / (source to target + disguise): {(distances_source_target.mean() - distances_disguise_target.mean()) / (distances_source_target.mean() + distances_disguise_target.mean())}")

    # # Plot embedding distance distribution
    # plot_embedding_distance_distribution(distances_disguise_target, distances_source_target, os.path.join(results_folder, f"{args.model.replace('/', '_')}_disguised-{args.disguise_as.replace('/', '_')}_embedding_distance_distribution.png"))
    # wandb.log({"embedding_distance_distribution": wandb.Image(os.path.join(results_folder, f"{args.model.replace('/', '_')}_disguised-{args.disguise_as.replace('/', '_')}_embedding_distance_distribution.png"))})
    # logging.info(f"Saved embedding distance distribution plot to {os.path.join(results_folder, f'{args.model.replace("/", "_")}_disguised-{args.disguise_as.replace("/", "_")}_embedding_distance_distribution.png')}")

if __name__ == "__main__":
    main()