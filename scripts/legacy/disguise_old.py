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

def get_model_response_path(model: str) -> str:
    base = model.replace('/', '_')
    return f"{base}.csv"

def load_data(args, data_dir: str) -> pd.DataFrame:
    # Resolve CSV path flexibly to support GSM8K filenames
    def resolve_csv_path(directory: str, model_name: str) -> str:
        expected = os.path.join(directory, get_model_response_path(model_name))
        if os.path.exists(expected):
            return expected
        # Fallbacks: try common GSM8K-style filenames
        base = model_name.replace('/', '_')
        candidates = []
        try:
            for fname in os.listdir(directory):
                if not fname.endswith('.csv'):
                    continue
                # exact base match anywhere or gsm8k variants
                if fname == f"{base}_gsm8k_test_500.csv" or fname.startswith(base):
                    candidates.append(os.path.join(directory, fname))
        except FileNotFoundError:
            pass
        if candidates:
            # Prefer gsm8k_test_500 if present
            for c in candidates:
                if c.endswith('_gsm8k_test_500.csv'):
                    return c
            return candidates[0]
        raise FileNotFoundError(f"Could not find CSV for model '{model_name}' in {directory}")

    disguise_path = resolve_csv_path(data_dir, args.disguise_as)
    model_path = resolve_csv_path(data_dir, args.model)
    disguise_df = pd.read_csv(disguise_path)
    model_df = pd.read_csv(model_path)
    disguise_df['prompt'] = disguise_df['prompt'].str.strip()
    model_df['prompt'] = model_df['prompt'].str.strip()
    df = disguise_df.merge(model_df, on="prompt", how="inner", suffixes=("_disguise", "_model"))
    df["target_model"] = args.disguise_as
    df["target_response"] = df["model_response_disguise"]
    df["target_response"] = df["target_response"].apply(remove_thinking_from_output)
    df["target_response_token_length"] = df["target_response"].apply(lambda x: get_token_count(x))
    df["source_model"] = args.model
    df["source_response"] = df["model_response_model"]
    df["source_response"] = df["source_response"].apply(remove_thinking_from_output)
    df["source_response_token_length"] = df["source_response"].apply(lambda x: get_token_count(x))
    df = df[["prompt", "target_model", "target_response", "target_response_token_length", "source_model", "source_response", "source_response_token_length"]]
    df = df.dropna(subset=["target_response", "source_response"])
    return df

def generate_disguised_prompts(df: pd.DataFrame, method, max_tokens: int) -> pd.DataFrame:
    df["disguised_prompt"] = df["prompt"].apply(lambda x: method.forward(x))
    df["disguised_prompt_token_length"] = df["disguised_prompt"].apply(lambda x: get_token_count(x[1]["content"]))
    df = df[df["disguised_prompt_token_length"] <= max_tokens]
    return df.drop(columns=["disguised_prompt_token_length"])

def generate_responses(df: pd.DataFrame, llm, sampling_params, batch_size: int, model_name: str) -> list:
    responses = []
    
    # Use LiteLLM path if no local llm is provided (i.e., using OpenAI-compatible server)
    if llm is None:
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
                        custom_llm_provider="openai",
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

def clean_response(old_responses: List[str], llm, sampling_params, batch_size: int, model_name: str) -> List[str]:
    prompt = "In the LLM output below, remove any text at the beginning which mentions generating a response or mimicing a style (e.g. 'Here is the response in the style of ...'). If there is no mention of this, return the original output. Do not alter the original output in any other way.\nLLM output: {response}"
    messages = [format_prompt(prompt.format(response=response)) for response in old_responses]
    responses = []
    
    # Check if it's an OpenAI model
    if model_name.startswith("gpt-") or "gpt" in model_name.lower():
        # Use LiteLLM for OpenAI models
        for i in tqdm(range(0, len(messages), batch_size), desc="Cleaning responses in batches"):
            batch_messages = messages[i:i+batch_size]
            batch_responses = []
            for message in batch_messages:
                try:
                    response = completion(
                        model=model_name,
                        messages=message,
                        max_tokens=sampling_params.max_tokens,
                        temperature=sampling_params.temperature,
                        top_p=sampling_params.top_p
                    )
                    batch_responses.append(response.choices[0].message.content)
                except Exception as e:
                    logging.error(f"Error cleaning response: {e}")
                    batch_responses.append("")
            responses.extend(batch_responses)
    else:
        # Use VLLM for other models
        for i in tqdm(range(0, len(messages), batch_size), desc="Cleaning responses in batches"):
            responses.extend(llm.chat(messages=messages[i:i+batch_size], sampling_params=sampling_params))
        responses = [response.outputs[0].text for response in responses]
    
    return responses

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
    
    try:
        fig.write_image(out_path)
    except Exception as e:
        # Fallback to HTML if static export is unavailable (e.g., Chrome not installed for kaleido)
        html_path = out_path.replace('.png', '.html')
        logging.warning(f"Static image export failed ({e}). Writing HTML to {html_path} instead.")
        fig.write_html(html_path, include_plotlyjs='cdn')

def plot_embedding_distance_distribution(distances_to_target: np.ndarray, distances_to_source: np.ndarray, out_path: str):
"""  # moved to scripts/legacy
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
    
    try:
        fig.write_image(out_path)
    except Exception as e:
        # Fallback to HTML if static export is unavailable
        html_path = out_path.replace('.png', '.html')
        logging.warning(f"Static image export failed ({e}). Writing HTML to {html_path} instead.")
        fig.write_html(html_path, include_plotlyjs='cdn')

def remove_thinking_from_output(output):
    # Handle NaN / non-string values gracefully
    if output is None:
        return ""
    try:
        # Remove content between <think> tags
        pattern = r'<think>.*?</think>'
        text = output if isinstance(output, str) else str(output)
        cleaned_output = re.sub(pattern, '', text, flags=re.DOTALL)
        # Remove any extra whitespace that might be left
        cleaned_output = re.sub(r'\n\s*\n', '\n\n', cleaned_output)
        return cleaned_output.strip()
    except Exception:
        # Fallback to string conversion
        return str(output)

def prep_wandb_table(df: pd.DataFrame) -> wandb.Table:
    # cast all columns to strings
    df = df.astype(str)
    return wandb.Table(dataframe=df)

def truncate_to_token_limit(text: str, encoding: tiktoken.Encoding, max_tokens: int = 8192) -> str:
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

def get_model_embeddings(embeddings: List[str], encoding: tiktoken.Encoding, max_tokens: int = 8192) -> np.ndarray:
    """
    Get the embeddings for the source and target model's responses using litellm
    """
    embeddings_list = []
    for row in tqdm(embeddings, desc="Getting embeddings"):
        safe_input = truncate_to_token_limit(row, encoding, max_tokens)
        embedding_vec = embedding(
            model="text-embedding-3-small",
            input=safe_input,
            caching=True,
        )
        embedding_vec = embedding_vec["data"][0]["embedding"]
        embeddings_list.append(embedding_vec)
    return np.array(embeddings_list)

def main():
    setup_logging()
    # Load environment variables from .env (e.g., OPENAI_API_KEY)
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3-8B")
    parser.add_argument("--disguise_as", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_dir", type=str, default="disguising/model-responses/base_500_all_models", help="Data directory. Use 'gsm8k' for GSM8K dataset.")
    parser.add_argument("--output_dir", type=str, default="disguising/model-responses/disguised")
    parser.add_argument("--method", type=str, default="random_sample_3_examples")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--enable_thinking", type=bool, default=False)
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--max_prompt_tokens", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    # Optional: use an OpenAI-compatible server (e.g., your running vLLM server)
    parser.add_argument("--use_openai_server", action="store_true")
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
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
        # Methods expect a DataFrame with a 'model_response' column for examples.
        # Map from our merged 'target_response' to the expected name.
        disguise_examples_df = df.rename(columns={"target_response": "model_response"})
        method = get_method(args.method, args.model, args.disguise_as, disguise_df=disguise_examples_df)
        df = generate_disguised_prompts(df, method, args.max_prompt_tokens)
        if args.test:
            df = df.head(10)
        print(df.head())

        if not args.skip_generation:
            # Decide how to generate responses
            llm = None
            use_openai_mode = args.model.startswith("gpt-") or "gpt" in args.model.lower() or args.use_openai_server
            if not use_openai_mode:
                # Local vLLM instance
                llm = LLM(model=args.model, trust_remote_code=True, max_model_len=args.max_tokens, tensor_parallel_size=args.tensor_parallel_size)
                
            # Configure LiteLLM/OpenAI client if using remote server
            if use_openai_mode and args.use_openai_server:
                # Configure via environment variables for litellm
                if args.api_base:
                    os.environ["OPENAI_API_BASE"] = args.api_base
                if args.api_key:
                    os.environ["OPENAI_API_KEY"] = args.api_key

            sampling_params = SamplingParams(max_tokens=args.max_prompt_tokens, temperature=args.temperature, top_p=args.top_p)
            df["disguised_response_raw"] = generate_responses(df, llm, sampling_params, BATCH_SIZE, args.model)
            df["disguised_response"] = df["disguised_response_raw"].apply(remove_thinking_from_output)
        else:
            # Skip generation (for quick sanity checks of prompt formation)
            df["disguised_response"] = ""

        # df["disguised_response"] = clean_response(df["disguised_response_raw"], llm, sampling_params, BATCH_SIZE, args.model)
        df["disguised_response_token_length"] = df["disguised_response"].apply(get_token_count)
        df["model"] = f"{args.model.replace('/', '_')}_disguised-{args.disguise_as.replace('/', '_')}"
        df["source_model"] = args.model
        df["target_model"] = args.disguise_as
        df["method"] = args.method
        
        df.to_csv(out_csv, index=False)
        logging.info(f"Saved disguised responses to {out_csv}")

    # Normalize text columns to avoid NaNs causing errors downstream
    for col in ["disguised_response", "target_response", "source_response"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    wandb.log({"data": prep_wandb_table(df)})
    wandb.summary["average_disguised_response_token_length"] = df["disguised_response_token_length"].mean()

    out_png = out_csv.replace('.csv', '.png')
    plot_token_length_distribution(df, out_png)
    try:
        if os.path.exists(out_png):
            wandb.log({"disguised_response_token_length_distribution": wandb.Image(out_png)})
            logging.info(f"Saved distribution plot to {out_png}")
        else:
            out_html = out_png.replace('.png', '.html')
            logging.info(f"Saved distribution plot to {out_html}")
    except Exception as e:
        logging.warning(f"Failed to log token length distribution image: {e}")

    # get average normalized difference in length
    def safe_len(s: str) -> int:
        try:
            return len(s) if isinstance(s, str) else len(str(s))
        except Exception:
            return 0
    length_diff = [(safe_len(row["disguised_response"]) - safe_len(row["target_response"])) / max(safe_len(row["disguised_response"]), safe_len(row["target_response"])) if max(safe_len(row["disguised_response"]), safe_len(row["target_response"])) > 0 else 0 for _, row in df.iterrows()]
    wandb.summary["length_diff"] = sum(length_diff) / len(length_diff)

    # get embeddings (skip if skipping generation)
    distances_disguise_target = None
    distances_source_target = None
    if not args.skip_generation:
        encoding = tiktoken.encoding_for_model("text-embedding-3-small")
        MAX_TOKENS = 8192
        embeddings = get_model_embeddings(df["disguised_response"].tolist(), encoding, MAX_TOKENS)
        embeddings_target = get_model_embeddings(df["target_response"].tolist(), encoding, MAX_TOKENS)
        embeddings_source = get_model_embeddings(df["source_response"].tolist(), encoding, MAX_TOKENS)
        # get pairwise distances between embeddings and target embeddings
        distances_disguise_target = np.linalg.norm(embeddings - embeddings_target, axis=1)
        wandb.summary["average_distance_disguise_target"] = distances_disguise_target.mean()
        # get pairwise distances between embeddings and source embeddings
        distances_source_target = np.linalg.norm(embeddings_source - embeddings_target, axis=1)
        wandb.summary["average_distance_source_target"] = distances_source_target.mean()
        wandb.summary["diff_avg_embedding_distance"] = (distances_source_target.mean() - distances_disguise_target.mean()) / (distances_disguise_target.mean() + distances_source_target.mean())

    # Compute heuristics
    heuristic_table = compute_heuristics(df["disguised_response"].tolist(), df["target_response"].tolist())
    # Save to the new hierarchical_math_disguise scores structure
    # Normalize naming to match stylistic_clustering:
    #   disguising/scores/{method}/source_<family>/<model>/target_<family>/<model>
    def split_family_model(name: str):
        if '/' in name:
            family, model = name.split('/', 1)
        else:
            # Derive a family heuristically from the prefix before first '-'
            parts = name.split('-', 1)
            family = parts[0]
            model = name
        return family, model

    source_family, source_model_name = split_family_model(args.model)
    target_family, target_model_name = split_family_model(args.disguise_as)
    scores_dir = (
        f"disguising/scores/{args.method}/"
        f"source_{source_family}/{source_model_name}/"
        f"target_{target_family}/{target_model_name}"
    )
    os.makedirs(scores_dir, exist_ok=True)
    heuristic_file = os.path.join(scores_dir, "heuristic_table.csv")
    heuristic_table.to_csv(heuristic_file, index=False)
    wandb.log({"style_heuristics": wandb.Table(dataframe=heuristic_table)})
    wandb.summary["heuristic_avg_score"] = heuristic_table["match"].mean()
    for i, row in heuristic_table.iterrows():
        wandb.summary[row['style_function']] = row["match"]

    # Compute heuristics comparing source and target responses
    heuristic_table_target_source = compute_heuristics(df["target_response"].tolist(), df["source_response"].tolist())
    heuristic_file_target_source = os.path.join(scores_dir, "heuristic_table_target_source.csv")
    heuristic_table_target_source.to_csv(heuristic_file_target_source, index=False)
    wandb.log({"style_heuristics_target_source": wandb.Table(dataframe=heuristic_table_target_source)})
    wandb.summary["heuristic_avg_score_source_target"] = heuristic_table_target_source["match"].mean()
    for i, row in heuristic_table_target_source.iterrows():
        wandb.summary[f"source_target_{row['style_function']}"] = row["match"]
    
    wandb.summary["heuristic_diff"] = heuristic_table["match"].mean() - heuristic_table_target_source["match"].mean()
    wandb.summary["heuristic_diff_normalized"] = (heuristic_table["match"].mean() - heuristic_table_target_source["match"].mean()) / heuristic_table_target_source["match"].mean()
    print(f"Heuristic diff: {wandb.summary['heuristic_diff']}")
    print(f"Heuristic diff normalized: {wandb.summary['heuristic_diff_normalized']}")

    if distances_disguise_target is not None and distances_source_target is not None:
        print("Embedding distances:")
        print(f"Average distance to target: {distances_disguise_target.mean()}")
        print(f"Average distance to source: {distances_source_target.mean()}")
        print(f"source to target - disguise / (source to target + disguise): {(distances_source_target.mean() - distances_disguise_target.mean()) / (distances_source_target.mean() + distances_disguise_target.mean())}")

    # Plot embedding distance distribution
    plot_path = os.path.join(scores_dir, f"{args.model.replace('/', '_')}_disguised-{args.disguise_as.replace('/', '_')}_embedding_distance_distribution.png")
    if distances_disguise_target is not None and distances_source_target is not None:
        plot_embedding_distance_distribution(distances_disguise_target, distances_source_target, plot_path)
        try:
            if os.path.exists(plot_path):
                wandb.log({"embedding_distance_distribution": wandb.Image(plot_path)})
                logging.info(f"Saved embedding distance distribution plot to {plot_path}")
            else:
                plot_html = plot_path.replace('.png', '.html')
                logging.info(f"Saved embedding distance distribution plot to {plot_html}")
        except Exception as e:
            logging.warning(f"Failed to log embedding distance distribution image: {e}")

if __name__ == "__main__":
    main()

    
