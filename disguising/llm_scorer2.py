import argparse
import wandb
import pandas as pd
import re
import os
import signal
import sys

from stylistic_analysis import compute_heuristics
from utils import get_token_count

def remove_thinking_from_output(output):
    # Remove content between <think> tags
    pattern = r'<think>.*?</think>'
    cleaned_output = re.sub(pattern, '', output, flags=re.DOTALL)
    # Remove any extra whitespace that might be left
    cleaned_output = re.sub(r'\n\s*\n', '\n\n', cleaned_output)
    return cleaned_output.strip()

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\nGracefully shutting down...")
        wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_a", type=str, required=True)
    parser.add_argument("--input_file_b", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="disguising-method-scoring")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Extract model names from file paths for wandb naming
    model_a = os.path.basename(args.input_file_a).replace('.csv', '')
    model_b = os.path.basename(args.input_file_b).replace('.csv', '')
    
    wandb.init(project=args.wandb_project, name=f"{model_a}_vs_{model_b}", group="base_comparison")
    wandb.config.update(args)

    try:
        # Load both CSV files
        df_a = pd.read_csv(args.input_file_a)
        df_b = pd.read_csv(args.input_file_b)
        
        print(f"Loaded {len(df_a)} rows from {args.input_file_a}")
        print(f"Loaded {len(df_b)} rows from {args.input_file_b}")
        
        # Merge on prompt column
        df = df_a.merge(df_b, on="prompt", how="inner", suffixes=("_a", "_b"))
        print(f"After merging: {len(df)} rows")
        
        # Rename columns for clarity
        df = df.rename(columns={
            "model_response_a": "response_a",
            "model_response_b": "response_b"
        })
        
        # Remove any thinking from the responses
        df["response_a"] = df["response_a"].apply(remove_thinking_from_output)
        df["response_b"] = df["response_b"].apply(remove_thinking_from_output)
        
        # Calculate token lengths if not present
        if "response_a_token_length" not in df.columns:
            df["response_a_token_length"] = df["response_a"].apply(get_token_count)
        if "response_b_token_length" not in df.columns:
            df["response_b_token_length"] = df["response_b"].apply(get_token_count)

        # get average normalized difference in length
        length_diff = [(len(row["response_a"]) - len(row["response_b"])) / max(len(row["response_a"]), len(row["response_b"])) for _, row in df.iterrows()]
        wandb.summary["length_diff"] = sum(length_diff) / len(length_diff)

        # Compute heuristics comparing responses from both models
        heuristic_table = compute_heuristics(df["response_a"].tolist(), df["response_b"].tolist())
        heuristic_file = os.path.join(output_dir, "heuristic_table.csv")
        heuristic_table.to_csv(heuristic_file, index=False)
        wandb.log({"style_heuristics": wandb.Table(dataframe=heuristic_table)})
        wandb.summary["heuristic_avg_score"] = heuristic_table["match"].mean()
        
        # Log individual heuristic scores
        for i, row in heuristic_table.iterrows():
            wandb.summary[row['style_function']] = row["match"]

        # Save response table
        wandb.log({"response_table": wandb.Table(data=df)})
        
        # Save the merged data
        df.to_csv(args.output_file, index=False)
        print(f"Saved comparison results to {args.output_file}")
        print(f"Saved heuristic table to {heuristic_file}")
        
        print(f"Heuristic average score: {heuristic_table['match'].mean():.3f}")
        print(f"Length difference: {wandb.summary['length_diff']:.3f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        wandb.finish()
        sys.exit(1)
    finally:
        wandb.finish()
        sys.exit(0) 