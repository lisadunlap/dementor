#!/usr/bin/env python3

import pandas as pd  # moved to scripts/legacy
import os
from stylistic_analysis import compute_heuristics

def main():
    # Load the disguised responses
    disguised_file = "disguising/model-responses/disguised/hierarchical_math_disguise/microsoft_Phi-4-mini-instruct_disguised-gpt-5.csv"
    base_file = "disguising/model-responses/gsm8k/microsoft_Phi-4-mini-instruct.csv"
    
    print("Loading disguised responses...")
    df_disguised = pd.read_csv(disguised_file)
    print(f"Loaded {len(df_disguised)} disguised responses")
    
    print("Loading base responses...")
    df_base = pd.read_csv(base_file)
    print(f"Loaded {len(df_base)} base responses")
    
    # Get the target responses (GPT-5 responses from the disguised file)
    # Handle NaN values by converting to empty strings
    target_responses = df_disguised["target_response"].fillna("").tolist()
    disguised_responses = df_disguised["disguised_response"].fillna("").tolist()
    source_responses = df_disguised["source_response"].fillna("").tolist()
    
    print("Computing heuristics: disguised vs target...")
    heuristic_table = compute_heuristics(disguised_responses, target_responses)
    
    print("Computing heuristics: target vs source...")
    heuristic_table_target_source = compute_heuristics(target_responses, source_responses)
    
    # Create scores directory
    scores_dir = "disguising/scores/hierarchical_math_disguise/source_microsoft/Phi-4-mini-instruct/target_gpt/gpt-5"
    os.makedirs(scores_dir, exist_ok=True)
    
    # Save heuristic tables
    heuristic_file = os.path.join(scores_dir, "heuristic_table.csv")
    heuristic_table.to_csv(heuristic_file, index=False)
    print(f"Saved heuristic table to {heuristic_file}")
    
    heuristic_file_target_source = os.path.join(scores_dir, "heuristic_table_target_source.csv")
    heuristic_table_target_source.to_csv(heuristic_file_target_source, index=False)
    print(f"Saved target-source heuristic table to {heuristic_file_target_source}")
    
    # Print summary
    print("\n=== HEURISTIC ANALYSIS RESULTS ===")
    print(f"Average style match (disguised vs target): {heuristic_table['match'].mean():.3f}")
    print(f"Average style match (target vs source): {heuristic_table_target_source['match'].mean():.3f}")
    print(f"Style improvement: {heuristic_table['match'].mean() - heuristic_table_target_source['match'].mean():.3f}")
    
    print("\n=== DETAILED BREAKDOWN ===")
    print("Disguised vs Target:")
    for _, row in heuristic_table.iterrows():
        print(f"  {row['style_function']}: {row['match']:.3f}")
    
    print("\nTarget vs Source:")
    for _, row in heuristic_table_target_source.iterrows():
        print(f"  {row['style_function']}: {row['match']:.3f}")

if __name__ == "__main__":
    main()
