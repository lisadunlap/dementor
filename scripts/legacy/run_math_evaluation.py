#!/usr/bin/env python3
# moved to scripts/legacy

import pandas as pd
import os
import sys

# Add the methods/utils directory to the path
sys.path.append('disguising/methods/utils')

from math_evaluation import MathDisguiseEvaluator

def main():
    # Load the disguised responses
    disguised_file = "disguising/model-responses/disguised/hierarchical_math_disguise/microsoft_Phi-4-mini-instruct_disguised-gpt-5.csv"
    
    print("Loading disguised responses...")
    df = pd.read_csv(disguised_file)
    print(f"Loaded {len(df)} disguised responses")
    
    # Handle NaN values
    df["disguised_response"] = df["disguised_response"].fillna("")
    df["target_response"] = df["target_response"].fillna("")
    df["source_response"] = df["source_response"].fillna("")
    
    # Get the response lists
    source_responses = df["source_response"].tolist()
    target_responses = df["target_response"].tolist()
    disguised_responses = df["disguised_response"].tolist()
    
    print("Running math-specific evaluation...")
    
    # Initialize the math evaluator
    evaluator = MathDisguiseEvaluator(
        source_responses=source_responses,
        target_responses=target_responses,
        disguised_responses=disguised_responses
    )
    
    # Get evaluation results
    results = evaluator.evaluation_results
    
    # Create scores directory
    scores_dir = "disguising/scores/hierarchical_math_disguise/source_microsoft/Phi-4-mini-instruct/target_gpt/gpt-5"
    os.makedirs(scores_dir, exist_ok=True)
    
    # Save math evaluation results
    math_eval_file = os.path.join(scores_dir, "math_evaluation_results.csv")
    
    # Convert results to DataFrame and save
    if isinstance(results, dict):
        # If results is a dict, convert to DataFrame
        results_df = pd.DataFrame([results])
    else:
        # If it's already a DataFrame
        results_df = results
    
    results_df.to_csv(math_eval_file, index=False)
    print(f"Saved math evaluation results to {math_eval_file}")
    
    # Print summary
    print(f"\n=== MATH EVALUATION RESULTS ===")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
