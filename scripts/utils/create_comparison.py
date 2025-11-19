#!/usr/bin/env python3
"""Create a comparison CSV for scoring."""
import pandas as pd

# Load both datasets
print("Loading GPT-4o responses...")
gpt4o_df = pd.read_csv('results/gpt4o_responses_fixed.csv', quoting=1)
print(f"GPT-4o: {len(gpt4o_df)} responses")

print("Loading Llama responses...")
llama_df = pd.read_csv('results/llama_3.1_8b_responses.csv')
print(f"Llama: {len(llama_df)} responses")

# Merge on prompt to create comparison pairs
print("Creating comparison pairs...")
comparison_df = pd.merge(
    gpt4o_df[['prompt', 'model_response']].rename(columns={'model_response': 'target_response'}),
    llama_df[['prompt', 'model_response']],
    on='prompt',
    how='inner'
)

print(f"Comparison pairs: {len(comparison_df)}")

# Add metadata
comparison_df['target_model'] = 'gpt-4o'
comparison_df['source_model'] = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

# Save comparison file
output_file = 'results/comparison_gpt4o_vs_llama.csv'
comparison_df.to_csv(output_file, index=False)
print(f"Saved comparison to: {output_file}")

# Show sample
print("\nSample comparison:")
print(comparison_df[['prompt', 'target_response', 'model_response']].head(2))
