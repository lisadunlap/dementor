#!/usr/bin/env python3
"""
Create a GSM8K dataset with existing GPT-4o responses.
This avoids needing new API calls and uses your existing data.
"""

import os
import pandas as pd

def create_gsm8k_gpt4o_dataset():
    """Create a GSM8K dataset with existing GPT-4o responses."""
    
    print("📊 Creating GSM8K dataset with existing GPT-4o responses...")
    
    # Load GSM8K test problems
    gsm8k_file = 'data/gsm8k/gsm8k_test.csv'
    if not os.path.exists(gsm8k_file):
        print(f"❌ GSM8K test file not found: {gsm8k_file}")
        return
    
    gsm8k_df = pd.read_csv(gsm8k_file)
    print(f"✅ Loaded {len(gsm8k_df)} GSM8K test problems")
    
    # Load existing GPT-4o responses
    gpt4o_file = 'disguising/model-responses/base_500_all_models/gpt-4o.csv'
    if not os.path.exists(gpt4o_file):
        print(f"❌ GPT-4o responses file not found: {gpt4o_file}")
        return
    
    gpt4o_df = pd.read_csv(gpt4o_file)
    print(f"✅ Loaded {len(gpt4o_df)} existing GPT-4o responses")
    
    # Take first 500 GSM8K problems
    gsm8k_500 = gsm8k_df.head(500).copy()
    
    # Sample 500 random GPT-4o responses to pair with the math problems
    gpt4o_sample = gpt4o_df.sample(n=500, random_state=42).copy()
    
    # Create the combined dataset
    combined_data = []
    
    for i, (_, gsm8k_row) in enumerate(gsm8k_500.iterrows()):
        gpt4o_row = gpt4o_sample.iloc[i]
        
        combined_data.append({
            'gsm8k_prompt': gsm8k_row['prompt'],
            'gsm8k_answer': gsm8k_row.get('answer', ''),
            'gpt4o_prompt': gpt4o_row['prompt'],
            'gpt4o_response': gpt4o_row['model_response'],
            'split': 'test'
        })
    
    # Save the combined dataset
    output_file = 'disguising/model-responses/gsm8k/gsm8k_500_with_gpt4o_examples.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(output_file, index=False)
    
    print(f"✅ Created dataset with {len(combined_df)} problems")
    print(f"📁 Saved to: {output_file}")
    
    # Show some examples
    print("\n📝 Example entries:")
    for i in range(min(3, len(combined_df))):
        print(f"\n--- Entry {i+1} ---")
        print(f"GSM8K Problem: {combined_df.iloc[i]['gsm8k_prompt'][:100]}...")
        print(f"GPT-4o Example: {combined_df.iloc[i]['gpt4o_prompt'][:100]}...")
        print(f"GPT-4o Response: {combined_df.iloc[i]['gpt4o_response'][:100]}...")
    
    return combined_df

if __name__ == "__main__":
    create_gsm8k_gpt4o_dataset()
