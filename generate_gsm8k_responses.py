#!/usr/bin/env python3
"""
Generate GPT-5 responses for GSM8K math problems.
Requires OpenAI API key in environment variable OPENAI_API_KEY
"""

import os
import pandas as pd
import openai
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_gpt5_response(prompt, max_retries=3):
    """Generate response from GPT-5 for a math problem."""
    
    system_prompt = """You are a helpful math tutor. Solve the given math problem step by step, showing your reasoning clearly. 
    Format your response in a clear, educational manner. Make sure to:
    1. Break down the problem into steps
    2. Show your calculations
    3. Provide the final answer
    4. Use clear, simple language"""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}")
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts for prompt: {prompt[:50]}...")
                return f"ERROR: {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff
            continue

def generate_responses_for_gsm8k():
    """Generate GPT-5 responses for 500 GSM8K test problems."""
    
    # Load the GSM8K test data
    gsm8k_file = 'data/gsm8k/gsm8k_test.csv'
    if not os.path.exists(gsm8k_file):
        print(f"GSM8K test file not found: {gsm8k_file}")
        return
    
    df = pd.read_csv(gsm8k_file)
    print(f"Loaded {len(df)} GSM8K test problems")
    
    # Take first 500 problems
    df = df.head(500)
    print(f"Generating responses for {len(df)} problems...")
    
    responses = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['prompt']
        
        # Generate response
        response = generate_gpt5_response(prompt)
        
        responses.append({
            'prompt': prompt,
            'model_response': response,
            'model': 'gpt-5',
            'split': 'test'
        })
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
        
        # Save progress every 50 responses
        if (idx + 1) % 50 == 0:
            temp_df = pd.DataFrame(responses)
            temp_df.to_csv('disguising/model-responses/gsm8k/gpt-5_gsm8k_test_500.csv', index=False)
            print(f"Saved {len(responses)} responses so far...")
    
    # Save final results
    final_df = pd.DataFrame(responses)
    final_df.to_csv('disguising/model-responses/gsm8k/gpt-5_gsm8k_test_500.csv', index=False)
    
    print(f"Generated responses for {len(responses)} problems")
    print(f"Saved to disguising/model-responses/gsm8k/gpt-5_gsm8k_test_500.csv")
    
    return final_df

if __name__ == "__main__":
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    # Generate responses for 500 test problems
    print("Generating GPT-5 responses for 500 GSM8K test problems...")
    generate_responses_for_gsm8k()
