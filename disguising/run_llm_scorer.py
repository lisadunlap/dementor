#!/usr/bin/env python3

import pandas as pd
import requests
import json
import os
import re
from stylistic_analysis import compute_heuristics
import time

# Use your vLLM server
VLLM_SERVER_URL = "http://localhost:8010"

comparison_system_prompt = """You are an expert evaluator of AI model responses. Your task is to rate the similarity between two responses in two key dimensions:

1. Semantic Meaning (1-4 points) - this ONLY considers the meaning of the response, not the style, formatting, or tone:
   - 4: Responses convey nearly identical meaning and information, with only trivial differences in wording or phrasing. If the format of the responses are different but the core meaning is the same, the score should be 4.
   - 3: Responses share the same core intent or function, even if one includes minor additional elements (e.g., both are greetings, both answer the same question, both provide the same instruction, etc.).
   - 2: Responses have some overlap in meaning, with significant differences in what is being communicated.
   - 1: Responses are unrelated in meaning or intent.

2. Stylistic Similarity (1-4 points) - this ONLY considers the style, formatting, and tone of the response, not the content:
   - 4: Response style is identical (formatting, tone, formality, etc.)
   - 3: Response style is similar but not identical (e.g., both are informal, both use bullet points, etc.)
   - 2: Response style has some elements in common (e.g., both are informal or conversational), but also significant differences in expressiveness, complexity, or structure
   - 1: Response style is completely different (e.g., one is formal, one is slang; one is a list, one is a paragraph)

For each comparison, provide:
1. The distribution of scores.
2. A brief explanation of the score breakdown
3. Specific examples of similarities and differences

Here are the two responses to compare:
Response 1:
{response1}


Response 2:
{response2}

Think through your response and provide a breakdown of the scores at the end. Your response should end with the scores in the following format exactly:
Thought process: [your thought process]
Breakdown:
- Semantic Meaning: [meaning_score]/4 
- Stylistic Similarity: [stylistic_score]/4 

There should not be any other text after the scores.
"""

def parse_score(score):
    try:
        original_score = score
        
        # Look for the breakdown section specifically, case insensitive
        breakdown_match = re.search(r'Breakdown:(.*?)(?=\n\n|$)', score, re.DOTALL | re.IGNORECASE)
        if not breakdown_match:
            print(f"Failed to find breakdown section in: {original_score}")
            return None, None
            
        breakdown_text = breakdown_match.group(1)
        
        # Find semantic and stylistic scores in the breakdown section, case insensitive and handling bold
        semantic_match = re.search(r'Semantic Meaning:\s*(\d+(?:\.\d+)?)/4', breakdown_text, re.IGNORECASE)
        stylistic_match = re.search(r'Stylistic Similarity:\s*(\d+(?:\.\d+)?)/4', breakdown_text, re.IGNORECASE)
        
        if semantic_match and stylistic_match:
            return float(semantic_match.group(1)), float(stylistic_match.group(1))
        
        print(f"Failed to parse scores from breakdown: {breakdown_text}")
        return None, None
    except Exception as e:
        print(f"Error parsing score: {str(e)}")
        print(f"Problematic score text: {score}")
        return None, None

def call_vllm_server(messages, max_tokens=4096, temperature=0.0):
    """Call the vLLM server via HTTP API"""
    payload = {
        "model": "microsoft/Phi-4-mini-instruct",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(f"{VLLM_SERVER_URL}/v1/chat/completions", 
                               json=payload, 
                               timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling vLLM server: {e}")
        return None

def format_prompt(response1, response2):
    user_question = comparison_system_prompt.format(response1=response1, response2=response2)
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_question}
    ]

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
    
    # Compute heuristics first
    print("Computing heuristics...")
    heuristic_table = compute_heuristics(df["disguised_response"].tolist(), df["target_response"].tolist())
    heuristic_table_target_source = compute_heuristics(df["target_response"].tolist(), df["source_response"].tolist())
    
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
    
    # Now run LLM scoring
    print("Starting LLM scoring...")
    semantic_scores = []
    stylistic_scores = []
    comparison_outputs = []
    
    # Process in batches to avoid overwhelming the server
    batch_size = 5
    num_rows = len(df)
    
    for i, batch_start in enumerate(range(0, num_rows, batch_size)):
        print(f"Processing batch {i+1} of {(num_rows + batch_size - 1) // batch_size}")
        batch_end = min(batch_start + batch_size, num_rows)
        batch_rows = df.iloc[batch_start:batch_end]
        
        for _, row in batch_rows.iterrows():
            messages = format_prompt(row["disguised_response"], row["target_response"])
            output = call_vllm_server(messages)
            
            if output:
                comparison_outputs.append(output)
                semantic_score, stylistic_score = parse_score(output)
                semantic_scores.append(semantic_score)
                stylistic_scores.append(stylistic_score)
            else:
                comparison_outputs.append("")
                semantic_scores.append(None)
                stylistic_scores.append(None)
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
    
    # Add results to dataframe
    df["comparison_results"] = comparison_outputs
    df["semantic_score"] = semantic_scores
    df["stylistic_score"] = stylistic_scores
    
    # Save results
    comparison_file = os.path.join(scores_dir, "comparison_results.csv")
    df.to_csv(comparison_file, index=False)
    print(f"Saved comparison results to {comparison_file}")
    
    # Print summary
    valid_semantic = [s for s in semantic_scores if s is not None]
    valid_stylistic = [s for s in stylistic_scores if s is not None]
    
    print(f"\n=== LLM SCORING RESULTS ===")
    print(f"Valid semantic scores: {len(valid_semantic)}/{len(semantic_scores)}")
    print(f"Valid stylistic scores: {len(valid_stylistic)}/{len(stylistic_scores)}")
    
    if valid_semantic:
        print(f"Average semantic score: {sum(valid_semantic) / len(valid_semantic):.3f}")
    if valid_stylistic:
        print(f"Average stylistic score: {sum(valid_stylistic) / len(valid_stylistic):.3f}")
    
    print(f"\n=== HEURISTIC ANALYSIS RESULTS ===")
    print(f"Average style match (disguised vs target): {heuristic_table['match'].mean():.3f}")
    print(f"Average style match (target vs source): {heuristic_table_target_source['match'].mean():.3f}")
    print(f"Style improvement: {heuristic_table['match'].mean() - heuristic_table_target_source['match'].mean():.3f}")

if __name__ == "__main__":
    main()
