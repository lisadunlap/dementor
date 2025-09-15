import pandas as pd
import json
import numpy as np
from openai import OpenAI
import random
from typing import List, Dict, Tuple
import os
from tqdm import tqdm

def _load_embeddings(ratings_df: pd.DataFrame, trait_axes: Dict) -> Tuple[List[List[float]], List[int]]:
    '''
    Load existing embeddings and response indices from a ratings DataFrame.
    '''
    trait_names = list(trait_axes.keys())
    
    response_indices = ratings_df['response_id'].tolist()
    
    embeddings = []
    for _, row in ratings_df.iterrows():
        rating_vector = [row[trait] for trait in trait_names]
        embeddings.append(rating_vector)
    
    return embeddings, response_indices

def extract_vibe_features(responses_source: pd.Series, responses_target: pd.Series, n_samples: int,
                          path_to_axes: str, path_to_ratings: str, n_axes: int = 10) -> Tuple[List[List[float]], Dict, pd.DataFrame]:
    """
    Extract vibe features that are most distinctive between source and target models. 
    1) Identify trait axes
    2) Rate target model responses on the trait axes
    
    Args:
        responses_source: pandas Series containing source model responses
        responses_target: pandas Series containing target model responses  
        n_samples: number of samples to extract features from
        path_to_axes: path to save trait axes
        path_to_ratings: path to save ratings
        n_axes: number of axes to extract
    Returns:
        embeddings: List of embeddings (list of 10 integers for each target response)
        response_indices: List of response indices (indices of the target responses successfully rated)
    """

    if os.path.exists(path_to_axes) and os.path.exists(path_to_ratings):
        trait_axes = json.load(open(path_to_axes))['trait_axes']
        ratings_df = pd.read_csv(path_to_ratings)
        embeddings, response_indices = _load_embeddings(ratings_df, trait_axes)
        return embeddings, response_indices
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Step 1: Sample 10 responses from source and target
    source_sample = responses_source.sample(n=min(n_samples, len(responses_source)), random_state=42)
    target_sample = responses_target.sample(n=min(n_samples, len(responses_target)), random_state=42)
    
    # Step 2: Create prompt to identify trait axes
    sample_responses_text_source = "\n\n---\n\n".join([f"Response {i+1}: {resp}" for i, resp in enumerate(source_sample)])
    sample_responses_text_target = "\n\n---\n\n".join([f"Response {i+1}: {resp}" for i, resp in enumerate(target_sample)])

    axes_prompt = f"""You are an expert in analyzing and comparing model response styles. Your goal is to identify the {n_axes} most distinctive trait or vibe axes that differentiate target model's response style from source model's response style, regardless of the content of the responses.
When analyzing the model style, consider the following factors in detail (this is not an exhaustive list, but a good starting point):

1. Response Style:
   - Length and verbosity of responses
   - Use of bullet points, numbered lists, or paragraphs
   - Level of formality or casualness
   - Use of technical jargon vs. plain language

2. Tone and Personality:
   - Emotional tone (friendly, professional, academic, etc.)
   - Use of humor or wit
   - Level of confidence in responses
   - Personal pronouns and self-references

3. Content Structure:
   - How information is organized
   - Use of headers, sections, or subsections
   - Introduction and conclusion patterns
   - Handling of multiple questions or topics

4. Language Patterns:
   - Common phrases or expressions
   - Commonly used introductions
   - Sentence structure and complexity
   - Use of metaphors or analogies
   - Transitional phrases and connectors

5. Technical Aspects:
   - Level of detail in explanations
   - Use of examples or demonstrations
   - Handling of uncertainty or ambiguity
   - Approach to problem-solving

6. Interaction Style:
   - How questions are addressed
   - Use of follow-up questions
   - Handling of edge cases or errors
   - Response to unclear or ambiguous prompts

Based on these sample responses from the source and target models, identify the {n_axes} top trait axes that the two models differ the most in. For each trait, provide a definition for the least expression (0) and most expression (10) of that trait.

Sample responses from source model:
{sample_responses_text_source}

Sample responses from target model:
{sample_responses_text_target}

Output format (JSON):
{{
  "trait_name_1": {{
    "0": "definition of least expression of this trait",
    "10": "definition of most expression of this trait"
  }},
  "trait_name_2": {{
    "0": "definition of least expression of this trait", 
    "10": "definition of most expression of this trait"
  }},
  ...
}}

Example:
{{
  "Assertiveness": {{
    "0": "Uses tentative or uncertain language",
    "10": "Uses definitive, confident statements"
  }}
}}

Provide exactly {n_axes} trait axes."""

    # Get trait axes from GPT-4o-mini
    try:
        axes_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": axes_prompt}],
            temperature=0.3
        )
        
        axes_text = axes_response.choices[0].message.content
        # Extract JSON from response
        start_idx = axes_text.find('{')
        end_idx = axes_text.rfind('}') + 1
        axes_json = axes_text[start_idx:end_idx]
        trait_axes = json.loads(axes_json)
        
    except Exception as e:
        raise ValueError(f"Error getting trait axes: {e}")
    
    print(f"Identified {len(trait_axes)} trait axes")
    
    # Step 3: Rate each target response on the trait axes
    target_ratings = []
    embeddings = []
    response_indices = []
    trait_names = list(trait_axes.keys())
    
    for idx, response in enumerate(tqdm(responses_target)):        
        # Create rating prompt
        traits_description = "\n".join([
            f"{trait}: 0={definitions['0']}, 10={definitions['10']}" 
            for trait, definitions in trait_axes.items()
        ])
        
        rating_prompt = f"""You are an expert in analyzing model response styles. Your goal is to rate the following response on the given 10 trait axes that characterize a model's response style, regardless of the content of the responses. 
Rate the following response on these {n_axes} trait axes, providing a score from 0-10 for each trait:

{traits_description}

Response to rate:
{response}

Output format (JSON):
{{
  "{trait_names[0]}": score,
  "{trait_names[1]}": score,
  ...
}}

Provide exactly one integer score (0-10) for each of the {n_axes} traits."""

        try:
            rating_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": rating_prompt}],
                temperature=0.1
            )
            
            rating_text = rating_response.choices[0].message.content
            # Extract JSON from response
            start_idx = rating_text.find('{')
            end_idx = rating_text.rfind('}') + 1
            rating_json = rating_text[start_idx:end_idx]
            ratings = json.loads(rating_json)
            
            # Ensure we have ratings for all traits
            rating_vector = []
            for trait in trait_names:
                if trait in ratings:
                    rating_vector.append(float(ratings[trait]))
                else:
                    raise ValueError(f"Trait {trait} not found in ratings.")
                    
        except Exception as e:
            print(f"WARNING: Error rating response {idx}: {e}. Skipping response.")
            # skip the response if error
            continue
        
        # Store results
        rating_dict = {
            'response_id': idx,
            'response_text': response
        }
        for i, trait in enumerate(trait_names):
            rating_dict[trait] = rating_vector[i]
        
        target_ratings.append(rating_dict)
        embeddings.append(rating_vector)
        response_indices.append(idx)
    
    # Step 4: Create outputs
    # Save trait axes to JSON
    trait_axes_with_samples = {}
    trait_axes_with_samples['trait_axes'] = trait_axes
    trait_axes_with_samples['samples'] = target_sample.to_list()
    with open(path_to_axes, 'w') as f:
        json.dump(trait_axes_with_samples, f, indent=2)
    
    # Create DataFrame with ratings
    ratings_df = pd.DataFrame(target_ratings)
    
    # Save to CSV
    ratings_df.to_csv(path_to_ratings, index=False)
    
    print(f"Processed {len(responses_target)} responses, {len(embeddings)} responses were successfully rated.")
    print(f"Trait/vibe axes saved to {path_to_axes}")
    print(f"Ratings saved to {path_to_ratings}")
    
    return embeddings, response_indices