import os
import json
import pandas as pd
from serve.utils_llm import get_llm_output
from typing import List

proposer_onesided = """You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions and seeing if these differences correspond with user preferences. Write down as many properties as you can find that are present in Model 1 but not in Model 2. Please format your differences as a list of properties that appear more in one output than the other.

{combined_responses}

The format should be a list of properties that appear more in the output of Model 1 than the output of Model 2 in the format of a short description of the property. Respond with a list of properties, each on a new line.

Consider differences on many different axes such as tone, language, structure, content, safety, and any other axis that you can think of. If the questions have a specific property or cover a specific topic (e.g. coding, creative writing, math, etc.), also consider differences which are relevant to that property or topic.
    
Remember that these differences should be human interpretable and that the differences should be concise, substantive and objective. Write down as many properties as you can find. Do not explain which model has which property, simply describe the property. Your response should not include any mention of Model 1 or Model 2, only the properties that are present more in Model 1 than Model 2.
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

reduce_freeform = """Below is a list of properties that are found in LLM outputs. I would like to summarize this list to a set of representative properties with clear and concise descriptions that cover the recurring themes in the data. Are there any interesting overarching properties that are present in a large number of the properties? Please return a list of properties that are seen in the data, where each property represents one type of behavior that is seen in the data.

Here is the list of properties:
{differences}

A human should be able to understand the property and its meaning, and this property should provide insight into the model's behavior or personality. Do not include subjective analysis about these properties, simply describe the property. For instance "the model is more advanced in its understanding" and "the model uses historical context" is not a good property because it is too vague and does not provide interesting insight into the model's behavior. Similarly, these properties should be on a per prompt basis, so "the model provides a consistent tone across prompts" or "the model varies its tone from formal to informal" is not a good property because a person could not make a judgement only looking at a single prompt.

These properties should be something that a human could reasonably expect to see in the model's output when given new prompts. If the property is specific to a type of task (e.g. coding), please ensure that the axis is named in a way that makes it clear what type of task it applies to in the axis name.

Order your final list of properties by how much they are seen in the data. Your response should be a list deliniated with "-"
"""

reduce_freeform_fixed = """Below is a list of properties that are found in LLM outputs. I would like to summarize this list to a set of at most {num_final_vibes} representative properties with clear and concise descriptions that cover the recurring themes in the data. Are there any interesting overarching properties that are present in a large number of the properties? Please return a list of properties that are seen in the data, where each property represents one type of behavior that is seen in the data.

Here is the list of properties:
{differences}

A human should be able to understand the property and its meaning, and this property should provide insight into the model's behavior or personality. Do not include subjective analysis about these properties, simply describe the property. For instance "the model is more advanced in its understanding" and "the model uses historical context" is not a good property because it is too vague and does not provide interesting insight into the model's behavior. Similarly, these properties should be on a per prompt basis, so "the model provides a consistent tone across prompts" or "the model varies its tone from formal to informal" is not a good property because a person could not make a judgement only looking at a single prompt.

These properties should be something that a human could reasonably expect to see in the model's output when given new prompts. If the property is specific to a type of task (e.g. coding), please ensure that the axis is named in a way that makes it clear what type of task it applies to in the axis name.

Order your final list of at most {num_final_vibes} properties by how much they are seen in the data. Your response should be a list deliniated with "-"
"""

def parse_bullets(text: str):
    lines = text.split("\n")
    bullets = []
    current_bullet = ""
    found_first_bullet = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-") or stripped.startswith("*"):
            found_first_bullet = True
            # Save previous bullet if exists
            if current_bullet:
                bullets.append(current_bullet.strip())
            current_bullet = stripped.replace("* ", "", 1).replace("- ", "", 1).replace("**", "")
        # Continuation of current bullet
        elif stripped and found_first_bullet:
            current_bullet += " " + stripped.replace("**", "")
    
    # Add the last bullet if there is one
    if current_bullet:
        bullets.append(current_bullet.strip())
    
    return [bullet.strip() for bullet in bullets if bullet.strip()]

def get_differences(df_batch: pd.DataFrame) -> List[str]:
    prompt_str = "\n\n".join([f"Prompt: {row['prompt']}\n\nModel 1: {row['model_1']}\n\nModel 2: {row['model_2']}" for _, row in df_batch.iterrows()])
    prompt = proposer_onesided.format(combined_responses=prompt_str)
    output = get_llm_output(prompt, "gpt-4o")
    return parse_bullets(output)

def reduce_properties(differences: List[str], num_final_vibes: int = False) -> List[str]:
    if num_final_vibes:
        prompt = reduce_freeform_fixed.format(differences=differences, num_final_vibes=num_final_vibes)
    else:
        prompt = reduce_freeform.format(differences=differences)
    output = get_llm_output(prompt, "gpt-4o")
    return parse_bullets(output)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--num_final_vibes", type=int, default=10)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    df["model_1"] = args.models[0]
    df["model_2"] = args.models[1]
    all_differences = []
    for i in range(args.num_rounds):
        batch_df = df.sample(args.batch_size)
        print(f"Round {i+1} of {args.num_rounds}")
        differences = get_differences(batch_df)
        all_differences.extend(differences)
    print(all_differences)
    reduced_differences = reduce_properties(all_differences, args.num_final_vibes)
    print(reduced_differences)
    print(len(reduced_differences))
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(reduced_differences, f)
