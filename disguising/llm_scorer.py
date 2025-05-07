import argparse
import wandb
import pandas as pd
import re
from vllm import LLM, SamplingParams
import os

from stylistic_analysis import compute_heuristics
from utils import get_token_count

comparison_system_prompt = """You are an expert evaluator of AI model responses. Your task is to rate the similarity between two responses in two key dimensions:

1. Semantic Meaning (1-4 points):
   - 4: Responses convey nearly identical meaning and information, with only trivial differences in wording or phrasing. If the format of the responses are different but the core meaning is the same, the score should be 4.
   - 3: Responses share the same core intent or function, even if one includes minor additional elements (e.g., both are greetings, both answer the same question, both provide the same instruction, etc.).
   - 2: Responses have some overlap in meaning, with significant differences in what is being communicated.
   - 1: Responses are unrelated in meaning or intent.

2. Stylistic Similarity (1-4 points):
   - 4: Response style is identical (formatting, tone, formality, etc.)
   - 3: Response style is similar but not identical (e.g., both are informal, both use bullet points, etc.)
   - 2: Response style has some elements in common (e.g., both are informal or conversational), but also significant differences in expressiveness, complexity, or structure
   - 1: Response style is completely different (e.g., one is formal, one is slang; one is a list, one is a paragraph)

For each comparison, provide:
1. The distribution of scores.
2. A brief explanation of the score breakdown
3. Specific examples of similarities and differences

Example format:
Scores: 3/4, 2/4
Breakdown:
- Semantic Meaning: 3/4 
    Justification: both responses are greetings, conveying the same intent, but one adds "how are you"
- Stylistic Similarity: 2/4 
    Justification: both responses are informal with minimal formatting, but one is much more enthusiastic in tone

Here are the two responses to compare:
Response 1:
{response1}


Response 2:
{response2}
"""

def parse_score(score):
    original_score = score
    # get first line of score
    score = score.split("\n")[0]
    
    # First try to match the exact format "Scores: X/Y, Z/W"
    match = re.search(r'Scores:\s*(\d+)/(\d+),\s*(\d+)/(\d+)', score)
    if match:
        return int(match.group(1)), int(match.group(3))
    
    # Try to find two numbers in the format X/Y, Z/W
    match = re.search(r'(\d+)/(\d+),\s*(\d+)/(\d+)', score)
    if match:
        return int(match.group(1)), int(match.group(3))
    
    # Try to find two numbers separated by comma or slash
    match = re.search(r'(\d+)[,/]\s*(\d+)', score)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # If that fails, try to find any two numbers in the line
    matches = re.findall(r'\b(\d+)\b', score)
    if len(matches) >= 2:
        return int(matches[0]), int(matches[1])
    
    print(f"Failed to parse score from: {original_score}")
    return None, None

def parse_similarity_score(score):
    # Use regex to find the first integer between 1 and 10 in the output
    match = re.search(r"\b([1-9]|10)\b", score)
    if match:
        return int(match.group(1))
    print(f"Failed to parse similarity score from: {score.splitlines()[0]}")
    return None

def format_prompt(response1, response2):
    user_question =  comparison_system_prompt.format(response1=response1, response2=response2)
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_question}
    ]

def format_similarity_prompt(response1, response2):
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": f"Given the following two model responses to a question, rate how similar the responses are on a scale of 1-10.\n\nResponse 1:\n{response1}\n\nResponse 2:\n{response2}\n\nSimilarity score (1-10):"}
    ]

def format_prompt(response1, response2):
    user_question =  comparison_system_prompt.format(response1=response1, response2=response2)
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_question}
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response1", type=str)
    parser.add_argument("--response2", type=str)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--input_file_a", type=str)
    parser.add_argument("--input_file_b", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--wandb_project", type=str, default="disguising")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--compute_heuristics_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    wandb.init(project=args.wandb_project, name=f"{args.input_file_a.split('/')[-1].replace('.csv', '')}_vs_{args.input_file_b.split('/')[-1].replace('.csv', '')}", group="comparison")
    wandb.config.update(args)

    if not args.input_file_a and not args.input_file_b:
        llm = LLM(model="microsoft/Phi-4-mini-instruct", trust_remote_code=True, max_model_len=4096)
        sampling_params = SamplingParams(
            max_tokens=500,
            temperature=0.0,
            )
        messages = format_prompt(args.response1, args.response2)
        output = llm.chat(messages=messages, sampling_params=sampling_params)
        print(output[0].outputs[0].text)
    else:
        assert args.input_file_a and args.input_file_b, "input_file_a and input_file_b must be provided"
        outputs = []
        semantic_scores = []
        stylistic_scores = []
        similarity_scores = []
        df_a = pd.read_csv(args.input_file_a).dropna(subset=["model_response"])
        df_b = pd.read_csv(args.input_file_b).dropna(subset=["model_response"])
        df_a["response_a"] = df_a["model_response"]
        df_b["response_b"] = df_b["model_response"]
        df_a = df_a[["prompt", "response_a"]]
        df_b = df_b[["prompt", "response_b"]]
        df = pd.merge(df_a, df_b, on="prompt", how="inner").drop_duplicates(subset=["prompt"]).reset_index(drop=True)
        print(f"Loaded {len(df)} rows")
        if args.test:
            df = df.head(10)
        
        # get token count of each response
        df["token_count_a"] = df["response_a"].apply(get_token_count)
        df["token_count_b"] = df["response_b"].apply(get_token_count)
        df["over_token_limit"] = df["token_count_a"]  + df["token_count_b"]  + 400 > 4096
        print(f"Found {len(df[df['over_token_limit']])} rows that are over the token limit")
        wandb.summary["over_token_limit"] = len(df[df["over_token_limit"]])
        df = df[~df["over_token_limit"]]

        # get average normalized difference in length
        length_diff = [(len(row["response_a"]) - len(row["response_b"])) / max(len(row["response_a"]), len(row["response_b"])) for _, row in df.iterrows()]
        wandb.summary["length_diff"] = sum(length_diff) / len(length_diff)

        # Compute heuristics
        heuristic_table = compute_heuristics(df["response_a"].tolist(), df["response_b"].tolist())
        heuristic_file= args.output_file.replace(".csv", "_heuristic_table.csv")
        heuristic_table.to_csv(heuristic_file, index=False)
        wandb.log({"style_heuristics": wandb.Table(dataframe=heuristic_table)})
        wandb.summary["heuristic_avg_score"] = heuristic_table["match"].mean()
        if args.compute_heuristics_only:
            wandb.finish()
            exit()
        
        llm = LLM(model="microsoft/Phi-4-mini-instruct", trust_remote_code=True, max_model_len=4096)
        sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0.0,
            )
        outputs = []
        batch_size = 100  # You can make this configurable
        num_rows = len(df)
        for i, batch_start in enumerate(range(0, num_rows, batch_size)):
            print(f"Processing batch {i+1} of {num_rows // batch_size}")
            batch_end = min(batch_start + batch_size, num_rows)
            batch_rows = df.iloc[batch_start:batch_end]

            # Prepare batch prompts for comparison
            comparison_messages_batch = [format_prompt(row["response_a"], row["response_b"]) for _, row in batch_rows.iterrows()]
            comparison_outputs = llm.chat(messages=comparison_messages_batch, sampling_params=sampling_params)

            for output in comparison_outputs:
                outputs.append(output.outputs[0].text)
                semantic_score, stylistic_score = parse_score(output.outputs[0].text)
                semantic_scores.append(semantic_score)
                stylistic_scores.append(stylistic_score)

            # Prepare batch prompts for similarity
            similarity_messages_batch = [format_similarity_prompt(row["response_a"], row["response_b"]) for _, row in batch_rows.iterrows()]
            similarity_outputs = llm.chat(messages=similarity_messages_batch, sampling_params=sampling_params)

            for output in similarity_outputs:
                similarity_score = parse_similarity_score(output.outputs[0].text)
                similarity_scores.append(similarity_score)

        df["comparison_results"] = outputs
        df["semantic_score"] = semantic_scores
        df["stylistic_score"] = stylistic_scores
        df["similarity_score"] = similarity_scores
        df.to_csv(args.output_file, index=False)
        wandb.log({"comparison_results": wandb.Table(data=df)})

        # find any rows where there are non or non-integer scores
        non_integer_scores = df[df["semantic_score"].isna() | df["stylistic_score"].isna() | df["similarity_score"].isna()]
        print(f"Found {len(non_integer_scores)} rows with nan scores")
        semantic_scores = [score for score in semantic_scores if score is not None]
        stylistic_scores = [score for score in stylistic_scores if score is not None]
        similarity_scores = [score for score in similarity_scores if score is not None]

        # log semantic and stylistic scores
        wandb.summary["semantic_score"] = sum(semantic_scores) / len(semantic_scores)
        wandb.summary["stylistic_score"] = sum(stylistic_scores) / len(stylistic_scores)
        wandb.summary["similarity_score_1_10"] = sum(similarity_scores) / len(similarity_scores)
        wandb.summary["parsing_errors"] = len(non_integer_scores)
        wandb.finish()
        exit()
