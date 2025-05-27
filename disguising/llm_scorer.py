import argparse
import wandb
import pandas as pd
import re
from vllm import LLM, SamplingParams
import os
import signal
import sys

from stylistic_analysis import compute_heuristics
from utils import get_token_count

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

Think through your resopnse and provide a breakdown of the scores at the end. Your response should end with the scores in the following format exactly:
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

def parse_similarity_score(score):
    try:
        # Look for the exact format "Similarity score (1-10): X", case insensitive and handling bold
        match = re.search(r'Similarity score \(1-10\):\s*(\d+(?:\.\d+)?)', score, re.IGNORECASE)
        if match:
            score_value = float(match.group(1))
            if 1 <= score_value <= 10:
                return score_value
        
        print(f"Failed to parse similarity score from: {score.splitlines()[0]}")
        return None
    except Exception as e:
        print(f"Error parsing similarity score: {str(e)}")
        print(f"Problematic score text: {score}")
        return None

def format_prompt(response1, response2):
    user_question =  comparison_system_prompt.format(response1=response1, response2=response2)
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_question}
    ]

pure_similarity_prompt = """
Given the following two model responses to a question, rate how similar the responses are on a scale of 1-10, where 1 is completely different and 10 is nearly identical in content and style.

Response 1:
{response1}

Response 2:
{response2}

After the response, provide a score between 1 and 10, where 1 is completely different and 10 is nearly identical in content and style. Your response should end with the score in the following format:

Thought process: [your thought process]
Similarity score (1-10): [score]

There should not be any other text after the score.
"""
def format_similarity_prompt(response1, response2):
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": pure_similarity_prompt.format(response1=response1, response2=response2)}
    ]

def format_prompt(response1, response2):
    user_question =  comparison_system_prompt.format(response1=response1, response2=response2)
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_question}
    ]

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\nGracefully shutting down...")
        if 'llm' in locals():
            del llm
        wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--response1", type=str)
    parser.add_argument("--response2", type=str)
    parser.add_argument("--input_file_a", type=str)
    parser.add_argument("--input_file_b", type=str)
    parser.add_argument("--output_dir", type=str, default="disguising/results/base_500")
    parser.add_argument("--wandb_project", type=str, default="disguising-baseline-scoring")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--compute_heuristics_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    wandb.init(project=args.wandb_project, name=f"{args.input_file_a.split('/')[-1].replace('.csv', '')}_vs_{args.input_file_b.split('/')[-1].replace('.csv', '')}", group="comparison")
    wandb.config.update(args)

    try:
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
                df = df.head(30)
            
            # get token count of each response
            df["token_count_a"] = df["response_a"].apply(get_token_count)
            df["token_count_b"] = df["response_b"].apply(get_token_count)
            df["over_token_limit"] = df["token_count_a"]  + df["token_count_b"]  + 800 > 4096
            print(f"Found {len(df[df['over_token_limit']])} rows that are over the token limit")
            wandb.summary["over_token_limit"] = len(df[df["over_token_limit"]])
            df = df[~df["over_token_limit"]]

            # get average normalized difference in length
            length_diff = [(len(row["response_a"]) - len(row["response_b"])) / max(len(row["response_a"]), len(row["response_b"])) for _, row in df.iterrows()]
            wandb.summary["length_diff"] = sum(length_diff) / len(length_diff)

            # Compute heuristics
            heuristic_table = compute_heuristics(df["response_a"].tolist(), df["response_b"].tolist())
            heuristic_file= os.path.join(args.output_dir, "heuristic_table.csv")
            heuristic_table.to_csv(heuristic_file, index=False)
            wandb.log({"style_heuristics": wandb.Table(dataframe=heuristic_table)})
            wandb.summary["heuristic_avg_score"] = heuristic_table["match"].mean()
            for i, row in heuristic_table.iterrows():
                wandb.summary[row['style_function']] = row["match"]
            if args.compute_heuristics_only:
                wandb.finish()
                sys.exit(0)
            
            llm = LLM(model="microsoft/Phi-4-mini-instruct", trust_remote_code=True, max_model_len=4096)
            sampling_params = SamplingParams(
                max_tokens=4096,
                temperature=0.0,
                )
            outputs = []
            semantic_outputs = []
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
                    semantic_outputs.append(output.outputs[0].text)
                    similarity_score = parse_similarity_score(output.outputs[0].text)
                    similarity_scores.append(similarity_score)

            df["comparison_results"] = outputs
            df["semantic_score"] = semantic_scores
            df["stylistic_score"] = stylistic_scores
            df["similarity_score"] = similarity_scores
            df["semantic_output"] = semantic_outputs
            print(df.head())
            df.to_csv(os.path.join(args.output_dir, "comparison_results.csv"), index=False)
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
    # except Exception as e:
    #     print(f"Error: {str(e)}")
    #     wandb.finish()
    #     sys.exit(1)
    finally:
        # Cleanup
        if 'llm' in locals():
            del llm
        wandb.finish()
        sys.exit(0)
