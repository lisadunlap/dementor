from serve.utils_llm import get_llm_output
import argparse
import wandb
import pandas as pd
import re
from vllm import LLM, SamplingParams

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
3, 1
Breakdown:
- Semantic Meaning: 3/4 (both responses are greetings, conveying the same intent, even though one adds "how are you")
- Stylistic Similarity: 2/4 (both responses are informal, but one is more expressive)

Here are the two responses to compare:
Response 1:
{response1}


Response 2:
{response2}
"""

def parse_score(score):
    # get first line of score
    score = score.split("\n")[0]
    # split on comma
    try:
        return int(score.split(",")[0]), int(score.split(",")[1])
    except Exception as e:
        print(f"Failed to parse score from: {score.splitlines()[0]}")
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
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=f"{args.input_file_a.split('/')[-1].replace('.csv', '')}_vs_{args.input_file_b.split('/')[-1].replace('.csv', '')}", group="comparison")
    wandb.config.update(args)

    if not args.input_file_a and not args.input_file_b:
        llm = LLM(model="microsoft/Phi-4-mini-instruct", trust_remote_code=True, max_model_len=2048)
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

        llm = LLM(model="microsoft/Phi-4-mini-instruct", trust_remote_code=True, max_model_len=4096)
        sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0.0,
            )
        outputs = []
        batch_size = 100  # You can make this configurable
        num_rows = len(df)
        for batch_start in range(0, num_rows, batch_size):
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
        print(f"Found {len(non_integer_scores)} rows with non-integer scores")
        semantic_scores = [score for score in semantic_scores if score is not None]
        stylistic_scores = [score for score in stylistic_scores if score is not None]
        similarity_scores = [score for score in similarity_scores if score is not None]

        # log semantic and stylistic scores
        wandb.summary["semantic_score"] = sum(semantic_scores) / len(semantic_scores)
        wandb.summary["stylistic_score"] = sum(stylistic_scores) / len(stylistic_scores)
        wandb.summary["similarity_score_1_10"] = sum(similarity_scores) / len(similarity_scores)
        wandb.finish()
