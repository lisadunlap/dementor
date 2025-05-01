from serve.utils_llm import get_llm_output
import argparse
import wandb
import pandas as pd
from vllm import LLM, SamplingParams

llm = LLM(model="microsoft/Phi-4-mini-instruct", trust_remote_code=True, max_model_len=2048)
sampling_params = SamplingParams(
  max_tokens=500,
  temperature=0.0,
)


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
    return int(score.split(",")[0]), int(score.split(",")[1])

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
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--wandb_project", type=str, default="disguising")
    parser.add_argument("--col1", type=str)
    parser.add_argument("--col2", type=str)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if not args.input_file:
        messages = format_prompt(args.response1, args.response2)
        output = llm.chat(messages=messages, sampling_params=sampling_params)
        print(output[0].outputs[0].text)
    else:
        assert args.col1 and args.col2, "col1 and col2 must be provided if input_file is provided"
        outputs = []
        semantic_scores = []
        stylistic_scores = []
        df = pd.read_csv(args.input_file)
        if args.test:
            df = df.head(10)
        outputs = []
        for index, row in df.iterrows():
            messages = format_prompt(row[args.col1], row[args.col2])
            print(messages)
            output = llm.chat(messages=messages, sampling_params=sampling_params)
            outputs.append(output[0].outputs[0].text)
            semantic_score, stylistic_score = parse_score(output[0].outputs[0].text)
            semantic_scores.append(semantic_score)
            stylistic_scores.append(stylistic_score)

        wandb.init(project=args.wandb_project)
        df["comparison_results"] = outputs
        df["semantic_score"] = semantic_scores
        df["stylistic_score"] = stylistic_scores
        df.to_csv(args.output_file, index=False)
        wandb.log({"comparison_results": wandb.Table(data=df)})

        # log semantic and stylistic scores
        wandb.summary["semantic_score"] = sum(semantic_scores) / len(semantic_scores)
        wandb.summary["stylistic_score"] = sum(stylistic_scores) / len(stylistic_scores)
        wandb.finish()
