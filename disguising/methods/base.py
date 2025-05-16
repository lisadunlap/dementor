import pandas as pd
from typing import List
from litellm import completion
import wandb

from transformers import AutoTokenizer
from utils import get_token_count

class MethodBase:
    """
    Base class for all methods.
    """
    def __init__(self, model: str, disguise_as: str) -> None:
        self.model = model
        self.disguise_as = disguise_as

    def make_disguise_prompt(self, examples, prompt):
        """
        Create a prompt string for disguise, given a dataframe of examples and a new prompt.
        examples: pd.DataFrame with columns 'prompt' and 'model_response'
        prompt: str, the new question to answer
        """
        disguise_prompt  = """Your task is to answer the following prompt in the style of another AI assistant. I will provide you with examples of responses from the other AI assistant to different prompts to help you understand the style. Your goal is to mimic the formatting, tone, level of detail, and phrasing – not to copy content exactly.
## Prompt to answer:
{prompt}

## Examples from the other AI assistant:
{examples}

Do not respond to my instructions above (e.g. "Here is the prompt in the style you want") in your output, only respond with the answer to the prompt in the desired style. Here is the prompt to answer (repeated for clarity):
prompt: {prompt}

response:
"""
        formatted_examples = ""
        examples = examples.reset_index(drop=True)
        for i, row in examples.iterrows():
            formatted_examples += f"### Example {i + 1}:\n"
            formatted_examples += f"prompt: {row['prompt']}\n"
            formatted_examples += f"response: {row['model_response']}\n\n"
        # strip the first " from the prompt if it exists at the beginning (this is a hack to get around errors with the original prompts, i think it was a bug)
        prompt = prompt.lstrip('"')
        return disguise_prompt.format(examples=formatted_examples, prompt=prompt)
    
    def forward(self, prompt: str) -> str:
        """
        Given a prompt, return the disguised prompt to use for the model.
        """
        return prompt
        
class RandomSampleDisguise(MethodBase):
    """
    Disguise the prompt by randomly sampling from the base model's responses.
    """
    def __init__(self, model: str, disguise_as: str, num_samples: int = 1000, num_samples_per_disguise: int = 5, seed: int = None,
                 disguise_df: pd.DataFrame = None) -> None:
        """
        Num_samples is the number of samples to use from the base model, used to read in the responses from the model-responses/base folder.
        Num_samples_per_disguise is the number of samples to use for each disguise.
        Seed is the seed to use for the random sampling, if it is None, then no seed is used and the samples will be different each time.
        """
        super().__init__(model, disguise_as)
        self.num_samples = num_samples
        self.num_samples_per_disguise = num_samples_per_disguise
        self.seed = seed
        self.disguise_df = disguise_df
        self.disguise_df["token_length"] = self.disguise_df["model_response"].apply(lambda x: get_token_count(x))
        # truncate the responses to 256 tokens
        self.disguise_df["model_response"] = self.disguise_df["model_response"].apply(lambda x: f"{x[:256]}...(truncated)" if get_token_count(x) > 256 else x)

    def forward(self, prompt: str) -> str:
        """
        Given a prompt, return the disguised prompt to use for the model.
        """
        disguise_df_sample = self.disguise_df.sample(n=self.num_samples_per_disguise, random_state=self.seed)
        disguise_prompt = self.make_disguise_prompt(disguise_df_sample, prompt)
        return [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": disguise_prompt}]

class JustNameIt(MethodBase):
    """
    Just ask the model to act like the other model.
    """
    def __init__(self, model: str, disguise_as: str) -> None:
        super().__init__(model, disguise_as)

    def forward(self, prompt: str) -> str:
        return [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"Answer the following prompt in the style of {self.disguise_as}:\n{prompt}"}]
    
class VibeBasedDisguise(MethodBase):
    """
    Disguise the prompt by using the vibe of the other model.
    """
    def __init__(self, model: str, disguise_as: str, num_samples: int = 1000, num_samples_per_disguise: int = 10, seed: int = None,
                 disguise_df: pd.DataFrame = None, model_df: pd.DataFrame = None) -> None:
        super().__init__(model, disguise_as)
        self.num_samples = num_samples
        self.num_samples_per_disguise = num_samples_per_disguise
        self.seed = seed
        self.disguise_df = disguise_df
        self.model_df = model_df
        # join the two dataframes on the prompt column
        self.model_df = self.model_df.merge(self.disguise_df, on="prompt", suffixes=("_model", "_disguise"))
        self.model_tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.system_prompts, self.logs = [], []
        for i in range(5):
            system_prompt, log = self.get_vibe_system_prompt(self.model_df.sample(n=num_samples_per_disguise))
            self.system_prompts.append(system_prompt)
            self.logs.append(log)
        self.vibe_prompt, log = self.aggregate_vibe_system_prompts(self.system_prompts)
        self.logs.append(log)
        self.log_to_wandb(self.logs)

    def log_to_wandb(self, logs: List[dict]) -> None:
        """
        Log the logs to wandb.
        """
        wandb.log({"logs": wandb.Table(dataframe=pd.DataFrame(logs))})
    
    def get_vibe_system_prompt(self, df_batch: pd.DataFrame) -> List[str]:
        proposer_systems_prompt = """You are a machine learning engineer tasked with constructing a systems prompt that will cause Model 1 to act like Model 2 such that users will not be able to tell the difference. To do this, you will be given examples of prompts and responses from both models. 

When analyzing the differences between the models, consider the following factors in detail (this is not an exhaustive list, but a good starting point):

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

If there are also behaviors that Model 1 has that are unique, you should instruct the model to not do these things (e.g. "Do not start your response with 'Hello, I'm ...'")

This system prompt will be given to Model 1 directly, so it should be written in the style of a helpful assistant and should not mention model 2 as it will not know anything about it. The prompt should be specific about the exact behaviors the model should have and may include examples of the desired behavior or specific phrases that should be used.

Think through your response step by step, then respond with your response in the following format:
Analysis: [your analysis of the differences between the two models]
System prompt: You are a helpful assistant ... [your proposed system prompt for Model 1 to cause it to act like Model 2]
"""
        prompt_str = "\n\n".join([f"### Prompt: {row['prompt']}\n\n### Model 1:\n{row['model_response_model']}\n\n### Model 2:\n{row['model_response_disguise']}" for _, row in df_batch.iterrows()])
        prompt = f"Examples:\n{prompt_str}"
        response = completion(
            model="openai/gpt-4o",
            messages=[{"content": proposer_systems_prompt, "role": "system"}, {"content": prompt, "role": "user"}],
            caching=True
        )
        response = response["choices"][0]["message"]["content"]
        parsed_response = response.split("System prompt:")[1].strip()
        logs = {"input": prompt, "output": response, "parsed_output": parsed_response}
        return parsed_response, logs
    
    def aggregate_vibe_system_prompts(self, systems_prompts: List[str]) -> str:
        """
        Aggregate the vibe system prompts into a single system prompt.
        """
        systems_prompt = """You are a machine learning engineer tasked with constructing a systems prompt that will cause Model 1 to act like Model 2 such that users will not be able to tell the difference.  To do this you will be given a list of different systems prompts that have been proposed for this task. Your job is to aggregate these into a single system prompt that will be given to Model 1. This prompt should generalize to new prompts, but be specific about the exact behaviors the model should have. You may include examples of the exact behavior you want the model to have or specific phrases that the model should use if you think it will help.
This system prompt will be given to Model 1 directly, so it should be written in the style of a helpful assistant and should not mention model 2 as it will not know anything about it. Remeber to be as specific as possible in your system prompt.

Respond with your response in the following format:
Analysis: [your analysis of the differences between the two models]
System prompt: [the final system prompt for Model 1 to cause it to act like Model 2]
"""
        prompt = f"System prompts:\n\n{systems_prompts}"
        print(prompt)
        response = completion(
            model="openai/gpt-4o",
            messages=[{"content": systems_prompt, "role": "system"}, {"content": prompt, "role": "user"}],
            caching=True
        )
        response = response["choices"][0]["message"]["content"]
        parsed_response = response.split("System prompt:")[1].strip()
        logs = {"input": prompt, "output": response, "parsed_output": parsed_response}
        return parsed_response, logs
    
    def forward(self, prompt: str) -> str:
        return [{"role": "system", "content": self.vibe_prompt}, {"role": "user", "content": prompt}]
