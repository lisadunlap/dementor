import pandas as pd
from typing import List
from litellm import completion, embedding
import numpy as np
import wandb
from sklearn.cluster import KMeans

from transformers import AutoTokenizer
from tqdm import tqdm
try:
    from ..utils import get_token_count
except ImportError:
    # Fallback for different import contexts
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
#         disguise_prompt  = """Your task is to answer the following prompt in the style of another AI assistant. I will provide you with examples of responses from the other AI assistant to different prompts to help you understand the style. Your goal is to mimic the formatting, tone, level of detail, and phrasing – not to copy content exactly.
# ## Prompt to answer:
# {prompt}

# ## Examples from the other AI assistant:
# {examples}

# Do not respond to my instructions above (e.g. "Here is the prompt in the style you want") in your output, only respond with the answer to the prompt in the desired style. Here is the prompt to answer (repeated for clarity):
# prompt: {prompt}

# response:
# """
        disguise_systems_prompt  = """Your task is to answer the following prompt in the style of another AI assistant. I will provide you with examples of responses from the other AI assistant to different prompts to help you understand the style. Your goal is to mimic the formatting, tone, level of detail, and phrasing – not to copy content exactly.

## Examples from the other AI assistant:
{examples}

Do not respond to my instructions above (e.g. "Here is the prompt in the style you want") in your output, only respond with the answer to the prompt in the desired style.
"""
        formatted_examples = ""
        examples = examples.reset_index(drop=True)
        for i, row in examples.iterrows():
            formatted_examples += f"### Example {i + 1}:\n"
            formatted_examples += f"prompt: {row['prompt']}\n"
            formatted_examples += f"response (length: {row['token_length']} total tokens): {row['target_response']}\n\n"
        # # strip the first " from the prompt if it exists at the beginning (this is a hack to get around errors with the original prompts, i think it was a bug)
        # prompt = prompt.lstrip('"')
        return disguise_systems_prompt.format(examples=formatted_examples,)
    
    def forward(self, prompt: str) -> str:
        """
        Given a prompt, return the disguised prompt to use for the model.
        """
        return prompt
        
class RandomSampleDisguise(MethodBase):
    """
    Disguise the prompt by randomly sampling from the base model's responses.
    """
    def __init__(self, model: str, disguise_as: str, num_samples_per_disguise: int = 5, seed: int = None,
                 disguise_df: pd.DataFrame = None) -> None:
        """
        Num_samples_per_disguise is the number of samples to use for each disguise.
        Seed is the seed to use for the random sampling, if it is None, then no seed is used and the samples will be different each time.
        """
        super().__init__(model, disguise_as)
        self.num_samples_per_disguise = num_samples_per_disguise
        self.seed = seed
        self.disguise_df = disguise_df.copy()
        self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(lambda x: get_token_count(x))
        # truncate the responses to 256 tokens

    def forward(self, prompt: str) -> str:
        """
        Given a prompt, return the disguised prompt to use for the model.
        """
        disguise_df_sample = self.disguise_df.sample(n=self.num_samples_per_disguise, random_state=self.seed)
        disguise_df_sample["target_response"] = disguise_df_sample["target_response"].apply(lambda x: f"{x[:256]}...(truncated)" if get_token_count(x) > 256 else x)
        disguise_prompt = self.make_disguise_prompt(disguise_df_sample, prompt)
        
        # Format for Gemma models
        if "gemma" in self.model.lower():
            # For Gemma models, we need to format the prompt using their specific template
            formatted_prompt = f"""<start_of_turn>user
{disguise_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted_prompt}]
        else:
            # Standard chat format for other models
            return [{"role": "system", "content": disguise_prompt}, {"role": "user", "content": prompt}]
    
class JustNameIt(MethodBase):
    """
    Just ask the model to act like the other model.
    """
    def __init__(self, model: str, disguise_as: str) -> None:
        super().__init__(model, disguise_as)

    def forward(self, prompt: str) -> str:
        return [{"role": "system", "content": f"You are a helpful assisant. Given a prompt, you will answer it in the style of {self.disguise_as}."}, {"role": "user", "content": prompt}]
    
class VibeBasedDisguise(MethodBase):
    """
    Disguise the prompt by using the vibe of the other model.
    """
    def __init__(self, model: str, disguise_as: str, num_samples_per_disguise: int = 5, seed: int = None,
                 disguise_df: pd.DataFrame = None) -> None:
        super().__init__(model, disguise_as)
        self.num_samples_per_disguise = num_samples_per_disguise
        self.seed = seed
        self.model_tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.system_prompts, self.logs = [], []
        self.model_df = disguise_df.copy()
        for i in range(3):
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

If there are also behaviors that Model 1 has that are unique, you should instruct the model to not do these things (e.g. "Do not start your response with 'Hello, I'm ...'")

This system prompt will be given to Model 1 directly, so it should be written in the style of a helpful assistant and should not mention model 2 as it will not know anything about it. The prompt should be specific about the exact behaviors the model should have and include examples of the desired behavior or specific phrases that should be used. Instructions should be specific and detailed; things like "be helpful" or "be friendly" are not specific enough.

Think through your response step by step, then respond with your response in the following format EXACTLY:
Analysis: [your analysis of the differences between the two models]

System prompt: [your proposed system prompt for Model 1 to cause it to act like Model 2]
"""
        prompt_str = "\n\n".join([f"# Prompt: {row['prompt']}\n\n# Model 1:\n{row['source_response']}\n\n# Model 2:\n{row['target_response']}\n-----------------" for _, row in df_batch.iterrows()])
        prompt = f"Examples:\n{prompt_str}"
        response = completion(
            model="openai/gpt-4o",
            messages=[{"content": proposer_systems_prompt, "role": "system"}, {"content": prompt, "role": "user"}],
            caching=True,
            max_tokens=4096
        )
        response = response["choices"][0]["message"]["content"]
        try:
            for header in ["System prompt:", "**System Prompt:**", "system prompt:"]:
                try:
                    parsed_response = response.split(header)[1].strip()
                    break
                except IndexError:
                    continue
            else:
                raise Exception("Could not find system prompt section in response")
        except Exception as e:
            print(f"Error parsing vibe system prompts: {e}")
            print(response)
            exit()
        logs = {"input": prompt, "output": response, "parsed_output": parsed_response}
        return parsed_response, logs
    
    def aggregate_vibe_system_prompts(self, systems_prompts: List[str]) -> str:
        """
        Aggregate the vibe system prompts into a single system prompt.
        """
        systems_prompt = """You are a machine learning engineer tasked with constructing a systems prompt that will cause Model 1 to act like Model 2 such that users will not be able to tell the difference. To do this you will be given a list of different systems prompts that have been proposed for this task. Your job is to aggregate these into a single system prompt that will be given to Model 1. This prompt should generalize to new prompts, but be specific about the exact behaviors the model should have. You should include examples of the exact behavior you want the model to have or specific phrases that the model should use if you think it will help. These examples should be taken from the systems prompts you are given.
This system prompt will be given to Model 1 directly, so it should be written in the style of a helpful assistant and should not mention model 2 as it will not know anything about it. Remeber to be as specific as possible in your system prompt, things like "be helpful" or "be friendly" are not specific enough. Remeber that you should leave as much detail from the system prompts you are given as possible in your final system prompt.

Respond with your response in the following format:
Analysis: [your analysis of the differences between the two models]
System prompt: [the final system prompt for Model 1 to cause it to act like Model 2]
"""
        prompt = f"System prompts:\n\n{systems_prompts}"
        print(prompt)
        try:
            response = completion(
                model="openai/gpt-4o",
                messages=[{"content": systems_prompt, "role": "system"}, {"content": prompt, "role": "user"}],
                caching=True,
                max_tokens=4096
            )
            response = response["choices"][0]["message"]["content"]
            print(response)
            try:
                parsed_response = response.split("System prompt:")[1].strip()
            except Exception as e:
                print(f"Error parsing vibe system prompts: {e}")
                exit()
            logs = {"input": prompt, "output": response, "parsed_output": parsed_response}
        except Exception as e:
            print(f"Error aggregating vibe system prompts: {e}")
            exit()
        return parsed_response, logs
    
    def forward(self, prompt: str) -> str:
        return [{"role": "system", "content": self.vibe_prompt}, {"role": "user", "content": prompt}]
    
class VibeBasedDisguiseOneSided(MethodBase):
    """
    Disguise the prompt by using the vibe of the other model.
    """
    def __init__(self, model: str, disguise_as: str, num_samples_per_disguise: int = 10, seed: int = None,
                 disguise_df: pd.DataFrame = None) -> None:
        super().__init__(model, disguise_as)
        self.num_samples_per_disguise = num_samples_per_disguise
        self.seed = seed
        self.model_tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.system_prompts, self.logs = [], []
        self.model_df = disguise_df.copy()
        for i in range(1):
            system_prompt, log = self.get_vibe_system_prompt(self.model_df.sample(n=num_samples_per_disguise))
            self.system_prompts.append(system_prompt)
            self.logs.append(log)
        # self.vibe_prompt, log = self.aggregate_vibe_system_prompts(self.system_prompts)
        self.vibe_prompt = self.system_prompts[0]
        self.logs.append(log)
        self.log_to_wandb(self.logs)

    def log_to_wandb(self, logs: List[dict]) -> None:
        """
        Log the logs to wandb.
        """
        wandb.log({"logs": wandb.Table(dataframe=pd.DataFrame(logs))})
    
    def get_vibe_system_prompt(self, df_batch: pd.DataFrame) -> List[str]:
        proposer_systems_prompt = """You are a machine learning engineer tasked with constructing a systems prompt that will cause my model to act like a target model such that users will not be able to tell the difference. To do this, you will be given examples of prompts and responses from the target model. 

Consider the following factors in detail to create a systems prompt that will cause my model to act like the target model (this is not an exhaustive list, but a good starting point):

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

This system prompt should be specific about the exact behaviors the model should have and include examples of the desired behavior or specific phrases that should be used. Instructions should be specific and detailed; things like "be helpful" or "be friendly" are not specific enough.

Think through your response step by step, then respond with your response in the following format:
Analysis: [your analysis of the differences between the two models]
System prompt: [your proposed system prompt for my model to cause it to act like the target model]
"""
        prompt_str = "\n\n".join([f"# Prompt: {row['prompt']}\n\n# Target Model:\n{row['target_response']}\n-----------------" for _, row in df_batch.iterrows()])
        prompt = f"Examples:\n{prompt_str}"
        response = completion(
            model="openai/gpt-4o",
            messages=[{"content": proposer_systems_prompt, "role": "system"}, {"content": prompt, "role": "user"}],
            caching=True,
            max_tokens=4096
        )
        response = response["choices"][0]["message"]["content"]
        parsed_response = response.split("System prompt:")[1].strip()
        logs = {"input": prompt, "output": response, "parsed_output": parsed_response}
        return parsed_response, logs
    
    def aggregate_vibe_system_prompts(self, systems_prompts: List[str]) -> str:
        """
        Aggregate the vibe system prompts into a single system prompt.
        """
        systems_prompt = """You are a machine learning engineer tasked with constructing a systems prompt that will cause Model 1 to act like Model 2 such that users will not be able to tell the difference. To do this you will be given a list of different systems prompts that have been proposed for this task. Your job is to aggregate these into a single system prompt that will be given to Model 1. This prompt should generalize to new prompts, but be specific about the exact behaviors the model should have. You should include examples of the exact behavior you want the model to have or specific phrases that the model should use if you think it will help. These examples should be taken from the systems prompts you are given.
This system prompt will be given to my model directly, so it should be written in the style of a helpful assistant and should not mention the target model as it will not know anything about it. Remeber to be as specific as possible in your system prompt, things like "be helpful" or "be friendly" are not specific enough. Remeber that you should leave as much detail from the system prompts you are given as possible in your final system prompt.

Respond with your response in the following format:
Analysis: [your analysis of the differences between the two models]
System prompt: [the final system prompt for my model to cause it to act like the target model]
"""
        prompt = f"System prompts:\n\n{systems_prompts}"
        print(prompt)
        try:
            response = completion(
                model="openai/gpt-4o",
                messages=[{"content": systems_prompt, "role": "system"}, {"content": prompt, "role": "user"}],
                caching=True,
                max_tokens=4096
            )
            response = response["choices"][0]["message"]["content"]
            print(response)
            parsed_response = response.split("System prompt:")[1].strip()
            logs = {"input": prompt, "output": response, "parsed_output": parsed_response}
        except Exception as e:
            print(f"Error aggregating vibe system prompts: {e}")
            exit()
        return parsed_response, logs
    
    def forward(self, prompt: str) -> str:
        return [{"role": "system", "content": self.vibe_prompt}, {"role": "user", "content": prompt}]
    