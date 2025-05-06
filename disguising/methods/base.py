import pandas as pd
from transformers import AutoTokenizer
from utils import get_token_count

class MethodBase:
    """
    Base class for all methods.
    """
    def __init__(self, model: str, disguise_as: str) -> None:
        self.model = model
        self.disguise_as = disguise_as
    
    def forward(self, prompt: str) -> str:
        """
        Given a prompt, return the disguised prompt to use for the model.
        """
        return prompt
    
class RandomSampleDisguise(MethodBase):
    """
    Disguise the prompt by randomly sampling from the base model's responses.
    """
    def __init__(self, model: str, disguise_as: str, num_samples: int = 1000, num_samples_per_disguise: int = 5, seed: int = None) -> None:
        """
        Num_samples is the number of samples to use from the base model, used to read in the responses from the model-responses/base folder.
        Num_samples_per_disguise is the number of samples to use for each disguise.
        Seed is the seed to use for the random sampling, if it is None, then no seed is used and the samples will be different each time.
        """
        super().__init__(model, disguise_as)
        self.num_samples = num_samples
        self.num_samples_per_disguise = num_samples_per_disguise
        self.seed = seed
        self.disguise_df = pd.read_csv(f"disguising/model-responses/base/{self.disguise_as.replace('/', '_')}_responses-{self.num_samples}.csv")
        self.disguise_df["token_length"] = self.disguise_df["model_response"].apply(lambda x: get_token_count(x))
        # truncate the responses to 256 tokens
        self.disguise_df["model_response"] = self.disguise_df["model_response"].apply(lambda x: f"{x[:256]}...(truncated)" if get_token_count(x) > 256 else x)
        
    @staticmethod
    def make_disguise_prompt(examples, prompt):
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
        disguise_df_sample = self.disguise_df.sample(n=self.num_samples_per_disguise, random_state=self.seed)
        disguise_prompt = self.make_disguise_prompt(disguise_df_sample, prompt)
        return disguise_prompt

class JustNameIt(MethodBase):
    """
    Just ask the model to act like the other model.
    """
    def __init__(self, model: str, disguise_as: str) -> None:
        super().__init__(model, disguise_as)

    def forward(self, prompt: str) -> str:
        return f"Answer the following prompt in the style of {self.disguise_as}:\n{prompt}"
