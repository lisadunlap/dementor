"""
Random sampling disguise method.
Samples k target examples and builds a clean system prompt.
"""
from typing import List, Dict
import pandas as pd

try:
    from .base import MethodBase, get_token_count
except ImportError:
    try:
        from dementor.methods.base import MethodBase, get_token_count
    except ImportError:
        from base import MethodBase, get_token_count


class RandomSamplingSystemPrompting(MethodBase):
    """Random sampling with system prompting (baseline)."""
    
    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None,
                 num_samples: int = 5, seed: int = None, max_tokens_per_example: int = 256):
        super().__init__(model, disguise_as)
        self.disguise_df = disguise_df.copy() if disguise_df is not None else None
        self.num_samples = num_samples
        self.seed = seed
        self.max_tokens_per_example = max_tokens_per_example
        
        if disguise_df is not None:
            self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(get_token_count)

    def forward(self, prompt: str) -> List[Dict[str, str]]:
        if self.disguise_df is None or len(self.disguise_df) == 0:
            system_prompt = f"You are {self.disguise_as}. Respond in the style and manner of {self.disguise_as}."
        else:
            # Exclude the current prompt to avoid leakage
            df = self.disguise_df
            if 'prompt' in df.columns:
                df = df[df['prompt'] != prompt]
            sample_size = min(self.num_samples, len(df))
            examples = (
                df.sample(n=sample_size, random_state=self._random_state(prompt))
                if sample_size > 0 else df.head(0)
            )
            
            system_prompt = f"""You are {self.disguise_as}. Study these examples of {self.disguise_as}'s responses and mimic the style, tone, formatting, and approach:

Examples:
"""
            for _, row in examples.iterrows():
                response = row['target_response']
                if get_token_count(response) > self.max_tokens_per_example:
                    response = response[:self.max_tokens_per_example*4] + "...(truncated)"
                system_prompt += f"\nQ: {row['prompt']}\nA: {response}\n"
            system_prompt += f"\n\nNow respond to the following prompt in the same style as {self.disguise_as}. Match the formatting, tone, level of detail, and approach shown in the examples."

        return self._wrap(system_prompt, prompt)
