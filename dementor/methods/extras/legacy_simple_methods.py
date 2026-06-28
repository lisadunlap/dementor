"""
Legacy/simple disguise methods retained for reference.

These implementations are deprecated in favor of the streamlined methods under
dementor/methods/*. They are not used by the current pipeline.
"""
from typing import List
import pandas as pd

from ..base import MethodBase
from dementor.data_utils import get_token_count


class RandomSampleDisguise(MethodBase):
    def __init__(self, model: str, disguise_as: str, num_samples_per_disguise: int = 5, seed: int = None,
                 disguise_df: pd.DataFrame = None) -> None:
        super().__init__(model, disguise_as)
        self.num_samples_per_disguise = num_samples_per_disguise
        self.seed = seed
        self.disguise_df = disguise_df.copy() if disguise_df is not None else pd.DataFrame()
        if not self.disguise_df.empty:
            self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(lambda x: get_token_count(x))

    def forward(self, prompt: str):
        if self.disguise_df.empty:
            system = f"You are {self.disguise_as}. Respond in this style."
            return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        disguise_df_sample = self.disguise_df.sample(n=min(self.num_samples_per_disguise, len(self.disguise_df)),
                                                    random_state=self.seed)
        disguise_df_sample["target_response"] = disguise_df_sample["target_response"].apply(
            lambda x: f"{x[:256]}...(truncated)" if get_token_count(x) > 256 else x)
        disguise_prompt = self.make_disguise_prompt(disguise_df_sample, prompt)
        return [{"role": "system", "content": disguise_prompt}, {"role": "user", "content": prompt}]


class JustNameIt(MethodBase):
    def forward(self, prompt: str):
        system = f"You are a helpful assistant. Answer in the style of {self.disguise_as}."
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


class VibeBasedDisguise(MethodBase):
    def __init__(self, model: str, disguise_as: str, num_samples_per_disguise: int = 5, seed: int = None,
                 disguise_df: pd.DataFrame = None) -> None:
        super().__init__(model, disguise_as)
        self.num_samples_per_disguise = num_samples_per_disguise
        self.seed = seed
        self.model_df = disguise_df.copy() if disguise_df is not None else pd.DataFrame()
        self.vibe_prompt = f"Respond with the characteristic communication style and personality of {self.disguise_as}."

    def forward(self, prompt: str):
        return [{"role": "system", "content": self.vibe_prompt}, {"role": "user", "content": prompt}]

