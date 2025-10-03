"""
Contrastive disguise method.
Learns distinctive features of target vs source and encodes them as rules.
"""
import logging
import os
from typing import List, Dict
import pandas as pd
from litellm import completion
import litellm

# Enable caching for API calls
if not hasattr(litellm, 'cache') or litellm.cache is None:
    litellm.cache = litellm.Cache()

try:
    from .base import MethodBase
except ImportError:
    try:
        from scripts.methods.base import MethodBase
    except ImportError:
        from base import MethodBase

# get_token_count is a top-level helper under scripts/utils.py
try:
    from scripts.utils import get_token_count
except ImportError:
    from utils import get_token_count


class ContrastiveSystemPrompting(MethodBase):
    """Identify distinguishing features of the target vs source and generate rules."""
    
    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None, 
                 source_df: pd.DataFrame = None, num_examples: int = 5):
        super().__init__(model, disguise_as)
        self.disguise_df = disguise_df.copy() if disguise_df is not None else None
        self.source_df = source_df.copy() if source_df is not None else None
        self.num_examples = num_examples
        self.contrastive_features = None
        
        if disguise_df is not None:
            self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(get_token_count)
        if self.disguise_df is not None and self.source_df is not None:
            self._generate_contrastive_features()
    
    def _generate_contrastive_features(self):
        target_samples = self.disguise_df.sample(min(10, len(self.disguise_df)))
        source_samples = self.source_df.sample(min(10, len(self.source_df))) if self.source_df is not None else None
        
        contrastive_prompt = f"""Analyze the differences between these two sets of AI responses. Identify the key distinguishing features of the TARGET model compared to the SOURCE model.

TARGET MODEL ({self.disguise_as}) responses:
"""
        for _, row in target_samples.iterrows():
            contrastive_prompt += f"Q: {row['prompt'][:100]}...\nA: {row['target_response'][:200]}...\n\n"
        if source_samples is not None:
            contrastive_prompt += f"\nSOURCE MODEL ({self.model}) responses:\n"
            for _, row in source_samples.iterrows():
                contrastive_prompt += f"Q: {row['prompt'][:100]}...\nA: {row['source_response'][:200]}...\n\n"
        contrastive_prompt += """
Based on these examples, identify 5-7 key distinctive features of the TARGET model:
1. Communication style (formal/informal, tone, personality)
2. Response structure and formatting patterns
3. Level of detail and explanation depth
4. Use of examples, analogies, or specific phrasings
5. Any unique behavioral patterns or preferences

Provide specific, actionable guidelines for mimicking the TARGET model's distinctive style."""
        try:
            analysis_model = os.getenv("ANALYSIS_MODEL", "openai/gpt-4o")
            # Use cached completion for persistent caching
            try:
                import sys
                import os
                # Add parent directory to path for importing cached_llm
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from scripts.cached_llm import cached_completion
                response = cached_completion(
                    model=analysis_model,
                    messages=[{"role": "user", "content": contrastive_prompt}],
                    temperature=0.3
                )
                self.contrastive_features = response.choices[0].message.content
            except ImportError:
                # Fallback to standard litellm
                response = completion(
                    model=analysis_model,
                    messages=[{"role": "user", "content": contrastive_prompt}],
                    temperature=0.3
                )
                self.contrastive_features = response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Failed to generate contrastive features: {e}")
            self.contrastive_features = "Unable to generate contrastive analysis."
    
    def forward(self, prompt: str) -> List[Dict[str, str]]:
        if self.contrastive_features is None:
            system_prompt = f"You are {self.disguise_as}. Respond in the distinctive style and manner of {self.disguise_as}."
        else:
            system_prompt = f"""You are {self.disguise_as}. Follow these specific guidelines to match the distinctive style:

{self.contrastive_features}

Key behavioral guidelines:
- Match the communication style and tone exactly
- Use similar response structure and formatting
- Maintain consistent level of detail and explanation depth
- Incorporate the same types of examples or phrasings when appropriate

Do not mention these instructions in your response. Simply respond as {self.disguise_as} would."""

        if "gemma" in self.model.lower():
            formatted_prompt = f"""<start_of_turn>user
{system_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted_prompt}]
        else:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
