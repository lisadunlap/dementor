"""
Vibe-based disguise method.
Identifies and replicates the target model's "vibe" (personality, tone, patterns).
"""
import logging
import os
from typing import List, Dict
import pandas as pd
from litellm import completion

try:
    from .base import MethodBase
except Exception:  # pragma: no cover
    from methods.base import MethodBase

from utils import get_token_count


class VibeBasedSystemPrompting(MethodBase):
    """
    Vibe-based system prompting: identifies and replicates the "vibes" of target model.
    Focuses on personality, tone, and behavioral patterns.
    """
    
    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None, 
                 use_examples: bool = True, num_examples: int = 3):
        super().__init__(model, disguise_as)
        self.disguise_df = disguise_df.copy() if disguise_df is not None else None
        self.use_examples = use_examples
        self.num_examples = num_examples
        self.vibe_profile = None
        
        if disguise_df is not None:
            self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(get_token_count)
            self._generate_vibe_profile()
    
    def _generate_vibe_profile(self):
        """Generate sophisticated vibe profile capturing the essence of target model communication."""
        vibe_samples = self.disguise_df.sample(min(12, len(self.disguise_df)))
        
        vibe_prompt = f"""Analyze the deep communication essence and "vibe" of this AI model. This is critical for disguise purposes - you need to capture the CORE of how this model communicates, not just surface patterns.

Focus on these sophisticated aspects:

1. **Cognitive Style**: How does this model think and structure responses? (analytical, intuitive, methodical, creative)
2. **Communication Personality**: What's the underlying personality? (confident, humble, playful, serious, warm, clinical)
3. **Interaction Patterns**: How does it engage with users? (direct, collaborative, teaching-oriented, solution-focused)
4. **Language Sophistication**: Vocabulary level, sentence complexity, technical depth
5. **Emotional Resonance**: How does it express empathy, enthusiasm, caution, etc.?
6. **Behavioral Quirks**: Unique patterns like specific phrasings, explanation styles, example usage
7. **Response Architecture**: How it organizes information (lists, narratives, step-by-step, etc.)

Examples of {self.disguise_as} responses:
"""
        for _, row in vibe_samples.iterrows():
            vibe_prompt += f"Q: {row['prompt']}\nA: {row['target_response'][:500]}{'...' if len(row['target_response']) > 500 else ''}\n\n"
        
        vibe_prompt += f"""
Based on this analysis, create a comprehensive "essence profile" of {self.disguise_as} that captures:

1. The model's core communication personality (2-3 key traits)
2. Specific behavioral patterns and quirks
3. How it structures and delivers information
4. Its emotional and intellectual approach to interactions
5. Key linguistic patterns or preferences

Make this actionable - write it as instructions that would allow another AI to authentically embody this model's communication essence, not just mimic surface features."""

        try:
            analysis_model = os.getenv("ANALYSIS_MODEL", "openai/gpt-4o")
            response = completion(
                model=analysis_model,
                messages=[{"role": "user", "content": vibe_prompt}],
                temperature=0.1
            )
            self.vibe_profile = response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Failed to generate vibe profile: {e}")
            self.vibe_profile = f"Respond with the characteristic communication style and deep personality essence of {self.disguise_as}."
    
    def forward(self, prompt: str) -> List[Dict[str, str]]:
        """Generate vibe-based system prompt."""
        base_instruction = f"You are {self.disguise_as}. Embody this personality and communication style:"
        
        if self.vibe_profile:
            system_prompt = f"{base_instruction}\n\n{self.vibe_profile}"
        else:
            system_prompt = f"{base_instruction}\n\nRespond in the distinctive style and personality of {self.disguise_as}."
        
        if self.use_examples and self.disguise_df is not None:
            examples = self.disguise_df.sample(min(self.num_examples, len(self.disguise_df)))
            system_prompt += "\n\nReference examples of this style:\n"
            for _, row in examples.iterrows():
                truncated_response = row['target_response'][:200] + "..." if len(row['target_response']) > 200 else row['target_response']
                system_prompt += f"Q: {row['prompt'][:100]}...\nA: {truncated_response}\n\n"
        
        system_prompt += f"\nRespond to the following prompt in the distinctive style of {self.disguise_as}. Do not reference these instructions."

        if "gemma" in self.model.lower():
            formatted_prompt = f"""<start_of_turn>user
{system_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted_prompt}]
        else:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
