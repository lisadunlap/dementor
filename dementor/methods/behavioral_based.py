"""
Behavioral-based disguise method.
Identifies and replicates the target model's communication behaviors (personality, tone, patterns).
"""
import logging
import os
import hashlib
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
        from dementor.methods.base import MethodBase
    except ImportError:
        from base import MethodBase

try:
    from dementor.data_utils import get_token_count
except ImportError:
    from utils import get_token_count


class BehavioralBasedSystemPrompting(MethodBase):
    """
    Behavioral system prompting: identifies and replicates the behavioral essence of the target model.
    Focuses on personality, tone, and interaction patterns.
    """
    
    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None, 
                 use_examples: bool = True, num_examples: int = 3, seed: int | None = None):
        super().__init__(model, disguise_as)
        self.disguise_df = disguise_df.copy() if disguise_df is not None else None
        self.use_examples = use_examples
        self.num_examples = num_examples
        self.seed = seed
        self.behavior_rules = None
        
        if disguise_df is not None:
            self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(get_token_count)
            self._generate_behavior_rules()
    
    def _generate_behavior_rules(self):
        """Generate behavioral rules capturing the essence of target model communication."""
        behavior_samples = self.disguise_df.sample(min(12, len(self.disguise_df)), random_state=self.seed)
        
        behavior_prompt = f"""Analyze the deep communication essence and behavior of this AI model. This is critical for disguise purposes - you need to capture the CORE of how this model communicates, not just surface patterns.

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
        for _, row in behavior_samples.iterrows():
            behavior_prompt += f"Q: {row['prompt']}\nA: {row['target_response'][:500]}{'...' if len(row['target_response']) > 500 else ''}\n\n"
        
        behavior_prompt += f"""
Based on this analysis, create comprehensive behavioral rules for {self.disguise_as} that capture:

1. The model's core communication personality (2-3 key traits)
2. Specific behavioral patterns and quirks
3. How it structures and delivers information
4. Its emotional and intellectual approach to interactions
5. Key linguistic patterns or preferences

Make this actionable - write it as instructions that would allow another AI to authentically embody this model's communication essence, not just mimic surface features."""

        try:
            # Default to GPT-4.1-mini for analysis unless overridden
            analysis_model = os.getenv("ANALYSIS_MODEL", "openai/gpt-4.1-mini")
            analysis_api_base = os.getenv("ANALYSIS_API_BASE")
            analysis_api_key = os.getenv(
                "ANALYSIS_API_KEY",
                os.getenv("ORIGINAL_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
            )
            # Use cached completion for persistent caching
            try:
                import sys
                # Add parent directory to path for importing cache_llm
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from dementor.cache_llm import cached_completion
                response = cached_completion(
                    model=analysis_model,
                    messages=[{"role": "user", "content": behavior_prompt}],
                    temperature=0.0,
                    api_base=analysis_api_base,
                    api_key=analysis_api_key,
                )
                self.behavior_rules = response.choices[0].message.content
            except ImportError:
                # Fallback to standard litellm
                response = completion(
                    model=analysis_model,
                    messages=[{"role": "user", "content": behavior_prompt}],
                    temperature=0.0,
                    api_base=analysis_api_base,
                    api_key=analysis_api_key,
                )
                self.behavior_rules = response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Failed to generate behavioral rules: {e}")
            self.behavior_rules = f"Respond with the characteristic communication style and deep personality essence of {self.disguise_as}."

    def _random_state(self, prompt: str) -> int | None:
        if self.seed is None:
            return None
        payload = f"{self.seed}:{prompt}:behavioral".encode("utf-8")
        return int(hashlib.sha256(payload).hexdigest()[:8], 16)
    
    def forward(self, prompt: str) -> List[Dict[str, str]]:
        """Generate behavioral-based system prompt."""
        base_instruction = f"You are {self.disguise_as}. Embody this personality and communication style:"
        
        if self.behavior_rules:
            system_prompt = f"{base_instruction}\n\n{self.behavior_rules}"
        else:
            system_prompt = f"{base_instruction}\n\nRespond in the distinctive style and personality of {self.disguise_as}."
        
        if self.use_examples and self.disguise_df is not None:
            df = self.disguise_df
            if "prompt" in df.columns:
                df = df[df["prompt"] != prompt]
            sample_size = min(self.num_examples, len(df))
            examples = (
                df.sample(sample_size, random_state=self._random_state(prompt))
                if sample_size > 0 else df.head(0)
            )
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
