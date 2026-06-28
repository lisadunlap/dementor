"""
Stylistic disguise method focusing on measurable surface features.
"""
from typing import List, Dict
import hashlib
import numpy as np
import pandas as pd

try:
    from .base import MethodBase
except ImportError:
    try:
        from dementor.methods.base import MethodBase
    except ImportError:
        from base import MethodBase

# get_token_count is defined in scripts/utils.py (top-level module)
try:
    from dementor.data_utils import get_token_count
except ImportError:
    from utils import get_token_count
try:
    from .utils.stylistic_analysis import (
        has_markdown, contains_list, contains_header, contains_code,
        contains_question, contains_exclamation, average_sentence_length
    )
except ImportError:
    try:
        from dementor.methods.utils.stylistic_analysis import (
            has_markdown, contains_list, contains_header, contains_code,
            contains_question, contains_exclamation, average_sentence_length
        )
    except ImportError:
        from utils.stylistic_analysis import (
            has_markdown, contains_list, contains_header, contains_code,
            contains_question, contains_exclamation, average_sentence_length
        )


class StylisticSystemPrompting(MethodBase):
    """Analyze and enforce surface-level stylistic patterns."""
    
    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None,
                 num_examples: int = 3, use_examples: bool = True, seed: int | None = None):
        super().__init__(model, disguise_as)
        self.disguise_df = disguise_df.copy() if disguise_df is not None else None
        self.num_examples = num_examples
        self.use_examples = use_examples
        self.seed = seed
        self.stylistic_rules = None
        
        if disguise_df is not None:
            self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(get_token_count)
            self._generate_stylistic_rules()

    def _random_state(self, prompt: str) -> int | None:
        if self.seed is None:
            return None
        payload = f"{self.seed}:{prompt}:stylistic".encode("utf-8")
        return int(hashlib.sha256(payload).hexdigest()[:8], 16)
    
    def _generate_stylistic_rules(self):
        responses = self.disguise_df['target_response'].tolist()
        style_analysis = {
            'avg_length': np.mean([len(r.split()) for r in responses]),
            'uses_markdown': np.mean([has_markdown(r) for r in responses]) > 0.3,
            'uses_bullets': np.mean([contains_list(r) for r in responses]) > 0.3,
            'uses_code': np.mean([contains_code(r) for r in responses]) > 0.3,
            'uses_headers': np.mean([contains_header(r) for r in responses]) > 0.3,
            'uses_questions': np.mean([contains_question(r) for r in responses]) > 0.3,
            'uses_exclamations': np.mean([contains_exclamation(r) for r in responses]) > 0.3,
            'avg_sentence_len': np.mean([average_sentence_length(r) for r in responses]),
        }
        guidelines = []
        if style_analysis['avg_length'] < 100:
            guidelines.append("Keep responses concise and brief (under 100 words typically)")
        elif style_analysis['avg_length'] > 300:
            guidelines.append("Provide detailed, comprehensive responses (typically 300+ words)")
        else:
            guidelines.append(f"Aim for moderate length responses (around {int(style_analysis['avg_length'])} words)")
        if style_analysis['uses_markdown']:
            guidelines.append("Use markdown formatting including headers (# ## ###)")
        if style_analysis['uses_bullets']:
            guidelines.append("Frequently use bullet points and lists for organization")
        if style_analysis['uses_code']:
            guidelines.append("Include code blocks and inline code formatting when relevant")
        if style_analysis['uses_headers']:
            guidelines.append("Use headers and section breaks to structure responses")
        if style_analysis['uses_questions']:
            guidelines.append("Ask clarifying questions or rhetorical questions")
        if style_analysis['uses_exclamations']:
            guidelines.append("Use exclamation points for enthusiasm and emphasis")
        self.stylistic_rules = f"""Surface-level stylistic patterns for {self.disguise_as}:

{chr(10).join('• ' + g for g in guidelines)}

Key measurable characteristics:
- Average response length: {int(style_analysis['avg_length'])} words
- Average sentence length: {style_analysis['avg_sentence_len']:.1f} words
- Uses markdown formatting: {'Yes' if style_analysis['uses_markdown'] else 'No'}
- Uses bullet points: {'Yes' if style_analysis['uses_bullets'] else 'No'}
- Includes code blocks: {'Yes' if style_analysis['uses_code'] else 'No'}
- Uses headers: {'Yes' if style_analysis['uses_headers'] else 'No'}
"""
    
    def forward(self, prompt: str) -> List[Dict[str, str]]:
        base_instruction = f"You are {self.disguise_as}. Match these specific stylistic patterns:"
        if self.stylistic_rules:
            system_prompt = f"{base_instruction}\n\n{self.stylistic_rules}"
        else:
            system_prompt = f"{base_instruction}\n\nFocus on matching the formatting, structure, and surface-level style patterns of {self.disguise_as}."
        if self.use_examples and self.disguise_df is not None:
            df = self.disguise_df
            if "prompt" in df.columns:
                df = df[df["prompt"] != prompt]
            sample_size = min(self.num_examples, len(df))
            examples = (
                df.sample(sample_size, random_state=self._random_state(prompt))
                if sample_size > 0 else df.head(0)
            )
            system_prompt += f"\n\nReference examples showing these patterns:\n"
            for _, row in examples.iterrows():
                truncated_response = row['target_response'][:300] + "..." if len(row['target_response']) > 300 else row['target_response']
                system_prompt += f"Q: {row['prompt'][:100]}...\nA: {truncated_response}\n\n"
        system_prompt += f"\nRespond to the following prompt matching these stylistic patterns exactly. Focus on surface features: formatting, length, structure, and presentation style."
        if "gemma" in self.model.lower():
            formatted_prompt = f"""<start_of_turn>user
{system_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted_prompt}]
        else:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
