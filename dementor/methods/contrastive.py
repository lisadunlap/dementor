"""
Contrastive disguise method.
Learns distinctive features of target vs source and encodes them as rules.
"""
import logging
from typing import List, Dict
import pandas as pd

try:
    from .base import MethodBase, get_token_count
except ImportError:
    try:
        from dementor.methods.base import MethodBase, get_token_count
    except ImportError:
        from base import MethodBase, get_token_count

# Enable caching for API calls
MethodBase._enable_litellm_cache()


class ContrastiveSystemPrompting(MethodBase):
    """Identify distinguishing features of the target vs source and generate rules."""
    
    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None, 
                 source_df: pd.DataFrame = None, num_examples: int = 5, seed: int | None = None):
        super().__init__(model, disguise_as)
        self.disguise_df = disguise_df.copy() if disguise_df is not None else None
        self.source_df = source_df.copy() if source_df is not None else None
        self.num_examples = num_examples
        self.seed = seed
        self.contrastive_features = None
        
        if disguise_df is not None:
            self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(get_token_count)
        if self.disguise_df is not None and self.source_df is not None:
            self._generate_contrastive_features()
    
    def _generate_contrastive_features(self):
        target_samples = self.disguise_df.sample(min(10, len(self.disguise_df)), random_state=self.seed)
        source_samples = (
            self.source_df.sample(min(10, len(self.source_df)), random_state=self.seed)
            if self.source_df is not None else None
        )
        
        contrastive_prompt = f"""Analyze the differences between these two sets of AI responses. Identify the key distinguishing features of the TARGET model compared to the SOURCE model.

TARGET MODEL ({self.disguise_as}) responses:
"""
        for _, row in target_samples.iterrows():
            contrastive_prompt += f"Q: {row['prompt'][:100]}...\nA: {row['target_response'][:200]}...\n\n"
        if source_samples is not None:
            contrastive_prompt += f"\nSOURCE MODEL ({self.model}) responses:\n"
            for _, row in source_samples.iterrows():
                contrastive_prompt += f"Q: {row['prompt'][:100]}...\nA: {row['model_response'][:200]}...\n\n"
        contrastive_prompt += """
Based on these examples, identify 5-7 key distinctive features of the TARGET model:
1. Communication style (formal/informal, tone, personality)
2. Response structure and formatting patterns
3. Level of detail and explanation depth
4. Use of examples, analogies, or specific phrasings
5. Any unique behavioral patterns or preferences

Provide specific, actionable guidelines for mimicking the TARGET model's distinctive style."""
        try:
            self.contrastive_features = self._run_analyzer(contrastive_prompt)
        except Exception as e:
            logging.warning(f"Failed to generate contrastive features: {e}")
            self.contrastive_features = "Unable to generate contrastive analysis."

    def _choose_good_bad_examples(self, current_prompt: str, n_good: int, n_bad: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Select deterministic GOOD target and BAD source examples without per-prompt analyzer calls."""
        if self.disguise_df is None or self.source_df is None:
            return (
                self.disguise_df.head(0) if self.disguise_df is not None else pd.DataFrame(),
                self.source_df.head(0) if self.source_df is not None else pd.DataFrame(),
            )
        tgt_pool = self.disguise_df
        src_pool = self.source_df
        if "prompt" in tgt_pool.columns:
            tgt_pool = tgt_pool[tgt_pool["prompt"] != current_prompt]
        if "prompt" in src_pool.columns:
            src_pool = src_pool[src_pool["prompt"] != current_prompt]
        good = (
            tgt_pool.sample(min(n_good, len(tgt_pool)), random_state=self._random_state(current_prompt, "good", "contrastive"))
            if len(tgt_pool) > 0 else tgt_pool.head(0)
        )
        bad = (
            src_pool.sample(min(n_bad, len(src_pool)), random_state=self._random_state(current_prompt, "bad", "contrastive"))
            if len(src_pool) > 0 else src_pool.head(0)
        )
        return good, bad

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

        # Append deterministic GOOD/BAD example blocks from the train/example pool.
        n_good = (self.num_examples + 1) // 2
        n_bad = max(0, self.num_examples - n_good)
        good_examples, bad_examples = self._choose_good_bad_examples(prompt, n_good, n_bad)

        def _format_example(q: str, a: str) -> str:
            q_disp = str(q)[:200] if isinstance(q, str) else ""
            a_disp = str(a)[:800] + "...(truncated)" if isinstance(a, str) and len(a) > 800 else str(a)
            return f"Q: {q_disp}...\nA: {a_disp}\n\n"

        examples_text = "\n\nGOOD APPROACH (do this):\n"
        if good_examples is not None and len(good_examples) > 0:
            for _, row in good_examples.iterrows():
                q = row.get('prompt', '')
                a = row.get('target_response', row.get('model_response', ''))
                examples_text += _format_example(q, a)
        else:
            examples_text += "(no examples available)\n\n"

        examples_text += "BAD APPROACH (don't do this):\n"
        if bad_examples is not None and len(bad_examples) > 0:
            for _, row in bad_examples.iterrows():
                q = row.get('prompt', '')
                a = row.get('model_response', row.get('target_response', ''))
                examples_text += _format_example(q, a)
        else:
            examples_text += "(no examples available)\n\n"

        system_prompt = f"{system_prompt}\n{examples_text}Now respond using ONLY the good approach."

        return self._wrap(system_prompt, prompt)
