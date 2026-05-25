"""
Contrastive disguise method.
Learns distinctive features of target vs source and encodes them as rules.
"""
import logging
import os
from typing import List, Dict
import json
import sys
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
            # Use GPT-4.1-mini by default for contrastive analysis (override with ANALYSIS_MODEL)
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
                from scripts.cache_llm import cached_completion
                response = cached_completion(
                    model=analysis_model,
                    messages=[{"role": "user", "content": contrastive_prompt}],
                    temperature=0.0,
                    api_base=analysis_api_base,
                    api_key=analysis_api_key,
                )
                self.contrastive_features = response.choices[0].message.content
            except ImportError:
                # Fallback to standard litellm
                response = completion(
                    model=analysis_model,
                    messages=[{"role": "user", "content": contrastive_prompt}],
                    temperature=0.0,
                    api_base=analysis_api_base,
                    api_key=analysis_api_key,
                )
                self.contrastive_features = response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Failed to generate contrastive features: {e}")
            self.contrastive_features = "Unable to generate contrastive analysis."
    
    def _choose_good_bad_examples_via_llm(
        self,
        current_prompt: str,
        n_good: int,
        n_bad: int,
        *,
        pool_size: int = 10,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Use analysis model to select GOOD (target) and BAD (source) examples consistently.

        Returns: (good_df, bad_df), empty frames if selection fails.
        """
        try:
            if self.disguise_df is None or self.source_df is None:
                return (
                    self.disguise_df.head(0) if self.disguise_df is not None else pd.DataFrame(),
                    self.source_df.head(0) if self.source_df is not None else pd.DataFrame(),
                )

            tgt_pool = self.disguise_df
            src_pool = self.source_df
            if 'prompt' in tgt_pool.columns:
                tgt_pool = tgt_pool[tgt_pool['prompt'] != current_prompt]
            if 'prompt' in src_pool.columns:
                src_pool = src_pool[src_pool['prompt'] != current_prompt]

            tgt_pool = tgt_pool.sample(n=min(pool_size, len(tgt_pool))) if len(tgt_pool) > 0 else tgt_pool.head(0)
            src_pool = src_pool.sample(n=min(pool_size, len(src_pool))) if len(src_pool) > 0 else src_pool.head(0)

            def _row_repr(idx: int, q: str, a: str) -> str:
                q_disp = str(q)[:160] if isinstance(q, str) else ""
                a_disp = str(a)[:400] if isinstance(a, str) else ""
                return f"[{idx}] Q: {q_disp}...\nA: {a_disp}...\n"

            tgt_block = "".join(
                _row_repr(int(i), row.get('prompt', ''), row.get('target_response', row.get('model_response', '')))
                for i, row in tgt_pool.iterrows()
            )
            src_block = "".join(
                _row_repr(int(i), row.get('prompt', ''), row.get('model_response', row.get('target_response', '')))
                for i, row in src_pool.iterrows()
            )

            analysis_model = os.getenv("ANALYSIS_MODEL", "openai/gpt-4.1-mini")
            analysis_api_base = os.getenv("ANALYSIS_API_BASE")
            analysis_api_key = os.getenv(
                "ANALYSIS_API_KEY",
                os.getenv("ORIGINAL_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
            )
            selection_prompt = (
                "You are selecting examples to teach a model to mimic the TARGET style and avoid the SOURCE style.\n"
                "From the TARGET_CANDIDATES, select the K_good indices that BEST illustrate the TARGET's distinctive style.\n"
                "From the SOURCE_CANDIDATES, select the K_bad indices that BEST illustrate traits we want to avoid.\n"
                "Return strict JSON ONLY with keys 'good_indices' and 'bad_indices' as integer arrays. No commentary.\n\n"
                f"K_good = {max(0, int(n_good))}\nK_bad = {max(0, int(n_bad))}\n\n"
                "TARGET_CANDIDATES (index-tagged):\n" + tgt_block + "\n"
                "SOURCE_CANDIDATES (index-tagged):\n" + src_block + "\n"
                "JSON: {\"good_indices\": [...], \"bad_indices\": [...]}"
            )

            try:
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from scripts.cache_llm import cached_completion
                sel_response = cached_completion(
                    model=analysis_model,
                    messages=[{"role": "user", "content": selection_prompt}],
                    temperature=0.0,
                    api_base=analysis_api_base,
                    api_key=analysis_api_key,
                )
                content = sel_response.choices[0].message.content
            except ImportError:
                sel_response = completion(
                    model=analysis_model,
                    messages=[{"role": "user", "content": selection_prompt}],
                    temperature=0.0,
                    api_base=analysis_api_base,
                    api_key=analysis_api_key,
                )
                content = sel_response.choices[0].message.content

            data = json.loads(content)
            good_idx = [int(i) for i in (data.get('good_indices') or []) if int(i) in tgt_pool.index]
            bad_idx = [int(i) for i in (data.get('bad_indices') or []) if int(i) in src_pool.index]

            good_df = tgt_pool.loc[good_idx] if len(good_idx) else tgt_pool.head(0)
            bad_df = src_pool.loc[bad_idx] if len(bad_idx) else src_pool.head(0)
            return good_df, bad_df
        except Exception:
            return (
                self.disguise_df.head(0) if self.disguise_df is not None else pd.DataFrame(),
                self.source_df.head(0) if self.source_df is not None else pd.DataFrame(),
            )

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

        # Append GOOD/BAD example blocks selected via analysis model (consistent prompt)
        n_good = (self.num_examples + 1) // 2
        n_bad = max(0, self.num_examples - n_good)
        good_examples, bad_examples = self._choose_good_bad_examples_via_llm(prompt, n_good, n_bad)

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

        if "gemma" in self.model.lower():
            formatted_prompt = f"""<start_of_turn>user
{system_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted_prompt}]
        else:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
