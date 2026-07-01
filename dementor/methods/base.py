import hashlib
import os
from typing import Dict, List

import pandas as pd

# get_token_count resolution is centralized here so every disguise method can
# import it from this module instead of repeating the fallback in each header.
try:
    from dementor.data_utils import get_token_count
except ImportError:
    from utils import get_token_count


class MethodBase:
    """Base class for all disguise methods."""

    def __init__(self, model: str, disguise_as: str) -> None:
        self.model = model
        self.disguise_as = disguise_as

    def make_disguise_prompt(self, examples: pd.DataFrame, prompt: str) -> str:
        """Build a simple disguise system prompt from example rows."""
        disguise_systems_prompt = (
            "Your task is to answer the following prompt in the style of another AI assistant. "
            "I will provide you with examples of responses to help you understand the style. "
            "Your goal is to mimic formatting, tone, level of detail, and phrasing — not to copy content exactly.\n\n"
            "## Examples from the other AI assistant:\n{examples}\n\n"
            "Do not echo these instructions; only answer the prompt in the desired style."
        )
        formatted_examples = ""
        examples = examples.reset_index(drop=True)
        for i, row in examples.iterrows():
            formatted_examples += f"### Example {i + 1}:\n"
            formatted_examples += f"prompt: {row['prompt']}\n"
            formatted_examples += f"response: {row['target_response']}\n\n"
        return disguise_systems_prompt.format(examples=formatted_examples)

    def _random_state(self, prompt: str, *labels: str) -> int | None:
        """Deterministic per-prompt random seed shared by all methods.

        Each method passes its own label suffix(es) so the derived seeds stay
        byte-identical to the per-method implementations they replace. The
        hashed payload is ``"{seed}:{prompt}[:{label}...]"``.
        """
        if self.seed is None:
            return None
        payload = ":".join([str(self.seed), prompt, *labels]).encode("utf-8")
        return int(hashlib.sha256(payload).hexdigest()[:8], 16)

    @staticmethod
    def _enable_litellm_cache() -> None:
        """Enable litellm response caching once (idempotent)."""
        import litellm
        if not hasattr(litellm, 'cache') or litellm.cache is None:
            litellm.cache = litellm.Cache()

    def _run_analyzer(self, analysis_prompt: str) -> str:
        """Resolve analyzer config from env and call the cached completion,
        falling back to standard ``litellm.completion`` when ``cache_llm`` is
        unavailable. Returns the completion's message content."""
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
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.0,
                api_base=analysis_api_base,
                api_key=analysis_api_key,
            )
        except ImportError:
            # Fallback to standard litellm
            from litellm import completion
            response = completion(
                model=analysis_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.0,
                api_base=analysis_api_base,
                api_key=analysis_api_key,
            )
        return response.choices[0].message.content

    def _wrap(self, system_prompt: str, prompt: str) -> List[Dict[str, str]]:
        """Wrap a system prompt + user prompt into chat messages, applying the
        gemma single-turn template when the target model is a gemma model."""
        if "gemma" in self.model.lower():
            formatted_prompt = f"""<start_of_turn>user
{system_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted_prompt}]
        else:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

    def forward(self, prompt: str) -> str:
        """Default no-op forward; subclasses should override and return chat messages."""
        return prompt
