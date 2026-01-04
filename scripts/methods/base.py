from typing import Optional

import pandas as pd


class MethodBase:
    """Base class for all disguise methods."""

    def __init__(self, model: str, disguise_as: str) -> None:
        self.model = model
        self.disguise_as = disguise_as
        self._last_display_system_prompt: Optional[str] = None

    def _placeholder_example_block(self, count: int, label: str = "Example") -> str:
        if count <= 0:
            return ""
        lines = []
        for i in range(count):
            idx = i + 1
            lines.append(f"### {label} {idx}:")
            lines.append(f"prompt: <random_example_prompt_{idx}>")
            lines.append(f"response: <random_example_response_{idx}>")
            lines.append("")
        return "\n".join(lines)

    def summarize_system_prompt(self, actual_prompt: str) -> str:
        """Return a display-friendly prompt (default: actual prompt)."""
        return self._last_display_system_prompt or actual_prompt

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
        actual = disguise_systems_prompt.format(examples=formatted_examples)
        placeholder_block = self._placeholder_example_block(len(examples))
        self._last_display_system_prompt = disguise_systems_prompt.format(examples=placeholder_block or "No examples provided.")
        return actual

    def forward(self, prompt: str) -> str:
        """Default no-op forward; subclasses should override and return chat messages."""
        return prompt
