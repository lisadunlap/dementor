import pandas as pd


class MethodBase:
    """Base class for all disguise methods."""

    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None) -> None:
        self.model = model
        self.disguise_as = disguise_as
        self.disguise_df = disguise_df

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

    def forward(self, prompt: str) -> list:
        """Default forward returns a simple chat message list."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

