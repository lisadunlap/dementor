from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from workflows.pipeline import DPOWorkflowConfig, OpenAIDPOParams, run_dpo_workflow
from workflows.dpo import PreferenceDatasetConfig


def main() -> None:
    cfg = DPOWorkflowConfig(
        provider="openai",
        dataset=PreferenceDatasetConfig(
            dataset_csv=Path("data/results/tinker_dpo/gsm8k_gpt-4.1-mini_preference_pairs.csv"),
            prompt_column="prompt",
            chosen_column="chosen_response",
            rejected_column="rejected_response",
            train_size=300,
            eval_size=200,
            seed=42,
        ),
        output_dir=Path("data/results/workflows/dpo_gpt-4.1-mini_as_llama-3.1-8b-instruct"),
        openai=OpenAIDPOParams(
            model="gpt-4.1-mini-2025-04-14",
            epochs=1,
            batch_size=25,
            beta=0.1,
        ),
    )
    run_dpo_workflow(cfg)


if __name__ == "__main__":
    main()
