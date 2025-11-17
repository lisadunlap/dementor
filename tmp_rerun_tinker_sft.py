from pathlib import Path

from workflows.pipeline import SFTWorkflowConfig, TinkerSFTParams, run_sft_workflow
from workflows.tinker import SFTDatasetConfig


def main() -> None:
    cfg = SFTWorkflowConfig(
        provider="tinker",
        dataset=SFTDatasetConfig(
            train_csv=Path(
                "data/model-responses/gsm8k/splits/seed42/train_300/openai_gpt-4.1-mini_responses_train300_seed42.csv"
            ),
            eval_csv=Path(
                "data/model-responses/gsm8k/splits/seed42/eval_200/openai_gpt-4.1-mini_responses_eval200_seed42.csv"
            ),
            prompt_column="prompt",
            completion_column="model_response",
        ),
        output_dir=Path("data/results/workflows/sft_llama-3.1-8b-instruct_as_gpt-4.1-mini"),
        tinker=TinkerSFTParams(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            weights_name="gsm8k_llama-3.1-8b-instruct",
            batch_size=16,
            epochs=6,
        ),
    )
    run_sft_workflow(cfg)


if __name__ == "__main__":
    main()
