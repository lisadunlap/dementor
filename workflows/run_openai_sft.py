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

from workflows.pipeline import SFTWorkflowConfig, run_sft_workflow
from workflows.openai import OpenAISFTJobConfig
from workflows.tinker import SFTDatasetConfig

cfg = SFTWorkflowConfig(
    provider="openai",
    dataset=SFTDatasetConfig(
        train_csv=Path("data/model-responses/gsm8k/splits/seed42/train_300/openai_gpt-4.1-mini_responses_train300_seed42.csv"),
        eval_csv=Path("data/model-responses/gsm8k/splits/seed42/eval_200/openai_gpt-4.1-mini_responses_eval200_seed42.csv"),
        prompt_column="prompt",
        completion_column="model_response",
    ),
    output_dir=Path("data/results/workflows/sft_gpt-4.1-mini_as_llama-3.1-8b-instruct"),
    openai=OpenAISFTJobConfig(
        model="gpt-4.1-mini-2025-04-14",
        system_prompt="You are a patient math tutor.",
        epochs=3,
        batch_size=25,
    ),
)
run_sft_workflow(cfg)
