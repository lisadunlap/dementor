from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI as OpenAIClient
from dementor.training.openai import collect_job_metrics
from dementor.training.plots import plot_convergence, series_from_metric_rows


def sync_openai_metrics(job_map: Dict[str, Path]) -> None:
    client = OpenAIClient()

    # Resolve job ids (fine_tuned_model -> job id)
    pending = {model_id: None for model_id in job_map}
    jobs = client.fine_tuning.jobs.list(limit=50)
    for job in jobs.data:
        ft_model = getattr(job, "fine_tuned_model", None)
        if ft_model in pending:
            pending[ft_model] = job.id

    for model_id, job_id in pending.items():
        if not job_id:
            print(f"[sync] Skipping {model_id}: job id not found in recent history.")
            continue

        metrics = collect_job_metrics(client, job_id)
        if not metrics:
            print(f"[sync] No metrics returned for job {job_id}")
            continue

        series = series_from_metric_rows(metrics, "train_loss", label=model_id)
        if not series:
            print(f"[sync] No train_loss entries for job {job_id}")
            continue

        out_dir = job_map[model_id] / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_name = "openai_sft.png" if "sft_" in str(job_map[model_id]) else "openai_dpo.png"
        plot_path = out_dir / plot_name
        plot_convergence([series], plot_path, title=f"{model_id} convergence")
        print(f"[sync] Wrote {plot_path}")


if __name__ == "__main__":
    job_map = {
        "ft:gpt-4.1-mini-2025-04-14:uc-berkeley-prof-trevor-darrell-group::Ca7onZrx": Path(
            "data/results/workflows/sft_gpt-4.1-mini_as_llama-3.1-8b-instruct"
        ),
        "ft:gpt-4.1-mini-2025-04-14:uc-berkeley-prof-trevor-darrell-group::Ca7nFPxf": Path(
            "data/results/workflows/dpo_gpt-4.1-mini_as_llama-3.1-8b-instruct"
        ),
    }
    sync_openai_metrics(job_map)
