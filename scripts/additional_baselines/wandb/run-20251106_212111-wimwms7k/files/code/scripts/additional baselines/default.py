#!/usr/bin/env python3
"""
Compute the “default” baseline: score source-model responses directly against
target-model responses on the same prompts (no disguise applied).

Example:
    python scripts/additional baselines/default.py \
        --source data/model-responses/call_center/meta-llama_Meta-Llama-3-8B-Instruct-500.csv \
        --target data/model-responses/call_center/gpt-4.1-mini.csv \
        --output-dir data/results/call_center/comparisons/baseline/default
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd  # type: ignore
import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.scorer import score_model_comparison

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _resolve_output_path(source: str, target: str, output_dir: str | None, run_name: str | None) -> Path:
    """
    Build the merged-comparison CSV path for the baseline run.

    Args:
        source: Path to the source-model CSV (`prompt`, `model_response`).
        target: Path to the target-model CSV (`prompt`, `model_response`).
        output_dir: Optional directory to place artifacts in.
        run_name: Optional explicit filename (without extension).

    Returns:
        Absolute `Path` where the merged comparison CSV should be written.
    """
    src = Path(source)
    tgt = Path(target)
    base_dir = Path(output_dir) if output_dir else src.parent / "baseline_default"
    base_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{run_name}.csv" if run_name else f"{src.stem}_vs_{tgt.stem}.csv"
    return (base_dir / filename).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score source vs. target responses without disguise.")
    parser.add_argument("--source", required=True, help="CSV with source-model outputs (columns: prompt, model_response).")
    parser.add_argument("--target", required=True, help="CSV with target-model outputs (columns: prompt, model_response).")
    parser.add_argument("--output-dir", default=None, help="Directory for merged + scored artifacts (default: <source_dir>/baseline_default).")
    parser.add_argument("--run-name", default=None, help="Optional filename stem for the merged comparison CSV.")
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini", help="LLM judge (prefix with openai/ to route via OpenAI API).")
    parser.add_argument("--heuristics-only", action="store_true", help="Skip LLM judging and emit heuristic features only.")
    parser.add_argument("--openai-api-base", default=None, help="Override OPENAI_API_BASE for judge routing.")
    parser.add_argument("--openai-api-key", default=None, help="Override OPENAI_API_KEY for judge routing.")
    args = parser.parse_args()

    merged_path = _resolve_output_path(args.source, args.target, args.output_dir, args.run_name)

    wandb_run = None
    wandb_name = args.run_name or "additional baselines"
    wandb_config = {**vars(args), "baseline": "default"}
    try:
        wandb.login()
        wandb_run = wandb.init(
            project="dementor-disguise",
            name=wandb_name,
            group="additional-baselines",
            job_type="baseline-default",
            config=wandb_config,
        )
    except Exception:
        wandb_run = None

    scored_df = None
    try:
        scored_df = score_model_comparison(
            source_file=args.source,
            target_file=args.target,
            output_file=str(merged_path),
            judge_model=args.judge_model,
            heuristics_only=args.heuristics_only,
            openai_api_base=args.openai_api_base,
            openai_api_key=args.openai_api_key,
        )
    finally:
        if wandb_run is not None and scored_df is not None:
            preview_limit = 1000
            table_df = scored_df if len(scored_df) <= preview_limit else scored_df.head(preview_limit).copy()
            try:
                wandb_run.log({"scored_table_preview": wandb.Table(dataframe=table_df)})
            except Exception:
                pass

            scored_dir = merged_path.parent / "scores" / merged_path.stem
            metrics_path = scored_dir / "scored_metrics.csv"

            if metrics_path.exists():
                try:
                    metrics_df = pd.read_csv(metrics_path)
                    for row in metrics_df.itertuples(index=False):
                        try:
                            value = float(row.value)
                        except Exception:
                            value = row.value
                        wandb_run.summary[str(row.metric)] = value
                except Exception as exc:
                    wandb_run.log({"metrics_parse_error": str(exc)}, step=0)

            artifact = wandb.Artifact(
                name=f"default-baseline-{merged_path.stem}",
                type="baseline-results",
                metadata={"baseline": "default"},
            )
            try:
                artifact.add_file(str(merged_path), name="merged_comparison.csv")
            except Exception:
                pass
            try:
                scored_path = scored_dir / "scored.csv"
                if scored_path.exists():
                    artifact.add_file(str(scored_path), name="scored.csv")
            except Exception:
                pass
            if metrics_path.exists():
                try:
                    artifact.add_file(str(metrics_path), name="scored_metrics.csv")
                except Exception:
                    pass

            wandb_run.log_artifact(artifact)

        if wandb_run is not None:
            wandb.finish()

    print(f"[baseline-default] merged comparison CSV: {merged_path}")
    print(f"[baseline-default] scored CSV: {merged_path.parent / 'scores' / merged_path.stem / 'scored.csv'}")


if __name__ == "__main__":
    main()