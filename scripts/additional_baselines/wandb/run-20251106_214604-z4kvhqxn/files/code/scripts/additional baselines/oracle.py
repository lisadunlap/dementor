#!/usr/bin/env python3
"""
Compute the “oracle” baseline: score one target-model generation against an
independent second generation from the same target model on matching prompts.

Example:
    python scripts/additional baselines/oracle.py \
        --reference data/model-responses/call_center/gpt-4o-run1.csv \
        --comparison data/model-responses/call_center/gpt-4o-run2.csv \
        --output-dir data/results/call_center/comparisons/baseline/oracle
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


def _resolve_output_path(reference: str, comparison: str, output_dir: str | None, run_name: str | None) -> Path:
    """
    Build the merged-comparison CSV path for the oracle run.

    Args:
        reference: First target-generation CSV (`prompt`, `model_response`).
        comparison: Second target-generation CSV (`prompt`, `model_response`).
        output_dir: Optional directory to place artifacts in.
        run_name: Optional explicit filename (without extension).

    Returns:
        Absolute `Path` where the merged comparison CSV should be written.
    """
    ref = Path(reference)
    cmp_path = Path(comparison)
    base_dir = Path(output_dir) if output_dir else ref.parent / "baseline_oracle"
    base_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{run_name}.csv" if run_name else f"{ref.stem}_vs_{cmp_path.stem}.csv"
    return (base_dir / filename).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score two target-model generations against each other (oracle baseline).")
    parser.add_argument("--reference", required=True, help="Reference target CSV (columns: prompt, model_response).")
    parser.add_argument("--comparison", required=True, help="Independent target CSV to compare (columns: prompt, model_response).")
    parser.add_argument("--output-dir", default=None, help="Directory for merged + scored artifacts (default: <reference_dir>/baseline_oracle).")
    parser.add_argument("--run-name", default=None, help="Optional filename stem for the merged comparison CSV.")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to log/preview (default: 500).")
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini", help="LLM judge (prefix with openai/ to route via OpenAI API).")
    parser.add_argument("--heuristics-only", action="store_true", help="Skip LLM judging and emit heuristic features only.")
    parser.add_argument("--openai-api-base", default=None, help="Override OPENAI_API_BASE for judge routing.")
    parser.add_argument("--openai-api-key", default=None, help="Override OPENAI_API_KEY for judge routing.")
    args = parser.parse_args()

    merged_path = _resolve_output_path(args.reference, args.comparison, args.output_dir, args.run_name)

    wandb_run = None
    wandb_name = args.run_name or "additional baselines"
    wandb_config = {**vars(args), "baseline": "oracle"}
    try:
        wandb.login()
        wandb_run = wandb.init(
            project="dementor-disguise",
            name=wandb_name,
            group="additional-baselines",
            job_type="baseline-oracle",
            config=wandb_config,
        )
        wandb_run.summary["num_samples"] = args.num_samples
    except Exception:
        wandb_run = None

    scored_df = None
    try:
        scored_df = score_model_comparison(
            source_file=args.reference,
            target_file=args.comparison,
            output_file=str(merged_path),
            judge_model=args.judge_model,
            heuristics_only=args.heuristics_only,
            openai_api_base=args.openai_api_base,
            openai_api_key=args.openai_api_key,
        )
    finally:
        if wandb_run is not None and scored_df is not None:
            preview_limit = args.num_samples
            table_df = scored_df if len(scored_df) <= preview_limit else scored_df.head(preview_limit).copy()
            try:
                wandb_run.log({"scored_table_preview": wandb.Table(dataframe=table_df)})
            except Exception:
                pass

            scored_dir = merged_path.parent / "scores" / merged_path.stem
            metrics_path = scored_dir / "scored_metrics.csv"
            metrics_payload: dict[str, float | int | str] = {}

            if metrics_path.exists():
                try:
                    metrics_df = pd.read_csv(metrics_path)
                    for row in metrics_df.itertuples(index=False):
                        try:
                            value = float(row.value)
                        except Exception:
                            value = row.value
                        metric_name = str(row.metric)
                        metrics_payload[metric_name] = value
                        wandb_run.summary[metric_name] = value
                except Exception as exc:
                    wandb_run.log({"metrics_parse_error": str(exc)}, step=0)
            if metrics_payload:
                try:
                    wandb_run.log(metrics_payload, step=0)
                except Exception:
                    pass

            artifact = wandb.Artifact(
                name=f"oracle-baseline-{merged_path.stem}",
                type="baseline-results",
                metadata={"baseline": "oracle"},
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

    print(f"[baseline-oracle] merged comparison CSV: {merged_path}")
    print(f"[baseline-oracle] scored CSV: {merged_path.parent / 'scores' / merged_path.stem / 'scored.csv'}")


if __name__ == "__main__":
    main()