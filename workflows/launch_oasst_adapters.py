"""
Launch SFT, self-SFT, and DPO LoRA adapters for the OpenAssistant (oasst1) dataset.

Adapter matrix (source = Llama-3.1-8B-Instruct):
  SFT     : Llama → Qwen3.6-27B        ×3 seeds  (3 jobs)
  SFT     : Llama → Nemotron-Nano       ×3 seeds  (3 jobs)
  SFT     : Llama → gpt-oss-20b         ×3 seeds  (3 jobs)
  self-SFT: Llama → Llama               ×3 seeds  (3 jobs)
  DPO     : Llama → Qwen3.6-27B        ×3 seeds  (3 jobs, depends on SFT)
  DPO     : Llama → Nemotron-Nano       ×3 seeds  (3 jobs, depends on SFT)
  DPO     : Llama → gpt-oss-20b         ×3 seeds  (3 jobs, depends on SFT)
  Total: 21 Tinker jobs

DPO preference pairs: chosen = target response, rejected = Llama baseline response.
DPO is initialized from the corresponding SFT checkpoint (stacked SFT→DPO ladder).

Usage:
  export TINKER_API_KEY='tml-...'
  python -m workflows.launch_oasst_adapters               # all stages
  python -m workflows.launch_oasst_adapters --stage sft   # SFT + self-SFT only
  python -m workflows.launch_oasst_adapters --stage dpo   # DPO only (SFT must exist)
  python -m workflows.launch_oasst_adapters --dry-run     # print config, no launch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe to call from worker threads
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
TRAIN_DIR = DATA_DIR / "model-responses" / "openasisstant" / "train"
TEST_DIR = DATA_DIR / "model-responses" / "openasisstant" / "test"
PREF_DIR = DATA_DIR / "results" / "openasisstant" / "dpo_preference_pairs"
REGISTRY = DATA_DIR / "tinker_adapters.json"

SOURCE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RENDERER = "llama3"

TARGETS: List[Dict[str, str]] = [
    {
        "model_id": "Qwen/Qwen3.6-27B",
        "slug": "qwen3.6-27b",
        "train_csv": "train_Qwen_Qwen3.6-27B_500.csv",
        "test_csv": "test_Qwen_Qwen3.6-27B_1000.csv",
    },
    {
        "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "slug": "nemotron-nano",
        "train_csv": "train_nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_500.csv",
        "test_csv": "test_nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_1000.csv",
    },
    {
        "model_id": "openai/gpt-oss-20b",
        "slug": "gpt-oss-20b",
        "train_csv": "train_openai_gpt-oss-20b_500.csv",
        "test_csv": "test_openai_gpt-oss-20b_1000.csv",
    },
]

LLAMA_TRAIN_CSV = TRAIN_DIR / "train_meta-llama_Llama-3.1-8B-Instruct_500.csv"
LLAMA_TEST_CSV = TEST_DIR / "test_meta-llama_Llama-3.1-8B-Instruct_1000.csv"

SEEDS = [42]
EVAL_SIZE = 50        # rows from test split used for Tinker's in-training eval loop
TRAIN_SIZE = 450   # 450 train + 50 eval = 500 total preference pairs

# SFT / self-SFT hyperparams (match experiment plan defaults)
SFT_BATCH_SIZE = 16
SFT_EPOCHS = 3
SFT_LR = 1e-4
SFT_LORA_RANK = 32
SFT_MAX_SAMPLE_TOKENS = 256

# DPO hyperparams
DPO_BATCH_SIZE = 16
DPO_EPOCHS = 3
DPO_LR = 1e-4
DPO_BETA = 0.1
DPO_LORA_RANK = 32
DPO_MAX_LENGTH = 4096

# Prompt template for conversational (oasst1 is free-form chat)
PROMPT_TEMPLATE = "{prompt}"
COMPLETION_TEMPLATE = "{completion}"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _read_registry() -> dict:
    if REGISTRY.exists():
        try:
            return json.loads(REGISTRY.read_text())
        except Exception:
            pass
    return {}


def _adapter_name(target_slug: str, stage: str, seed: int) -> str:
    if stage == "self_sft":
        return f"oasst_llama-3.1-8b_self_sft_seed{seed}"
    return f"oasst_llama-3.1-8b_as_{target_slug}_{stage}_seed{seed}"


def _get_sft_checkpoint(target_slug: str, seed: int) -> Optional[str]:
    """Return the SFT checkpoint path (state URI) from the registry."""
    reg = _read_registry()
    name = _adapter_name(target_slug, "sft", seed)
    entry = reg.get(name, {})
    # prefer downloadable state checkpoint for DPO; fall back to sampler path
    return entry.get("checkpoint_path") or entry.get("path")


# ---------------------------------------------------------------------------
# Eval CSV preparation
# ---------------------------------------------------------------------------

def _make_eval_csv(test_csv: Path, target_slug: str, seed: int) -> Path:
    """Take the first EVAL_SIZE rows of the test CSV for Tinker's in-training eval."""
    out_dir = DATA_DIR / "results" / "openasisstant" / "sft_eval_slices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{target_slug}_seed{seed}_{EVAL_SIZE}.csv"
    if out_path.exists():
        return out_path
    df = pd.read_csv(test_csv)
    df.head(EVAL_SIZE).to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# DPO preference pair preparation
# ---------------------------------------------------------------------------

def _make_preference_csv(target: Dict[str, str]) -> Path:
    """
    Build the DPO preference CSV for one target.
    chosen = target response, rejected = Llama baseline response.
    Keyed and merged on 'prompt'.
    """
    PREF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREF_DIR / f"oasst_llama_as_{target['slug']}_pref.csv"
    if out_path.exists():
        print(f"  [pref] Using existing {out_path.name}")
        return out_path

    target_df = pd.read_csv(TRAIN_DIR / target["train_csv"])[["prompt", "model_response"]].rename(
        columns={"model_response": "chosen_response"}
    )
    llama_df = pd.read_csv(LLAMA_TRAIN_CSV)[["prompt", "model_response"]].rename(
        columns={"model_response": "rejected_response"}
    )
    merged = target_df.merge(llama_df, on="prompt", how="inner")
    if len(merged) < TRAIN_SIZE:
        print(f"  WARNING: only {len(merged)} matched preference pairs for {target['slug']}")
    merged[["prompt", "chosen_response", "rejected_response"]].to_csv(out_path, index=False)
    print(f"  [pref] Wrote {len(merged)} preference pairs → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# SFT job runner
# ---------------------------------------------------------------------------

def run_sft_job(
    *,
    train_csv: Path,
    test_csv: Path,
    target_slug: str,
    stage: str,           # "sft" or "self_sft"
    seed: int,
    dry_run: bool,
) -> Tuple[str, Optional[str]]:
    """Submit one SFT job to Tinker and return (adapter_name, sampler_path)."""
    from workflows.pipeline import SFTWorkflowConfig, TinkerSFTParams, run_sft_workflow
    from workflows.tinker import SFTDatasetConfig, EvaluationConfig

    name = _adapter_name(target_slug, stage, seed)
    eval_csv = _make_eval_csv(test_csv, target_slug, seed)
    output_dir = DATA_DIR / "results" / "openasisstant" / f"tinker_{stage}" / name

    dataset = SFTDatasetConfig(
        train_csv=train_csv,
        eval_csv=eval_csv,
        prompt_column="prompt",
        completion_column="model_response",
    )
    params = TinkerSFTParams(
        base_model=SOURCE_MODEL,
        batch_size=SFT_BATCH_SIZE,
        epochs=SFT_EPOCHS,
        learning_rate=SFT_LR,
        prompt_template=PROMPT_TEMPLATE,
        completion_template=COMPLETION_TEMPLATE,
        evaluation_config=EvaluationConfig(max_sample_tokens=SFT_MAX_SAMPLE_TOKENS),
        weights_name=name,
        registry_path=REGISTRY,
        seed=seed,
        lora_kwargs={"rank": SFT_LORA_RANK},
    )
    config = SFTWorkflowConfig(
        provider="tinker",
        dataset=dataset,
        output_dir=output_dir,
        tinker=params,
    )

    if dry_run:
        print(f"  [dry-run] {name}  train={train_csv.name}  seed={seed}")
        return name, None

    print(f"  [launch] {name}")
    result = run_sft_workflow(config)
    sampler_path = result.artifacts.get("sampler_path")
    print(f"  [done]   {name}  sampler_path={sampler_path}")
    return name, sampler_path


# ---------------------------------------------------------------------------
# DPO job runner
# ---------------------------------------------------------------------------

def run_dpo_job(
    *,
    preference_csv: Path,
    target_slug: str,
    seed: int,
    sft_checkpoint: Optional[str],
    dry_run: bool,
) -> Tuple[str, Optional[str]]:
    """Submit one DPO job to Tinker and return (adapter_name, log_path)."""
    from workflows.pipeline import DPOWorkflowConfig, TinkerDPOParams, run_dpo_workflow
    from workflows.dpo import PreferenceDatasetConfig

    name = _adapter_name(target_slug, "dpo", seed)
    log_path = DATA_DIR / "results" / "openasisstant" / "tinker_dpo" / name / "logs"
    output_dir = DATA_DIR / "results" / "openasisstant" / "tinker_dpo" / name

    dataset = PreferenceDatasetConfig(
        dataset_csv=preference_csv,
        prompt_column="prompt",
        chosen_column="chosen_response",
        rejected_column="rejected_response",
        train_size=TRAIN_SIZE,
        eval_size=EVAL_SIZE,
        seed=seed,
    )
    tinker_params = TinkerDPOParams(
        model_name=SOURCE_MODEL,
        reference_model_name=SOURCE_MODEL,
        renderer_name=RENDERER,
        log_path=log_path,
        learning_rate=DPO_LR,
        dpo_beta=DPO_BETA,
        num_epochs=DPO_EPOCHS,
        batch_size=DPO_BATCH_SIZE,
        max_length=DPO_MAX_LENGTH,
        lora_rank=DPO_LORA_RANK,
        save_every=50,
        load_checkpoint_path=sft_checkpoint,
    )
    config = DPOWorkflowConfig(
        provider="tinker",
        dataset=dataset,
        output_dir=output_dir,
        tinker=tinker_params,
    )

    if dry_run:
        print(f"  [dry-run] {name}  pref={preference_csv.name}  sft_ckpt={sft_checkpoint}  seed={seed}")
        return name, None

    print(f"  [launch] {name}  sft_ckpt={sft_checkpoint}")
    result = run_dpo_workflow(config)
    log = str(result.artifacts.get("log_path", ""))
    print(f"  [done]   {name}  log={log}")
    return name, log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch oasst SFT / self-SFT / DPO Tinker jobs.")
    p.add_argument("--stage", choices=["sft", "dpo", "all"], default="all",
                   help="Which adapter stages to run (default: all).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print job configs without submitting to Tinker.")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                   help="Seeds to run (default: 1 2 3).")
    p.add_argument("--workers", type=int, default=256,
                   help="Max concurrent Tinker jobs per phase (default: 256).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dry_run and not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY is not set. export TINKER_API_KEY='tml-...' first.")

    # ---- prepare preference pair CSVs (cheap; always done up front) --------
    pref_csvs: Dict[str, Path] = {}
    if args.stage in ("dpo", "all"):
        print("\n[prep] Building DPO preference pair CSVs …")
        for target in TARGETS:
            pref_csvs[target["slug"]] = _make_preference_csv(target)

    # ---- Phase 1: SFT + self-SFT -------------------------------------------
    sft_results: Dict[Tuple[str, int], Optional[str]] = {}   # (slug, seed) → sampler_path

    if args.stage in ("sft", "all"):
        sft_jobs = []
        # cross-SFT
        for target in TARGETS:
            for seed in args.seeds:
                sft_jobs.append({
                    "train_csv": TRAIN_DIR / target["train_csv"],
                    "test_csv": TEST_DIR / target["test_csv"],
                    "target_slug": target["slug"],
                    "stage": "sft",
                    "seed": seed,
                })
        # self-SFT
        for seed in args.seeds:
            sft_jobs.append({
                "train_csv": LLAMA_TRAIN_CSV,
                "test_csv": LLAMA_TEST_CSV,
                "target_slug": "self",
                "stage": "self_sft",
                "seed": seed,
            })

        print(f"\n[Phase 1] Submitting {len(sft_jobs)} SFT / self-SFT jobs "
              f"(max {args.workers} concurrent) …")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_key: Dict[Future, Tuple[str, int]] = {}
            for job in sft_jobs:
                key = (job["target_slug"], job["seed"])
                f = pool.submit(
                    run_sft_job,
                    train_csv=job["train_csv"],
                    test_csv=job["test_csv"],
                    target_slug=job["target_slug"],
                    stage=job["stage"],
                    seed=job["seed"],
                    dry_run=args.dry_run,
                )
                future_to_key[f] = key

            for fut in as_completed(future_to_key):
                key = future_to_key[fut]
                try:
                    name, sampler_path = fut.result()
                    sft_results[key] = sampler_path
                    print(f"  ✓ {name}")
                except Exception as exc:
                    print(f"  ✗ {key}: {exc}")
                    sft_results[key] = None

    # ---- Phase 2: DPO (stacked on SFT) -------------------------------------
    if args.stage in ("dpo", "all"):
        dpo_jobs = []
        for target in TARGETS:
            for seed in args.seeds:
                # prefer SFT checkpoint from this run; fall back to registry
                sft_ckpt = sft_results.get((target["slug"], seed)) or _get_sft_checkpoint(
                    target["slug"], seed
                )
                if sft_ckpt is None and not args.dry_run:
                    print(f"  WARNING: no SFT checkpoint for {target['slug']} seed={seed} — "
                          f"DPO will train from base model.")
                dpo_jobs.append({
                    "preference_csv": pref_csvs[target["slug"]],
                    "target_slug": target["slug"],
                    "seed": seed,
                    "sft_checkpoint": sft_ckpt,
                })

        print(f"\n[Phase 2] Submitting {len(dpo_jobs)} DPO jobs "
              f"(max {args.workers} concurrent) …")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_name: Dict[Future, str] = {}
            for job in dpo_jobs:
                name = _adapter_name(job["target_slug"], "dpo", job["seed"])
                f = pool.submit(
                    run_dpo_job,
                    preference_csv=job["preference_csv"],
                    target_slug=job["target_slug"],
                    seed=job["seed"],
                    sft_checkpoint=job["sft_checkpoint"],
                    dry_run=args.dry_run,
                )
                future_to_name[f] = name

            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    _, log = fut.result()
                    print(f"  ✓ {name}  log={log}")
                except Exception as exc:
                    print(f"  ✗ {name}: {exc}")

    print("\nAll done." if not args.dry_run else "\nDry-run complete — no jobs submitted.")


if __name__ == "__main__":
    main()
