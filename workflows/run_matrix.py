"""Matrix dispatcher for the AAAI conference experiment.

Drives the 4-model symmetric Tinker matrix:
- 4 sources × 3 cross targets × 3 train datasets × 3 seeds = 108 SFT jobs
- Same shape = 108 DPO jobs (Phase D — wired separately later)

Subcommands:
  generate-target-responses: generate baseline responses on TRAIN splits for use as SFT completions
  build-sft-data: join (train_prompt, target_response) pairs into SFT CSVs
  launch-sft: submit SFT jobs to Tinker (one per cell × seed)
  list-cells: print the matrix without launching anything
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATASETS = DATA / "datasets"
BASELINES_DIR = DATA / "model-responses" / "matrix_baselines"
SFT_DATA_DIR = DATA / "results" / "matrix" / "sft_data"
SFT_OUTPUT_DIR = DATA / "results" / "matrix" / "sft_runs"


# ============================================================================
# Matrix definition
# ============================================================================

MODELS: list[str] = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3.6-27B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "openai/gpt-oss-20b",
]

# Short slug for filenames and adapter names
MODEL_SLUG: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
    "Qwen/Qwen3.6-27B": "qwen3.6-27b",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nemotron-nano-30b-a3b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
}

# Per-model kwargs for tokenizer.apply_chat_template — disables thinking traces
# so the responses are comparable across models for fingerprint analysis.
CHAT_TEMPLATE_KWARGS: dict[str, dict] = {
    "meta-llama/Llama-3.1-8B-Instruct": {},
    "Qwen/Qwen3.6-27B": {"enable_thinking": False},
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {"enable_thinking": False},
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
}


def clean_response(model: str, raw: str) -> str:
    """Strip per-model chat-template artifacts to leave just the assistant message."""
    text = raw
    if model == "openai/gpt-oss-20b":
        # gpt-oss emits multi-channel: <|channel|>analysis<|message|>...<|end|>
        # <|start|>assistant<|channel|>final<|message|>ACTUAL<|return|>
        # Keep only the 'final' channel content.
        final_marker = "<|channel|>final<|message|>"
        if final_marker in text:
            text = text.split(final_marker, 1)[1]
    # Strip common chat tokens across all models
    for tok in (
        "<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|end|>",
        "<|return|>", "<|endoftext|>",
    ):
        text = text.replace(tok, "")
    return text.strip()

# Train splits per dataset (HumanEval is eval-only)
TRAIN_DATASETS: dict[str, Path] = {
    "gsm8k": DATASETS / "gsm8k" / "gsm8k_prompts_train_500_seed42.csv",
    "chatbot_arena": DATASETS / "chatbot_arena" / "chatbot_arena_prompts_train_500_seed42.csv",
    "writingprompts": DATASETS / "writingprompts" / "writingprompts_train_500_seed42.csv",
}

# Per-dataset templates for SFT
DATASET_TEMPLATES: dict[str, tuple[str, str]] = {
    "gsm8k": ("Question: {prompt}\nAnswer:", " {completion}\n"),
    "chatbot_arena": ("{prompt}", "{completion}"),
    "writingprompts": ("Prompt: {prompt}\nStory:", " {completion}\n"),
}

SEEDS: list[int] = [1, 2, 3]
LLAMA_RETIREMENT = "2026-06-12"


# ============================================================================
# Cell iteration with Llama-first priority
# ============================================================================


@dataclass(frozen=True)
class Cell:
    source: str
    target: str
    dataset: str
    seed: int

    @property
    def slug(self) -> str:
        return f"{self.dataset}_{MODEL_SLUG[self.source]}_as_{MODEL_SLUG[self.target]}_seed{self.seed}"

    @property
    def llama_critical(self) -> bool:
        return self.source == "meta-llama/Llama-3.1-8B-Instruct"


def iter_cells() -> Iterable[Cell]:
    """Iterate all (source, target, dataset, seed) cells. Llama-source cells first."""
    cells: list[Cell] = []
    for source in MODELS:
        for target in MODELS:
            if source == target:
                continue
            for dataset in TRAIN_DATASETS:
                for seed in SEEDS:
                    cells.append(Cell(source=source, target=target, dataset=dataset, seed=seed))
    # Llama-source cells first (hard deadline)
    cells.sort(key=lambda c: (not c.llama_critical, c.source, c.target, c.dataset, c.seed))
    return cells


# ============================================================================
# Target-response generation (Phase A.2 — for SFT training data)
# ============================================================================


def baseline_path(model: str, dataset: str) -> Path:
    return BASELINES_DIR / dataset / f"{MODEL_SLUG[model]}_train.csv"


def generate_target_responses(
    *,
    models: list[str],
    datasets: list[str],
    dry_run: bool = False,
    parallel: int = 1,
) -> int:
    """Generate one canonical response per (model, train_prompt) for SFT training data.

    Skips combos that already have a cache file with the right number of rows.
    Returns total number of new generations.

    parallel: number of in-flight Tinker sample() requests per model. 1 = sequential.
    """
    from dotenv import load_dotenv

    load_dotenv()
    import tinker
    from tinker import types
    from tinker_cookbook.model_info import get_recommended_renderer_name

    service = tinker.ServiceClient() if not dry_run else None

    total_generated = 0
    for model in models:
        renderer = get_recommended_renderer_name(model)
        sampling = None
        tok = None
        for dataset in datasets:
            train_csv = TRAIN_DATASETS[dataset]
            prompts_df = pd.read_csv(train_csv)
            n_expected = len(prompts_df)
            out_path = baseline_path(model, dataset)

            if out_path.exists():
                existing = pd.read_csv(out_path)
                if len(existing) == n_expected:
                    print(f"  [skip] {model} on {dataset}: {n_expected} responses already cached")
                    continue

            if dry_run:
                print(f"  [dry-run] would generate {n_expected} responses: {model} on {dataset}")
                continue

            print(f"  [gen] {model} on {dataset}: {n_expected} prompts (renderer={renderer})")
            if sampling is None:
                sampling = service.create_sampling_client(base_model=model)
                if hasattr(sampling, "get_tokenizer"):
                    tok = sampling.get_tokenizer()
                else:
                    train_client = service.create_lora_training_client(base_model=model)
                    tok = train_client.get_tokenizer()

            params = types.SamplingParams(max_tokens=512, temperature=0.0, stop=None)
            chat_kwargs = CHAT_TEMPLATE_KWARGS.get(model, {})
            prompts_list = prompts_df["prompt"].tolist()
            rows: list[dict] = []
            t0 = time.time()

            def _submit(prompt: str):
                messages = [{"role": "user", "content": prompt}]
                rendered = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
                )
                encoded = tok.encode(rendered, add_special_tokens=True)
                mi = types.ModelInput.from_ints(tokens=encoded)
                return sampling.sample(prompt=mi, sampling_params=params, num_samples=1)

            def _result_with_retry(prompt: str, fut, max_retries: int = 4):
                """Collect future; on transient API errors, resubmit + retry."""
                current = fut
                last_err = None
                for attempt in range(max_retries):
                    try:
                        return current.result()
                    except Exception as e:
                        last_err = e
                        print(
                            f"    [retry {attempt + 1}/{max_retries}] {type(e).__name__}: {str(e)[:100]}",
                            flush=True,
                        )
                        current = _submit(prompt)
                raise last_err

            # Chunked pipelining: submit `parallel` futures, then drain. Order preserved.
            for chunk_start in range(0, len(prompts_list), parallel):
                chunk = prompts_list[chunk_start : chunk_start + parallel]
                futures = [(p, _submit(p)) for p in chunk]
                for p, f in futures:
                    reply_raw = tok.decode(_result_with_retry(p, f).sequences[0].tokens)
                    reply = clean_response(model, reply_raw)
                    rows.append({"prompt": p, "model_response": reply, "model": model})
                completed = chunk_start + len(chunk)
                # Report progress at ~every 25 prompts
                if completed // 25 != (completed - len(chunk)) // 25 or completed == n_expected:
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1e-6)
                    eta = (n_expected - completed) / max(rate, 1e-6)
                    print(f"    {completed}/{n_expected} ({rate:.2f} gens/s, ETA {eta:.0f}s)", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            total_generated += len(rows)
            print(f"  [wrote] {out_path} ({len(rows)} rows, {time.time() - t0:.0f}s)")

    return total_generated


# ============================================================================
# Build SFT training CSVs (Phase B — joins prompt + target_response per cell)
# ============================================================================


def sft_data_path(cell: Cell) -> Path:
    return SFT_DATA_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_train.csv"


def build_sft_data(*, dry_run: bool = False) -> int:
    """For each (source, target, dataset) cell, build the SFT training CSV.

    SFT input = train prompt
    SFT completion = TARGET's response to that prompt (from the baseline cache)

    Source is irrelevant for the data itself (it determines which base model gets
    fine-tuned, not what the training data contains). So we de-dup across seeds
    and sources: per (target, dataset) we just slice the cache. Cell-level CSVs
    point to that shared cache for clarity.
    """
    n_built = 0
    seen: set[tuple[str, str]] = set()
    for cell in iter_cells():
        if cell.seed != SEEDS[0]:
            continue  # SFT data is the same across seeds for the same (S,T,D)
        if (cell.source, cell.target, cell.dataset) in seen:
            continue
        seen.add((cell.source, cell.target, cell.dataset))

        target_cache = baseline_path(cell.target, cell.dataset)
        out_path = sft_data_path(cell)
        if not target_cache.exists():
            print(f"  [missing] {target_cache} (target={cell.target} on {cell.dataset})")
            continue
        if dry_run:
            print(f"  [dry-run] would build {out_path} from {target_cache}")
            continue

        df = pd.read_csv(target_cache)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df[["prompt", "model_response"]].to_csv(out_path, index=False)
        n_built += 1
        print(f"  [built] {out_path} ({len(df)} rows)")
    return n_built


# ============================================================================
# Launch SFT (Phase C — Tinker LoRA, one job per cell)
# ============================================================================


def launch_sft(
    *,
    cells: list[Cell],
    dry_run: bool,
    only_llama: bool = False,
    max_jobs: int | None = None,
) -> dict:
    """Launch SFT jobs for the given cells.

    Returns a manifest of submitted jobs (or what would be submitted in dry-run).
    """
    from dotenv import load_dotenv

    load_dotenv()
    from tinker_cookbook.model_info import get_recommended_renderer_name

    if only_llama:
        cells = [c for c in cells if c.llama_critical]
    if max_jobs is not None:
        cells = cells[:max_jobs]

    manifest: list[dict] = []
    for cell in cells:
        train_csv = sft_data_path(cell)
        if not train_csv.exists() and not dry_run:
            print(f"  [missing-data] {cell.slug}: {train_csv} not built — skipping")
            continue

        prompt_template, completion_template = DATASET_TEMPLATES[cell.dataset]
        renderer_name = get_recommended_renderer_name(cell.source)
        weights_name = f"sft_{cell.slug}"
        output_dir = SFT_OUTPUT_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_seed{cell.seed}"

        record = {
            "cell": cell.slug,
            "source": cell.source,
            "target": cell.target,
            "dataset": cell.dataset,
            "seed": cell.seed,
            "weights_name": weights_name,
            "renderer_name": renderer_name,
            "base_model": cell.source,
            "train_csv": str(train_csv),
            "prompt_template": prompt_template,
            "completion_template": completion_template,
            "output_dir": str(output_dir),
            "llama_critical": cell.llama_critical,
        }

        if dry_run:
            manifest.append(record)
            continue

        from workflows.pipeline import SFTWorkflowConfig, TinkerSFTParams, run_sft_workflow
        from workflows.tinker import EvaluationConfig, SFTDatasetConfig

        ds_cfg = SFTDatasetConfig(
            train_csv=train_csv,
            eval_csv=None,
            prompt_column="prompt",
            completion_column="model_response",
            train_size=500,
            eval_size=0,
            seed=cell.seed,
        )
        sft_cfg = SFTWorkflowConfig(
            provider="tinker",
            dataset=ds_cfg,
            output_dir=output_dir,
            tinker=TinkerSFTParams(
                base_model=cell.source,
                batch_size=16,
                epochs=3,
                learning_rate=1e-4,
                prompt_template=prompt_template,
                completion_template=completion_template,
                evaluation_config=EvaluationConfig(max_sample_tokens=256),
                weights_name=weights_name,
                registry_path=DATA / "tinker_adapters.json",
                seed=cell.seed,
            ),
        )
        print(f"[launch] {cell.slug}")
        t0 = time.time()
        result = run_sft_workflow(sft_cfg)
        elapsed = time.time() - t0
        record["sampler_path"] = result.artifacts.get("sampler_path")
        record["elapsed_seconds"] = elapsed
        manifest.append(record)
        print(f"  -> {record['sampler_path']} ({elapsed:.0f}s)")

    return {"jobs": manifest, "n_jobs": len(manifest)}


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list-cells", help="Print the matrix without launching anything")
    list_p.add_argument("--only-llama", action="store_true")

    gen_p = sub.add_parser(
        "generate-target-responses",
        help="Generate target-model responses on training prompts (SFT completions).",
    )
    gen_p.add_argument("--dry-run", action="store_true")
    gen_p.add_argument("--models", nargs="*", default=MODELS, help="Subset of models to run")
    gen_p.add_argument("--datasets", nargs="*", default=list(TRAIN_DATASETS), help="Subset of datasets")
    gen_p.add_argument("--parallel", type=int, default=1, help="In-flight Tinker sample() requests per model")

    build_p = sub.add_parser("build-sft-data", help="Build SFT training CSVs from baselines.")
    build_p.add_argument("--dry-run", action="store_true")

    sft_p = sub.add_parser("launch-sft", help="Submit SFT jobs to Tinker.")
    sft_p.add_argument("--dry-run", action="store_true")
    sft_p.add_argument("--only-llama", action="store_true")
    sft_p.add_argument("--max-jobs", type=int, default=None)
    sft_p.add_argument("--manifest-out", type=Path, default=None)

    args = parser.parse_args()

    if args.cmd == "list-cells":
        cells = list(iter_cells())
        if args.only_llama:
            cells = [c for c in cells if c.llama_critical]
        for c in cells:
            tag = "[LLAMA-CRIT]" if c.llama_critical else "           "
            print(f"  {tag} {c.slug}")
        print(f"\nTotal cells: {len(cells)} (of {4*3*3*3} expected for full matrix)")
        return 0

    if args.cmd == "generate-target-responses":
        n = generate_target_responses(
            models=args.models,
            datasets=args.datasets,
            dry_run=args.dry_run,
            parallel=args.parallel,
        )
        print(f"\nGenerated {n} new responses")
        return 0

    if args.cmd == "build-sft-data":
        n = build_sft_data(dry_run=args.dry_run)
        print(f"\nBuilt {n} SFT training CSVs")
        return 0

    if args.cmd == "launch-sft":
        cells = list(iter_cells())
        manifest = launch_sft(
            cells=cells,
            dry_run=args.dry_run,
            only_llama=args.only_llama,
            max_jobs=args.max_jobs,
        )
        out_path = args.manifest_out or (DATA / "results" / "matrix" / "sft_manifest.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nManifest: {out_path}")
        print(f"Total jobs: {manifest['n_jobs']}")
        if args.dry_run:
            print("(dry-run — no jobs submitted)")
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    sys.exit(main())
