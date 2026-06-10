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
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Matplotlib backend must be set BEFORE pyplot or any plotting helper imports.
# Worker threads in ThreadPoolExecutor can't use the GUI MacOS backend.
import matplotlib
matplotlib.use("Agg")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATASETS = DATA / "datasets"
BASELINES_DIR = DATA / "model-responses" / "matrix_baselines"
SFT_DATA_DIR = DATA / "results" / "matrix" / "sft_data"
SFT_OUTPUT_DIR = DATA / "results" / "matrix" / "sft_runs"
SELF_SFT_DATA_DIR = DATA / "results" / "matrix" / "self_sft_data"
SELF_SFT_OUTPUT_DIR = DATA / "results" / "matrix" / "self_sft_runs"
DPO_DATA_DIR = DATA / "results" / "matrix" / "dpo_data"
DPO_OUTPUT_DIR = DATA / "results" / "matrix" / "dpo_runs"
PEFT_ADAPTER_DIR = DATA / "adapters" / "peft"


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
    # B2a confound-breaker sources (high-capability models from "launderer" lineages).
    # Added to the slug/chat maps only (NOT to MODELS) so the global matrix is unchanged;
    # B2a trains an explicit 8-cell subset via launch_sft(cells=...).
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
    "Qwen/Qwen3-32B": "qwen3-32b",
    # B2b confound-breaker: high-capability SMALL model (4B, MATH 0.64). Breaks the
    # capability<->size confound B2a introduced — completes the small+strong cell of the
    # capability x size 2x2 (the only missing corner).
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3-4b",
}

# Per-model kwargs for tokenizer.apply_chat_template — disables thinking traces
# so the responses are comparable across models for fingerprint analysis.
CHAT_TEMPLATE_KWARGS: dict[str, dict] = {
    "meta-llama/Llama-3.1-8B-Instruct": {},
    "Qwen/Qwen3.6-27B": {"enable_thinking": False},
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {"enable_thinking": False},
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
    "meta-llama/Llama-3.3-70B-Instruct": {},
    "Qwen/Qwen3-32B": {"enable_thinking": False},
    "Qwen/Qwen3-4B-Instruct-2507": {},
}


def clean_response(model: str, raw: str) -> str:
    """Strip per-model chat-template artifacts to leave just the assistant message.

    gpt-oss uses the harmony multi-channel format
    (``<|channel|>analysis<|message|>CoT<|end|>…<|channel|>final<|message|>ANSWER``).
    Normally the answer lives in the ``final`` channel. But a model fine-tuned —
    especially with DPO — toward another model frequently STOPS emitting the
    ``<|channel|>final<|message|>`` marker and dumps the answer straight under the
    ``analysis`` (or ``commentary``) channel header. The previous version only
    stripped when ``final`` was present, so post-DPO outputs leaked raw channel
    markup (and reasoning) into the stored response, spuriously inflating measured
    persistence. We now fall back to the last channel header when ``final`` is
    absent, and strip any residual harmony scaffolding tokens for all models.
    """
    text = raw
    if model == "openai/gpt-oss-20b":
        final = "<|channel|>final<|message|>"
        if final in text:
            text = text.rsplit(final, 1)[1]                      # the genuine final answer
        else:
            headers = list(re.finditer(r"<\|channel\|>\w+<\|message\|>", text))
            if headers:
                text = text[headers[-1].end():]                 # answer dumped under analysis/commentary
    # Strip any residual harmony / chat scaffolding tokens (harmless for other models).
    text = re.sub(r"<\|(?:start|end|return|channel|message|constrain|"
                  r"eot_id|eom_id|im_start|im_end|endoftext)\|>", "", text)
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


def renderer_for(model: str) -> str:
    try:
        from tinker_cookbook.model_info import get_recommended_renderer_name

        return get_recommended_renderer_name(model)
    except ModuleNotFoundError:
        return {
            "meta-llama/Llama-3.1-8B-Instruct": "llama3",
            "Qwen/Qwen3.6-27B": "qwen3",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nemotron",
            "openai/gpt-oss-20b": "gpt-oss",
            "meta-llama/Llama-3.3-70B-Instruct": "llama3",
            "Qwen/Qwen3-32B": "qwen3",
            "Qwen/Qwen3-4B-Instruct-2507": "qwen3",
        }.get(model, "unknown")


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


def iter_self_sft_cells(
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
) -> Iterable[Cell]:
    """Small post-training drift control: each model trains on its own outputs."""
    models = models or MODELS
    datasets = datasets or list(TRAIN_DATASETS)
    seeds = seeds or [SEEDS[0]]
    cells = [
        Cell(source=model, target=model, dataset=dataset, seed=seed)
        for model in models
        for dataset in datasets
        for seed in seeds
    ]
    cells.sort(key=lambda c: (not c.llama_critical, c.source, c.dataset, c.seed))
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
    service = tinker.ServiceClient() if not dry_run else None

    total_generated = 0
    for model in models:
        renderer = renderer_for(model)  # guarded helper (cookbook with dict fallback)
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


def self_sft_data_path(cell: Cell) -> Path:
    return SELF_SFT_DATA_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_self_train.csv"


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


def build_self_sft_data(*, dry_run: bool = False) -> int:
    """Build self-SFT control CSVs: prompt -> same model's own response."""
    n_built = 0
    seen: set[tuple[str, str]] = set()
    for cell in iter_self_sft_cells():
        key = (cell.source, cell.dataset)
        if key in seen:
            continue
        seen.add(key)

        source_cache = baseline_path(cell.source, cell.dataset)
        out_path = self_sft_data_path(cell)
        if not source_cache.exists():
            print(f"  [missing] {source_cache} ({cell.source} on {cell.dataset})")
            continue
        if dry_run:
            print(f"  [dry-run] would build {out_path} from {source_cache}")
            continue

        df = pd.read_csv(source_cache)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df[["prompt", "model_response"]].to_csv(out_path, index=False)
        n_built += 1
        print(f"  [built] {out_path} ({len(df)} rows)")
    return n_built


# ============================================================================
# Launch SFT (Phase C — Tinker LoRA, one job per cell)
# ============================================================================


def dpo_data_path(cell: Cell) -> Path:
    return DPO_DATA_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_pairs.csv"


def build_dpo_data(*, dry_run: bool = False) -> int:
    """For each (source, target, dataset) cell, build a DPO preference CSV.

    Schema: (prompt, chosen_response, rejected_response).
    chosen = TARGET's response to train prompt
    rejected = SOURCE's response to the SAME train prompt
    Both sides come from data/model-responses/matrix_baselines/{dataset}/{model_slug}_train.csv.
    """
    n_built = 0
    seen: set[tuple[str, str, str]] = set()
    for cell in iter_cells():
        if cell.seed != SEEDS[0]:
            continue  # DPO data is the same across seeds for the same (S,T,D)
        key = (cell.source, cell.target, cell.dataset)
        if key in seen:
            continue
        seen.add(key)

        source_cache = baseline_path(cell.source, cell.dataset)
        target_cache = baseline_path(cell.target, cell.dataset)
        out_path = dpo_data_path(cell)
        if not source_cache.exists() or not target_cache.exists():
            print(f"  [missing] {cell.slug}: source or target baseline missing")
            continue
        if dry_run:
            print(f"  [dry-run] would build {out_path}")
            continue

        src = pd.read_csv(source_cache)[["prompt", "model_response"]].rename(
            columns={"model_response": "rejected_response"}
        )
        tgt = pd.read_csv(target_cache)[["prompt", "model_response"]].rename(
            columns={"model_response": "chosen_response"}
        )
        merged = pd.merge(src, tgt, on="prompt", how="inner")
        merged = merged[merged["chosen_response"].str.len() > 0]
        merged = merged[merged["rejected_response"].str.len() > 0]
        merged = merged[["prompt", "chosen_response", "rejected_response"]]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        n_built += 1
        print(f"  [built] {out_path} ({len(merged)} pairs)")
    return n_built


def _find_final_dpo_checkpoint(log_path: Path) -> tuple[str | None, str | None]:
    """Parse cookbook DPO log to find the final saved checkpoint URIs.

    Returns (state_path, sampler_path) from the last entry in checkpoints.jsonl.
    """
    ck_path = log_path / "checkpoints.jsonl"
    if not ck_path.exists():
        return None, None
    state, sampler = None, None
    with ck_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            s = rec.get("state_path")
            sp = rec.get("sampler_path")
            if isinstance(s, str) and s.startswith("tinker://"):
                state = s
            if isinstance(sp, str) and sp.startswith("tinker://"):
                sampler = sp
    return state, sampler


def backfill_register_dpo() -> dict:
    """Scan dpo_runs/*/logs/checkpoints.jsonl and register DPO adapters in tinker_adapters.json."""
    from workflows.tinker import record_adapter_mapping
    registry_path = DATA / "tinker_adapters.json"
    registered = 0
    skipped = 0
    missing = 0
    for cell in iter_cells():
        log_path = DPO_OUTPUT_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_seed{cell.seed}" / "logs"
        state, sampler = _find_final_dpo_checkpoint(log_path)
        if not (state or sampler):
            missing += 1
            continue
        alias = f"dpo_{cell.slug}"
        # Sampler URI primary (for inference); state URI in metadata for future download
        record_adapter_mapping(
            alias,
            sampler or state,
            registry_path,
            metadata={
                "base_model": cell.source,
                "checkpoint_path": state,
                "sampler_path": sampler,
            },
        )
        registered += 1
    return {"registered": registered, "missing": missing, "skipped": skipped}


def launch_dpo(
    *,
    cells: list[Cell],
    dry_run: bool,
    only_llama: bool = False,
    max_jobs: int | None = None,
    parallel: int = 1,
) -> dict:
    """Launch DPO jobs (one per cell), starting from the corresponding SFT adapter."""
    from dotenv import load_dotenv
    from concurrent.futures import ThreadPoolExecutor, as_completed

    load_dotenv()

    registry_path = DATA / "tinker_adapters.json"
    registered: set[str] = set()
    sft_lookup: dict[str, dict] = {}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
            registered = set(registry.keys())
            sft_lookup = {k: v for k, v in registry.items() if k.startswith("sft_") and isinstance(v, dict)}
        except Exception:
            pass

    if only_llama:
        cells = [c for c in cells if c.llama_critical]
    if max_jobs is not None:
        cells = cells[:max_jobs]

    pending: list[tuple[Cell, dict, dict]] = []
    manifest: list[dict] = []
    for cell in cells:
        dpo_alias = f"dpo_{cell.slug}"
        if dpo_alias in registered and not dry_run:
            print(f"  [skip] {cell.slug}: already in registry")
            continue
        pref_csv = dpo_data_path(cell)
        if not pref_csv.exists() and not dry_run:
            print(f"  [missing-data] {cell.slug}: {pref_csv} not built")
            continue
        sft_entry = sft_lookup.get(f"sft_{cell.slug}")
        if not sft_entry and not dry_run:
            print(f"  [missing-sft] {cell.slug}: no SFT adapter to start from")
            continue
        sft_state_path = sft_entry.get("checkpoint_path") if sft_entry else None

        record = {
            "cell": cell.slug,
            "source": cell.source,
            "target": cell.target,
            "dataset": cell.dataset,
            "seed": cell.seed,
            "dpo_alias": dpo_alias,
            "base_model": cell.source,
            "renderer_name": renderer_for(cell.source),
            "pref_csv": str(pref_csv),
            "sft_checkpoint_path": sft_state_path,
        }
        if dry_run:
            manifest.append(record)
            continue
        pending.append((cell, record, sft_entry))

    if dry_run:
        return {"jobs": manifest, "n_jobs": len(manifest)}

    print(f"\nLaunching {len(pending)} DPO jobs with parallel={parallel}\n", flush=True)

    def _run_one(cell: Cell, record: dict, sft_entry: dict) -> dict:
        from workflows.pipeline import DPOWorkflowConfig, TinkerDPOParams, run_dpo_workflow
        from workflows.dpo import PreferenceDatasetConfig

        pref_csv = Path(record["pref_csv"])
        output_dir = DPO_OUTPUT_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_seed{cell.seed}"
        log_path = output_dir / "logs"
        sft_state_path = sft_entry.get("checkpoint_path")

        ds_cfg = PreferenceDatasetConfig(
            dataset_csv=pref_csv,
            prompt_column="prompt",
            chosen_column="chosen_response",
            rejected_column="rejected_response",
            train_size=500,
            eval_size=0,
            seed=cell.seed,
        )
        params = TinkerDPOParams(
            model_name=cell.source,
            renderer_name=record["renderer_name"],
            log_path=log_path,
            learning_rate=1e-5,
            dpo_beta=0.1,
            num_epochs=1,
            batch_size=16,
            max_length=4096,
            lora_rank=32,
            save_every=50,
            load_checkpoint_path=sft_state_path,
        )
        cfg = DPOWorkflowConfig(
            provider="tinker",
            dataset=ds_cfg,
            output_dir=output_dir,
            tinker=params,
        )

        print(f"[launch] {cell.slug}", flush=True)
        t0 = time.time()
        last_err: Exception | None = None
        result = None
        for attempt in range(1, 5):
            try:
                result = run_dpo_workflow(cfg)
                break
            except Exception as e:
                last_err = e
                wait_s = min(60, 5 * 2 ** (attempt - 1))
                print(
                    f"  [retry dpo {attempt}/4] {cell.slug} {type(e).__name__}: {str(e)[:120]} — sleep {wait_s}s",
                    flush=True,
                )
                time.sleep(wait_s)
        if result is None:
            print(f"  [DPO FAILED after 4 attempts] {cell.slug}: {last_err}", flush=True)
            record["error"] = str(last_err)
            return record
        elapsed = time.time() - t0

        # Pull the final DPO checkpoint URIs out of the cookbook's checkpoints.jsonl
        state_path, sampler_path = _find_final_dpo_checkpoint(log_path)
        record["dpo_state_path"] = state_path
        record["dpo_sampler_path"] = sampler_path
        record["elapsed_seconds"] = elapsed
        # Register in tinker_adapters.json so downstream code can find this adapter
        if state_path or sampler_path:
            from workflows.tinker import record_adapter_mapping
            record_adapter_mapping(
                record["dpo_alias"],
                sampler_path or state_path,
                registry_path,
                metadata={
                    "base_model": cell.source,
                    "sft_parent": sft_state_path,
                    "checkpoint_path": state_path,
                    "sampler_path": sampler_path,
                },
            )
        print(f"  -> {cell.slug} dpo done ({elapsed:.0f}s) | sampler={sampler_path}", flush=True)
        return record

    if parallel <= 1:
        for cell, record, sft_entry in pending:
            manifest.append(_run_one(cell, record, sft_entry))
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_run_one, cell, record, sft_entry): cell.slug
                for cell, record, sft_entry in pending
            }
            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                done += 1
                try:
                    manifest.append(fut.result())
                except Exception as e:
                    cell_slug = futures[fut]
                    print(f"  [worker exception] {cell_slug}: {e}", flush=True)
                    manifest.append({"cell": cell_slug, "error": str(e)})
                print(f"[progress] {done}/{total} DPO jobs complete", flush=True)

    return {"jobs": manifest, "n_jobs": len(manifest)}


def export_adapter_to_peft(
    *,
    tinker_path: str,
    base_model: str,
    output_dir: Path,
    max_retries: int = 6,
) -> dict:
    """Download a Tinker sampler checkpoint and unpack it as a PEFT adapter dir.

    Bypasses tinker_cookbook.weights.download which goes through the high-level
    SDK and consistently times out at ~80s before Tinker can respond. Uses
    direct httpx against the archive endpoint instead.

    Returns a status dict; raises on failure after max_retries.
    """
    import os
    import tarfile
    import httpx

    if (output_dir / "adapter_config.json").exists():
        return {"status": "already_exported", "output_dir": str(output_dir)}

    import tinker
    from tinker import types as tinker_types

    parsed = tinker_types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
    base_url = os.environ.get(
        "TINKER_BASE_URL", "https://tinker.thinkingmachines.dev/services/tinker-prod"
    )
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY not set")
    archive_url = (
        f"{base_url}/api/v1/training_runs/{parsed.training_run_id}"
        f"/checkpoints/{parsed.checkpoint_id}/archive"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tar_tmp = output_dir.parent / f".{output_dir.name}.tar"

    last_err: Exception | None = None
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        for attempt in range(1, max_retries + 1):
            try:
                # 1. Resolve signed URL (302 when ready, 503 when archive is being built)
                resp = client.get(
                    archive_url,
                    headers={"X-API-Key": api_key, "Accept": "application/gzip"},
                    follow_redirects=False,
                )
                if resp.status_code == 503:
                    wait_s = 30
                    print(
                        f"  [archive building] {output_dir.name}: 503, sleeping {wait_s}s",
                        flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                if resp.status_code != 302:
                    raise RuntimeError(
                        f"unexpected status {resp.status_code}: {resp.text[:200]}"
                    )
                signed_url = resp.headers["Location"]

                # 2. Stream the tar to disk
                with client.stream("GET", signed_url) as r:
                    r.raise_for_status()
                    with tar_tmp.open("wb") as f:
                        for chunk in r.iter_bytes(64 * 1024):
                            f.write(chunk)

                # 3. Safely extract (reject symlinks + path traversal)
                base = output_dir.resolve()
                with tarfile.open(tar_tmp, "r") as tar:
                    for member in tar.getmembers():
                        if member.issym() or member.islnk():
                            raise RuntimeError(f"unsafe symlink in archive: {member.name}")
                        member_path = (output_dir / member.name).resolve()
                        if not member_path.is_relative_to(base):
                            raise RuntimeError(f"path traversal: {member.name}")
                    tar.extractall(path=output_dir)
                tar_tmp.unlink(missing_ok=True)

                (output_dir / "dementor_tinker_export.json").write_text(
                    json.dumps(
                        {
                            "tinker_path": tinker_path,
                            "base_model": base_model,
                            "output_dir": str(output_dir),
                            "format": "peft",
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                return {"status": "exported", "output_dir": str(output_dir)}
            except Exception as e:
                last_err = e
                wait_s = min(60, 10 * 2 ** (attempt - 1))
                print(
                    f"  [retry export {attempt}/{max_retries}] {output_dir.name} {type(e).__name__}: {str(e)[:120]} — sleep {wait_s}s",
                    flush=True,
                )
                time.sleep(wait_s)
    raise RuntimeError(f"Export failed after {max_retries} attempts: {last_err}")


def upload_adapter_to_hf(
    *,
    peft_dir: Path,
    repo_id: str,
    base_model: str,
    alias: str,
    private: bool = False,
) -> dict:
    """Create (or reuse) an HF repo and upload the PEFT adapter dir.

    Adds a README with provenance. Returns {'status', 'repo_url'}.
    """
    from huggingface_hub import HfApi, create_repo, upload_folder

    api = HfApi()
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        # Already exists; check for adapter_model.safetensors at the repo root
        if any(s.rfilename == "adapter_model.safetensors" for s in info.siblings):
            return {"status": "already_on_hub", "repo_url": f"https://huggingface.co/{repo_id}"}
    except Exception:
        pass

    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    # Write a README into the dir before upload
    readme = peft_dir / "README.md"
    if not readme.exists():
        # Parse cell info from alias: sft_{dataset}_{source}_as_{target}_seed{N}
        parts = alias.split("_")
        stage = parts[0]
        readme.write_text(
            f"""---
base_model: {base_model}
library_name: peft
tags:
- lora
- {stage}
- dementor-research
---

# {alias}

LoRA adapter trained via [Tinker](https://thinkingmachines.ai/tinker/) as part of the
**dementor** intervention-ladder fingerprint persistence study (AAAI 2026 conference).

- **Base model:** `{base_model}`
- **Training stage:** {stage.upper()} (LoRA rank 32, target_modules=all-linear)
- **Alias:** `{alias}`

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
tok = AutoTokenizer.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}")
```

Part of the dementor matrix: 4 source models × 3 cross-targets × 3 train datasets × 3 seeds × 2 stages = 216 adapters.
""",
            encoding="utf-8",
        )

    upload_folder(
        repo_id=repo_id,
        folder_path=str(peft_dir),
        repo_type="model",
        commit_message=f"Upload {alias} adapter from dementor matrix",
    )
    return {"status": "uploaded", "repo_url": f"https://huggingface.co/{repo_id}"}


def push_adapters_to_hf(
    *,
    namespace: str = "dementor-research",
    only_llama: bool = False,
    kinds: tuple[str, ...] = ("sft", "dpo"),
    delete_local_after: bool = True,
    private: bool = False,
    max_retries: int = 4,
    parallel: int = 1,
) -> dict:
    """Upload all registered adapters to HuggingFace Hub.

    For each cell: if local PEFT exists, upload + (optionally) delete local.
    If not, download from Tinker to a temp dir, upload, delete temp.
    """
    from dotenv import load_dotenv
    import shutil

    load_dotenv()

    registry_path = DATA / "tinker_adapters.json"
    if not registry_path.exists():
        return {"uploaded": 0, "skipped": 0, "errors": []}
    registry = json.loads(registry_path.read_text())

    cell_to_source: dict[str, str] = {}
    cell_is_llama: dict[str, bool] = {}
    for cell in iter_cells():
        for prefix in ("sft", "dpo"):
            alias = f"{prefix}_{cell.slug}"
            cell_to_source[alias] = cell.source
            cell_is_llama[alias] = cell.llama_critical

    # Sort: Llama-source first (we have those locally, fastest path)
    def sort_key(item):
        alias = item[0]
        is_llama = cell_is_llama.get(alias, False)
        return (not is_llama, alias)

    # Build the pending list
    pending: list[tuple[str, dict]] = []
    skipped = 0
    errors: list[dict] = []
    for alias, entry in sorted(registry.items(), key=sort_key):
        prefix = alias.split("_", 1)[0]
        if prefix not in kinds or alias not in cell_to_source:
            continue
        source = cell_to_source[alias]
        if only_llama and source != "meta-llama/Llama-3.1-8B-Instruct":
            continue
        tinker_path = entry.get("path") if isinstance(entry, dict) else entry
        if not tinker_path:
            errors.append({"alias": alias, "error": "no tinker path"})
            continue
        pending.append((alias, {"source": source, "tinker_path": tinker_path}))

    def _handle_one(alias: str, meta: dict) -> dict:
        from huggingface_hub import HfApi
        source = meta["source"]
        tinker_path = meta["tinker_path"]
        repo_id = f"{namespace}/{alias}"
        local_dir = PEFT_ADAPTER_DIR / alias
        try:
            # Cheap HF-side existence check FIRST — skip if already uploaded
            api = HfApi()
            try:
                info = api.repo_info(repo_id=repo_id, repo_type="model")
                if any(s.rfilename == "adapter_model.safetensors" for s in info.siblings):
                    print(f"[skip] {alias}: already on hub", flush=True)
                    return {"alias": alias, "status": "already_on_hub",
                            "repo_url": f"https://huggingface.co/{repo_id}"}
            except Exception:
                pass  # repo doesn't exist; will create on upload

            if not (local_dir / "adapter_config.json").exists():
                print(f"[download] {alias}", flush=True)
                export_adapter_to_peft(
                    tinker_path=tinker_path, base_model=source, output_dir=local_dir
                )
            print(f"[upload] {alias} -> {repo_id}", flush=True)
            last_err: Exception | None = None
            res = None
            for attempt in range(1, max_retries + 1):
                try:
                    res = upload_adapter_to_hf(
                        peft_dir=local_dir,
                        repo_id=repo_id,
                        base_model=source,
                        alias=alias,
                        private=private,
                    )
                    break
                except Exception as e:
                    last_err = e
                    wait_s = min(60, 10 * 2 ** (attempt - 1))
                    print(
                        f"  [retry upload {attempt}/{max_retries}] {alias} {type(e).__name__}: {str(e)[:120]} — sleep {wait_s}s",
                        flush=True,
                    )
                    time.sleep(wait_s)
            if res is None:
                raise RuntimeError(f"upload failed after {max_retries} attempts: {last_err}")
            print(f"  -> {alias} {res['status']}: {res['repo_url']}", flush=True)
            if delete_local_after and local_dir.exists():
                shutil.rmtree(local_dir)
                print(f"  -> {alias} local cleaned", flush=True)
            return {"alias": alias, "status": res["status"], "repo_url": res["repo_url"]}
        except Exception as e:
            print(f"  [UPLOAD FAILED] {alias}: {e}", flush=True)
            return {"alias": alias, "error": str(e)}

    print(f"\nProcessing {len(pending)} adapters with parallel={parallel}\n", flush=True)

    uploaded = 0
    if parallel <= 1:
        for alias, meta in pending:
            res = _handle_one(alias, meta)
            if "error" in res:
                errors.append(res)
            else:
                uploaded += 1
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_handle_one, alias, meta): alias for alias, meta in pending}
            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                done += 1
                try:
                    res = fut.result()
                    if "error" in res:
                        errors.append(res)
                    else:
                        uploaded += 1
                except Exception as e:
                    cell_alias = futures[fut]
                    print(f"  [worker exception] {cell_alias}: {e}", flush=True)
                    errors.append({"alias": cell_alias, "error": str(e)})
                print(f"[progress] {done}/{total} uploads complete", flush=True)

    return {"uploaded": uploaded, "skipped": skipped, "errors": errors}


def backfill_export_adapters(*, only_llama: bool = False, kinds: tuple[str, ...] = ("sft", "dpo")) -> dict:
    """Download all already-registered SFT and/or DPO adapters to local PEFT format.

    Uses direct httpx against Tinker's archive endpoint (bypasses SDK timeouts).
    Llama-source adapters are downloaded first (June 12 retirement deadline).
    """
    from dotenv import load_dotenv

    load_dotenv()

    registry_path = DATA / "tinker_adapters.json"
    if not registry_path.exists():
        return {"exported": 0, "skipped": 0, "errors": []}
    registry = json.loads(registry_path.read_text())

    # Build cell-name -> base_model lookup for both SFT and DPO aliases
    cell_to_source: dict[str, str] = {}
    cell_is_llama: dict[str, bool] = {}
    for cell in iter_cells():
        for prefix in ("sft", "dpo"):
            alias = f"{prefix}_{cell.slug}"
            cell_to_source[alias] = cell.source
            cell_is_llama[alias] = cell.llama_critical

    # Sort: Llama-source first, then by alias
    def sort_key(item):
        alias = item[0]
        is_llama = cell_is_llama.get(alias, False)
        return (not is_llama, alias)

    exported = 0
    skipped = 0
    errors: list[dict] = []
    for alias, entry in sorted(registry.items(), key=sort_key):
        prefix = alias.split("_", 1)[0]
        if prefix not in kinds:
            continue
        if alias not in cell_to_source:
            skipped += 1
            continue
        source = cell_to_source[alias]
        if only_llama and source != "meta-llama/Llama-3.1-8B-Instruct":
            continue
        # Always use the sampler URI (path field). State URIs are NOT downloadable
        # via get_checkpoint_archive_url — Tinker rejects them as "not a sampler weights checkpoint".
        if isinstance(entry, dict):
            tinker_path = entry.get("path")
        else:
            tinker_path = entry
        if not tinker_path:
            errors.append({"alias": alias, "error": "no path in registry entry"})
            continue
        out_dir = PEFT_ADAPTER_DIR / alias
        print(f"[export] {alias}", flush=True)
        try:
            res = export_adapter_to_peft(
                tinker_path=tinker_path, base_model=source, output_dir=out_dir
            )
            print(f"  -> {res['status']} at {res['output_dir']}", flush=True)
            exported += 1
        except Exception as e:
            print(f"  [EXPORT FAILED] {alias}: {e}", flush=True)
            errors.append({"alias": alias, "error": str(e)})

    return {"exported": exported, "skipped": skipped, "errors": errors}


def launch_sft(
    *,
    cells: list[Cell],
    dry_run: bool,
    only_llama: bool = False,
    max_jobs: int | None = None,
    parallel: int = 1,
    alias_prefix: str = "sft",
    data_path_fn=sft_data_path,
    output_root: Path = SFT_OUTPUT_DIR,
) -> dict:
    """Launch SFT jobs for the given cells.

    Returns a manifest of submitted jobs (or what would be submitted in dry-run).

    parallel: number of SFT jobs to run concurrently via ThreadPoolExecutor.
        Each worker thread owns its own training_client. Registry writes are
        serialized by a threading.Lock in workflows.tinker.
    """
    from dotenv import load_dotenv
    from concurrent.futures import ThreadPoolExecutor, as_completed

    load_dotenv()
    if only_llama:
        cells = [c for c in cells if c.llama_critical]
    if max_jobs is not None:
        cells = cells[:max_jobs]

    # Skip cells whose adapter is already registered (resumability).
    registry_path = DATA / "tinker_adapters.json"
    registered: set[str] = set()
    if registry_path.exists() and not dry_run:
        try:
            registered = set(json.loads(registry_path.read_text()).keys())
        except Exception:
            registered = set()

    # Build per-cell plans first (cheap), then dispatch in parallel.
    pending: list[tuple[Cell, dict, Any]] = []  # (cell, record, sft_cfg)
    manifest: list[dict] = []

    from workflows.pipeline import SFTWorkflowConfig, TinkerSFTParams, run_sft_workflow
    from workflows.tinker import EvaluationConfig, SFTDatasetConfig

    for cell in cells:
        train_csv = data_path_fn(cell)
        if not train_csv.exists() and not dry_run:
            print(f"  [missing-data] {cell.slug}: {train_csv} not built — skipping")
            continue
        weights_name_check = f"{alias_prefix}_{cell.slug}"
        if weights_name_check in registered and not dry_run:
            print(f"  [skip] {cell.slug}: already in registry")
            continue

        prompt_template, completion_template = DATASET_TEMPLATES[cell.dataset]
        renderer_name = renderer_for(cell.source)
        weights_name = f"{alias_prefix}_{cell.slug}"
        output_dir = output_root / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_seed{cell.seed}"

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
        pending.append((cell, record, sft_cfg))

    if dry_run:
        return {"jobs": manifest, "n_jobs": len(manifest)}

    print(f"\nLaunching {len(pending)} SFT jobs with parallel={parallel}\n", flush=True)

    def _run_one(cell: Cell, record: dict, sft_cfg: Any) -> dict:
        print(f"[launch] {cell.slug}", flush=True)
        t0 = time.time()
        last_err: Exception | None = None
        result = None
        for attempt in range(1, 5):
            try:
                result = run_sft_workflow(sft_cfg)
                break
            except Exception as e:
                last_err = e
                wait_s = min(60, 5 * 2 ** (attempt - 1))
                print(
                    f"  [retry sft {attempt}/4] {cell.slug} {type(e).__name__}: {str(e)[:120]} — sleeping {wait_s}s",
                    flush=True,
                )
                time.sleep(wait_s)
        if result is None:
            print(f"  [SFT FAILED after 4 attempts] {cell.slug}: {last_err}", flush=True)
            record["error"] = str(last_err)
            return record
        elapsed = time.time() - t0
        sampler_path = result.artifacts.get("sampler_path")
        record["sampler_path"] = sampler_path
        record["elapsed_seconds"] = elapsed

        # Look up the downloadable checkpoint_path that _save_sampler_checkpoint
        # stored in the registry (different endpoint from sampler_path).
        checkpoint_path = None
        try:
            registry = json.loads((DATA / "tinker_adapters.json").read_text())
            entry = registry.get(f"sft_{cell.slug}", {})
            if isinstance(entry, dict):
                checkpoint_path = entry.get("checkpoint_path")
        except Exception:
            pass
        record["checkpoint_path"] = checkpoint_path

        # NOTE: inline PEFT export disabled — Tinker's get_checkpoint_archive_url
        # consistently times out at ~60s ("Creating checkpoint archive ... this may
        # take a while" then APIConnectionError). Adapters are saved on Tinker as
        # both sampler URIs (path) and state URIs (checkpoint_path). Use
        # workflows.run_matrix backfill-export later once Tinker download is fixed.
        print(f"  -> {cell.slug} sft done ({elapsed:.0f}s) | sampler={sampler_path}", flush=True)
        return record

    if parallel <= 1:
        for cell, record, sft_cfg in pending:
            manifest.append(_run_one(cell, record, sft_cfg))
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_run_one, cell, record, sft_cfg): cell.slug
                for cell, record, sft_cfg in pending
            }
            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                done += 1
                try:
                    manifest.append(fut.result())
                except Exception as e:
                    cell_slug = futures[fut]
                    print(f"  [worker exception] {cell_slug}: {e}", flush=True)
                    manifest.append({"cell": cell_slug, "error": str(e)})
                print(f"[progress] {done}/{total} SFT jobs complete", flush=True)

    return {"jobs": manifest, "n_jobs": len(manifest)}


def monitor_self_sft_controls() -> dict:
    """Summarize self-SFT controls from the adapter registry and local logs."""
    registry_path = DATA / "tinker_adapters.json"
    registry = {}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())

    rows = []
    for cell in iter_self_sft_cells():
        alias = f"self_sft_{cell.slug}"
        out_dir = SELF_SFT_OUTPUT_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_seed{cell.seed}"
        entry = registry.get(alias)
        status = "registered" if entry else "missing"
        if not entry and out_dir.exists():
            status = "started"
        rows.append(
            {
                "alias": alias,
                "model": cell.source,
                "dataset": cell.dataset,
                "seed": cell.seed,
                "status": status,
                "output_dir": str(out_dir),
                "sampler_path": entry.get("path") if isinstance(entry, dict) else None,
                "checkpoint_path": entry.get("checkpoint_path") if isinstance(entry, dict) else None,
            }
        )

    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    return {"counts": counts, "controls": rows}


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list-cells", help="Print the matrix without launching anything")
    list_p.add_argument("--only-llama", action="store_true")

    self_list_p = sub.add_parser("list-self-sft-controls", help="Print self-SFT drift-control cells")
    self_list_p.add_argument("--models", nargs="*", default=MODELS)
    self_list_p.add_argument("--datasets", nargs="*", default=list(TRAIN_DATASETS))

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

    self_build_p = sub.add_parser("build-self-sft-data", help="Build self-SFT drift-control CSVs.")
    self_build_p.add_argument("--dry-run", action="store_true")

    sft_p = sub.add_parser("launch-sft", help="Submit SFT jobs to Tinker.")
    sft_p.add_argument("--dry-run", action="store_true")
    sft_p.add_argument("--only-llama", action="store_true")
    sft_p.add_argument("--max-jobs", type=int, default=None)
    sft_p.add_argument("--manifest-out", type=Path, default=None)
    sft_p.add_argument("--parallel", type=int, default=1, help="Number of concurrent SFT jobs (ThreadPoolExecutor)")

    self_sft_p = sub.add_parser("launch-self-sft", help="Submit self-SFT drift-control jobs to Tinker.")
    self_sft_p.add_argument("--dry-run", action="store_true")
    self_sft_p.add_argument("--max-jobs", type=int, default=None)
    self_sft_p.add_argument("--manifest-out", type=Path, default=None)
    self_sft_p.add_argument("--parallel", type=int, default=1)
    self_sft_p.add_argument("--models", nargs="*", default=MODELS)
    self_sft_p.add_argument("--datasets", nargs="*", default=list(TRAIN_DATASETS))

    self_mon_p = sub.add_parser("monitor-self-sft", help="Summarize self-SFT drift-control registry/log status.")
    self_mon_p.add_argument("--json", action="store_true")

    dpd_p = sub.add_parser("build-dpo-data", help="Build DPO preference CSVs from baselines.")
    dpd_p.add_argument("--dry-run", action="store_true")

    dpo_p = sub.add_parser("launch-dpo", help="Submit DPO jobs to Tinker (on top of SFT adapters).")
    dpo_p.add_argument("--dry-run", action="store_true")
    dpo_p.add_argument("--only-llama", action="store_true")
    dpo_p.add_argument("--max-jobs", type=int, default=None)
    dpo_p.add_argument("--manifest-out", type=Path, default=None)
    dpo_p.add_argument("--parallel", type=int, default=1)

    bf_p = sub.add_parser(
        "backfill-export",
        help="Download already-registered SFT adapters to local PEFT.",
    )
    bf_p.add_argument("--only-llama", action="store_true")

    bd_p = sub.add_parser(
        "backfill-register-dpo",
        help="Scan dpo_runs/*/logs/checkpoints.jsonl and register DPO adapter URIs.",
    )

    hf_p = sub.add_parser(
        "push-to-hf",
        help="Upload all registered adapters to HuggingFace Hub.",
    )
    hf_p.add_argument("--namespace", default="dementor-research", help="HF user or org name")
    hf_p.add_argument("--only-llama", action="store_true")
    hf_p.add_argument("--kinds", nargs="+", default=["sft", "dpo"], choices=["sft", "dpo"])
    hf_p.add_argument("--keep-local", action="store_true", help="Don't delete local PEFT after upload")
    hf_p.add_argument("--private", action="store_true", help="Create private repos (default: public)")
    hf_p.add_argument("--parallel", type=int, default=1, help="Concurrent uploads")

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

    if args.cmd == "list-self-sft-controls":
        cells = list(iter_self_sft_cells(models=args.models, datasets=args.datasets))
        for c in cells:
            tag = "[LLAMA-CRIT]" if c.llama_critical else "           "
            print(f"  {tag} self_sft_{c.slug}")
        print(f"\nTotal self-SFT controls: {len(cells)}")
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

    if args.cmd == "build-self-sft-data":
        n = build_self_sft_data(dry_run=args.dry_run)
        print(f"\nBuilt {n} self-SFT control CSVs")
        return 0

    if args.cmd == "launch-sft":
        cells = list(iter_cells())
        manifest = launch_sft(
            cells=cells,
            dry_run=args.dry_run,
            only_llama=args.only_llama,
            max_jobs=args.max_jobs,
            parallel=args.parallel,
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

    if args.cmd == "launch-self-sft":
        cells = list(iter_self_sft_cells(models=args.models, datasets=args.datasets))
        manifest = launch_sft(
            cells=cells,
            dry_run=args.dry_run,
            max_jobs=args.max_jobs,
            parallel=args.parallel,
            alias_prefix="self_sft",
            data_path_fn=self_sft_data_path,
            output_root=SELF_SFT_OUTPUT_DIR,
        )
        out_path = args.manifest_out or (DATA / "results" / "matrix" / "self_sft_manifest.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nManifest: {out_path}")
        print(f"Total jobs: {manifest['n_jobs']}")
        if args.dry_run:
            print("(dry-run — no jobs submitted)")
        return 0

    if args.cmd == "monitor-self-sft":
        result = monitor_self_sft_controls()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Self-SFT control status:")
            for key, value in sorted(result["counts"].items()):
                print(f"  {key}: {value}")
            for row in result["controls"]:
                print(f"  {row['status']:10s} {row['alias']}")
        return 0

    if args.cmd == "build-dpo-data":
        n = build_dpo_data(dry_run=args.dry_run)
        print(f"\nBuilt {n} DPO preference CSVs")
        return 0

    if args.cmd == "launch-dpo":
        cells = list(iter_cells())
        manifest = launch_dpo(
            cells=cells,
            dry_run=args.dry_run,
            only_llama=args.only_llama,
            max_jobs=args.max_jobs,
            parallel=args.parallel,
        )
        out_path = args.manifest_out or (DATA / "results" / "matrix" / "dpo_manifest.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nManifest: {out_path}")
        print(f"Total jobs: {manifest['n_jobs']}")
        return 0

    if args.cmd == "backfill-export":
        result = backfill_export_adapters(only_llama=args.only_llama)
        print(f"\nExported {result['exported']}, skipped {result['skipped']}, errors {len(result['errors'])}")
        for err in result["errors"]:
            print(f"  ERROR {err['alias']}: {err['error']}")
        return 0 if not result["errors"] else 1

    if args.cmd == "backfill-register-dpo":
        result = backfill_register_dpo()
        print(f"\nRegistered {result['registered']} DPO adapters; {result['missing']} missing checkpoint logs")
        return 0

    if args.cmd == "push-to-hf":
        result = push_adapters_to_hf(
            namespace=args.namespace,
            only_llama=args.only_llama,
            kinds=tuple(args.kinds),
            delete_local_after=not args.keep_local,
            private=args.private,
            parallel=args.parallel,
        )
        print(f"\nUploaded {result['uploaded']}, errors {len(result['errors'])}")
        for err in result["errors"][:10]:
            print(f"  ERROR {err['alias']}: {err['error']}")
        return 0 if not result["errors"] else 1

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    sys.exit(main())
