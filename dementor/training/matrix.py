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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Matplotlib backend must be set BEFORE pyplot or any plotting helper imports.
# Worker threads in ThreadPoolExecutor can't use the GUI MacOS backend.
import matplotlib
matplotlib.use("Agg")

import pandas as pd

from dementor import config

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATASETS = DATA / "datasets"
BASELINES_DIR = DATA / "model-responses" / "matrix_baselines"
SFT_DATA_DIR = DATA / "results" / "matrix" / "sft_data"
SFT_OUTPUT_DIR = DATA / "results" / "matrix" / "sft_runs"
SELF_SFT_DATA_DIR = DATA / "results" / "matrix" / "self_sft_data"
SELF_SFT_OUTPUT_DIR = DATA / "results" / "matrix" / "self_sft_runs"
DPO_DATA_DIR = DATA / "results" / "matrix" / "dpo_data"
DPO_OUTPUT_DIR = DATA / "results" / "matrix" / "dpo_runs"
SAFETY_SFT_DATA_DIR = DATA / "results" / "matrix" / "safety_sft_data"
SAFETY_SFT_OUTPUT_DIR = DATA / "results" / "matrix" / "safety_sft_runs"
SAFETY_DPO_DATA_DIR = DATA / "results" / "matrix" / "safety_dpo_data"
SAFETY_DPO_OUTPUT_DIR = DATA / "results" / "matrix" / "safety_dpo_runs"
PEFT_ADAPTER_DIR = DATA / "adapters" / "peft"
REFUSAL_DIR = DATA / "datasets" / "benchmarks" / "refusal"
DEFAULT_SAFETY_PROMPTS = REFUSAL_DIR / "advbench_harmful.csv"
DEFAULT_SAFETY_EXCLUDE_PROMPTS = REFUSAL_DIR / "eval_120h_80o_seed42.csv"
SAFETY_NATIVE_REFUSAL_DIR = ROOT / "results" / "safety" / "native_refusal"
DEFAULT_SAFETY_REPLAY_SIZE = 120
GENERIC_REFUSAL_RESPONSE = "I can't help with that request."
GENERIC_NONREFUSAL_RESPONSE = "I can help with that request."


# ============================================================================
# Concurrency helpers (shared retry + ThreadPool dispatch scaffolding)
# ============================================================================

def _retry_call(
    fn,
    *,
    attempts: int,
    base_wait: float,
    label: str,
    name: str = "",
    err_trunc: int = 120,
    sleep_verb: str = "sleep",
    max_wait: float = 60,
    indent: str = "  ",
):
    """Call ``fn()`` up to ``attempts`` times, retrying on any exception.

    On each failure prints ``{indent}[retry {label} {attempt}/{attempts}] ...``
    then sleeps ``min(max_wait, base_wait * 2 ** (attempt - 1))`` seconds before
    the next try. Returns ``fn()``'s value on the first success; re-raises the
    last exception if every attempt fails.
    """
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            wait_s = min(max_wait, base_wait * 2 ** (attempt - 1))
            name_part = f"{name} " if name else ""
            print(
                f"{indent}[retry {label} {attempt}/{attempts}] "
                f"{name_part}{type(e).__name__}: {str(e)[:err_trunc]} — {sleep_verb} {wait_s}s",
                flush=True,
            )
            time.sleep(wait_s)
    if last_err is None:  # attempts <= 0: loop never ran — avoid `raise None`
        raise RuntimeError(f"{label} failed after {attempts} attempts")
    raise last_err


def _dispatch(items, worker, *, parallel, label, key, on_success, on_error):
    """Run ``worker(item)`` over ``items``, sequentially or via a thread pool.

    When ``parallel <= 1`` results are handed to ``on_success`` in item order and
    worker exceptions propagate (matching the original inline loops). Otherwise a
    ``ThreadPoolExecutor(max_workers=parallel)`` fans the work out: results reach
    ``on_success`` in completion order, and each worker exception is logged as
    ``[worker exception] {key}`` then routed to ``on_error(key, exc)``. A
    ``[progress] done/total {label} complete`` line prints per completion.
    """
    if parallel <= 1:
        for item in items:
            on_success(worker(item))
        return
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(worker, item): key(item) for item in items}
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            done += 1
            try:
                on_success(fut.result())
            except Exception as e:
                k = futures[fut]
                print(f"  [worker exception] {k}: {e}", flush=True)
                on_error(k, e)
            print(f"[progress] {done}/{total} {label} complete", flush=True)


# ============================================================================
# Matrix definition
# ============================================================================

# Derived from config.yaml (the single source of truth). MODELS is the active
# 10-model roster; the slug/chat maps also cover roster_legacy so existing adapters
# and the B2 cell-subset drivers still resolve. Module-level MUTABLE dicts on
# purpose — extra_model_census.py registers extra models by mutating CHAT_TEMPLATE_KWARGS.
MODELS: list[str] = [m["id"] for m in config.roster()]
MODEL_SLUG: dict[str, str] = {m["id"]: m["slug"] for m in config.roster(include_legacy=True)}
CHAT_TEMPLATE_KWARGS: dict[str, dict] = {
    m["id"]: dict(m.get("chat_template_kwargs", {})) for m in config.roster(include_legacy=True)
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
    if model.startswith("openai/gpt-oss"):  # 20b and 120b both use the harmony format
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

# Train splits + per-dataset SFT templates (from config.yaml).
TRAIN_DATASETS: dict[str, Path] = {
    n: config.resolve_path(config.dataset(n)["train_csv"]) for n in config.dataset_names()
}
DATASET_TEMPLATES: dict[str, tuple[str, str]] = {
    n: (config.dataset(n)["prompt_template"], config.dataset(n)["completion_template"])
    for n in config.dataset_names()
}

SEEDS: list[int] = config.seeds()


def source_id_of(alias: str) -> str | None:
    """Source model id from an adapter alias
    ``{stage}_{dataset}_{srcslug}_as_{tgtslug}_seed{N}`` (legacy-inclusive slug map)."""
    if "_as_" not in alias:
        return None
    head = alias.split("_as_", 1)[0]                   # {stage}_{dataset}_{srcslug}
    for stage in ("safety_sft_", "safety_dpo_", "self_sft_", "sft_", "dpo_"):  # longest prefix first
        if head.startswith(stage):
            body = head[len(stage):]                    # {dataset}_{srcslug}
            break
    else:
        return None
    for ds in config.dataset_names():
        if body.startswith(ds + "_"):
            return config.slug_to_id().get(body[len(ds) + 1:])
    return None


def renderer_for(model: str) -> str:
    # Local (non-Tinker) models aren't in the tinker cookbook; use the config renderer.
    if config.backend_for(model) == "local":
        return config.renderer_for(model)
    try:
        from tinker_cookbook.model_info import get_recommended_renderer_name

        return get_recommended_renderer_name(model)
    except Exception:  # cookbook absent, or doesn't recognize the model
        try:
            return config.renderer_for(model)
        except KeyError:
            return "unknown"


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
    """All cross (source != target) cells over MODELS x datasets x seeds."""
    return [
        Cell(source=s, target=t, dataset=d, seed=seed)
        for s in MODELS for t in MODELS if s != t
        for d in TRAIN_DATASETS for seed in SEEDS
    ]


def iter_self_sft_cells(
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
) -> Iterable[Cell]:
    """Small post-training drift control: each model trains on its own outputs."""
    models = models or MODELS
    datasets = datasets or list(TRAIN_DATASETS)
    seeds = seeds or SEEDS
    return [
        Cell(source=model, target=model, dataset=dataset, seed=seed)
        for model in models
        for dataset in datasets
        for seed in seeds
    ]


# ============================================================================
# Target-response generation (Phase A.2 — for SFT training data)
# ============================================================================


def baseline_path(model: str, dataset: str) -> Path:
    return BASELINES_DIR / dataset / f"{MODEL_SLUG[model]}_train.csv"


def _build_sft_cfg(*, cell: Cell, ds_cfg, output_dir: Path, weights_name: str,
                   prompt_template: str, completion_template: str):
    """SFTWorkflowConfig for a cell — backend from config, hyperparameters from config.sft()."""
    from dementor.training.pipeline import LocalSFTParams, SFTWorkflowConfig, TinkerSFTParams
    from dementor.training.tinker_backend import EvaluationConfig

    hp = config.sft()
    common = dict(
        base_model=cell.source, batch_size=hp["batch_size"], epochs=hp["epochs"],
        learning_rate=hp["learning_rate"], prompt_template=prompt_template,
        completion_template=completion_template, weights_name=weights_name,
        registry_path=config.registry_path(), seed=cell.seed,
    )
    if config.backend_for(cell.source) == "local":
        return SFTWorkflowConfig(provider="local", dataset=ds_cfg, output_dir=output_dir,
                                 local=LocalSFTParams(**common))
    return SFTWorkflowConfig(provider="tinker", dataset=ds_cfg, output_dir=output_dir,
                             tinker=TinkerSFTParams(**common,
                                 evaluation_config=EvaluationConfig(max_sample_tokens=256)))


def _build_dpo_cfg(*, cell: Cell, ds_cfg, output_dir: Path, sft_state_path,
                   renderer_name: str, log_path: Path, weights_name: str | None = None):
    """DPOWorkflowConfig for a cell — backend from config, hyperparameters from config.dpo()."""
    from dementor.training.pipeline import DPOWorkflowConfig, LocalDPOParams, TinkerDPOParams

    hp = config.dpo()
    weights_name = weights_name or f"dpo_{cell.slug}"
    if config.backend_for(cell.source) == "local":
        return DPOWorkflowConfig(provider="local", dataset=ds_cfg, output_dir=output_dir,
            local=LocalDPOParams(model_name=cell.source, load_checkpoint_path=sft_state_path,
                weights_name=weights_name, registry_path=config.registry_path(),
                learning_rate=hp["learning_rate"], dpo_beta=hp["dpo_beta"], num_epochs=hp["num_epochs"],
                batch_size=hp["batch_size"], max_length=hp["max_length"], lora_rank=hp["lora_rank"],
                seed=cell.seed))
    return DPOWorkflowConfig(provider="tinker", dataset=ds_cfg, output_dir=output_dir,
        tinker=TinkerDPOParams(model_name=cell.source, renderer_name=renderer_name, log_path=log_path,
            learning_rate=hp["learning_rate"], dpo_beta=hp["dpo_beta"], num_epochs=hp["num_epochs"],
            batch_size=hp["batch_size"], max_length=hp["max_length"], lora_rank=hp["lora_rank"],
            save_every=hp["save_every"], load_checkpoint_path=sft_state_path))


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
    # `tinker` is imported lazily inside the Tinker branch below so an all-local
    # data-generation run does not require the optional `tinker` package.
    service = None  # created lazily on first Tinker model (an all-local run needs no Tinker)

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
            if config.backend_for(model) == "local":
                from dementor.training.local_backend import generate_local_responses
                prompts_list = prompts_df["prompt"].tolist()
                t0 = time.time()
                replies = generate_local_responses(
                    model=model, prompts=prompts_list,
                    chat_template_kwargs=CHAT_TEMPLATE_KWARGS.get(model, {}),
                    max_new_tokens=512, temperature=0.0,
                )
                rows = [{"prompt": p, "model_response": clean_response(model, r), "model": model}
                        for p, r in zip(prompts_list, replies)]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(out_path, index=False)
                total_generated += len(rows)
                print(f"  [wrote] {out_path} ({len(rows)} rows, {time.time() - t0:.0f}s, local)")
                continue
            # Tinker path — import lazily so all-local runs never need the `tinker` package.
            import tinker
            from tinker import types
            if sampling is None:
                if service is None:
                    service = tinker.ServiceClient()
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


# ============================================================================
# Safety-constrained imitation data (target imitation + refusal replay)
# ============================================================================


def safety_sft_data_path(cell: Cell) -> Path:
    return SAFETY_SFT_DATA_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_train.csv"


def safety_dpo_data_path(cell: Cell) -> Path:
    return SAFETY_DPO_DATA_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_pairs.csv"


def _manifest_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".manifest.json")


def _unique_training_data_cells(cells: Iterable[Cell] | None = None) -> Iterable[Cell]:
    """Yield one cell per (source, target, dataset); training data is shared across seeds."""
    seen: set[tuple[str, str, str]] = set()
    for cell in (cells or iter_cells()):
        key = (cell.source, cell.target, cell.dataset)
        if key in seen:
            continue
        seen.add(key)
        yield cell


def load_safety_replay_prompts(
    *,
    safety_prompts_file: Path = DEFAULT_SAFETY_PROMPTS,
    exclude_prompts_file: Path | None = DEFAULT_SAFETY_EXCLUDE_PROMPTS,
    replay_size: int | None = DEFAULT_SAFETY_REPLAY_SIZE,
    prompt_seed: int = 42,
    categories: Sequence[str] = ("harmful",),
) -> pd.DataFrame:
    """Load a deterministic, optionally eval-disjoint set of safety replay prompts."""
    if replay_size is not None and replay_size < 0:
        raise ValueError("replay_size must be non-negative or None")
    if replay_size == 0:
        return pd.DataFrame(columns=["prompt", "category", "expected"])
    if not safety_prompts_file.exists():
        raise FileNotFoundError(f"Safety prompt CSV not found: {safety_prompts_file}")
    df = pd.read_csv(safety_prompts_file)
    if "prompt" not in df.columns:
        raise ValueError(f"Safety prompt CSV must contain a prompt column: {safety_prompts_file}")
    df = df.copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df = df[df["prompt"].str.len() > 0].drop_duplicates("prompt")
    if categories and "category" in df.columns:
        df = df[df["category"].astype(str).isin(set(categories))]

    if exclude_prompts_file is not None and exclude_prompts_file.exists():
        excluded = pd.read_csv(exclude_prompts_file)
        if "prompt" in excluded.columns:
            excluded_prompts = set(excluded["prompt"].astype(str).str.strip())
            df = df[~df["prompt"].isin(excluded_prompts)]

    if replay_size is not None:
        if len(df) < replay_size:
            raise ValueError(
                f"Requested {replay_size} replay prompts but only {len(df)} are available "
                f"after filtering/exclusion from {safety_prompts_file}"
            )
        df = df.sample(n=replay_size, random_state=prompt_seed)
    else:
        df = df.sample(frac=1.0, random_state=prompt_seed)
    return df.reset_index(drop=True)


def _native_refusal_snippets(
    model: str,
    *,
    preferred_seed: int = 1,
) -> tuple[dict[str, str], Path | None, str | None]:
    """Return prompt -> native refusal snippet when cached safety verdicts exist."""
    candidate_seeds = []
    for seed in (preferred_seed, 1, 2, 3, 42, 43, 44):
        if seed not in candidate_seeds:
            candidate_seeds.append(seed)
    for seed in candidate_seeds:
        path = SAFETY_NATIVE_REFUSAL_DIR / f"{MODEL_SLUG[model]}_seed{seed}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "prompt" not in df.columns:
            return {}, path, None
        response_col = "model_response" if "model_response" in df.columns else "snippet_redacted"
        if response_col not in df.columns:
            return {}, path, None
        working = df.copy()
        if "refused" in working.columns:
            working = working[working["refused"].astype(int) == 1]
        working["prompt"] = working["prompt"].astype(str).str.strip()
        working[response_col] = working[response_col].fillna("").astype(str).str.strip()
        working = working[(working["prompt"].str.len() > 0) & (working[response_col].str.len() > 0)]
        return dict(zip(working["prompt"], working[response_col], strict=False)), path, response_col
    return {}, None, None


def _safety_sft_replay_rows(
    *,
    source: str,
    safety_prompts: pd.DataFrame,
    response_seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    snippets, snippet_path, snippet_col = _native_refusal_snippets(source, preferred_seed=response_seed)
    rows: list[dict[str, object]] = []
    n_native = 0
    for prompt in safety_prompts["prompt"].astype(str):
        response = snippets.get(prompt)
        response_source = "generic_refusal"
        if response:
            n_native += 1
            response_source = f"native_refusal:{snippet_path.name if snippet_path else 'unknown'}"
        else:
            response = GENERIC_REFUSAL_RESPONSE
        rows.append(
            {
                "prompt": prompt,
                "model_response": response,
                "row_type": "safety_replay",
                "safety_response_source": response_source,
            }
        )
    meta = {
        "native_refusal_path": str(snippet_path) if snippet_path else None,
        "native_refusal_column": snippet_col,
        "native_refusal_rows_used": n_native,
        "generic_refusal_rows_used": len(rows) - n_native,
    }
    return pd.DataFrame(rows), meta


def _safety_dpo_replay_rows(
    *,
    source: str,
    safety_prompts: pd.DataFrame,
    response_seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    snippets, snippet_path, snippet_col = _native_refusal_snippets(source, preferred_seed=response_seed)
    rows: list[dict[str, object]] = []
    n_native = 0
    for prompt in safety_prompts["prompt"].astype(str):
        chosen = snippets.get(prompt)
        response_source = "generic_refusal"
        if chosen:
            n_native += 1
            response_source = f"native_refusal:{snippet_path.name if snippet_path else 'unknown'}"
        else:
            chosen = GENERIC_REFUSAL_RESPONSE
        rows.append(
            {
                "prompt": prompt,
                "chosen_response": chosen,
                "rejected_response": GENERIC_NONREFUSAL_RESPONSE,
                "row_type": "safety_replay",
                "safety_response_source": response_source,
            }
        )
    meta = {
        "native_refusal_path": str(snippet_path) if snippet_path else None,
        "native_refusal_column": snippet_col,
        "native_refusal_rows_used": n_native,
        "generic_refusal_rows_used": len(rows) - n_native,
        "rejected_response_source": "generic_nonrefusal_stub",
    }
    return pd.DataFrame(rows), meta


def _nonempty_response_filter(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in columns:
        mask &= df[col].fillna("").astype(str).str.strip().str.len() > 0
    return mask


def _base_safety_manifest(
    *,
    cell: Cell,
    safety_prompts_file: Path,
    exclude_prompts_file: Path | None,
    replay_size: int | None,
    prompt_seed: int,
    response_seed: int,
    safety_prompts: pd.DataFrame,
) -> dict[str, object]:
    return {
        "cell": cell.slug,
        "source": cell.source,
        "target": cell.target,
        "dataset": cell.dataset,
        "safety_prompts_file": str(safety_prompts_file),
        "exclude_prompts_file": str(exclude_prompts_file) if exclude_prompts_file else None,
        "replay_size_requested": replay_size,
        "safety_prompt_seed": prompt_seed,
        "safety_response_seed": response_seed,
        "safety_replay_prompts": len(safety_prompts),
        "generic_refusal_response": GENERIC_REFUSAL_RESPONSE,
    }


def build_safety_sft_data(
    *,
    dry_run: bool = False,
    cells: list[Cell] | None = None,
    safety_prompts_file: Path = DEFAULT_SAFETY_PROMPTS,
    exclude_prompts_file: Path | None = DEFAULT_SAFETY_EXCLUDE_PROMPTS,
    replay_size: int | None = DEFAULT_SAFETY_REPLAY_SIZE,
    prompt_seed: int = 42,
    response_seed: int = 1,
) -> int:
    """Build SFT CSVs that mix target imitation with refusal replay rows."""
    safety_prompts = load_safety_replay_prompts(
        safety_prompts_file=safety_prompts_file,
        exclude_prompts_file=exclude_prompts_file,
        replay_size=replay_size,
        prompt_seed=prompt_seed,
    )
    n_built = 0
    for cell in _unique_training_data_cells(cells):
        target_cache = baseline_path(cell.target, cell.dataset)
        out_path = safety_sft_data_path(cell)
        if not target_cache.exists():
            print(f"  [missing] {target_cache} (target={cell.target} on {cell.dataset})")
            continue
        if dry_run:
            print(
                f"  [dry-run] would build {out_path} from {target_cache} "
                f"+ {len(safety_prompts)} safety replay rows"
            )
            continue

        imitation = pd.read_csv(target_cache)[["prompt", "model_response"]]
        imitation = imitation[_nonempty_response_filter(imitation, ["prompt", "model_response"])].copy()
        imitation["row_type"] = "imitation"
        imitation["safety_response_source"] = ""
        replay, replay_meta = _safety_sft_replay_rows(
            source=cell.source,
            safety_prompts=safety_prompts,
            response_seed=response_seed,
        )
        combined = pd.concat([imitation, replay], ignore_index=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        manifest = _base_safety_manifest(
            cell=cell,
            safety_prompts_file=safety_prompts_file,
            exclude_prompts_file=exclude_prompts_file,
            replay_size=replay_size,
            prompt_seed=prompt_seed,
            response_seed=response_seed,
            safety_prompts=safety_prompts,
        )
        manifest.update(
            {
                "stage": "safety_sft",
                "target_cache": str(target_cache),
                "imitation_rows": len(imitation),
                "safety_replay_rows": len(replay),
                "total_rows": len(combined),
                **replay_meta,
            }
        )
        _manifest_path(out_path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        n_built += 1
        print(
            f"  [built] {out_path} ({len(imitation)} imitation + {len(replay)} safety replay rows)"
        )
    return n_built


def build_safety_dpo_data(
    *,
    dry_run: bool = False,
    cells: list[Cell] | None = None,
    safety_prompts_file: Path = DEFAULT_SAFETY_PROMPTS,
    exclude_prompts_file: Path | None = DEFAULT_SAFETY_EXCLUDE_PROMPTS,
    replay_size: int | None = DEFAULT_SAFETY_REPLAY_SIZE,
    prompt_seed: int = 42,
    response_seed: int = 1,
) -> int:
    """Build DPO CSVs that mix target-vs-source imitation pairs with refusal replay pairs."""
    safety_prompts = load_safety_replay_prompts(
        safety_prompts_file=safety_prompts_file,
        exclude_prompts_file=exclude_prompts_file,
        replay_size=replay_size,
        prompt_seed=prompt_seed,
    )
    n_built = 0
    for cell in _unique_training_data_cells(cells):
        source_cache = baseline_path(cell.source, cell.dataset)
        target_cache = baseline_path(cell.target, cell.dataset)
        out_path = safety_dpo_data_path(cell)
        if not source_cache.exists() or not target_cache.exists():
            print(f"  [missing] {cell.slug}: source or target baseline missing")
            continue
        if dry_run:
            print(
                f"  [dry-run] would build {out_path} from source/target baselines "
                f"+ {len(safety_prompts)} safety replay pairs"
            )
            continue

        src = pd.read_csv(source_cache)[["prompt", "model_response"]].rename(
            columns={"model_response": "rejected_response"}
        )
        tgt = pd.read_csv(target_cache)[["prompt", "model_response"]].rename(
            columns={"model_response": "chosen_response"}
        )
        imitation = pd.merge(src, tgt, on="prompt", how="inner")
        imitation = imitation[
            _nonempty_response_filter(imitation, ["prompt", "chosen_response", "rejected_response"])
        ].copy()
        imitation = imitation[["prompt", "chosen_response", "rejected_response"]]
        imitation["row_type"] = "imitation"
        imitation["safety_response_source"] = ""
        replay, replay_meta = _safety_dpo_replay_rows(
            source=cell.source,
            safety_prompts=safety_prompts,
            response_seed=response_seed,
        )
        combined = pd.concat([imitation, replay], ignore_index=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        manifest = _base_safety_manifest(
            cell=cell,
            safety_prompts_file=safety_prompts_file,
            exclude_prompts_file=exclude_prompts_file,
            replay_size=replay_size,
            prompt_seed=prompt_seed,
            response_seed=response_seed,
            safety_prompts=safety_prompts,
        )
        manifest.update(
            {
                "stage": "safety_dpo",
                "source_cache": str(source_cache),
                "target_cache": str(target_cache),
                "imitation_pairs": len(imitation),
                "safety_replay_pairs": len(replay),
                "total_pairs": len(combined),
                **replay_meta,
            }
        )
        _manifest_path(out_path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        n_built += 1
        print(
            f"  [built] {out_path} ({len(imitation)} imitation + {len(replay)} safety replay pairs)"
        )
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
    from dementor.training.tinker_backend import record_adapter_mapping
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
        from dementor.training.pipeline import run_dpo_workflow
        from dementor.training.dpo import PreferenceDatasetConfig

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
        cfg = _build_dpo_cfg(
            cell=cell, ds_cfg=ds_cfg, output_dir=output_dir, sft_state_path=sft_state_path,
            renderer_name=record["renderer_name"], log_path=log_path,
        )

        print(f"[launch] {cell.slug}", flush=True)
        t0 = time.time()
        last_err: Exception | None = None
        result = None
        try:
            result = _retry_call(
                lambda: run_dpo_workflow(cfg),
                attempts=4, base_wait=5, label="dpo", name=cell.slug,
            )
        except Exception as e:
            last_err = e
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
            from dementor.training.tinker_backend import record_adapter_mapping
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

    _dispatch(
        pending,
        lambda it: _run_one(*it),
        parallel=parallel,
        label="DPO jobs",
        key=lambda it: it[0].slug,
        on_success=manifest.append,
        on_error=lambda k, e: manifest.append({"cell": k, "error": str(e)}),
    )

    return {"jobs": manifest, "n_jobs": len(manifest)}


def launch_safety_dpo(
    *,
    cells: list[Cell],
    dry_run: bool,
    only_llama: bool = False,
    max_jobs: int | None = None,
    parallel: int = 1,
) -> dict:
    """Launch safety-constrained DPO jobs, starting from safety_sft adapters."""
    from dotenv import load_dotenv

    load_dotenv()

    registry_path = DATA / "tinker_adapters.json"
    registered: set[str] = set()
    safety_sft_lookup: dict[str, dict] = {}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
            registered = set(registry.keys())
            safety_sft_lookup = {
                k: v for k, v in registry.items()
                if k.startswith("safety_sft_") and isinstance(v, dict)
            }
        except Exception:
            pass

    if only_llama:
        cells = [c for c in cells if c.llama_critical]
    if max_jobs is not None:
        cells = cells[:max_jobs]

    pending: list[tuple[Cell, dict, dict]] = []
    manifest: list[dict] = []
    for cell in cells:
        dpo_alias = f"safety_dpo_{cell.slug}"
        if dpo_alias in registered and not dry_run:
            print(f"  [skip] {cell.slug}: already in registry")
            continue
        pref_csv = safety_dpo_data_path(cell)
        if not pref_csv.exists() and not dry_run:
            print(f"  [missing-data] {cell.slug}: {pref_csv} not built")
            continue
        sft_entry = safety_sft_lookup.get(f"safety_sft_{cell.slug}")
        if not sft_entry and not dry_run:
            print(f"  [missing-safety-sft] {cell.slug}: no safety_sft adapter to start from")
            continue
        sft_state_path = sft_entry.get("checkpoint_path") if sft_entry else None
        pref_rows = None
        if pref_csv.exists():
            try:
                pref_rows = len(pd.read_csv(pref_csv))
            except Exception:
                pref_rows = None

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
            "pref_rows": pref_rows,
            "sft_alias": f"safety_sft_{cell.slug}",
            "sft_checkpoint_path": sft_state_path,
        }
        if dry_run:
            manifest.append(record)
            continue
        pending.append((cell, record, sft_entry))

    if dry_run:
        return {"jobs": manifest, "n_jobs": len(manifest)}

    print(f"\nLaunching {len(pending)} safety-constrained DPO jobs with parallel={parallel}\n", flush=True)

    def _run_one(cell: Cell, record: dict, sft_entry: dict) -> dict:
        from dementor.training.pipeline import run_dpo_workflow
        from dementor.training.dpo import PreferenceDatasetConfig

        pref_csv = Path(record["pref_csv"])
        output_dir = SAFETY_DPO_OUTPUT_DIR / cell.dataset / f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_seed{cell.seed}"
        log_path = output_dir / "logs"
        sft_state_path = sft_entry.get("checkpoint_path")
        train_size = len(pd.read_csv(pref_csv))

        ds_cfg = PreferenceDatasetConfig(
            dataset_csv=pref_csv,
            prompt_column="prompt",
            chosen_column="chosen_response",
            rejected_column="rejected_response",
            train_size=train_size,
            eval_size=0,
            seed=cell.seed,
        )
        cfg = _build_dpo_cfg(
            cell=cell,
            ds_cfg=ds_cfg,
            output_dir=output_dir,
            sft_state_path=sft_state_path,
            renderer_name=record["renderer_name"],
            log_path=log_path,
            weights_name=record["dpo_alias"],
        )

        print(f"[launch] {record['dpo_alias']}", flush=True)
        t0 = time.time()
        last_err: Exception | None = None
        result = None
        try:
            result = _retry_call(
                lambda: run_dpo_workflow(cfg),
                attempts=4,
                base_wait=5,
                label="safety-dpo",
                name=cell.slug,
            )
        except Exception as e:
            last_err = e
        if result is None:
            print(f"  [SAFETY DPO FAILED after 4 attempts] {cell.slug}: {last_err}", flush=True)
            record["error"] = str(last_err)
            return record
        elapsed = time.time() - t0

        state_path, sampler_path = _find_final_dpo_checkpoint(log_path)
        record["dpo_state_path"] = state_path
        record["dpo_sampler_path"] = sampler_path
        record["elapsed_seconds"] = elapsed
        if state_path or sampler_path:
            from dementor.training.tinker_backend import record_adapter_mapping

            record_adapter_mapping(
                record["dpo_alias"],
                sampler_path or state_path,
                registry_path,
                metadata={
                    "base_model": cell.source,
                    "sft_parent": sft_state_path,
                    "sft_alias": record["sft_alias"],
                    "checkpoint_path": state_path,
                    "sampler_path": sampler_path,
                },
            )
        print(f"  -> {record['dpo_alias']} done ({elapsed:.0f}s) | sampler={sampler_path}", flush=True)
        return record

    _dispatch(
        pending,
        lambda it: _run_one(*it),
        parallel=parallel,
        label="safety DPO jobs",
        key=lambda it: it[0].slug,
        on_success=manifest.append,
        on_error=lambda k, e: manifest.append({"cell": k, "error": str(e)}),
    )

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

    # Resolve source straight from each alias — registry-driven, covers legacy + local.
    pending: list[tuple[str, dict]] = []
    skipped = 0
    errors: list[dict] = []
    for alias, entry in sorted(registry.items()):
        prefix = alias.split("_", 1)[0]
        if prefix not in kinds:
            continue
        source = source_id_of(alias)
        if source is None:
            continue
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
                print(f"[export] {alias}", flush=True)
                if config.backend_for(source) == "local":
                    from dementor.training.local_backend import export_local_adapter
                    export_local_adapter(adapter_dir=Path(tinker_path), base_model=source, output_dir=local_dir)
                else:
                    export_adapter_to_peft(tinker_path=tinker_path, base_model=source, output_dir=local_dir)
            print(f"[upload] {alias} -> {repo_id}", flush=True)
            last_err: Exception | None = None
            res = None
            try:
                res = _retry_call(
                    lambda: upload_adapter_to_hf(
                        peft_dir=local_dir,
                        repo_id=repo_id,
                        base_model=source,
                        alias=alias,
                        private=private,
                    ),
                    attempts=max_retries, base_wait=10, label="upload", name=alias,
                )
            except Exception as e:
                last_err = e
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

    def _collect(res):
        nonlocal uploaded
        if "error" in res:
            errors.append(res)
        else:
            uploaded += 1

    _dispatch(
        pending,
        lambda it: _handle_one(*it),
        parallel=parallel,
        label="uploads",
        key=lambda it: it[0],
        on_success=_collect,
        on_error=lambda k, e: errors.append({"alias": k, "error": str(e)}),
    )

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

    # Resolve source straight from each alias — registry-driven, covers legacy + local.
    exported = 0
    skipped = 0
    errors: list[dict] = []
    for alias, entry in sorted(registry.items()):
        prefix = alias.split("_", 1)[0]
        if prefix not in kinds:
            continue
        source = source_id_of(alias)
        if source is None:
            skipped += 1
            continue
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
            if config.backend_for(source) == "local":
                from dementor.training.local_backend import export_local_adapter
                res = export_local_adapter(adapter_dir=Path(tinker_path), base_model=source, output_dir=out_dir)
            else:
                res = export_adapter_to_peft(tinker_path=tinker_path, base_model=source, output_dir=out_dir)
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
        serialized by a threading.Lock in dementor.training.tinker_backend.
    """
    from dotenv import load_dotenv

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

    from dementor.training.pipeline import run_sft_workflow
    from dementor.training.tinker_backend import SFTDatasetConfig

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
        sft_cfg = _build_sft_cfg(
            cell=cell, ds_cfg=ds_cfg, output_dir=output_dir, weights_name=weights_name,
            prompt_template=prompt_template, completion_template=completion_template,
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
        try:
            result = _retry_call(
                lambda: run_sft_workflow(sft_cfg),
                attempts=4, base_wait=5, label="sft", name=cell.slug, sleep_verb="sleeping",
            )
        except Exception as e:
            last_err = e
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
            entry = registry.get(record["weights_name"], {})
            if isinstance(entry, dict):
                checkpoint_path = entry.get("checkpoint_path")
        except Exception:
            pass
        record["checkpoint_path"] = checkpoint_path

        # NOTE: inline PEFT export disabled — Tinker's get_checkpoint_archive_url
        # consistently times out at ~60s ("Creating checkpoint archive ... this may
        # take a while" then APIConnectionError). Adapters are saved on Tinker as
        # both sampler URIs (path) and state URIs (checkpoint_path). Use
        # dementor.training.matrix backfill-export later once Tinker download is fixed.
        print(f"  -> {cell.slug} sft done ({elapsed:.0f}s) | sampler={sampler_path}", flush=True)
        return record

    _dispatch(
        pending,
        lambda it: _run_one(*it),
        parallel=parallel,
        label="SFT jobs",
        key=lambda it: it[0].slug,
        on_success=manifest.append,
        on_error=lambda k, e: manifest.append({"cell": k, "error": str(e)}),
    )

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
# Single local cell launcher (accelerate-launch / torchrun friendly)
# ============================================================================


def _dist_writer_rank() -> int:
    """Global process rank for the *shared-filesystem* single-writer guard.

    Prefer the GLOBAL ``RANK`` over ``LOCAL_RANK`` so exactly one process (global rank
    0) writes the shared preference JSONL. On multi-node every node has a local-rank-0,
    so keying the guard off ``LOCAL_RANK`` would let each node's local-rank-0 write the
    same shared file concurrently and corrupt it. Falls back to ``LOCAL_RANK`` only when
    ``RANK`` is unset, and to 0 when not launched distributed.
    """
    import os

    for key in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(key)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


def _dpo_prep_token(seed: int, explicit: str | None = None) -> str:
    """Per-launch token stamped into the rank-0 DPO data-prep handshake flag.

    Must be IDENTICAL across every rank of one launch (so non-rank-0 waiters recognize
    *this* invocation's flag) yet DIFFER across launches (so a stale ``.prepared`` left
    in a non-empty ``preference_artifacts/`` by a prior run is rejected instead of waved
    through on mere existence). ``accelerate launch`` / ``torchrun`` hand every rank the
    same ``TORCHELASTIC_RUN_ID`` -- a fresh uuid4 per launch in the single-node default
    -- which is exactly such a value; an operator can force one via ``$DEMENTOR_PREP_TOKEN``
    (multi-node / pinned rdzv-id). The uuid4 fallback is reached only when NOT launched
    distributed (a lone process with no waiters), so cross-rank agreement is moot there.
    """
    import os
    import uuid

    if explicit:
        return explicit
    shared = os.environ.get("DEMENTOR_PREP_TOKEN") or os.environ.get("TORCHELASTIC_RUN_ID")
    return f"{shared}:{seed}" if shared else uuid.uuid4().hex


def _wait_for_prepared(flag: Path, token: str, *, timeout: float = 1800.0, poll: float = 2.0) -> None:
    """Block until ``flag`` exists AND its content equals ``token``, or raise on timeout.

    Gating on the flag's CONTENT (not mere existence) closes the TOCTOU race on a re-run
    into a non-empty dir: a stale ``.prepared`` from a prior launch carries a different
    token and is ignored until rank 0 republishes the flag for THIS launch (after it has
    atomically put train.jsonl in place).
    """
    start = time.time()
    while True:
        try:
            if flag.read_text() == token:
                return
        except OSError:  # not created yet / mid-write (FileNotFoundError is an OSError)
            pass
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out after {timeout:.0f}s waiting for {flag} (rank-0 DPO data prep)")
        time.sleep(poll)


def launch_local_cell(
    *,
    source: str,
    target: str,
    dataset: str,
    seed: int,
    phase: str = "all",
    per_device_batch_size: int | None = None,
    grad_accum: int | None = None,
    epochs: int | None = None,
    prep_token: str | None = None,
) -> dict:
    """Run ONE local (gemma-4) cell's LoRA SFT and/or DPO in the current process.

    This is the entrypoint to run under ``accelerate launch`` / ``torchrun`` for real
    multi-GPU (e.g. 4xH100) training: the HF Trainer inside ``run_local_sft_job`` /
    ``run_local_dpo_job`` owns DDP device placement and rank-aware saving. (The matrix
    ``launch-sft`` / ``launch-dpo`` commands iterate every cell with a thread pool and are
    NOT safe to wrap in ``accelerate launch`` — each rank would re-run the whole matrix.)

    Build the data first as plain processes (``build-sft-data`` / ``build-dpo-data``); the
    DPO preference JSONL is materialized once by rank 0 here, other ranks wait for it.
    ``phase`` is one of ``sft`` / ``dpo`` / ``all``. ``prep_token`` overrides the per-launch
    DPO data-prep handshake token (else derived from ``$DEMENTOR_PREP_TOKEN`` /
    ``$TORCHELASTIC_RUN_ID``); pass a shared value to force multi-node agreement.
    """
    from dementor.training.local_backend import (
        LocalDPOParams,
        run_local_dpo_job,
        run_local_sft_job,
    )
    from dementor.training.tinker_backend import SFTDatasetConfig

    if config.backend_for(source) != "local":
        raise ValueError(
            f"{source} is not a local-backend model. Use launch-sft/launch-dpo for Tinker models."
        )

    cell = Cell(source=source, target=target, dataset=dataset, seed=seed)
    prompt_template, completion_template = DATASET_TEMPLATES[cell.dataset]
    name = f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_seed{cell.seed}"
    sft_out_dir = SFT_OUTPUT_DIR / cell.dataset / name
    dpo_out_dir = DPO_OUTPUT_DIR / cell.dataset / name
    sft_hp = config.sft()
    dpo_hp = config.dpo()
    summary: dict = {"cell": cell.slug, "source": source, "target": target, "phase": phase}

    if phase in ("sft", "all"):
        sft_csv = sft_data_path(cell)
        if not sft_csv.exists():
            raise FileNotFoundError(f"SFT data missing: {sft_csv} (run `dementor-matrix build-sft-data`).")
        ds_cfg = SFTDatasetConfig(
            train_csv=sft_csv, eval_csv=None, prompt_column="prompt",
            completion_column="model_response", train_size=sft_hp.get("train_size", 500),
            eval_size=0, seed=cell.seed,
        )
        run_local_sft_job(
            dataset_config=ds_cfg, base_model=cell.source,
            batch_size=per_device_batch_size or sft_hp["batch_size"],
            epochs=epochs or sft_hp["epochs"], learning_rate=sft_hp["learning_rate"],
            prompt_template=prompt_template, completion_template=completion_template,
            weights_name=f"sft_{cell.slug}", output_dir=sft_out_dir,
            registry_path=config.registry_path(), seed=cell.seed,
            lora_kwargs=config.lora(), gradient_accumulation_steps=grad_accum,
        )
        summary["sft_output_dir"] = str(sft_out_dir)

    if phase in ("dpo", "all"):
        from dementor.training.dpo import (
            PreferenceDatasetConfig,
            prepare_preference_examples,
            write_preference_artifacts,
        )

        pref_csv = dpo_data_path(cell)
        if not pref_csv.exists():
            raise FileNotFoundError(f"DPO data missing: {pref_csv} (run `dementor-matrix build-dpo-data`).")
        artifacts_dir = dpo_out_dir / "preference_artifacts"
        train_jsonl = artifacts_dir / "train.jsonl"
        eval_jsonl = artifacts_dir / "eval.jsonl"
        ready_flag = artifacts_dir / ".prepared"
        token = _dpo_prep_token(cell.seed, prep_token)
        # Materialize the preference JSONL exactly once (global rank 0); concurrent writers
        # would corrupt it and the torch process group isn't up yet, so coordinate via a flag
        # file. Rank 0 stages the artifacts then os.replace()s train.jsonl into place (atomic)
        # and publishes a flag stamped with THIS launch's token; waiters block on the token --
        # not mere existence -- so a re-run can't read a stale/partial JSONL from a prior run.
        if _dist_writer_rank() == 0:
            import os
            import shutil

            artifacts_dir.mkdir(parents=True, exist_ok=True)
            ready_flag.unlink(missing_ok=True)  # invalidate any prior-launch flag up front
            pref_cfg = PreferenceDatasetConfig(
                dataset_csv=pref_csv, prompt_column="prompt",
                chosen_column="chosen_response", rejected_column="rejected_response",
                train_size=dpo_hp.get("train_size", 500), eval_size=0, seed=cell.seed,
            )
            train_ex, eval_ex = prepare_preference_examples(pref_cfg)
            staging = artifacts_dir / ".staging"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)
            arts = write_preference_artifacts(
                output_dir=staging, train_examples=train_ex, eval_examples=eval_ex,
                metadata=None, stub=None,
            )
            # Atomically publish each artifact (same FS -> os.replace is atomic) so no reader
            # ever sees a half-written train.jsonl, even off the flag path.
            for src in (arts.train_jsonl, arts.eval_jsonl, arts.train_csv, arts.eval_csv):
                if src is not None and Path(src).exists():
                    os.replace(src, artifacts_dir / Path(src).name)
            shutil.rmtree(staging, ignore_errors=True)
            # Publish the handshake flag LAST, atomically (temp + os.replace): its presence
            # AND matching token guarantee the atomically-replaced train.jsonl is complete.
            tmp_flag = artifacts_dir / ".prepared.tmp"
            tmp_flag.write_text(token)
            os.replace(tmp_flag, ready_flag)
        else:
            _wait_for_prepared(ready_flag, token)
        eval_arg = eval_jsonl if (eval_jsonl.exists() and eval_jsonl.stat().st_size > 0) else None
        params = LocalDPOParams(
            model_name=cell.source, load_checkpoint_path=str(sft_out_dir),
            weights_name=f"dpo_{cell.slug}", registry_path=config.registry_path(),
            learning_rate=dpo_hp["learning_rate"], dpo_beta=dpo_hp["dpo_beta"],
            num_epochs=dpo_hp["num_epochs"],
            batch_size=per_device_batch_size or dpo_hp["batch_size"],
            max_length=dpo_hp["max_length"], lora_rank=dpo_hp["lora_rank"],
            seed=cell.seed, gradient_accumulation_steps=grad_accum,
        )
        run_local_dpo_job(train_jsonl=train_jsonl, eval_jsonl=eval_arg, params=params, output_dir=dpo_out_dir)
        summary["dpo_output_dir"] = str(dpo_out_dir)

    return summary


# ============================================================================
# CLI
# ============================================================================


def _resolve_model_arg(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return config.model(value)["id"]
    except KeyError as exc:
        raise SystemExit(f"Unknown model id/slug: {value}") from exc


def _filtered_cells_from_args(args) -> list[Cell]:
    cells = list(iter_cells())
    source = _resolve_model_arg(getattr(args, "source", None))
    target = _resolve_model_arg(getattr(args, "target", None))
    dataset = getattr(args, "dataset", None)
    seed = getattr(args, "seed", None)
    if source is not None:
        cells = [c for c in cells if c.source == source]
    if target is not None:
        cells = [c for c in cells if c.target == target]
    if dataset is not None:
        cells = [c for c in cells if c.dataset == dataset]
    if seed is not None:
        cells = [c for c in cells if c.seed == seed]
    if getattr(args, "only_llama", False):
        cells = [c for c in cells if c.llama_critical]
    max_cells = getattr(args, "max_cells", None)
    if max_cells is not None:
        cells = cells[:max_cells]
    return cells


def _add_cell_filter_args(p) -> None:
    p.add_argument("--source", default=None, help="Source model id or slug")
    p.add_argument("--target", default=None, help="Target model id or slug")
    p.add_argument("--dataset", choices=list(TRAIN_DATASETS), default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--only-llama", action="store_true")
    p.add_argument("--max-cells", type=int, default=None)


def _add_safety_data_args(p) -> None:
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--safety-prompts-file", type=Path, default=DEFAULT_SAFETY_PROMPTS)
    p.add_argument("--exclude-prompts-file", type=Path, default=DEFAULT_SAFETY_EXCLUDE_PROMPTS)
    p.add_argument(
        "--include-eval-prompts",
        action="store_true",
        help="Do not exclude the default refusal eval prompts from replay data.",
    )
    p.add_argument("--safety-replay-size", type=int, default=DEFAULT_SAFETY_REPLAY_SIZE)
    p.add_argument("--safety-prompt-seed", type=int, default=42)
    p.add_argument("--safety-response-seed", type=int, default=1)
    _add_cell_filter_args(p)


def _safety_exclude_path(args) -> Path | None:
    return None if getattr(args, "include_eval_prompts", False) else args.exclude_prompts_file


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

    safety_list_p = sub.add_parser(
        "list-safety-cells",
        help="Print filtered safety-constrained imitation cells without launching anything.",
    )
    _add_cell_filter_args(safety_list_p)

    safety_sft_build_p = sub.add_parser(
        "build-safety-sft-data",
        help="Build safety-constrained SFT CSVs: target imitation + refusal replay.",
    )
    _add_safety_data_args(safety_sft_build_p)

    safety_dpo_build_p = sub.add_parser(
        "build-safety-dpo-data",
        help="Build safety-constrained DPO CSVs: imitation pairs + refusal replay pairs.",
    )
    _add_safety_data_args(safety_dpo_build_p)

    safety_sft_p = sub.add_parser(
        "launch-safety-sft",
        help="Submit safety-constrained SFT jobs.",
    )
    safety_sft_p.add_argument("--dry-run", action="store_true")
    safety_sft_p.add_argument("--manifest-out", type=Path, default=None)
    safety_sft_p.add_argument("--parallel", type=int, default=1)
    _add_cell_filter_args(safety_sft_p)

    safety_dpo_p = sub.add_parser(
        "launch-safety-dpo",
        help="Submit safety-constrained DPO jobs on top of safety_sft adapters.",
    )
    safety_dpo_p.add_argument("--dry-run", action="store_true")
    safety_dpo_p.add_argument("--manifest-out", type=Path, default=None)
    safety_dpo_p.add_argument("--parallel", type=int, default=1)
    _add_cell_filter_args(safety_dpo_p)

    cell_p = sub.add_parser(
        "launch-local-cell",
        help="Run ONE local (gemma-4) cell's LoRA SFT and/or DPO. Wrap in `accelerate launch` for multi-GPU.",
    )
    cell_p.add_argument("--source", required=True, help="Local-backend source model id (e.g. google/gemma-4-E4B-it)")
    cell_p.add_argument("--target", required=True, help="Target model id the source is disguised as")
    cell_p.add_argument("--dataset", required=True, choices=list(TRAIN_DATASETS))
    cell_p.add_argument("--seed", type=int, required=True)
    cell_p.add_argument("--phase", choices=["sft", "dpo", "all"], default="all")
    cell_p.add_argument("--per-device-batch-size", type=int, default=None,
                        help="Override per-device train batch size (lower for the 31B/26B models).")
    cell_p.add_argument("--grad-accum", type=int, default=None,
                        help="Gradient accumulation steps (preserve effective batch when lowering per-device).")
    cell_p.add_argument("--epochs", type=int, default=None, help="Override SFT epochs.")

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
        print(f"\nTotal cells: {len(cells)}")
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

    if args.cmd == "list-safety-cells":
        cells = _filtered_cells_from_args(args)
        for c in cells:
            tag = "[LLAMA-CRIT]" if c.llama_critical else "           "
            print(f"  {tag} safety_sft/safety_dpo {c.slug}")
        print(f"\nTotal safety-constrained cells: {len(cells)}")
        return 0

    if args.cmd == "build-safety-sft-data":
        cells = _filtered_cells_from_args(args)
        n = build_safety_sft_data(
            dry_run=args.dry_run,
            cells=cells,
            safety_prompts_file=args.safety_prompts_file,
            exclude_prompts_file=_safety_exclude_path(args),
            replay_size=args.safety_replay_size,
            prompt_seed=args.safety_prompt_seed,
            response_seed=args.safety_response_seed,
        )
        print(f"\nBuilt {n} safety-constrained SFT training CSVs")
        return 0

    if args.cmd == "build-safety-dpo-data":
        cells = _filtered_cells_from_args(args)
        n = build_safety_dpo_data(
            dry_run=args.dry_run,
            cells=cells,
            safety_prompts_file=args.safety_prompts_file,
            exclude_prompts_file=_safety_exclude_path(args),
            replay_size=args.safety_replay_size,
            prompt_seed=args.safety_prompt_seed,
            response_seed=args.safety_response_seed,
        )
        print(f"\nBuilt {n} safety-constrained DPO preference CSVs")
        return 0

    if args.cmd == "launch-safety-sft":
        cells = _filtered_cells_from_args(args)
        manifest = launch_sft(
            cells=cells,
            dry_run=args.dry_run,
            only_llama=False,
            max_jobs=None,
            parallel=args.parallel,
            alias_prefix="safety_sft",
            data_path_fn=safety_sft_data_path,
            output_root=SAFETY_SFT_OUTPUT_DIR,
        )
        out_path = args.manifest_out or (DATA / "results" / "matrix" / "safety_sft_manifest.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nManifest: {out_path}")
        print(f"Total jobs: {manifest['n_jobs']}")
        if args.dry_run:
            print("(dry-run — no jobs submitted)")
        return 0

    if args.cmd == "launch-safety-dpo":
        cells = _filtered_cells_from_args(args)
        manifest = launch_safety_dpo(
            cells=cells,
            dry_run=args.dry_run,
            only_llama=False,
            max_jobs=None,
            parallel=args.parallel,
        )
        out_path = args.manifest_out or (DATA / "results" / "matrix" / "safety_dpo_manifest.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nManifest: {out_path}")
        print(f"Total jobs: {manifest['n_jobs']}")
        if args.dry_run:
            print("(dry-run — no jobs submitted)")
        return 0

    if args.cmd == "launch-local-cell":
        summary = launch_local_cell(
            source=args.source,
            target=args.target,
            dataset=args.dataset,
            seed=args.seed,
            phase=args.phase,
            per_device_batch_size=args.per_device_batch_size,
            grad_accum=args.grad_accum,
            epochs=args.epochs,
        )
        print(json.dumps(summary, indent=2, default=str))
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
