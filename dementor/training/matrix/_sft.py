"""SFT training-data builders and Tinker SFT job launchers.

Covers the cross-imitation SFT data (target's response as the completion), the
self-SFT drift controls (model trains on its own outputs), the generic SFT job
launcher (reused for self-SFT and safety-SFT via ``alias_prefix`` / ``data_path_fn``
/ ``output_root``), and the self-SFT status monitor.
"""
from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd

from ._cells import (
    Cell,
    baseline_path,
    iter_cells,
    iter_self_sft_cells,
    self_sft_data_path,
    sft_data_path,
)
from ._concurrency import _dispatch, _retry_call
from ._configs import _build_sft_cfg
from ._constants import (
    DATA,
    DATASET_TEMPLATES,
    MODEL_SLUG,
    SEEDS,
    SELF_SFT_OUTPUT_DIR,
    SFT_OUTPUT_DIR,
)
from ._models import renderer_for


def build_sft_data(*, dry_run: bool = False, datasets: list[str] | None = None) -> int:
    """For each (source, target, dataset) cell, build the SFT training CSV.

    SFT input = train prompt
    SFT completion = TARGET's response to that prompt (from the baseline cache)

    Source is irrelevant for the data itself (it determines which base model gets
    fine-tuned, not what the training data contains). So we de-dup across seeds
    and sources: per (target, dataset) we just slice the cache. Cell-level CSVs
    point to that shared cache for clarity.

    ``datasets`` optionally restricts building to a subset of dataset names (e.g.
    ``["chatbot_arena"]``) so callers that only need one dataset don't rebuild the rest.
    """
    n_built = 0
    seen: set[tuple[str, str]] = set()
    for cell in iter_cells():
        if datasets is not None and cell.dataset not in datasets:
            continue
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


def launch_sft(
    *,
    cells: list[Cell],
    dry_run: bool,
    only_llama: bool = False,
    max_jobs: int | None = None,
    parallel: int = 1,
    alias_prefix: str = "sft",
    data_path_fn=sft_data_path,
    output_root=SFT_OUTPUT_DIR,
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
