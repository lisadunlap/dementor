"""DPO preference-data builder, checkpoint-registry backfill, and DPO launchers.

Builds (prompt, chosen=target, rejected=source) preference CSVs, parses the
cookbook DPO logs to recover final checkpoint URIs, and launches both the plain
cross-imitation DPO jobs (starting from the matching SFT adapter) and the
safety-constrained DPO jobs (starting from the safety_sft adapter).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ._cells import (
    Cell,
    baseline_path,
    dpo_data_path,
    iter_cells,
    safety_dpo_data_path,
)
from ._concurrency import _dispatch, _retry_call
from ._configs import _build_dpo_cfg
from ._constants import (
    DATA,
    DPO_OUTPUT_DIR,
    MODEL_SLUG,
    SAFETY_DPO_OUTPUT_DIR,
    SEEDS,
)
from ._models import renderer_for


def build_dpo_data(*, dry_run: bool = False, datasets: list[str] | None = None) -> int:
    """For each (source, target, dataset) cell, build a DPO preference CSV.

    Schema: (prompt, chosen_response, rejected_response).
    chosen = TARGET's response to train prompt
    rejected = SOURCE's response to the SAME train prompt
    Both sides come from data/model-responses/matrix_baselines/{dataset}/{model_slug}_train.csv.

    ``datasets`` optionally restricts building to a subset of dataset names.
    """
    n_built = 0
    seen: set[tuple[str, str, str]] = set()
    for cell in iter_cells():
        if datasets is not None and cell.dataset not in datasets:
            continue
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
