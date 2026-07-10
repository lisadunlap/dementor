"""Safety-constrained imitation data: target imitation mixed with refusal replay.

Loads a deterministic (optionally eval-disjoint) set of harmful prompts, pairs
them with the source model's native refusal snippet (falling back to a generic
refusal), and folds those replay rows into the SFT / DPO training data so the
fine-tune preserves refusal behaviour while imitating the target. Each output
CSV is accompanied by a JSON manifest describing exactly how it was built.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from ._cells import (
    Cell,
    baseline_path,
    safety_dpo_data_path,
    safety_sft_data_path,
    _manifest_path,
    _unique_training_data_cells,
)
from ._constants import (
    DEFAULT_SAFETY_EXCLUDE_PROMPTS,
    DEFAULT_SAFETY_PROMPTS,
    DEFAULT_SAFETY_REPLAY_SIZE,
    GENERIC_NONREFUSAL_RESPONSE,
    GENERIC_REFUSAL_RESPONSE,
    MODEL_SLUG,
)


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
    # Resolve the directory through the package at call time so a monkeypatched
    # matrix.SAFETY_NATIVE_REFUSAL_DIR is still honoured after the split.
    from dementor.training import matrix as _matrix
    candidate_seeds = []
    for seed in (preferred_seed, 1, 2, 3, 42, 43, 44):
        if seed not in candidate_seeds:
            candidate_seeds.append(seed)
    for seed in candidate_seeds:
        path = _matrix.SAFETY_NATIVE_REFUSAL_DIR / f"{MODEL_SLUG[model]}_seed{seed}.csv"
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
