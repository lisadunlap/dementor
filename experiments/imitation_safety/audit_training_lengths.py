#!/usr/bin/env python3
"""Audit effective local-DPO sequence caps against the executed preference pairs.

The publication campaign used backend-specific memory controls.  This CPU-only audit
replays TRL's non-conversational tokenization rule (prompt + completion + EOS) with each
configured source tokenizer and counts examples for which either chosen or rejected
sequence exceeds the effective cap.  It does not approximate Tinker's renderer-side
tokenization, so Tinker settings are documented but excluded from truncation rates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dementor import config


REPO = config.project_root()
SUMMARY = REPO / "data" / "results" / "safety" / "erosion_seed42_summary.csv"
PAIRS = REPO / "data" / "results" / "matrix" / "dpo_data"
OUTPUT = REPO / "paper" / "naz_aaai2027" / "local_dpo_training_length_audit.json"


def effective_local_settings(model: dict) -> dict:
    """Return settings applied by the executed local launcher for this source."""
    hp = config.dpo()
    model_parallel = model.get("local_training") == "model_parallel"
    return {
        "profile": "local_model_parallel" if model_parallel else "local_single_gpu",
        "max_length": min(int(hp["max_length"]), 1024 if model_parallel else 1536),
        "per_device_batch_size": 1 if model_parallel else min(int(hp["batch_size"]), 2),
        "gradient_accumulation_steps": 8,
    }


def _lengths(tokenizer, frame: pd.DataFrame, batch_size: int) -> list[int]:
    eos = tokenizer.eos_token or ""
    prompts = frame["prompt"].fillna("").astype(str).tolist()
    chosen = frame["chosen_response"].fillna("").astype(str).tolist()
    rejected = frame["rejected_response"].fillna("").astype(str).tolist()
    if eos:
        chosen = [text if text.endswith(eos) else text + eos for text in chosen]
        rejected = [text if text.endswith(eos) else text + eos for text in rejected]

    output: list[int] = []
    for start in range(0, len(frame), batch_size):
        prompt_batch = prompts[start:start + batch_size]
        chosen_batch = chosen[start:start + batch_size]
        rejected_batch = rejected[start:start + batch_size]
        chosen_lengths = tokenizer(
            [prompt + response for prompt, response in zip(prompt_batch, chosen_batch)],
            add_special_tokens=True,
            truncation=False,
            return_length=True,
        )["length"]
        rejected_lengths = tokenizer(
            [prompt + response for prompt, response in zip(prompt_batch, rejected_batch)],
            add_special_tokens=True,
            truncation=False,
            return_length=True,
        )["length"]
        output.extend(
            max(int(chosen_length), int(rejected_length))
            for chosen_length, rejected_length in zip(chosen_lengths, rejected_lengths)
        )
    return output


def _summarize(lengths: list[int], cap: int) -> dict:
    values = np.asarray(lengths, dtype=int)
    truncated = values > cap
    return {
        "examples": int(len(values)),
        "truncated_examples": int(truncated.sum()),
        "truncated_pct": float(100.0 * truncated.mean()),
        "length_p50": float(np.percentile(values, 50)),
        "length_p95": float(np.percentile(values, 95)),
        "length_p99": float(np.percentile(values, 99)),
        "length_max": int(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY)
    parser.add_argument("--pairs-root", type=Path, default=PAIRS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    models = {model["slug"]: model for model in config.campaign_roster("imitation_safety")}
    summary = pd.read_csv(args.summary_csv)
    cells = summary[
        (summary["stage"] == "dpo")
        & (summary["source"] != summary["target"])
        & summary["source"].map(lambda slug: models[slug]["backend"] == "local")
    ].copy()
    if len(cells) != 396:
        raise ValueError(f"expected 396 local DPO cells, found {len(cells)}")

    result = {
        "method": {
            "unit": "DPO preference example",
            "length": "max(tokens(prompt+chosen+EOS), tokens(prompt+rejected+EOS))",
            "tokenizer": "configured source-model tokenizer",
            "truncated": "length exceeds the effective local DPO max_length",
            "scope": "local backend only; Tinker renderer tokenization is not reconstructed",
        },
        "configured_tinker_settings": {
            "max_length": int(config.dpo()["max_length"]),
            "batch_size": int(config.dpo()["batch_size"]),
        },
        "sources": {},
    }
    all_lengths_by_profile: dict[str, list[int]] = {}
    profile_cells: dict[str, int] = {}

    for source, source_cells in cells.groupby("source"):
        model = models[source]
        settings = effective_local_settings(model)
        tokenizer = AutoTokenizer.from_pretrained(
            model["id"], local_files_only=args.local_files_only
        )
        lengths: list[int] = []
        for row in source_cells.itertuples(index=False):
            pair_path = args.pairs_root / row.dataset / f"{row.source}_as_{row.target}_pairs.csv"
            if not pair_path.is_file():
                raise FileNotFoundError(pair_path)
            lengths.extend(
                _lengths(tokenizer, pd.read_csv(pair_path), args.tokenizer_batch_size)
            )
        summary_stats = _summarize(lengths, settings["max_length"])
        result["sources"][source] = {
            "model_id": model["id"],
            "cells": int(len(source_cells)),
            **settings,
            **summary_stats,
        }
        profile = settings["profile"]
        all_lengths_by_profile.setdefault(profile, []).extend(lengths)
        profile_cells[profile] = profile_cells.get(profile, 0) + len(source_cells)
        print(
            f"{source}: {len(lengths)} examples, cap={settings['max_length']}, "
            f"truncated={summary_stats['truncated_pct']:.3f}%",
            flush=True,
        )

    profiles = {}
    for profile, lengths in all_lengths_by_profile.items():
        cap = 1024 if profile == "local_model_parallel" else 1536
        profiles[profile] = {
            "cells": int(profile_cells[profile]),
            "max_length": cap,
            **_summarize(lengths, cap),
        }
    result["profiles"] = profiles
    total_examples = sum(profile["examples"] for profile in profiles.values())
    total_truncated = sum(profile["truncated_examples"] for profile in profiles.values())
    result["aggregate"] = {
        "cells": int(sum(profile["cells"] for profile in profiles.values())),
        "examples": int(total_examples),
        "truncated_examples": int(total_truncated),
        "truncated_pct": float(100.0 * total_truncated / total_examples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
