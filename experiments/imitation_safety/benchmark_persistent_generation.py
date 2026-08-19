#!/usr/bin/env python3
"""Exact-text adoption gate for persistent composed-adapter generation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import gc
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import erosion_common as EC  # noqa: E402
from run_erosion_source import attach_training_composition  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("item_id")
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=EC.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--gen-batch", type=int, default=32)
    parser.add_argument("--fresh-legacy-reference", action="store_true",
                        help="generate the reference with load_gen_model in this runtime")
    args = parser.parse_args()

    item = EC.find_item(args.item_id)
    if not item or not item.get("sft_parent"):
        raise SystemExit(f"not a local composed DPO item: {args.item_id}")
    reference = pd.read_csv(args.reference_csv).iloc[:args.limit]
    prompts = reference["prompt"].astype(str).tolist()
    from dementor.training.matrix import clean_response
    expected = [clean_response(item["base_model"], value)
                for value in reference["model_response"].fillna("").astype(str)]

    legacy_seconds = None
    if args.fresh_legacy_reference:
        legacy_started = time.time()
        legacy_tok, legacy_model, legacy_dev = EC.load_gen_model(
            item["base_model"], item["adapter_dir"], sft_parent=item["sft_parent"]
        )
        expected = EC.generate_responses(
            legacy_tok, legacy_model, legacy_dev, item["base_model"], prompts,
            max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch,
        )
        legacy_seconds = time.time() - legacy_started
        del legacy_model, legacy_tok
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    started = time.time()
    tok, model, input_dev = EC.load_gen_base(item["base_model"])
    loaded = time.time()
    model, _ = attach_training_composition(model, item)
    actual = EC.generate_responses(
        tok, model, input_dev, item["base_model"], prompts,
        max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch,
    )
    finished = time.time()
    exact = [left == right for left, right in zip(actual, expected)]
    result = {
        "item_id": item["id"],
        "reference_csv": str(args.reference_csv.resolve()),
        "composition": "resident_base+sft_merged+dpo_lora",
        "n": len(exact),
        "exact_text_matches": sum(exact),
        "exact_text_match_rate": sum(exact) / len(exact) if exact else None,
        "base_load_seconds": loaded - started,
        "generation_seconds": finished - loaded,
        "legacy_reference_seconds": legacy_seconds,
        "first_mismatches": [
            {"index": index, "expected": expected[index], "actual": actual[index]}
            for index, matches in enumerate(exact) if not matches
        ][:5],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(exact):
        raise SystemExit("persistent generation failed exact-text adoption gate")


if __name__ == "__main__":
    main()
