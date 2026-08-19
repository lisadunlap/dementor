#!/usr/bin/env python3
"""Generate a group of local DPO cells while loading their source model only once.

Each DPO checkpoint represents a delta trained after its SFT parent was merged.  The exact additive
``SFT + DPO`` LoRA is materialized once, attached to the resident base for one cell, then removed.
Safety and fidelity CSV checkpoints use the same paths and generation function as
``run_erosion_item.py``, so this worker is resumable and interchangeable with the legacy runner.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import erosion_common as EC  # noqa: E402
import fidelity_common as FC  # noqa: E402


def item_generation_done(item_id: str, benchmarks: list[str], also_fidelity: bool) -> bool:
    root = Path(EC.WORK) / item_id
    safety = all((root / benchmark / "all_gens.csv").is_file() for benchmark in benchmarks)
    return safety and (not also_fidelity or FC.gens_done(item_id))


def source_groups(items: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(item["base_model"], []).append(item)
    return groups


def attach_training_composition(model, item: dict):
    """Reproduce ``merge SFT, then attach DPO`` without reloading the resident base.

    PEFT's additive adapter composition is mathematically equivalent but bf16 rounding can change
    greedy decoding.  This path deliberately performs the same operations as the validated legacy
    loader.  Before merging, it snapshots only the LoRA-targeted base tensors; after generation
    those tensors are restored bit-for-bit.
    """
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer

    sft = PeftModel.from_pretrained(model, item["sft_parent"])
    snapshots = []
    for module in sft.modules():
        if not isinstance(module, LoraLayer):
            continue
        for parameter_name in ("weight", "bias"):
            parameter = getattr(module.base_layer, parameter_name, None)
            if parameter is not None:
                snapshots.append((parameter, parameter.detach().cpu().clone()))
    merged = sft.merge_and_unload()
    dpo = PeftModel.from_pretrained(merged, item["adapter_dir"])
    dpo.eval()
    return dpo, snapshots


def restore_resident_base(model, snapshots):
    """Unload the unmerged DPO wrapper and restore pristine targeted base tensors."""
    base = model.unload()
    for parameter, original in snapshots:
        parameter.data.copy_(original.to(parameter.device))
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return base


def generate_item(item: dict, tok, model, input_dev, args, benchmarks: list[str]):
    item_id = item["id"]
    output_dir = Path(EC.item_dir(item_id))
    logf = str(output_dir / "run.log")
    started = time.time()
    EC.log(f"=== PERSISTENT START {item_id} base={item['base_model']} CUDA="
           f"{os.environ.get('CUDA_VISIBLE_DEVICES')} ===", logf)
    snapshots = None
    failure = None
    try:
        needed = []
        for benchmark in benchmarks:
            benchmark_dir = output_dir / benchmark
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            if not (benchmark_dir / "all_gens.csv").is_file():
                needed.append(benchmark)
        fidelity_needed = args.also_fidelity and not FC.gens_done(item_id)
        if not needed and not fidelity_needed:
            EC.log(f"=== PERSISTENT SKIP {item_id}: generations complete ===", logf)
            return model

        model, snapshots = attach_training_composition(model, item)
        for benchmark in needed:
            frame = pd.read_csv(EC.get_subsample(
                benchmark, args.max_prompts, args.subsample_seed, logf
            ))
            prompts = frame["prompt"].astype(str).tolist()
            EC.log(f"[A] {benchmark}: generating {len(prompts)} prompts", logf)
            responses = EC.generate_responses(
                tok, model, input_dev, item["base_model"], prompts,
                max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch, logf=logf,
            )
            result = pd.DataFrame({
                "prompt": prompts,
                "model_response": [r if str(r).strip() else " " for r in responses],
                "benchmark": benchmark,
                "label": frame["label"] if "label" in frame else "harmful",
                "expected": frame["expected"] if "expected" in frame else "refuse",
                "category": (frame["category"].fillna("").astype(str)
                             if "category" in frame else ""),
            })
            EC._atomic_write_csv(result, str(output_dir / benchmark / "all_gens.csv"))

        if fidelity_needed:
            prompts = FC.load_heldout_prompts(item["dataset"])
            EC.log(f"[A] fidelity/{item['dataset']}: generating {len(prompts)} prompts", logf)
            responses = EC.generate_responses(
                tok, model, input_dev, item["base_model"], prompts,
                max_new_tokens=args.fidelity_max_new_tokens,
                batch_size=args.gen_batch, logf=logf,
            )
            FC._write_gens_csv(FC.adapter_gens_path(item_id), prompts, responses)

        error = output_dir / "ERROR.json"
        if error.exists():
            error.unlink()
        EC.log(f"=== PERSISTENT GENERATED {item_id} in {(time.time()-started)/60:.1f}m ===", logf)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        EC.log(f"=== PERSISTENT ERROR {item_id}: {exc} ===\n{tb[-2500:]}", logf)
        (output_dir / "ERROR.json").write_text(json.dumps({
            "id": item_id, "status": "error", "error": str(exc), "tb": tb[-2000:],
        }, indent=2) + "\n")
        failure = exc
    finally:
        if snapshots is not None:
            model = restore_resident_base(model, snapshots)
    if failure is not None:
        raise failure
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="source slug or exact base-model id")
    parser.add_argument("--items", help="optional comma-separated item IDs")
    parser.add_argument("--benchmarks", default=",".join(EC.DEFAULT_BENCHMARKS))
    parser.add_argument("--max-prompts", type=int, default=EC.DEFAULT_MAX_PROMPTS)
    parser.add_argument("--subsample-seed", type=int, default=EC.DEFAULT_SUBSAMPLE_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=EC.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--fidelity-max-new-tokens", type=int, default=512)
    parser.add_argument("--gen-batch", type=int, default=32)
    parser.add_argument("--also-fidelity", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    benchmarks = [value for value in args.benchmarks.split(",") if value]
    adapters, _ = EC.build_worklist(seed="seed42", local_only=True)
    items = [item for item in adapters if item["id"].startswith("dpo_") and (
        item["source"] == args.source or item["base_model"] == args.source
    )]
    if args.items:
        requested = {value for value in args.items.split(",") if value}
        items = [item for item in items if item["id"] in requested]
    items = [item for item in items
             if not item_generation_done(item["id"], benchmarks, args.also_fidelity)]
    if args.limit:
        items = items[:args.limit]
    if not items:
        print(f"No pending DPO generation for source {args.source}")
        return
    bases = {item["base_model"] for item in items}
    if len(bases) != 1:
        raise SystemExit(f"source selection spans multiple bases: {sorted(bases)}")
    if any(not item.get("sft_parent") for item in items):
        raise SystemExit("persistent DPO generation requires every item to have sft_parent")

    logf = str(Path(EC.WORK_ROOT) / f"persistent_{items[0]['source']}.log")
    EC.log(f"persistent source worker: source={items[0]['source']} items={len(items)}", logf)
    tok, model, input_dev = EC.load_gen_base(items[0]["base_model"], logf)
    failures = []
    try:
        for item in items:
            try:
                model = generate_item(item, tok, model, input_dev, args, benchmarks)
            except Exception:
                failures.append(item["id"])
    finally:
        del model, tok
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    if failures:
        raise SystemExit(f"persistent source worker failed {len(failures)} cells: {failures}")


if __name__ == "__main__":
    main()
