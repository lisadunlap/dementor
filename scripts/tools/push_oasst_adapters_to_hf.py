"""
Export oasst LoRA adapters from Tinker and push to HuggingFace.

Creates a new 'Dementor adapters openassistant' collection under dementor-research
and pushes each SFT/self-SFT adapter as a PEFT model repo, then adds it to the
collection.

Usage:
  python scripts/tools/push_oasst_adapters_to_hf.py --hf-token hf_...
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HF_ORG = "dementor-research"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REGISTRY = Path("data/tinker_adapters.json")

ADAPTERS = [
    {
        "alias": "oasst_llama-3.1-8b_as_qwen3.6-27b_sft_seed42",
        "repo_id": f"{HF_ORG}/oasst-llama-3.1-8b-as-qwen3.6-27b-sft-seed42",
        "description": "Llama-3.1-8B-Instruct SFT'd to imitate Qwen3.6-27B on OpenAssistant (oasst1, seed=42)",
    },
    {
        "alias": "oasst_llama-3.1-8b_as_nemotron-nano_sft_seed42",
        "repo_id": f"{HF_ORG}/oasst-llama-3.1-8b-as-nemotron-nano-sft-seed42",
        "description": "Llama-3.1-8B-Instruct SFT'd to imitate Nemotron-3-Nano on OpenAssistant (oasst1, seed=42)",
    },
    {
        "alias": "oasst_llama-3.1-8b_as_gpt-oss-20b_sft_seed42",
        "repo_id": f"{HF_ORG}/oasst-llama-3.1-8b-as-gpt-oss-20b-sft-seed42",
        "description": "Llama-3.1-8B-Instruct SFT'd to imitate gpt-oss-20b on OpenAssistant (oasst1, seed=42)",
    },
    {
        "alias": "oasst_llama-3.1-8b_self_sft_seed42",
        "repo_id": f"{HF_ORG}/oasst-llama-3.1-8b-self-sft-seed42",
        "description": "Llama-3.1-8B-Instruct self-SFT (trained on its own responses) on OpenAssistant (oasst1, seed=42)",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--private", action="store_true", default=False,
                   help="Make repos private (default: public).")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def _tinker_path_for(alias: str) -> str:
    reg = json.loads(REGISTRY.read_text())
    entry = reg.get(alias)
    if not entry:
        raise KeyError(f"Alias '{alias}' not found in {REGISTRY}")
    # prefer sampler path; fall back to checkpoint path
    path = entry.get("path") or entry.get("checkpoint_path")
    if not path:
        raise ValueError(f"No Tinker path for '{alias}'")
    return path


def push_one(adapter: dict, *, token: str, private: bool) -> str:
    from tinker_cookbook import weights
    from tinker_cookbook.weights import ModelCardConfig
    import tempfile

    alias = adapter["alias"]
    repo_id = adapter["repo_id"]
    tinker_path = _tinker_path_for(alias)

    model_card = ModelCardConfig(
        base_model=BASE_MODEL,
        datasets=["OpenAssistant/oasst1"],
        tags=["lora", "peft", "dementor", "behavioral-inertia", "openassistant", "oasst1"],
    )

    with tempfile.TemporaryDirectory() as tmp:
        print(f"  [download] {alias} from Tinker …")
        local_path = weights.download(tinker_path=tinker_path, output_dir=tmp)
        print(f"  [push]     {alias} → {repo_id}")
        url = weights.publish_to_hf_hub(
            model_path=local_path,
            repo_id=repo_id,
            private=private,
            token=token,
            model_card=model_card,
        )

    print(f"  [done]     {repo_id}  →  {url}")
    return url


def main() -> None:
    args = parse_args()
    if not args.hf_token:
        raise SystemExit("Provide --hf-token or set HF_TOKEN.")

    from huggingface_hub import HfApi

    api = HfApi(token=args.hf_token)

    # Create collection
    print("[1/3] Creating HuggingFace collection …")
    try:
        col = api.create_collection(
            title="Dementor adapters openassistant",
            namespace=HF_ORG,
            description=(
                "Llama-3.1-8B-Instruct LoRA adapters trained on OpenAssistant (oasst1) "
                "to imitate other models — part of the Dementor behavioral-inertia study."
            ),
            private=args.private,
            exists_ok=True,
            token=args.hf_token,
        )
        collection_slug = col.slug
        print(f"  Collection: https://huggingface.co/collections/{collection_slug}")
    except Exception as exc:
        print(f"  WARNING: could not create collection: {exc}")
        collection_slug = None

    # Push adapters in parallel
    print(f"\n[2/3] Pushing {len(ADAPTERS)} adapters (workers={args.workers}) …")
    pushed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(push_one, a, token=args.hf_token, private=args.private): a
            for a in ADAPTERS
        }
        for fut in as_completed(futures):
            adapter = futures[fut]
            try:
                url = fut.result()
                pushed[adapter["repo_id"]] = url
            except Exception as exc:
                print(f"  ✗ {adapter['alias']}: {exc}")

    # Add repos to collection
    if collection_slug and pushed:
        print(f"\n[3/3] Adding {len(pushed)} repos to collection …")
        for repo_id in pushed:
            try:
                api.add_collection_item(
                    collection_slug=collection_slug,
                    item_id=repo_id,
                    item_type="model",
                    token=args.hf_token,
                    exists_ok=True,
                )
                print(f"  ✓ added {repo_id}")
            except Exception as exc:
                print(f"  ✗ {repo_id}: {exc}")

    print("\nDone.")
    print(f"Collection: https://huggingface.co/collections/{collection_slug}")


if __name__ == "__main__":
    main()
