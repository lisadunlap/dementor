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
    # --- SFT: named sampler_weights/ paths from registry (required by tinker_cookbook download) ---
    {
        "tinker_path": "tinker://f6f93b9b-cd36-5593-8323-259763e8970c:train:0/sampler_weights/oasst_llama-3.1-8b_as_qwen3.6-27b_sft_seed42_20260601233903",
        "repo_id": f"{HF_ORG}/oasst-llama-3.1-8b-as-qwen3.6-27b-sft-seed42",
    },
    {
        "tinker_path": "tinker://59b80c47-0021-5b28-97d5-0c2e727580bf:train:0/sampler_weights/oasst_llama-3.1-8b_as_nemotron-nano_sft_seed42_20260601233903",
        "repo_id": f"{HF_ORG}/oasst-llama-3.1-8b-as-nemotron-nano-sft-seed42",
    },
    {
        "tinker_path": "tinker://e22fd856-d859-55a1-86ba-a8acb36d2550:train:0/sampler_weights/oasst_llama-3.1-8b_as_gpt-oss-20b_sft_seed42_20260601233939",
        "repo_id": f"{HF_ORG}/oasst-llama-3.1-8b-as-gpt-oss-20b-sft-seed42",
    },
    {
        "tinker_path": "tinker://ea27109e-27c3-5ef9-9480-cbba09651f25:train:0/sampler_weights/oasst_llama-3.1-8b_self_sft_seed42_20260601233905",
        "repo_id": f"{HF_ORG}/oasst-llama-3.1-8b-self-sft-seed42",
    },
    # --- DPO: from tinker_cookbook DPO logs ---
    {
        "tinker_path": "tinker://9d7273c8-06f7-568a-8448-b3c284e7d09b:train:0/weights/final",
        "repo_id": f"{HF_ORG}/oasst-llama-3.1-8b-as-qwen3.6-27b-dpo-seed42",
    },
    {
        "tinker_path": "tinker://84f7e912-7b21-57b2-bba8-342a712231a0:train:0/weights/final",
        "repo_id": f"{HF_ORG}/oasst-llama-3.1-8b-as-nemotron-nano-dpo-seed42",
    },
    {
        "tinker_path": "tinker://3bce597d-f243-589f-8cc1-bd5528200bdd:train:0/weights/final",
        "repo_id": f"{HF_ORG}/oasst-llama-3.1-8b-as-gpt-oss-20b-dpo-seed42",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--private", action="store_true", default=False,
                   help="Make repos private (default: public).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip repos that already exist on HuggingFace (default: True).")
    p.add_argument("--stage", choices=["sft", "dpo", "all"], default="all",
                   help="Which adapter stage to push (default: all).")
    return p.parse_args()


def push_one(adapter: dict, *, token: str, private: bool) -> str:
    from tinker_cookbook import weights
    from tinker_cookbook.weights import ModelCardConfig
    import tempfile

    tinker_path = adapter["tinker_path"]
    repo_id = adapter["repo_id"]

    model_card = ModelCardConfig(
        base_model=BASE_MODEL,
        datasets=["OpenAssistant/oasst1"],
        tags=["lora", "peft", "dementor", "behavioral-inertia", "openassistant", "oasst1"],
    )

    with tempfile.TemporaryDirectory() as tmp:
        print(f"  [download] {repo_id} from Tinker …")
        local_path = weights.download(tinker_path=tinker_path, output_dir=tmp)
        print(f"  [push]     {repo_id}")
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

    # Filter by stage
    from huggingface_hub import HfApi as _HfApi
    _api = _HfApi(token=args.hf_token)
    existing_repos = {m.id for m in _api.list_models(author=HF_ORG)}

    adapters_to_push = []
    for a in ADAPTERS:
        if args.stage != "all":
            if args.stage == "sft" and "dpo" in a["repo_id"]: continue
            if args.stage == "dpo" and "dpo" not in a["repo_id"]: continue
        if args.skip_existing and a["repo_id"] in existing_repos:
            print(f"  [skip] {a['repo_id']} (already on HF)")
            continue
        adapters_to_push.append(a)

    if not adapters_to_push:
        print("Nothing to push — all adapters already on HuggingFace.")
        return

    # Push adapters in parallel
    print(f"\n[2/3] Pushing {len(adapters_to_push)} adapters (workers={args.workers}) …")
    pushed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(push_one, a, token=args.hf_token, private=args.private): a
            for a in adapters_to_push
        }
        for fut in as_completed(futures):
            adapter = futures[fut]
            try:
                url = fut.result()
                pushed[adapter["repo_id"]] = url
            except Exception as exc:
                print(f"  ✗ {adapter['repo_id']}: {exc}")

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
