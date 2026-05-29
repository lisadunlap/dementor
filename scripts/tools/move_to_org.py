"""Relocate all Dementor adapters + the responses dataset into a HuggingFace org
via server-side transfers (no data re-upload), clearing the personal profile, then
rebuild per-dataset collections under the org and retire the stale personal ones.

Run AFTER creating the org on huggingface.co (the API can't create orgs). The
analysis pipeline is unaffected (it uses Tinker sampler paths, not HF repo IDs).

Usage:
  python -m scripts.tools.move_to_org --org <org_name> [--dry-run]
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

DATASETS = ["gsm8k", "chatbot_arena", "writingprompts"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, help="Target HuggingFace org (must already exist).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    from huggingface_hub import HfApi

    api = HfApi()
    me = api.whoami()["name"]
    models = sorted(
        m.id for m in api.list_models(author=me)
        if m.id.split("/", 1)[1].startswith(("sft_", "dpo_", "self_sft_"))
    )
    dsets = [d.id for d in api.list_datasets(author=me) if d.id.split("/", 1)[1].startswith("dementor-")]
    print(f"to move into org '{args.org}': {len(models)} adapters + {len(dsets)} datasets", flush=True)
    if args.dry_run:
        print("dry-run; nothing moved.")
        return

    moved = 0
    for rid in models:
        to = f"{args.org}/{rid.split('/', 1)[1]}"
        try:
            api.move_repo(from_id=rid, to_id=to, repo_type="model")
            moved += 1
        except Exception as exc:
            print(f"  [skip] {rid}: {str(exc)[:80]}", flush=True)
    print(f"moved {moved}/{len(models)} adapters", flush=True)
    for rid in dsets:
        to = f"{args.org}/{rid.split('/', 1)[1]}"
        try:
            api.move_repo(from_id=rid, to_id=to, repo_type="dataset")
            print(f"  moved dataset -> https://huggingface.co/datasets/{to}", flush=True)
        except Exception as exc:
            print(f"  [skip dataset] {rid}: {str(exc)[:80]}", flush=True)

    # Retire stale personal collections, rebuild under the org.
    try:
        for col in api.list_collections(owner=me):
            if str(col.title).startswith("Dementor adapters"):
                api.delete_collection(col.slug)
                print(f"  retired stale collection {col.slug}", flush=True)
    except Exception as exc:
        print(f"  [collections cleanup note] {str(exc)[:80]}", flush=True)

    org_models = [f"{args.org}/{m.split('/', 1)[1]}" for m in models]
    for ds in DATASETS:
        items = [m for m in org_models if f"_{ds}_" in m.split("/", 1)[1]]
        if not items:
            continue
        col = api.create_collection(
            title=f"Dementor adapters {ds}", namespace=args.org,
            description=f"SFT, DPO and self-SFT LoRA adapters for the {ds} dataset.", private=False,
        )
        for it in items:
            try:
                api.add_collection_item(col.slug, item_id=it, item_type="model", exists_ok=True)
            except Exception as exc:
                print(f"  [skip item] {it}: {str(exc)[:50]}", flush=True)
        print(f"{ds}: {len(items)} -> https://huggingface.co/collections/{col.slug}", flush=True)

    print("\nDone. Next: refresh the dataset README links to the org namespace.")


if __name__ == "__main__":
    main()
