"""Group the 228 Dementor adapters into HuggingFace Collections (one per dataset:
gsm8k / chatbot_arena / writingprompts), each holding that dataset's SFT + DPO +
self-SFT LoRAs. Non-destructive (repos are unchanged); gives browsable, single-link
views to share. Re-runnable (item adds are idempotent).

Usage: python -m scripts.tools.make_adapter_collections
"""
from __future__ import annotations

from dotenv import load_dotenv

DATASETS = ["gsm8k", "chatbot_arena", "writingprompts"]
DESC = "SFT, DPO and self-SFT LoRA adapters for the {ds} dataset (4 models, cross-targets, 3 seeds)."


def main() -> None:
    load_dotenv()
    from huggingface_hub import HfApi

    api = HfApi()
    me = api.whoami()["name"]
    adapters = sorted(
        m.id for m in api.list_models(author=me)
        if m.id.split("/", 1)[1].startswith(("sft_", "dpo_", "self_sft_"))
    )

    def dataset_of(repo: str) -> str | None:
        name = repo.split("/", 1)[1]
        for d in DATASETS:
            if f"_{d}_" in name:
                return d
        return None

    groups: dict[str, list[str]] = {d: [] for d in DATASETS}
    for a in adapters:
        d = dataset_of(a)
        if d:
            groups[d].append(a)

    for ds, items in groups.items():
        col = api.create_collection(
            title=f"Dementor adapters {ds}",
            namespace=me, description=DESC.format(ds=ds), private=False,
        )
        added = 0
        for a in items:
            try:
                api.add_collection_item(col.slug, item_id=a, item_type="model", exists_ok=True)
                added += 1
            except Exception as exc:  # already present / transient
                print(f"  [skip] {a}: {str(exc)[:60]}", flush=True)
        print(f"{ds}: {added}/{len(items)} adapters -> https://huggingface.co/collections/{col.slug}", flush=True)


if __name__ == "__main__":
    main()
