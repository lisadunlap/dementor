"""Relocate all Dementor adapters + the responses dataset into a HuggingFace org
via server-side transfers (no data re-upload), clearing the personal profile, then
rebuild per-dataset collections under the org and retire the stale personal ones.

Run AFTER creating the org on huggingface.co (the API can't create orgs) AND with a
token that has write access to that org. Fine-grained tokens only reach orgs that
were selected when the token was minted, so a token created before the org exists
will 403 on every move; the preflight below catches that up front. The analysis
pipeline is unaffected (it uses Tinker sampler paths, not HF repo IDs).

Usage:
  python -m scripts.tools.move_to_org --org <org_name> [--dry-run]
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

DATASETS = ["gsm8k", "chatbot_arena", "writingprompts"]
WRITE_ROLES = {"write", "admin", "contributor"}


def _org_token_role(api, org: str) -> str | None:
    """This token's role within `org`. None = member but token has no scoped
    access; '__not_a_member__' = the token's user isn't in the org at all."""
    for o in api.whoami().get("orgs") or []:
        if o.get("name") == org:
            return ((o.get("auth") or {}).get("accessToken") or {}).get("role")
    return "__not_a_member__"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, help="Target HuggingFace org (must already exist).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    me = who["name"]

    # Preflight: without write scope on the org, move_repo and create_collection
    # both 403. Fail fast with the exact fix instead of leaving a half-done state.
    # A classic write/admin token grants blanket write to every member org (its
    # per-org role field reads None); a fine-grained token must be org-scoped.
    top_role = ((who.get("auth") or {}).get("accessToken") or {}).get("role")
    org_role = _org_token_role(api, args.org)
    is_member = org_role != "__not_a_member__"
    can_write = is_member and (top_role in {"write", "admin"} or org_role in WRITE_ROLES)
    if not can_write:
        why = "your account is not a member of it" if not is_member \
            else f"this token has no write scope for it (token role={top_role!r}, org role={org_role!r})"
        print(f"ERROR: cannot write to org '{args.org}' — {why}.\n")
        print("Fix: mint a token with org-write access and put it in .env as HF_TOKEN=hf_...")
        print("  tokens page : https://huggingface.co/settings/tokens")
        print("  simplest    : a classic 'Write' token (writes to every org you belong to), or")
        print(f"  fine-grained: 'Read/write repos' + 'Manage collections', with the '{args.org}' org checked.")
        print("Then re-run this command (.env HF_TOKEN overrides the cached CLI login).")
        return

    models = sorted(
        m.id for m in api.list_models(author=me)
        if m.id.split("/", 1)[1].startswith(("sft_", "dpo_", "self_sft_"))
    )
    dsets = [d.id for d in api.list_datasets(author=me) if d.id.split("/", 1)[1].startswith("dementor-")]
    print(f"to move into org '{args.org}' (token role={top_role}): {len(models)} adapters + {len(dsets)} datasets", flush=True)
    if args.dry_run:
        print("dry-run; nothing moved.")
        return

    moved, skipped = 0, []
    for rid in models:
        to = f"{args.org}/{rid.split('/', 1)[1]}"
        try:
            api.move_repo(from_id=rid, to_id=to, repo_type="model")
            moved += 1
        except Exception as exc:
            skipped.append(rid)
            print(f"  [skip] {rid}: {str(exc)[:100]}", flush=True)
    tail = f" ({len(skipped)} skipped)" if skipped else ""
    print(f"moved {moved}/{len(models)} adapters{tail}", flush=True)

    for rid in dsets:
        to = f"{args.org}/{rid.split('/', 1)[1]}"
        try:
            api.move_repo(from_id=rid, to_id=to, repo_type="dataset")
            print(f"  moved dataset -> https://huggingface.co/datasets/{to}", flush=True)
        except Exception as exc:
            print(f"  [skip dataset] {rid}: {str(exc)[:100]}", flush=True)

    # Collections are a convenience layer; never let a collection error crash the
    # run after the (irreversible-ish) repo moves have already succeeded.
    try:
        for col in api.list_collections(owner=me):
            if str(col.title).startswith("Dementor adapters"):
                api.delete_collection(col.slug)
                print(f"  retired stale collection {col.slug}", flush=True)
    except Exception as exc:
        print(f"  [collections cleanup note] {str(exc)[:100]}", flush=True)

    org_models = [f"{args.org}/{m.split('/', 1)[1]}" for m in models]
    for ds in DATASETS:
        items = [m for m in org_models if f"_{ds}_" in m.split("/", 1)[1]]
        if not items:
            continue
        try:
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
        except Exception as exc:
            print(f"  [skip collection {ds}] {str(exc)[:100]}", flush=True)

    print("\nDone. Next: refresh the dataset README links to the org namespace.")


if __name__ == "__main__":
    main()
