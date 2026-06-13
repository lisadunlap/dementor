"""Upload the Phase-B new-source adapters to the dementor-research HF org.

The stock `run_matrix push-to-hf` only knows the original 4x4 matrix cells
(iter_cells()), so it skips our new sources. This driver selects new-source
adapters straight from the registry by base_model and reuses the same tested
primitives (export_adapter_to_peft -> upload_adapter_to_hf), with the identical
repo naming `{namespace}/{alias}`.

New sources: Qwen3-4B, Llama-3.3-70B, Qwen3-32B, each disguised as the four base
targets across gsm8k / writingprompts / chatbot_arena / oasst1, SFT + DPO.
Expected: 3 x 4 x 4 x 2 = 96 adapters.

Weights live on Tinker (registry stores tinker:// paths); each adapter is
downloaded to a temp PEFT dir, uploaded, and the local copy removed.

  python scripts/analysis/push_new_adapters_hf.py --parallel 4 [--dry-run] [--keep-local]
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from workflows.run_matrix import (DATA, MODEL_SLUG, PEFT_ADAPTER_DIR,
                                  export_adapter_to_peft, upload_adapter_to_hf)

NEW_SOURCES = {
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-32B",
}
# slug -> full model id (SFT registry entries lack base_model, so resolve from alias)
SLUG_TO_MODEL = {MODEL_SLUG[m]: m for m in NEW_SOURCES}
NEW_SLUGS = set(SLUG_TO_MODEL)
DATASETS = ("gsm8k", "writingprompts", "chatbot_arena", "oasst1")
NAMESPACE = "dementor-research"


def source_slug_of(alias: str) -> str | None:
    """alias = {stage}_{dataset}_{source}_as_{target}_seed{N} -> source slug."""
    head = alias.split("_as_", 1)[0]                # {stage}_{dataset}_{source}
    rest = head.split("_", 1)[1] if "_" in head else ""   # {dataset}_{source}
    for ds in DATASETS:
        if rest.startswith(ds + "_"):
            return rest[len(ds) + 1:]
    return None


def select() -> list[tuple[str, dict, str]]:
    reg = json.loads((DATA / "tinker_adapters.json").read_text())
    out = []
    for alias, entry in sorted(reg.items()):
        if not isinstance(entry, dict) or not entry.get("path"):
            continue
        if alias.split("_", 1)[0] not in ("sft", "dpo"):
            continue
        slug = source_slug_of(alias)
        if slug in NEW_SLUGS:
            out.append((alias, entry, SLUG_TO_MODEL[slug]))
    return out


def handle_one(alias: str, entry: dict, source: str, *, dry_run: bool, keep_local: bool,
               max_retries: int = 4) -> dict:
    from huggingface_hub import HfApi
    import time
    tinker_path = entry["path"]
    repo_id = f"{NAMESPACE}/{alias}"
    local_dir = PEFT_ADAPTER_DIR / alias

    # already on hub?
    try:
        info = HfApi().repo_info(repo_id=repo_id, repo_type="model")
        if any(s.rfilename == "adapter_model.safetensors" for s in info.siblings):
            print(f"[skip] {alias}: already on hub", flush=True)
            return {"alias": alias, "status": "already_on_hub"}
    except Exception:
        pass

    if dry_run:
        print(f"[dry-run] would upload {alias} -> {repo_id}", flush=True)
        return {"alias": alias, "status": "dry_run"}

    try:
        if not (local_dir / "adapter_config.json").exists():
            print(f"[download] {alias}", flush=True)
            export_adapter_to_peft(tinker_path=tinker_path, base_model=source, output_dir=local_dir)
        print(f"[upload] {alias} -> {repo_id}", flush=True)
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                res = upload_adapter_to_hf(peft_dir=local_dir, repo_id=repo_id,
                                           base_model=source, alias=alias, private=False)
                break
            except Exception as e:
                last_err = e
                wait = min(60, 10 * 2 ** (attempt - 1))
                print(f"  [retry {attempt}/{max_retries}] {alias} {type(e).__name__}: {str(e)[:100]} — sleep {wait}s", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f"upload failed after {max_retries}: {last_err}")
        print(f"  -> {alias} {res['status']}: {res['repo_url']}", flush=True)
        if not keep_local and local_dir.exists():
            shutil.rmtree(local_dir)
        return {"alias": alias, "status": res["status"], "repo_url": res["repo_url"]}
    except Exception as e:
        print(f"  [FAILED] {alias}: {e}", flush=True)
        return {"alias": alias, "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-local", action="store_true")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    pending = select()
    by_key: dict[str, int] = {}
    for alias, _, _ in pending:
        stage, ds = alias.split("_")[0], alias.split("_")[1]
        by_key[f"{ds}/{stage}"] = by_key.get(f"{ds}/{stage}", 0) + 1
    print(f"Selected {len(pending)} new-source adapters: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by_key.items())))

    results = []
    if args.parallel <= 1:
        for alias, entry, source in pending:
            results.append(handle_one(alias, entry, source, dry_run=args.dry_run, keep_local=args.keep_local))
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futs = {pool.submit(handle_one, a, e, s, dry_run=args.dry_run, keep_local=args.keep_local): a
                    for a, e, s in pending}
            for fut in as_completed(futs):
                results.append(fut.result())

    uploaded = sum(1 for r in results if r.get("status") == "uploaded")
    onhub = sum(1 for r in results if r.get("status") == "already_on_hub")
    errs = [r for r in results if "error" in r]
    print(f"\n=== done: uploaded={uploaded}, already_on_hub={onhub}, errors={len(errs)}, total={len(results)} ===")
    for e in errs[:20]:
        print(f"  ERROR {e['alias']}: {e['error'][:150]}")


if __name__ == "__main__":
    main()
