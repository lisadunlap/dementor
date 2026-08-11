"""Export Tinker/local adapters to PEFT format and push them to the HuggingFace Hub.

``export_adapter_to_peft`` streams a Tinker sampler checkpoint straight from the
archive endpoint (bypassing the SDK's timeouts); ``upload_adapter_to_hf`` writes
a provenance README and uploads a PEFT dir; ``push_adapters_to_hf`` and
``backfill_export_adapters`` iterate the adapter registry to do this in bulk.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from dementor import config

from ._concurrency import _dispatch, _retry_call
from ._constants import DATA, PEFT_ADAPTER_DIR
from ._models import source_id_of


def _alias_stage(alias: str) -> str:
    """Return the registry stage, preserving the compound ``self_sft`` prefix."""
    return "self_sft" if alias.startswith("self_sft_") else alias.split("_", 1)[0]


def export_adapter_to_peft(
    *,
    tinker_path: str,
    base_model: str,
    output_dir: Path,
    max_retries: int = 6,
) -> dict:
    """Download a Tinker sampler checkpoint and unpack it as a PEFT adapter dir.

    Bypasses tinker_cookbook.weights.download which goes through the high-level
    SDK and consistently times out at ~80s before Tinker can respond. Uses
    direct httpx against the archive endpoint instead.

    Returns a status dict; raises on failure after max_retries.
    """
    import os
    import tarfile
    import httpx

    if (output_dir / "adapter_config.json").exists():
        return {"status": "already_exported", "output_dir": str(output_dir)}

    import tinker
    from tinker import types as tinker_types

    parsed = tinker_types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
    base_url = os.environ.get(
        "TINKER_BASE_URL", "https://tinker.thinkingmachines.dev/services/tinker-prod"
    )
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY not set")
    archive_url = (
        f"{base_url}/api/v1/training_runs/{parsed.training_run_id}"
        f"/checkpoints/{parsed.checkpoint_id}/archive"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tar_tmp = output_dir.parent / f".{output_dir.name}.tar"

    last_err: Exception | None = None
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        for attempt in range(1, max_retries + 1):
            try:
                # 1. Resolve signed URL (302 when ready, 503 when archive is being built)
                resp = client.get(
                    archive_url,
                    headers={"X-API-Key": api_key, "Accept": "application/gzip"},
                    follow_redirects=False,
                )
                if resp.status_code == 503:
                    wait_s = 30
                    print(
                        f"  [archive building] {output_dir.name}: 503, sleeping {wait_s}s",
                        flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                if resp.status_code != 302:
                    raise RuntimeError(
                        f"unexpected status {resp.status_code}: {resp.text[:200]}"
                    )
                signed_url = resp.headers["Location"]

                # 2. Stream the tar to disk
                with client.stream("GET", signed_url) as r:
                    r.raise_for_status()
                    with tar_tmp.open("wb") as f:
                        for chunk in r.iter_bytes(64 * 1024):
                            f.write(chunk)

                # 3. Safely extract (reject symlinks + path traversal)
                base = output_dir.resolve()
                with tarfile.open(tar_tmp, "r") as tar:
                    for member in tar.getmembers():
                        if member.issym() or member.islnk():
                            raise RuntimeError(f"unsafe symlink in archive: {member.name}")
                        member_path = (output_dir / member.name).resolve()
                        if not member_path.is_relative_to(base):
                            raise RuntimeError(f"path traversal: {member.name}")
                    tar.extractall(path=output_dir)
                tar_tmp.unlink(missing_ok=True)

                (output_dir / "dementor_tinker_export.json").write_text(
                    json.dumps(
                        {
                            "tinker_path": tinker_path,
                            "base_model": base_model,
                            "output_dir": str(output_dir),
                            "format": "peft",
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                return {"status": "exported", "output_dir": str(output_dir)}
            except Exception as e:
                last_err = e
                wait_s = min(60, 10 * 2 ** (attempt - 1))
                print(
                    f"  [retry export {attempt}/{max_retries}] {output_dir.name} {type(e).__name__}: {str(e)[:120]} — sleep {wait_s}s",
                    flush=True,
                )
                time.sleep(wait_s)
    raise RuntimeError(f"Export failed after {max_retries} attempts: {last_err}")


def upload_adapter_to_hf(
    *,
    peft_dir: Path,
    repo_id: str,
    base_model: str,
    alias: str,
    private: bool = False,
) -> dict:
    """Create (or reuse) an HF repo and upload the PEFT adapter dir.

    Adds a README with provenance. Returns {'status', 'repo_url'}.
    """
    from huggingface_hub import HfApi, create_repo, upload_folder

    api = HfApi()
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        # Already exists; check for adapter_model.safetensors at the repo root
        if any(s.rfilename == "adapter_model.safetensors" for s in info.siblings):
            return {"status": "already_on_hub", "repo_url": f"https://huggingface.co/{repo_id}"}
    except Exception:
        pass

    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    # Write a README into the dir before upload
    readme = peft_dir / "README.md"
    if not readme.exists():
        # Parse cell info from alias: sft_{dataset}_{source}_as_{target}_seed{N}
        stage = _alias_stage(alias)
        campaign_models = len(config.campaign_roster())
        campaign_datasets = len(config.campaign_dataset_names())
        campaign_seeds = len(config.campaign_seeds())
        cross_cells_per_stage = (
            campaign_models * (campaign_models - 1) * campaign_datasets * campaign_seeds
        )
        stage_cells = (
            campaign_models * campaign_datasets * campaign_seeds
            if stage == "self_sft"
            else cross_cells_per_stage
        )
        readme.write_text(
            f"""---
base_model: {base_model}
library_name: peft
tags:
- lora
- {stage}
- dementor-research
---

# {alias}

LoRA adapter trained via [Tinker](https://thinkingmachines.ai/tinker/) as part of the
**dementor** configuration-defined behavioral-imitation study.

- **Base model:** `{base_model}`
- **Training stage:** {stage.upper()} (LoRA rank 32, target_modules=all-linear)
- **Alias:** `{alias}`

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
tok = AutoTokenizer.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}")
```

The named campaign contains {campaign_models} models, {campaign_datasets} datasets, and
{campaign_seeds} seed(s), yielding {stage_cells} configured cells for this stage. See
`config.yaml` in the code release for the exact cohort and hyperparameters.
""",
            encoding="utf-8",
        )

    upload_folder(
        repo_id=repo_id,
        folder_path=str(peft_dir),
        repo_type="model",
        commit_message=f"Upload {alias} adapter from dementor matrix",
    )
    return {"status": "uploaded", "repo_url": f"https://huggingface.co/{repo_id}"}


def push_adapters_to_hf(
    *,
    namespace: str = "dementor-research",
    only_llama: bool = False,
    kinds: tuple[str, ...] = ("sft", "dpo", "self_sft"),
    delete_local_after: bool = True,
    private: bool = False,
    max_retries: int = 4,
    parallel: int = 1,
) -> dict:
    """Upload all registered adapters to HuggingFace Hub.

    For each cell: if local PEFT exists, upload + (optionally) delete local.
    If not, download from Tinker to a temp dir, upload, delete temp.
    """
    from dotenv import load_dotenv
    import shutil

    load_dotenv()

    registry_path = DATA / "tinker_adapters.json"
    if not registry_path.exists():
        return {"uploaded": 0, "skipped": 0, "errors": []}
    registry = json.loads(registry_path.read_text())

    # Resolve source straight from each alias — registry-driven, covers legacy + local.
    pending: list[tuple[str, dict]] = []
    skipped = 0
    errors: list[dict] = []
    for alias, entry in sorted(registry.items()):
        prefix = _alias_stage(alias)
        if prefix not in kinds:
            continue
        source = source_id_of(alias)
        if source is None:
            continue
        if only_llama and source != "meta-llama/Llama-3.1-8B-Instruct":
            continue
        tinker_path = entry.get("path") if isinstance(entry, dict) else entry
        if not tinker_path:
            errors.append({"alias": alias, "error": "no tinker path"})
            continue
        pending.append((alias, {"source": source, "tinker_path": tinker_path}))

    def _handle_one(alias: str, meta: dict) -> dict:
        from huggingface_hub import HfApi
        source = meta["source"]
        tinker_path = meta["tinker_path"]
        repo_id = f"{namespace}/{alias}"
        local_dir = PEFT_ADAPTER_DIR / alias
        try:
            # Cheap HF-side existence check FIRST — skip if already uploaded
            api = HfApi()
            try:
                info = api.repo_info(repo_id=repo_id, repo_type="model")
                if any(s.rfilename == "adapter_model.safetensors" for s in info.siblings):
                    print(f"[skip] {alias}: already on hub", flush=True)
                    return {"alias": alias, "status": "already_on_hub",
                            "repo_url": f"https://huggingface.co/{repo_id}"}
            except Exception:
                pass  # repo doesn't exist; will create on upload

            if not (local_dir / "adapter_config.json").exists():
                print(f"[export] {alias}", flush=True)
                if config.backend_for(source) == "local":
                    from dementor.training.local_backend import export_local_adapter
                    export_local_adapter(adapter_dir=Path(tinker_path), base_model=source, output_dir=local_dir)
                else:
                    export_adapter_to_peft(tinker_path=tinker_path, base_model=source, output_dir=local_dir)
            print(f"[upload] {alias} -> {repo_id}", flush=True)
            last_err: Exception | None = None
            res = None
            try:
                res = _retry_call(
                    lambda: upload_adapter_to_hf(
                        peft_dir=local_dir,
                        repo_id=repo_id,
                        base_model=source,
                        alias=alias,
                        private=private,
                    ),
                    attempts=max_retries, base_wait=10, label="upload", name=alias,
                )
            except Exception as e:
                last_err = e
            if res is None:
                raise RuntimeError(f"upload failed after {max_retries} attempts: {last_err}")
            print(f"  -> {alias} {res['status']}: {res['repo_url']}", flush=True)
            if delete_local_after and local_dir.exists():
                shutil.rmtree(local_dir)
                print(f"  -> {alias} local cleaned", flush=True)
            return {"alias": alias, "status": res["status"], "repo_url": res["repo_url"]}
        except Exception as e:
            print(f"  [UPLOAD FAILED] {alias}: {e}", flush=True)
            return {"alias": alias, "error": str(e)}

    print(f"\nProcessing {len(pending)} adapters with parallel={parallel}\n", flush=True)

    uploaded = 0

    def _collect(res):
        nonlocal uploaded
        if "error" in res:
            errors.append(res)
        else:
            uploaded += 1

    _dispatch(
        pending,
        lambda it: _handle_one(*it),
        parallel=parallel,
        label="uploads",
        key=lambda it: it[0],
        on_success=_collect,
        on_error=lambda k, e: errors.append({"alias": k, "error": str(e)}),
    )

    return {"uploaded": uploaded, "skipped": skipped, "errors": errors}


def backfill_export_adapters(
    *,
    only_llama: bool = False,
    kinds: tuple[str, ...] = ("sft", "dpo", "self_sft"),
) -> dict:
    """Download registered adapters for the requested configured stages to local PEFT.

    Uses direct httpx against Tinker's archive endpoint (bypasses SDK timeouts).
    Llama-source adapters are downloaded first (June 12 retirement deadline).
    """
    from dotenv import load_dotenv

    load_dotenv()

    registry_path = DATA / "tinker_adapters.json"
    if not registry_path.exists():
        return {"exported": 0, "skipped": 0, "errors": []}
    registry = json.loads(registry_path.read_text())

    # Resolve source straight from each alias — registry-driven, covers legacy + local.
    exported = 0
    skipped = 0
    errors: list[dict] = []
    for alias, entry in sorted(registry.items()):
        prefix = _alias_stage(alias)
        if prefix not in kinds:
            continue
        source = source_id_of(alias)
        if source is None:
            skipped += 1
            continue
        if only_llama and source != "meta-llama/Llama-3.1-8B-Instruct":
            continue
        # Always use the sampler URI (path field). State URIs are NOT downloadable
        # via get_checkpoint_archive_url — Tinker rejects them as "not a sampler weights checkpoint".
        if isinstance(entry, dict):
            tinker_path = entry.get("path")
        else:
            tinker_path = entry
        if not tinker_path:
            errors.append({"alias": alias, "error": "no path in registry entry"})
            continue
        out_dir = PEFT_ADAPTER_DIR / alias
        print(f"[export] {alias}", flush=True)
        try:
            if config.backend_for(source) == "local":
                from dementor.training.local_backend import export_local_adapter
                res = export_local_adapter(adapter_dir=Path(tinker_path), base_model=source, output_dir=out_dir)
            else:
                res = export_adapter_to_peft(tinker_path=tinker_path, base_model=source, output_dir=out_dir)
            print(f"  -> {res['status']} at {res['output_dir']}", flush=True)
            exported += 1
        except Exception as e:
            print(f"  [EXPORT FAILED] {alias}: {e}", flush=True)
            errors.append({"alias": alias, "error": str(e)})

    return {"exported": exported, "skipped": skipped, "errors": errors}
