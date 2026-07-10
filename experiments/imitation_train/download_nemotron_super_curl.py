#!/usr/bin/env python
"""Direct-curl downloader for nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16.

The Hugging Face Xet transfer backend can stall on this repo. This script uses the
Hub API only for metadata, then downloads files with curl into a local snapshot
directory. It validates expected byte sizes and safetensors headers so truncated
shards fail before model load.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from huggingface_hub import HfApi
from safetensors import safe_open


REPO_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
SLUG = "nemotron-super-120b"


def _repo_root() -> Path:
    return Path(os.environ.get("DEMENTOR_REPO") or Path(__file__).resolve().parents[2])


def _keep(name: str) -> bool:
    suffixes = (
        ".json",
        ".safetensors",
        ".model",
        ".txt",
        ".jinja",
        ".py",
        ".md",
    )
    base = os.path.basename(name)
    return name.endswith(suffixes) or base.startswith(("tokenizer", "special_tokens"))


def _snapshot_dir(hf_home: Path, repo_id: str, sha: str) -> Path:
    owner, repo = repo_id.split("/", 1)
    return hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / sha


def _curl_one(repo_id: str, revision: str, rel: str, out: Path, token: str | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{quote(rel)}?download=1"
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "20",
        "--retry-delay",
        "5",
        "--continue-at",
        "-",
        "--output",
        str(out),
    ]
    run_kwargs = {}
    if token:
        cmd.extend(["--config", "-"])
        run_kwargs = {
            "input": f'header = "Authorization: Bearer {token}"\n',
            "text": True,
        }
    cmd.append(url)
    subprocess.run(cmd, check=True, **run_kwargs)


def _check_file(path: Path, expected_size: int | None) -> None:
    if not path.exists():
        raise RuntimeError(f"missing {path}")
    actual = path.stat().st_size
    if expected_size is not None and actual != expected_size:
        raise RuntimeError(f"size mismatch {path}: got {actual}, expected {expected_size}")
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as f:
            _ = list(f.keys())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--hf-home", default=os.environ.get("DEMENTOR_HF_HOME") or os.environ.get("HF_HOME"))
    ap.add_argument("--models-dir", default=os.environ.get("DEMENTOR_MODELS_DIR"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    root = _repo_root()
    load_dotenv(root / ".env")
    token = os.environ.get("HF_TOKEN")
    hf_home = Path(args.hf_home or (root / "data" / "huggingface")).expanduser().resolve()
    models_dir = Path(args.models_dir or (root / "data" / "models")).expanduser().resolve()

    api = HfApi(token=token)
    info = api.model_info(args.repo_id, revision=args.revision, files_metadata=True)
    sha = info.sha
    snap = _snapshot_dir(hf_home, args.repo_id, sha)
    files = [s for s in info.siblings if _keep(s.rfilename)]
    if not files:
        raise RuntimeError(f"no files selected from {args.repo_id}")
    snap.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] repo={args.repo_id} revision={args.revision} sha={sha}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] snapshot={snap}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] selected_files={len(files)} workers={args.workers}", flush=True)

    def done(s) -> bool:
        path = snap / s.rfilename
        try:
            _check_file(path, s.size)
            return True
        except Exception:
            return False

    pending = [s for s in files if not done(s)]
    print(f"[{time.strftime('%H:%M:%S')}] pending={len(pending)} already_ok={len(files) - len(pending)}", flush=True)
    if args.verify_only and pending:
        raise RuntimeError(f"{len(pending)} files missing or invalid")

    def fetch(s) -> str:
        rel = s.rfilename
        path = snap / rel
        print(f"[{time.strftime('%H:%M:%S')}] GET {rel}", flush=True)
        _curl_one(args.repo_id, sha, rel, path, token)
        _check_file(path, s.size)
        return rel

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(fetch, s): s.rfilename for s in pending}
        for fut in cf.as_completed(futs):
            rel = futs[fut]
            try:
                fut.result()
                print(f"[{time.strftime('%H:%M:%S')}] OK {rel}", flush=True)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] FAIL {rel}: {type(e).__name__}: {e}", flush=True)
                raise

    for s in files:
        _check_file(snap / s.rfilename, s.size)

    models_dir.mkdir(parents=True, exist_ok=True)
    link = models_dir / SLUG
    if link.exists() or link.is_symlink():
        if link.resolve() != snap:
            print(f"[{time.strftime('%H:%M:%S')}] model link exists: {link} -> {link.resolve()}", flush=True)
    else:
        link.symlink_to(snap, target_is_directory=True)
        print(f"[{time.strftime('%H:%M:%S')}] linked {link} -> {snap}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] COMPLETE {args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
