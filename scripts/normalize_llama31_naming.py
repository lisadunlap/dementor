#!/usr/bin/env python3
"""
Normalize artifact filenames from Meta-Llama-3-8B-Instruct → Meta-Llama-3.1-8B-Instruct.

This copies files (does not delete originals) so downstream scripts can rely on
the 3.1 naming convention. Safe to run multiple times; existing targets are skipped.

Usage:
  python scripts/normalize_llama31_naming.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = "Meta-Llama-3-8B-Instruct"
NEW = "Meta-Llama-3.1-8B-Instruct"


def maybe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
    except Exception as exc:
        print(f"WARN: failed to copy {src} -> {dst}: {exc}")


def main() -> int:
    candidates = []
    for rel_root in (
        ROOT / "data/model-responses",
        ROOT / "data/results",
    ):
        if not rel_root.exists():
            continue
        for path in rel_root.rglob("*"):
            if path.is_file() and OLD in path.name:
                candidates.append(path)

    if not candidates:
        print("No files found with old naming.")
        return 0

    for src in candidates:
        new_name = src.name.replace(OLD, NEW)
        dst = src.with_name(new_name)
        maybe_copy(src, dst)

    print("Done. Originals preserved; new 3.1-named copies created where missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






