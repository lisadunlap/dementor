#!/usr/bin/env python3
"""Build portable HF coordination files for disjoint remote generation lanes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import erosion_common as EC  # noqa: E402
import fidelity_common as FC  # noqa: E402


def complete(item: dict) -> bool:
    root = Path(EC.WORK) / item["id"]
    return all((root / b / "all_gens.csv").is_file() for b in EC.DEFAULT_BENCHMARKS) and FC.gens_done(item["id"])


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(value + "\n" for value in values))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-root", required=True, type=Path)
    args = parser.parse_args()

    items, _ = EC.build_worklist(seed="seed42", local_only=True)
    items = sorted((x for x in items if x["id"].startswith("dpo_")), key=lambda x: x["id"])
    pending = [x for x in items if not complete(x)]

    llama = [x for x in pending if x["source"] == "llama-3.3-70b"]
    llama_lanes = [[], [], [], []]
    for index, item in enumerate(llama):
        llama_lanes[index % 4].append(item["id"])

    ordinary_sources = {
        "node-4h100-ordinary.txt": {"ministral-8b"},
        "node-8h100-ordinary-a.txt": {"granite-4-h-small"},
        "node-8h100-ordinary-b.txt": {"llama-3.1-8b"},
        "main-ordinary.txt": {"gemma-4-31b", "gemma-4-e4b", "olmo-3-7b", "phi-4"},
    }
    manifests = {
        "node-4h100-llama70.txt": llama_lanes[1],
        "node-8h100-llama70-a.txt": llama_lanes[2],
        "node-8h100-llama70-b.txt": llama_lanes[3],
        "main-llama70.txt": llama_lanes[0],
    }
    for name, sources in ordinary_sources.items():
        manifests[name] = [x["id"] for x in pending if x["source"] in sources]

    assigned = [cell for values in manifests.values() for cell in values]
    if len(assigned) != len(set(assigned)):
        raise RuntimeError("manifest overlap")

    coordination = args.transfer_root / "coordination"
    for name, values in manifests.items():
        write_lines(coordination / "manifests" / name, values)

    portable = {}
    for item in items:
        cell = item["id"]
        portable[cell] = {
            "cell_id": cell,
            "base_model": item["base_model"],
            "dataset": item["dataset"],
            "source": item["source"],
            "target": item["target"],
            "seed": item["seed"],
            "sft_adapter": f"adapters/sft/{cell}",
            "dpo_adapter": f"adapters/dpo/{cell}",
        }
    coordination.mkdir(parents=True, exist_ok=True)
    (coordination / "portable_registry.json").write_text(json.dumps(portable, indent=2, sort_keys=True) + "\n")

    files = sorted((args.transfer_root / "adapters").rglob("adapter_*"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = pool.map(digest, files)
        with (coordination / "checksums.sha256").open("w") as handle:
            for path, checksum in zip(files, hashes):
                handle.write(f"{checksum}  {path.relative_to(args.transfer_root)}\n")

    summary = {
        "git_commit": "6e4f3f75627e5980e937a150fb9b6e751f4be3a4",
        "total_cells": len(items),
        "complete_at_freeze": len(items) - len(pending),
        "pending_at_freeze": len(pending),
        "manifests": {name: len(values) for name, values in manifests.items()},
        "adapter_files": len(files),
    }
    (coordination / "assignment_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
