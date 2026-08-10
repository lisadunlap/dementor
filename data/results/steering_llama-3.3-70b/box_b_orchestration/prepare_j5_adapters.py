"""Materialize the exact 60 DPO adapters used by Box B J5."""
from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ORG = "dementor-research"
DATASETS = ("chatbot_arena", "gsm8k", "oasst1", "writingprompts")
ROOT = Path("/home/ubuntu/dementor-runtime/dpo_runs")
WORKLIST = Path("/home/ubuntu/dementor-runtime/worklist_mp.txt")


def main() -> None:
    names = [model.id.split("/", 1)[1] for model in HfApi(token=True).list_models(author=ORG)]
    cells = sorted(
        name
        for name in names
        if name.startswith("dpo_")
        and "_llama-3.3-70b_as_" in name
        and name.endswith("_seed42")
    )
    assert len(cells) == 60, f"expected 60, got {len(cells)} -- STOP, do not run partial"

    parsed: list[tuple[str, str]] = []
    for name in cells:
        dataset = next((d for d in DATASETS if name.startswith(f"dpo_{d}_")), None)
        if dataset is None:
            raise RuntimeError(f"unparseable DPO adapter: {name}")
        parsed.append((dataset, name[len(f"dpo_{dataset}_"):]))
    counts = Counter(dataset for dataset, _ in parsed)
    assert all(counts[d] == 15 for d in DATASETS), f"expected 15 per dataset, got {counts}"
    print(f"ASSERT_OK 60 DPO adapters: {dict(sorted(counts.items()))}", flush=True)

    def pull(row: tuple[str, str]) -> tuple[str, str]:
        dataset, cell = row
        dest = ROOT / dataset / cell
        snapshot_download(f"{ORG}/dpo_{dataset}_{cell}", local_dir=dest, token=True)
        if not (dest / "adapter_config.json").is_file():
            raise RuntimeError(f"missing adapter_config.json after download: {dest}")
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(pull, row) for row in parsed]
        for i, future in enumerate(as_completed(futures), 1):
            dataset, cell = future.result()
            print(f"[{i}/60] {dataset}/{cell}", flush=True)

    WORKLIST.parent.mkdir(parents=True, exist_ok=True)
    WORKLIST.write_text("".join(f"llama-3.3-70b|{d}|{c}\n" for d, c in parsed))
    print(f"COMPLETE 60/60; worklist={WORKLIST}", flush=True)


if __name__ == "__main__":
    main()
