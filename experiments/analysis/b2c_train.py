"""B2c — cross-dataset replication of the Phase-B new sources.

B2a/B2b trained the three new sources (Llama-3.3-70B, Qwen3-32B, Qwen3-4B) on
gsm8k ONLY, and the result was suspicious: every new source "retained" only into
the nemotron target and laundered into the other three. That makes the per-source
means (and the capability verdict built on them) a single-target (->nemotron)
artifact unless it replicates on other datasets.

B2c runs the same three sources -> the four original targets on the remaining
datasets (writingprompts, chatbot_arena) PLUS oasst1 (added to align with the
partner's 4-dataset set). Decisive test:
  - ->nemotron concentration REPLICATES on wp/ca/oasst1  -> real source x target
    (or nemotron-as-target) effect; report it as a finding.
  - It does NOT replicate -> gsm8k was noise; the capability story is dead and
    Phase B is an honest null.

Same machinery as b2a/b2b_train.py (register sources in MODEL_SLUG/CHAT_TEMPLATE
only, build SFT+DPO data from baselines, launch the cell subset). Dataset and
source subsets are CLI-selectable so this one driver covers the whole sweep.

Stages:
  --build        create SFT + DPO data CSVs from baselines (needs source+target baselines)
  --train-sft    launch SFT jobs (Tinker spend)
  --train-dpo    launch DPO jobs (after SFT; Tinker spend)
  --datasets     subset of {writingprompts,chatbot_arena,oasst1} (default: all three)
  --sources      subset of the three new sources (default: all three)
  --dry-run      with a train stage, print the plan without spending
"""
from __future__ import annotations

import argparse

import pandas as pd

from dementor.training.matrix import (Cell, baseline_path, sft_data_path, dpo_data_path,
                                  launch_sft, launch_dpo)

NEW_SOURCES = ["Qwen/Qwen3-4B-Instruct-2507", "meta-llama/Llama-3.3-70B-Instruct",
               "Qwen/Qwen3-32B"]
ORIG_TARGETS = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3.6-27B",
                "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "openai/gpt-oss-20b"]
DATASETS = ["writingprompts", "chatbot_arena", "oasst1"]
SEED = 1


def my_cells(datasets: list[str], sources: list[str]) -> list[Cell]:
    # Exclude self-pairs (s==t); those are the self-SFT controls, not disguise cells.
    return [Cell(source=s, target=t, dataset=ds, seed=SEED)
            for ds in datasets for s in sources for t in ORIG_TARGETS if s != t]


def _col_str(df, col):
    return df[col].astype(str).fillna("")


def build_data(cells: list[Cell]):
    for cell in cells:
        tgt_cache = baseline_path(cell.target, cell.dataset)
        src_cache = baseline_path(cell.source, cell.dataset)
        if not tgt_cache.exists():
            raise SystemExit(f"missing target baseline: {tgt_cache}")
        if not src_cache.exists():
            raise SystemExit(f"missing source baseline (run generate-target-responses first): {src_cache}")
        tgt = pd.read_csv(tgt_cache)
        tgt["model_response"] = _col_str(tgt, "model_response")

        # SFT: prompt -> target's response
        sp = sft_data_path(cell)
        sp.parent.mkdir(parents=True, exist_ok=True)
        tgt[["prompt", "model_response"]].to_csv(sp, index=False)

        # DPO: (prompt, chosen=target, rejected=source)
        src = pd.read_csv(src_cache)
        src["model_response"] = _col_str(src, "model_response")
        rj = src[["prompt", "model_response"]].rename(columns={"model_response": "rejected_response"})
        ch = tgt[["prompt", "model_response"]].rename(columns={"model_response": "chosen_response"})
        merged = pd.merge(rj, ch, on="prompt", how="inner")
        merged = merged[(merged["chosen_response"].str.len() > 0)
                        & (merged["rejected_response"].str.len() > 0)]
        merged = merged[["prompt", "chosen_response", "rejected_response"]]
        dp = dpo_data_path(cell)
        dp.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(dp, index=False)
        print(f"  built {cell.slug}: SFT {len(tgt)} rows, DPO {len(merged)} pairs")


def _preflight():
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            "TINKER_API_KEY is not set in this environment and is absent from the project "
            ".env. Training cannot authenticate to Tinker. Re-run in a shell where "
            "TINKER_API_KEY is exported. The launch is idempotent — already-registered "
            "adapters are skipped, so it only fills the gaps.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train-sft", action="store_true")
    ap.add_argument("--train-dpo", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--sources", nargs="*", default=NEW_SOURCES)
    args = ap.parse_args()

    cells = my_cells(args.datasets, args.sources)
    print(f"B2c cells ({len(cells)}) over {args.datasets}:")
    for c in cells:
        print(f"  {c.slug}")

    if (args.train_sft or args.train_dpo) and not args.dry_run:
        _preflight()

    if args.build:
        print("=== building data ===")
        build_data(cells)

    if args.train_sft:
        print("=== launch SFT ===")
        m = launch_sft(cells=cells, dry_run=args.dry_run, parallel=args.parallel)
        print({k: (len(v) if isinstance(v, list) else v) for k, v in m.items()})

    if args.train_dpo:
        print("=== launch DPO ===")
        m = launch_dpo(cells=cells, dry_run=args.dry_run, parallel=args.parallel)
        print({k: (len(v) if isinstance(v, list) else v) for k, v in m.items()})


if __name__ == "__main__":
    main()
