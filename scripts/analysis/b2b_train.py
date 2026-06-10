"""B2b — break the capability<->size confound that B2a introduced.

B2a added two HIGH-capability sources, but both were LARGE (Llama-3.3-70B, Qwen3-32B),
so their retention is consistent with either capability or scale. B2b adds the missing
corner of the capability x size 2x2: a HIGH-capability SMALL model, Qwen3-4B (4B, MATH
0.64). The other three corners already exist (small+weak llama-3.1-8b launders;
large+weak qwen3.6-27b launders; large+strong nemotron/gpt-oss/llama-3.3-70b/qwen3-32b
retain).

Pre-registered test:
  - Qwen3-4B RETAINS (high DPO style persistence, like the strong models) -> capability
    drives durability, SIZE ruled out (a 4B retains; weak models launder at any size).
  - Qwen3-4B LAUNDERS (low persistence, like the small/weak llama-3.1-8b) -> B2a's
    retention was SIZE, not capability.

Same machinery as b2a_train.py: register Qwen3-4B in MODEL_SLUG/CHAT_TEMPLATE only (NOT
MODELS), build the 4 SFT + 4 DPO data CSVs from baselines, launch the 4-cell subset.

Stages:
  --build        create the 4 SFT + 4 DPO data CSVs from baselines (needs the source baseline)
  --train-sft    launch the 4 SFT jobs (Tinker spend)
  --train-dpo    launch the 4 DPO jobs (after SFT; Tinker spend)
  --dry-run      with a train stage, print the plan without spending
"""
from __future__ import annotations

import argparse

import pandas as pd

from workflows.run_matrix import (Cell, baseline_path, sft_data_path, dpo_data_path,
                                  launch_sft, launch_dpo)

NEW_SOURCES = ["Qwen/Qwen3-4B-Instruct-2507"]
ORIG_TARGETS = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3.6-27B",
                "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "openai/gpt-oss-20b"]
DATASET = "gsm8k"
SEED = 1


def my_cells() -> list[Cell]:
    return [Cell(source=s, target=t, dataset=DATASET, seed=SEED)
            for s in NEW_SOURCES for t in ORIG_TARGETS]


def _col_str(df, col):
    return df[col].astype(str).fillna("")


def build_data():
    for cell in my_cells():
        tgt_cache = baseline_path(cell.target, DATASET)
        src_cache = baseline_path(cell.source, DATASET)
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
    load_dotenv(str(Path(__file__).resolve().parents[2] / ".env"))
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
    args = ap.parse_args()

    cells = my_cells()
    print(f"B2b cells ({len(cells)}): " + ", ".join(c.slug for c in cells))

    if (args.train_sft or args.train_dpo) and not args.dry_run:
        _preflight()

    if args.build:
        print("=== building data ===")
        build_data()

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
