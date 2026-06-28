"""B2a — train the 8 confound-breaker cells: 2 new high-capability sources (from launderer
lineages) imitating the original 4 targets, gsm8k, seed1.

Targets EXACTLY 8 cells via launch_sft(cells=...) / launch_dpo(cells=...) — the 2 new models
are registered in MODEL_SLUG/CHAT_TEMPLATE_KWARGS but NOT in MODELS, so the global matrix is
unchanged and we build/launch only our subset. Data CSVs are created directly (build_sft_data /
build_dpo_data iterate the global matrix, which excludes our new sources).

Stages:
  --build        create the 8 SFT + 8 DPO data CSVs from baselines (cheap; needs new-source baselines)
  --train-sft    launch the 8 SFT jobs (Tinker spend)
  --train-dpo    launch the 8 DPO jobs (after SFT; Tinker spend)
  --dry-run      with a train stage, print the plan without spending
"""
from __future__ import annotations

import argparse

import pandas as pd

from dementor.training.matrix import (Cell, baseline_path, sft_data_path, dpo_data_path,
                                  launch_sft, launch_dpo)

NEW_SOURCES = ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3-32B"]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train-sft", action="store_true")
    ap.add_argument("--train-dpo", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()

    cells = my_cells()
    print(f"B2a cells ({len(cells)}): " + ", ".join(c.slug for c in cells))

    # Preflight: training delegates to Tinker (SFT) / tinker_cookbook (DPO), both of which
    # construct a tinker.ServiceClient() that reads TINKER_API_KEY from the environment. The
    # key is NOT in the project .env (it is harness-injected per-invocation), so a run that
    # lacks it would otherwise spawn N jobs that each burn 4×60s retries before failing at auth.
    # Fail fast with an actionable message instead.
    if args.train_sft or args.train_dpo:
        import os
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))
        if not os.environ.get("TINKER_API_KEY"):
            raise SystemExit(
                "TINKER_API_KEY is not set in this environment and is absent from the project "
                ".env. Training cannot authenticate to Tinker. Re-run this command in a shell "
                "where TINKER_API_KEY is exported (e.g. `TINKER_API_KEY=... ./.venv/bin/python "
                "experiments/analysis/b2a_train.py --train-dpo --parallel 4`). The launch is "
                "idempotent — already-registered adapters are skipped, so it only fills the gaps."
            )

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
