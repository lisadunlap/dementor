"""D1-FIX capability-transfer generation on MATH-500 (the benchmark the census found
to actually SPREAD the 4 models: gpt-oss 0.79 / qwen 0.50 / llama 0.45, spread 0.34 —
escapes the gsm8k ceiling that made the original D1 untestable).

For each cross-pair {source}_to_{target} we sample, on the SAME MATH-500 prompt set:
  - the SOURCE base model            -> acc_source (capability origin)
  - the TARGET base model            -> acc_target (capability destination)
  - the SFT and DPO gsm8k adapters   -> acc_imitator (does imitation import competence?)
All sampled through run_cell_pipeline._sample_messages with MAX_TOKENS bumped to 1024
(MATH needs reasoning room) at temp 0 (greedy), so base and adapter accuracies are
measured under identical settings. Adapter base_model = SOURCE (verified convention).

Idempotent: _sample_messages skips any unit whose CSV already has the full prompt set.
Outputs per-unit graded CSVs under results/d1fix_capgen/; the dissociation analysis
(cap_xfer vs style persistence) is computed separately in d1fix_analyze.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from scripts.analysis import run_cell_pipeline as RCP  # noqa: E402
from scripts.analysis import grade_mathbench as G  # noqa: E402
from scripts.analysis.census_mathbench import load_math500, MODELS  # noqa: E402
from scripts.analysis.run_cell_pipeline import _sample_messages  # noqa: E402

GEN_DIR = ROOT / "results/d1fix_capgen"
REGISTRY = ROOT / "data/tinker_adapters.json"
SLUG2FULL = {slug: full for full, slug in MODELS}
RUNGS = ["sft", "dpo"]


def ordered_pairs():
    slugs = [s for _, s in MODELS]
    return [(a, b) for a in slugs for b in slugs if a != b]


def grade_df(raw_path: Path, golds, out_path: Path):
    df = pd.read_csv(raw_path)
    n = len(df)
    df["gold"] = golds[:n]
    df["correct"] = [int(G.is_correct(G.math500_extract(str(r)), g))
                     for r, g in zip(df["model_response"], df["gold"])]
    df.to_csv(out_path, index=False)
    raw_path.unlink(missing_ok=True)
    return df["correct"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--cap", type=int, default=3000, help="hard cap on sampling calls")
    ap.add_argument("--diagonal", action="store_true",
                    help="A3 capability placebo: sample the 4 self-SFT gsm8k adapters on MATH-500")
    args = ap.parse_args()

    # Bump the reasoning budget for MATH (gsm8k default is 512) — affects every
    # _sample_messages call in this process.
    RCP.MAX_TOKENS = args.max_tokens

    load_dotenv()
    import tinker
    service = tinker.ServiceClient()

    items, _ = load_math500(args.n, args.seed)
    prompts = [it["prompt"] for it in items]
    golds = [it["gold"] for it in items]
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    reg = json.loads(REGISTRY.read_text())
    spent = {"calls": 0}

    def do_unit(label, base_full, model_path, out_final):
        if out_final.exists() and len(pd.read_csv(out_final)) == len(prompts):
            print(f"  [skip] {label}: cached", flush=True)
            return
        if spent["calls"] + len(prompts) > args.cap:
            print(f"  [BUDGET] would exceed cap {args.cap}; stopping at {label}", flush=True)
            return "stop"
        raw = GEN_DIR / f"_raw_{out_final.stem}.csv"
        _sample_messages(service, base_model=base_full, model_path=model_path,
                         messages_by_prompt=msgs, prompts=prompts,
                         clean_model=base_full, render_model=base_full,
                         temperature=args.temperature, seed=args.seed,
                         parallel=args.parallel, out_path=raw, label=label)
        spent["calls"] += len(prompts)
        acc = grade_df(raw, golds, out_final)
        print(f"  {label}: acc={acc:.3f}", flush=True)

    # 1) Base models (acc_source / acc_target) under the identical 1024-token regime.
    print("=== base models on MATH-500 ===", flush=True)
    for full, slug in MODELS:
        if do_unit(f"base {slug}", full, None, GEN_DIR / f"base_{slug}.csv") == "stop":
            return

    # 2) Adapter rungs (sft, dpo) for all 12 cross-pairs, base_model = SOURCE.
    print("=== adapter rungs on MATH-500 ===", flush=True)
    for (src, tgt) in ordered_pairs():
        for rung in RUNGS:
            alias = f"{rung}_gsm8k_{src}_as_{tgt}_seed1"
            entry = reg.get(alias)
            if not entry or not entry.get("path"):
                print(f"  [missing] {alias}", flush=True)
                continue
            out = GEN_DIR / f"{rung}_{src}_as_{tgt}.csv"
            if do_unit(f"adapter {alias}", SLUG2FULL[src], entry["path"], out) == "stop":
                return

    # 3) A3 capability placebo: self-SFT gsm8k adapters (model imitating its OWN gsm8k outputs).
    #    Expectation: MATH acc ≈ native base acc (self-imitation should not move capability).
    if args.diagonal:
        print("=== self-SFT gsm8k adapters on MATH-500 (capability placebo) ===", flush=True)
        for full, slug in MODELS:
            alias = f"self_sft_gsm8k_{slug}_as_{slug}_seed1"
            entry = reg.get(alias)
            if not entry or not entry.get("path"):
                print(f"  [missing] {alias}", flush=True)
                continue
            if do_unit(f"self {alias}", full, entry["path"], GEN_DIR / f"self_{slug}.csv") == "stop":
                return

    print(f"\n[done] sampling calls this run: {spent['calls']}", flush=True)


if __name__ == "__main__":
    main()
