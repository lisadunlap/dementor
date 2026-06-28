"""Fetch published refusal-benchmark PROMPTS (text only, no completions, no model spend).

Direction 3 — Safety-behavior laundering. This is the Phase-0 prompt-set assembly
step: download AdvBench `harmful_behaviors` and XSTest, normalize both to a common
`prompt,category,expected` schema, and write fixed seed-42 eval subsets plus the full
sets to `data/datasets/benchmarks/refusal/`.

DUAL-USE NOTE
-------------
This downloads only the *prompts* of standard, published refusal benchmarks. It does
NOT generate, store, or amplify any harmful completions. The downstream ladder runner
emits only a binary refuse/comply verdict + a redacted snippet. We do not author new
harmful prompts; AdvBench and XSTest are cited public benchmarks.

This script performs free HuggingFace `datasets` downloads only. No Tinker / paid API.

Usage:
    ./.venv/bin/python scripts/safety/fetch_refusal_prompts.py
    ./.venv/bin/python scripts/safety/fetch_refusal_prompts.py --n-harmful 120 --n-overrefusal 80
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OUT_DIR = Path("data/datasets/benchmarks/refusal")
SEED = 42  # mirror the repo-wide seed-42 eval-subset convention

# Published benchmark sources (prompts only). Non-gated public mirrors of the
# canonical sets (walledai/AdvBench and walledai/XSTest are gated on the Hub).
#   - AdvBench harmful_behaviors (Zou et al. 2023): 520 harmful instructions.
#     Mirror exposes columns `instruction` (the prompt) + `response` (an example
#     affirmative completion we DELIBERATELY ignore — never stored).
#   - XSTest v2 (Röttger et al. 2024): 250 safe over-refusal probes + 200 unsafe
#     `contrast_*` variants. Mirror's `prompts` split has `type` + `prompt`.
ADVBENCH_HF = ("ivnle/advbench_harmful_behaviors", "train")
XSTEST_HF = ("natolambert/xstest-v2-copy", "prompts")


def _normalize_advbench(df: pd.DataFrame) -> pd.DataFrame:
    """AdvBench harmful_behaviors -> prompt,category,expected (all expected=refuse).

    Uses the `instruction` column as the prompt. The mirror also ships a
    `response` column (an affirmative jailbreak-style completion); we never read,
    store, or surface it — dual-use mitigation.
    """
    col = "instruction" if "instruction" in df.columns else (
        "prompt" if "prompt" in df.columns else (
            "goal" if "goal" in df.columns else df.columns[0]))
    out = pd.DataFrame({"prompt": df[col].astype(str).str.strip()})
    out["category"] = "harmful"
    out["expected"] = "refuse"
    out = out.drop_duplicates("prompt").reset_index(drop=True)
    return out


def _normalize_xstest(df: pd.DataFrame) -> pd.DataFrame:
    """XSTest -> prompt,category,expected.

    In XSTest v2, `type` values prefixed `contrast_` are the genuinely-unsafe
    variants (expected=refuse); the rest are benign-but-trigger-y over-refusal
    probes (expected=comply). The `type` is kept as `category` for fine-grained
    over-refusal analysis. We drop the `completion` column (never stored).
    """
    pcol = "prompt" if "prompt" in df.columns else df.columns[0]
    typ = df["type"].astype(str) if "type" in df.columns else pd.Series(["xstest"] * len(df))
    is_unsafe = typ.str.startswith("contrast")
    out = pd.DataFrame({"prompt": df[pcol].astype(str).str.strip()})
    out["category"] = typ.values
    out["expected"] = ["refuse" if u else "comply" for u in is_unsafe]
    out = out.drop_duplicates("prompt").reset_index(drop=True)
    return out


def _load_hf(repo: str, split_hint: str) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(repo)
    if split_hint and split_hint in ds:
        split = split_hint
    elif "train" in ds:
        split = "train"
    else:
        split = list(ds.keys())[0]
    return ds[split].to_pandas()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-harmful", type=int, default=120,
                    help="Size of the fixed harmful eval subset (seed-42). 0 = keep full set only.")
    ap.add_argument("--n-overrefusal", type=int, default=80,
                    help="Size of the fixed XSTest-safe over-refusal eval subset (seed-42).")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[advbench] downloading {ADVBENCH_HF[0]} (prompts only) ...", flush=True)
    adv = _normalize_advbench(_load_hf(*ADVBENCH_HF))
    print(f"[xstest]   downloading {XSTEST_HF[0]} (prompts only) ...", flush=True)
    xs = _normalize_xstest(_load_hf(*XSTEST_HF))

    # Full sets.
    adv_full = args.out_dir / "advbench_harmful.csv"
    xs_full = args.out_dir / "xstest.csv"
    adv.to_csv(adv_full, index=False)
    xs.to_csv(xs_full, index=False)
    print(f"[write] {adv_full}  rows={len(adv)}")
    print(f"[write] {xs_full}  rows={len(xs)} "
          f"(comply/over-refusal={int((xs.expected=='comply').sum())}, refuse={int((xs.expected=='refuse').sum())})")

    # Fixed seed-42 eval subsets used as the default 200-prompt cost-bounded set.
    harmful = adv if args.n_harmful in (0, None) or args.n_harmful >= len(adv) \
        else adv.sample(n=args.n_harmful, random_state=SEED).reset_index(drop=True)
    xs_safe = xs[xs.expected == "comply"].reset_index(drop=True)
    over = xs_safe if args.n_overrefusal in (0, None) or args.n_overrefusal >= len(xs_safe) \
        else xs_safe.sample(n=args.n_overrefusal, random_state=SEED).reset_index(drop=True)

    eval_set = pd.concat([harmful, over], ignore_index=True)
    eval_path = args.out_dir / f"eval_{len(harmful)}h_{len(over)}o_seed{SEED}.csv"
    eval_set.to_csv(eval_path, index=False)
    print(f"[write] {eval_path}  rows={len(eval_set)} "
          f"(harmful={len(harmful)}, over-refusal={len(over)})")
    print("\nDone. Prompts only; no completions generated, no spend incurred.")


if __name__ == "__main__":
    main()
