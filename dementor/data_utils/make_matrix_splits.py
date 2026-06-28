#!/usr/bin/env python3
"""Build the prompt splits for the experiment matrix.

Creates 7 CSVs in data/datasets/ and verifies train/eval disjointness:
- gsm8k:        eval_1000 (from gsm8k.test) + train_500 (from gsm8k.train)
- chatbot_arena: eval_1000 + train_500 (disjoint indices from the 9640-row pool)
- writingprompts: eval_500 + train_500 (disjoint, pulled from euclaise/writingprompts)
- humaneval:    164 prompts, eval-only (pulled from openai/human-eval)

Seed 42 everywhere for reproducibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "data" / "datasets"
SEED = 42


def write_split(df: pd.DataFrame, path: Path, prompt_col: str = "prompt") -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"prompt": df[prompt_col].astype(str).str.strip()})
    out = out[out["prompt"].str.len() > 0].reset_index(drop=True)
    out.to_csv(path, index=False)
    return len(out)


def gsm8k_splits() -> tuple[Path, Path]:
    test_df = pd.read_csv(DATASETS / "gsm8k" / "gsm8k_test.csv")
    train_df = pd.read_csv(DATASETS / "gsm8k" / "gsm8k_train.csv")

    eval_df = test_df.sample(n=1000, random_state=SEED).reset_index(drop=True)
    train_split = train_df.sample(n=500, random_state=SEED).reset_index(drop=True)

    eval_path = DATASETS / "gsm8k" / "gsm8k_prompts_eval_1000_seed42.csv"
    train_path = DATASETS / "gsm8k" / "gsm8k_prompts_train_500_seed42.csv"
    n_eval = write_split(eval_df, eval_path)
    n_train = write_split(train_split, train_path)
    print(f"[gsm8k] wrote eval={n_eval} train={n_train}")
    return eval_path, train_path


def chatbot_arena_splits() -> tuple[Path, Path]:
    src = DATASETS / "chatbot_arena" / "chatbot_arena_prompts.csv"
    df = pd.read_csv(src)
    if "prompt" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "prompt"})

    shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    eval_df = shuffled.iloc[:1000].copy()
    train_df = shuffled.iloc[1000:1500].copy()

    eval_path = DATASETS / "chatbot_arena" / "chatbot_arena_prompts_eval_1000_seed42.csv"
    train_path = DATASETS / "chatbot_arena" / "chatbot_arena_prompts_train_500_seed42.csv"
    n_eval = write_split(eval_df, eval_path)
    n_train = write_split(train_df, train_path)
    print(f"[chatbot_arena] wrote eval={n_eval} train={n_train}")
    return eval_path, train_path


def writingprompts_splits() -> tuple[Path, Path]:
    from datasets import load_dataset

    ds = load_dataset("euclaise/writingprompts", split="train")
    prompt_col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
    df = pd.DataFrame({"prompt": ds[prompt_col]})
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df = df[df["prompt"].str.len() > 0].drop_duplicates().reset_index(drop=True)
    shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    eval_df = shuffled.iloc[:500].copy()
    train_df = shuffled.iloc[500:1000].copy()

    eval_path = DATASETS / "writingprompts" / "writingprompts_eval_500_seed42.csv"
    train_path = DATASETS / "writingprompts" / "writingprompts_train_500_seed42.csv"
    n_eval = write_split(eval_df, eval_path)
    n_train = write_split(train_df, train_path)
    print(f"[writingprompts] wrote eval={n_eval} train={n_train}")
    return eval_path, train_path


def humaneval_split() -> Path:
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    df = pd.DataFrame({"prompt": ds["prompt"]})
    out_path = DATASETS / "humaneval" / "humaneval_prompts.csv"
    n = write_split(df, out_path)
    print(f"[humaneval] wrote {n} prompts (eval-only)")
    return out_path


def audit_disjoint(eval_path: Path, train_path: Path, label: str) -> bool:
    eval_set = set(pd.read_csv(eval_path)["prompt"].tolist())
    train_set = set(pd.read_csv(train_path)["prompt"].tolist())
    overlap = eval_set & train_set
    if overlap:
        print(f"[AUDIT FAIL] {label}: {len(overlap)} overlapping prompts")
        return False
    print(f"[audit ok] {label}: no overlap (eval={len(eval_set)}, train={len(train_set)})")
    return True


def main() -> int:
    print("Building prompt splits...\n")
    gsm_eval, gsm_train = gsm8k_splits()
    arena_eval, arena_train = chatbot_arena_splits()
    wp_eval, wp_train = writingprompts_splits()
    he_path = humaneval_split()

    print("\nAuditing train/eval disjointness...")
    ok = True
    ok &= audit_disjoint(gsm_eval, gsm_train, "gsm8k")
    ok &= audit_disjoint(arena_eval, arena_train, "chatbot_arena")
    ok &= audit_disjoint(wp_eval, wp_train, "writingprompts")
    print(f"[humaneval] eval-only (n={len(pd.read_csv(he_path))}) — no train split to audit")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
