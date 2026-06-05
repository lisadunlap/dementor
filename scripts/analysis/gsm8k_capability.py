"""D2 Step 3 — gsm8k capability transfer (H2, gsm8k cells only).

Fetches gsm8k TEST gold once (free `datasets` download, no API key, no model
inference), grades source/target/each-rung disguised generations by exact-match,
and defines a signed capability-transfer fraction analogous to persistence
movement. H2: across gsm8k cells, capability transfer correlates with reasoning
transfer more strongly than with style transfer.

All intermediates land under data/results/reasoning/ (gold lookup included) — we
do NOT write to data/datasets/.

Pure CPU + one network download. No model generation.

Usage:
  PYTHONPATH=. ./.venv/bin/python -m scripts.analysis.gsm8k_capability
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.reasoning_structure import _read_responses
from scripts.analysis.structural_decomp import SHORT

DECONTAM_ROOT = Path("data/results/decontam")
OUT_DIR = Path("data/results/reasoning")
GOLD_CACHE = OUT_DIR / "gsm8k_test_gold_lookup.csv"  # our own intermediate, NOT data/datasets/
RUNGS = ["just_name_it", "random_sampling", "stylistic", "sft", "dpo"]
LOW_POWER = 0.05  # |acc_t - acc_s| below this => denominator too small for headline


def _norm_prompt(p: str) -> str:
    return re.sub(r"\s+", " ", str(p or "")).strip().lower()


def _norm_num(s: str) -> str | None:
    if s is None:
        return None
    s = s.strip().replace("$", "").replace(",", "").replace("%", "")
    s = s.rstrip(".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    val = m.group(0)
    # normalize trailing .0 so "18" == "18.0"
    if "." in val:
        val = val.rstrip("0").rstrip(".")
    return val if val else "0"


def build_gold() -> dict[str, str]:
    if GOLD_CACHE.exists():
        df = pd.read_csv(GOLD_CACHE)
        return {r.norm_prompt: _norm_num(str(r.gold)) for r in df.itertuples()}
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for q, a in zip(ds["question"], ds["answer"]):
        m = re.search(r"####\s*([-0-9.,]+)", a)
        gold = _norm_num(m.group(1)) if m else None
        rows.append({"norm_prompt": _norm_prompt(q), "gold": gold})
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOLD_CACHE, index=False)
    return {r.norm_prompt: r.gold for r in df.itertuples()}


_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_HASH = re.compile(r"####\s*([-0-9.,$%]+)")
_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def extract_pred(text: str) -> str | None:
    s = str(text or "")
    boxed = _BOXED.findall(s)
    if boxed:
        return _norm_num(boxed[-1])
    hashed = _HASH.findall(s)
    if hashed:
        return _norm_num(hashed[-1])
    # last number in the final non-empty line, else last number anywhere
    lines = [ln for ln in s.splitlines() if ln.strip()]
    for ln in reversed(lines[-3:] if lines else []):
        nums = _NUM.findall(ln)
        if nums:
            return _norm_num(nums[-1])
    nums = _NUM.findall(s)
    return _norm_num(nums[-1]) if nums else None


def _grade(prompts: list[str], responses: list[str], gold: dict[str, str]) -> tuple[float, int, int]:
    correct = 0
    graded = 0
    unparsed = 0
    for p, r in zip(prompts, responses):
        g = gold.get(_norm_prompt(p))
        if g is None:
            continue  # prompt not in gold (shouldn't happen for gsm8k test)
        graded += 1
        pred = extract_pred(r)
        if pred is None:
            unparsed += 1
            continue
        if pred == g:
            correct += 1
    acc = correct / graded if graded else float("nan")
    return acc, graded, unparsed


def compute() -> tuple[pd.DataFrame, dict]:
    gold = build_gold()
    rows = []
    total_unparsed = 0
    total_graded = 0
    for dpo_csv in sorted((DECONTAM_ROOT / "gsm8k").glob("*/gen/rung_dpo.csv")):
        gen = dpo_csv.parent
        pair = gen.parts[gen.parts.index("decontam") + 2]
        src_full, tgt_full = pair.split("_to_")
        src, tgt = SHORT[src_full], SHORT[tgt_full]

        sdf = pd.read_csv(gen / "source_seed1.csv")
        tdf = pd.read_csv(gen / "target_seed1.csv")
        acc_s, gs, us = _grade(sdf["prompt"].tolist(), sdf["model_response"].tolist(), gold)
        acc_t, gt, ut = _grade(tdf["prompt"].tolist(), tdf["model_response"].tolist(), gold)
        total_unparsed += us + ut
        total_graded += gs + gt

        for rung in RUNGS:
            ddf = pd.read_csv(gen / f"rung_{rung}.csv")
            acc_d, gd, ud = _grade(ddf["prompt"].tolist(), ddf["model_response"].tolist(), gold)
            total_unparsed += ud
            total_graded += gd
            denom = acc_t - acc_s
            low_power = abs(denom) < LOW_POWER
            if abs(denom) > 1e-9:
                cap = (acc_d - acc_s) / denom
                cap = float(np.clip(cap, -0.5, 1.5))
            else:
                cap = float("nan")
            rows.append({
                "source": src, "target": tgt, "rung": rung,
                "acc_source": acc_s, "acc_target": acc_t, "acc_disguised": acc_d,
                "cap_transfer": cap, "low_power": low_power,
            })
    meta = {
        "extraction_failure_rate": total_unparsed / max(total_graded, 1),
        "n_graded": total_graded,
    }
    return pd.DataFrame(rows), meta


def h2_stats(cap: pd.DataFrame, reasoning_csv: Path, style_csv: Path) -> tuple[pd.DataFrame, dict]:
    """Correlate cap_transfer vs (1-persist_reasoning) and vs (1-persist_style)."""
    from scipy.stats import pearsonr, spearmanr

    rs = pd.read_csv(reasoning_csv)
    rs = rs[rs["dataset"] == "gsm8k"][["source", "target", "rung", "persist_reasoning"]]
    st = pd.read_csv(style_csv)
    st = st[st["dataset"] == "gsm8k"][["source", "target", "rung", "persist_style_recomp"]]

    df = cap.merge(rs, on=["source", "target", "rung"], how="left")
    df = df.merge(st, on=["source", "target", "rung"], how="left")
    df["transfer_reasoning"] = 1.0 - df["persist_reasoning"]
    df["transfer_style"] = 1.0 - df["persist_style_recomp"]

    rows = []
    for rung_set, label in [(["sft", "dpo"], "sft+dpo"), (["sft"], "sft"), (["dpo"], "dpo")]:
        sub = df[df["rung"].isin(rung_set) & (~df["low_power"])].copy()
        sub = sub.dropna(subset=["cap_transfer", "transfer_reasoning", "transfer_style"])
        if len(sub) < 3:
            continue
        pr_rs, _ = pearsonr(sub["cap_transfer"], sub["transfer_reasoning"])
        pr_st, _ = pearsonr(sub["cap_transfer"], sub["transfer_style"])
        sp_rs, _ = spearmanr(sub["cap_transfer"], sub["transfer_reasoning"])
        sp_st, _ = spearmanr(sub["cap_transfer"], sub["transfer_style"])
        rows.append({
            "rungs": label, "n": len(sub),
            "pearson_reasoning": float(pr_rs), "pearson_style": float(pr_st),
            "spearman_reasoning": float(sp_rs), "spearman_style": float(sp_st),
            "reasoning_wins": abs(pr_rs) > abs(pr_st),
        })
    stat = pd.DataFrame(rows)
    return df, {"h2_table": stat}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap, meta = compute()
    cap.to_csv(OUT_DIR / "gsm8k_capability_percell.csv", index=False)

    df, h2 = h2_stats(
        cap,
        OUT_DIR / "reasoning_persistence_percell.csv",
        OUT_DIR / "reasoning_vs_style_percell.csv",
    )
    df.to_csv(OUT_DIR / "gsm8k_capability_joined.csv", index=False)
    h2["h2_table"].to_csv(OUT_DIR / "h2_capability_corr.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("=" * 80)
    print("D2 H2 — gsm8k capability transfer vs reasoning/style transfer")
    print("=" * 80)
    print(f"extraction failure rate: {meta['extraction_failure_rate']:.3f}  (n_graded={meta['n_graded']})")
    print("\n--- mean accuracy by rung (across 12 gsm8k cells) ---")
    print(cap.groupby("rung")[["acc_source", "acc_target", "acc_disguised", "cap_transfer"]]
          .mean().reindex(RUNGS).round(3).to_string())
    print(f"\nlow-power cells (|acc_t-acc_s|<{LOW_POWER}) excluded from headline: "
          f"{int(cap['low_power'].sum())}/{len(cap)}")
    print("\n--- H2: |corr(cap_transfer, reasoning)| vs |corr(cap_transfer, style)| ---")
    print(h2["h2_table"].round(3).to_string(index=False))
    print(f"\nwrote -> {OUT_DIR}/gsm8k_capability_percell.csv, h2_capability_corr.csv")


if __name__ == "__main__":
    main()
