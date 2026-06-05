"""D1 capability axis: deterministic GSM8K grader (NO model calls, NO API spend).

Builds a prompt->gold lookup from the free HuggingFace `openai/gsm8k` (config
`main`) test split, implements a final-answer extractor + exact-match grader, and
grades every cached condition for all 12 gsm8k cross-pair cells under
`data/results/decontam/gsm8k/{source}_to_{target}/gen/`.

Direction convention (verified against scripts/analysis/run_cell_pipeline.py):
  For a cell `{source}_to_{target}`:
    - base_model = source  -> source is the imitator's OWN base.
    - the LoRA adapter is trained on the TARGET's outputs (tgt_train).
    - rung_*.csv = the source base wearing the disguise adapter, imitating target.
  Therefore on the CAPABILITY axis:
    - acc_source  = capability ORIGIN  (imitator's own competence; source_seed*).
    - acc_target  = capability DESTINATION (the imitated model; target_seed*).
    - acc_B(rung) = the disguised imitator's competence (rung_*.csv).
  This is semantically coherent with the style `persistence` metric, which
  measures RETENTION OF THE SOURCE style: persistence->0 means the imitator has
  shed its own (source) style and moved onto the target's. The capability dual
  asks: does competence move from acc_source toward acc_target as the same thing
  happens? cap_xfer = (acc_B - acc_source) / (acc_target - acc_source).

Outputs (all free, local):
  data/datasets/gsm8k/gsm8k_test_gold.csv   prompt -> gold lookup
  results/d1_gsm8k_accuracy_long.csv        per cell x condition accuracy
  results/d1_gsm8k_peritem_correct.csv      per-item correctness (bootstrap/McNemar)
  results/d1_grader_audit.csv               spot-check audit rows
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.common import read_csv_robust  # noqa: E402

DECONTAM = ROOT / "data/results/decontam/gsm8k"
GOLD_PATH = ROOT / "data/datasets/gsm8k/gsm8k_test_gold.csv"
ACC_LONG = ROOT / "results/d1_gsm8k_accuracy_long.csv"
PERITEM = ROOT / "results/d1_gsm8k_peritem_correct.csv"
AUDIT = ROOT / "results/d1_grader_audit.csv"

CONDITIONS = [
    "source_seed1", "source_seed2",
    "target_seed1", "target_seed2",
    "rung_just_name_it", "rung_random_sampling", "rung_stylistic",
    "rung_sft", "rung_dpo",
]


# ---------------------------------------------------------------------------
# Prompt normalization (curly vs straight apostrophe, whitespace)
# ---------------------------------------------------------------------------
def norm_prompt(p: str) -> str:
    p = str(p)
    p = p.replace("’", "'").replace("‘", "'")
    p = p.replace("“", '"').replace("”", '"')
    p = re.sub(r"\s+", " ", p).strip()
    return p


# ---------------------------------------------------------------------------
# Gold answer parsing from the GSM8K `answer` field (text after final ####).
# ---------------------------------------------------------------------------
def gold_from_answer(answer: str) -> Optional[str]:
    m = re.search(r"####\s*(.+)\s*$", str(answer).strip())
    raw = m.group(1) if m else str(answer)
    return canon_num(raw)


def canon_num(raw: str) -> Optional[str]:
    """Strip $ , % whitespace; cast to float; return canonical numeric string."""
    if raw is None:
        return None
    s = str(raw).strip()
    s = s.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
    s = s.replace("\\", "").replace("*", "").strip()
    m = re.search(r"-?\d*\.?\d+", s)
    if not m:
        return None
    try:
        f = float(m.group(0))
    except ValueError:
        return None
    # canonical form: int if integral, else trimmed float
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return repr(round(f, 6))


# ---------------------------------------------------------------------------
# Final-answer extractor from a model response.
# Priority: GSM8K `#### N`  >  "answer is / = N"  >  last number in text.
# ---------------------------------------------------------------------------
NUM_RE = r"-?\d[\d,]*(?:\.\d+)?"


def extract_pred(text: str) -> Optional[str]:
    if text is None:
        return None
    t = str(text)
    # strip latex \( \) \[ \] and bold markers that wrap final answers
    t_clean = t.replace(" ", "").replace(" ", "")

    # 1) GSM8K-style #### N
    m = re.search(r"####\s*\$?(" + NUM_RE + r")", t_clean)
    if m:
        return canon_num(m.group(1))

    # 2) "answer is X" / "= X" / "answer: X" near the end (take last such match)
    tie = list(re.finditer(
        r"(?:answer\s*(?:is|:)?\s*|=\s*|\bis\s*)\$?(" + NUM_RE + r")",
        t_clean, flags=re.IGNORECASE))
    # 3) last number anywhere (standard GSM8K eval fallback)
    nums = list(re.finditer(NUM_RE, t_clean))

    # Prefer an explicit "answer is/=" only if it occurs at/after the last number's
    # position minus a small window (i.e. it's the concluding statement). Simpler &
    # robust: use last number, but if the very last token region is "= N" or
    # "answer is N", that's already the last number. So last-number suffices and
    # the tie list mainly helps when trailing prose follows the answer.
    if nums:
        last_num = nums[-1].group(0)
        # If there is an "answer is/=" pattern strictly after the last bare number
        # start, prefer it (handles "...= 28 hours." where 28 is last number anyway).
        if tie and tie[-1].start() >= nums[-1].start():
            return canon_num(tie[-1].group(1))
        return canon_num(last_num)
    if tie:
        return canon_num(tie[-1].group(1))
    return None


def is_correct(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return str(pred).strip() == str(gold).strip()


# ---------------------------------------------------------------------------
# Step 0: build gold lookup
# ---------------------------------------------------------------------------
def build_gold() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for q, a in zip(ds["question"], ds["answer"]):
        rows.append({"prompt": q, "prompt_norm": norm_prompt(q), "gold": gold_from_answer(a)})
    gold = pd.DataFrame(rows)
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    gold[["prompt", "gold"]].to_csv(GOLD_PATH, index=False)
    print(f"[gold] wrote {len(gold)} rows -> {GOLD_PATH}")
    return gold


# ---------------------------------------------------------------------------
# Step 2: grade every condition per cell
# ---------------------------------------------------------------------------
def main() -> None:
    gold = build_gold()
    gmap = dict(zip(gold["prompt_norm"], gold["gold"]))

    cells = sorted(p.name for p in DECONTAM.iterdir() if p.is_dir())
    acc_rows = []
    item_rows = []
    audit_rows = []
    join_report = []

    for cell in cells:
        source, target = cell.split("_to_")
        gen = DECONTAM / cell / "gen"
        for cond in CONDITIONS:
            fp = gen / f"{cond}.csv"
            if not fp.exists():
                continue
            df = read_csv_robust(fp)
            df = df.dropna(subset=["prompt"]).drop_duplicates("prompt")
            df["prompt_norm"] = df["prompt"].map(norm_prompt)
            df["gold"] = df["prompt_norm"].map(gmap)
            joined = df["gold"].notna().sum()
            join_report.append((cell, cond, len(df), int(joined)))
            df = df[df["gold"].notna()].copy()
            df["pred"] = df["model_response"].map(extract_pred)
            df["correct"] = [is_correct(p, g) for p, g in zip(df["pred"], df["gold"])]

            role = ("source" if cond.startswith("source") else
                    "target" if cond.startswith("target") else
                    cond.replace("rung_", ""))
            n = len(df)
            ncorr = int(df["correct"].sum())
            acc_rows.append({
                "dataset": "gsm8k", "source": source, "target": target,
                "condition": cond, "rung_or_role": role,
                "n": n, "n_correct": ncorr, "acc": ncorr / n if n else float("nan"),
            })
            for _, r in df.iterrows():
                item_rows.append({
                    "source": source, "target": target, "condition": cond,
                    "rung_or_role": role, "prompt_norm": r["prompt_norm"],
                    "correct": int(r["correct"]),
                })

    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(ACC_LONG, index=False)
    pd.DataFrame(item_rows).to_csv(PERITEM, index=False)
    print(f"[acc] wrote {len(acc_df)} rows -> {ACC_LONG}")
    print(f"[peritem] wrote {len(item_rows)} rows -> {PERITEM}")

    # join report
    jr = pd.DataFrame(join_report, columns=["cell", "cond", "n_rows", "n_joined"])
    bad = jr[jr["n_joined"] < jr["n_rows"] - 1]
    print(f"[join] min joined/rows: {jr['n_joined'].min()}/{jr['n_rows'].max()}; "
          f"{len(bad)} cond(s) with >1 unjoined")
    if len(bad):
        print(bad.to_string(index=False))

    # Step 1.3 audit: 10 graded items each for 2 cells, source+target+dpo
    audit_cells = ["llama-3.1-8b_to_gpt-oss-20b", "gpt-oss-20b_to_llama-3.1-8b"]
    for cell in audit_cells:
        source, target = cell.split("_to_")
        gen = DECONTAM / cell / "gen"
        for cond in ["source_seed1", "target_seed1", "rung_dpo"]:
            fp = gen / f"{cond}.csv"
            if not fp.exists():
                continue
            df = read_csv_robust(fp).dropna(subset=["prompt"]).drop_duplicates("prompt")
            df["prompt_norm"] = df["prompt"].map(norm_prompt)
            df["gold"] = df["prompt_norm"].map(gmap)
            df = df[df["gold"].notna()].head(10)
            for _, r in df.iterrows():
                pred = extract_pred(r["model_response"])
                audit_rows.append({
                    "cell": cell, "cond": cond,
                    "resp_tail": str(r["model_response"])[-120:].replace("\n", " "),
                    "pred": pred, "gold": r["gold"],
                    "correct": is_correct(pred, r["gold"]),
                })
    pd.DataFrame(audit_rows).to_csv(AUDIT, index=False)
    print(f"[audit] wrote {len(audit_rows)} rows -> {AUDIT}")

    # quick accuracy summary by role
    print("\n=== accuracy by condition (mean over 12 cells) ===")
    print(acc_df.groupby("rung_or_role")["acc"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
