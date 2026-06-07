"""D1-FIX capability-spread census: deterministic graders for HARDER math benchmarks.

Pure local grading (NO model calls). Three benchmark formats are supported, each
with a deterministic exact-match extractor that an auditor can read off by eye:

  - gsm_symbolic : GSM-Symbolic (apple/GSM-Symbolic, config main). Gold = number
                   after the final '####' in the dataset `answer`. Prediction =
                   '#### N' if emitted, else "answer is/=" tail, else last number.
                   (Reuses the gsm8k extractor convention.)
  - math500      : HuggingFaceH4/MATH-500, levels 3-5, restricted to items whose
                   gold `answer` is a plain integer/decimal (so numeric exact-match
                   is reliable). Prediction = \\boxed{...} content if present, else
                   last number. Symbolic-answer items are excluded up front.
  - mmlu_pro     : TIGER-Lab/MMLU-Pro (a category subset, e.g. math). Gold = the
                   correct option LETTER. Prediction = a chosen letter parsed from
                   the response (boxed letter / 'answer is (X)' / standalone 'X').

All three reduce to is_correct(pred, gold) exact match. The grader is shared by
the census driver census_mathbench.py.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Numeric canonicalization + GSM-style extraction (mirrors grade_gsm8k.py)
# ---------------------------------------------------------------------------
NUM_RE = r"-?\d[\d,]*(?:\.\d+)?"


def canon_num(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    s = s.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
    s = s.replace("\\", "").replace("*", "").strip()
    m = re.search(r"-?\d*\.?\d+", s)
    if not m:
        return None
    try:
        f = float(m.group(0))
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(f):  # guard: a very long digit string overflows to inf -> round(inf) raises
        return None
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return repr(round(f, 6))


def is_plain_num(a: str) -> bool:
    s = str(a).replace("\\", "").replace("$", "").replace(",", "").replace("%", "").strip()
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))


# ---------------------------------------------------------------------------
# GSM-Symbolic gold + prediction
# ---------------------------------------------------------------------------
def gsm_gold(answer: str) -> Optional[str]:
    m = re.search(r"####\s*(.+)\s*$", str(answer).strip())
    raw = m.group(1) if m else str(answer)
    return canon_num(raw)


def gsm_extract(text: str) -> Optional[str]:
    if text is None:
        return None
    t = str(text).replace(" ", "").replace(" ", "")
    m = re.search(r"####\s*\$?(" + NUM_RE + r")", t)
    if m:
        return canon_num(m.group(1))
    tie = list(re.finditer(
        r"(?:answer\s*(?:is|:)?\s*|=\s*|\bis\s*)\$?(" + NUM_RE + r")", t, flags=re.IGNORECASE))
    nums = list(re.finditer(NUM_RE, t))
    if nums:
        if tie and tie[-1].start() >= nums[-1].start():
            return canon_num(tie[-1].group(1))
        return canon_num(nums[-1].group(0))
    if tie:
        return canon_num(tie[-1].group(1))
    return None


# ---------------------------------------------------------------------------
# MATH-500 gold + prediction (numeric-answer subset only)
# ---------------------------------------------------------------------------
def math500_gold(answer: str) -> Optional[str]:
    return canon_num(answer)


def _boxed_content(t: str) -> Optional[str]:
    # extract the LAST \boxed{...} accounting for nested braces
    idx = t.rfind("\\boxed")
    if idx == -1:
        return None
    j = t.find("{", idx)
    if j == -1:
        return None
    depth, k = 0, j
    while k < len(t):
        if t[k] == "{":
            depth += 1
        elif t[k] == "}":
            depth -= 1
            if depth == 0:
                return t[j + 1:k]
        k += 1
    return None


def math500_extract(text: str) -> Optional[str]:
    if text is None:
        return None
    t = str(text)
    box = _boxed_content(t)
    if box is not None:
        c = canon_num(box)
        if c is not None:
            return c
    # fall back: "answer is/=" then last number
    return gsm_extract(t)


# ---------------------------------------------------------------------------
# MMLU-Pro gold + prediction (letter choice)
# ---------------------------------------------------------------------------
def mmlu_extract(text: str, n_options: int) -> Optional[str]:
    if text is None:
        return None
    t = str(text)
    valid = {chr(ord("A") + i) for i in range(n_options)}
    # 1) \boxed{X}
    box = _boxed_content(t)
    if box:
        b = box.strip().strip("()").upper()
        if b in valid:
            return b
    # 2) "answer is (X)" / "answer: X" / "answer is X" (take last)
    pats = [
        r"answer\s*(?:is|:)?\s*\(?([A-Z])\)?",
        r"\bthe\s+answer\s+is\s+\(?([A-Z])\)?",
        r"\boption\s+\(?([A-Z])\)?",
    ]
    cands = []
    for p in pats:
        for m in re.finditer(p, t, flags=re.IGNORECASE):
            cands.append((m.start(), m.group(1).upper()))
    cands = [(s, l) for s, l in cands if l in valid]
    if cands:
        return max(cands, key=lambda x: x[0])[1]
    # 3) a standalone parenthesized letter near the end: (X)
    paren = [(m.start(), m.group(1).upper())
             for m in re.finditer(r"\(([A-Z])\)", t) if m.group(1).upper() in valid]
    if paren:
        return max(paren, key=lambda x: x[0])[1]
    # 4) last bare standalone capital letter token that is a valid option
    bare = [(m.start(), m.group(0).upper())
            for m in re.finditer(r"\b([A-Z])\b", t) if m.group(0).upper() in valid]
    if bare:
        return max(bare, key=lambda x: x[0])[1]
    return None


# ---------------------------------------------------------------------------
def is_correct(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (ValueError, TypeError):
        return str(pred).strip() == str(gold).strip()
