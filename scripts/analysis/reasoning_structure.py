"""D2 Step 0 — reasoning-trace structure feature extractor.

Deterministic, regex/counting only, CPU, no MiniLM, no API. Mirrors the
``structural_matrix(texts) -> (X, names)`` signature of
``structural_decomp.structural_matrix`` but produces a feature vector that is
DISJOINT from the 32-D ``_style_*`` style basis: where the style features count
markdown/greeting/bullets, these features count the *shape of the chain of
thought* — number of reasoning steps, granularity per step, formal-derivation
density, self-correction, branching, answer-restatement, logical connectives.

The motivating contrast (verified on the de-confounded ``decontam/`` gens):
nemotron writes explicit "**Step-by-step reasoning** / 1. / 2. ..." enumerations
while gpt-oss writes compact LaTeX derivations. Those are different reasoning
*structures* the imitator may or may not pick up, independent of surface style.

Usage (smoke test):
  PYTHONPATH=. ./.venv/bin/python -m scripts.analysis.reasoning_structure
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RESPONSE_COL = "model_response"
DECONTAM_ROOT = Path("data/results/decontam")

# CoT-leak guard tokens — if any appears we are reading a non-decontam/confounded
# source and must abort (R3). Never re-introduce the gpt-oss chat-template leak.
LEAK_TOKENS = ("<|channel|>", "<|message|>", "analysis<|", "<|start|>", "<|end|>")

# --- compiled patterns -------------------------------------------------------
_STEP_LINE = re.compile(
    r"^\s*(?:\d+[.)]|[-*•]|Step\s*\d+|First|Second|Third|Then|Next|Finally)\b",
    re.IGNORECASE,
)
_STEP_MARKER = re.compile(r"step\s*[-\s]?(?:by[-\s]?step|\d+)", re.IGNORECASE)
_LETME = re.compile(
    r"\b(?:let me|let's|we have|we need|we get|we can|note that|observe that)\b",
    re.IGNORECASE,
)
_LATEX_BLOCK = re.compile(r"\\\[|\\\]|\\\(|\\\)")
_CALC_MARKER = re.compile(r"<<.*?>>")
_RESTATE_CUE = re.compile(
    r"(?:answer is|the answer|therefore|so the|\*\*answer|####|"
    r"final answer|in total|altogether|=\s*[-0-9.,$]+\s*\w*\.?\s*$)",
    re.IGNORECASE,
)
_SELFCORRECT = re.compile(
    r"\b(?:wait|actually|re-?check|reconsider|that's wrong|that is wrong|"
    r"on second thought|correction|let me redo|hmm|oops)\b",
    re.IGNORECASE,
)
_BRANCHING = re.compile(
    r"\b(?:case\s*\d+|if\b.*\bthen\b|otherwise|either\b|alternatively|"
    r"on the other hand|in the other case)\b",
    re.IGNORECASE,
)
_CONNECTIVE = re.compile(
    r"\b(?:therefore|thus|hence|because|since|so that|which means|consequently)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w])-?\d+(?:[.,]\d+)?")

FEATURE_NAMES = [
    "rs_step_count",
    "rs_tokens_per_step",
    "rs_step_markers",
    "rs_letme_markers",
    "rs_equation_density",
    "rs_restatement",
    "rs_selfcorrection",
    "rs_branching",
    "rs_connective_density",
    "rs_n_equations_total",
    "rs_n_numbers",
]


def _features_one(text: str) -> list[float]:
    value = str(text or "")
    words = value.split()
    n_words = len(words)
    lines = [ln for ln in value.splitlines() if ln.strip()]

    step_count = float(sum(1 for ln in lines if _STEP_LINE.match(ln)))
    tokens_per_step = float(n_words) / max(step_count, 1.0)
    step_markers = float(len(_STEP_MARKER.findall(value)))
    letme = float(len(_LETME.findall(value)))

    n_eq = float(value.count("="))
    n_latex = float(len(_LATEX_BLOCK.findall(value)))
    n_calc = float(len(_CALC_MARKER.findall(value)))
    eq_total = n_eq + n_latex + n_calc
    per100 = 100.0 / max(n_words, 1.0)
    eq_density = eq_total * per100

    # answer-restatement: cue present in the final non-empty line, OR a global cue count
    last_line = lines[-1] if lines else ""
    restatement = float(bool(_RESTATE_CUE.search(last_line)))

    selfcorrect = float(len(_SELFCORRECT.findall(value)))
    branching = float(len(_BRANCHING.findall(value)))
    connective_density = float(len(_CONNECTIVE.findall(value))) * per100
    n_numbers = float(len(_NUMBER.findall(value)))

    return [
        step_count,
        tokens_per_step,
        step_markers,
        letme,
        eq_density,
        restatement,
        selfcorrect,
        branching,
        connective_density,
        eq_total,
        n_numbers,
    ]


def reasoning_matrix(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Reasoning-structure feature matrix (raw, un-scaled). Disjoint from style."""
    rows = [_features_one(t) for t in texts]
    X = np.asarray(rows, dtype=float) if rows else np.zeros((0, len(FEATURE_NAMES)))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, list(FEATURE_NAMES)


def assert_no_leak(texts: list[str], where: str = "") -> None:
    for t in texts:
        s = str(t or "")
        for tok in LEAK_TOKENS:
            if tok in s:
                raise SystemExit(
                    f"CoT-leak token {tok!r} found in {where}; refusing to measure on "
                    f"confounded text. Only the de-confounded decontam/ gens are allowed."
                )


def _read_responses(path: Path) -> list[str]:
    texts = pd.read_csv(path)[RESPONSE_COL].astype(str).tolist()
    assert_no_leak(texts, where=str(path))
    return texts


def _smoke() -> None:
    """Eyeball 3 source/target pairs: nemotron should out-enumerate gpt-oss."""
    out_rows = []
    pairs = [
        ("gsm8k", "nemotron-nano-30b-a3b_to_gpt-oss-20b"),
        ("gsm8k", "gpt-oss-20b_to_llama-3.1-8b"),
        ("gsm8k", "qwen3.6-27b_to_nemotron-nano-30b-a3b"),
    ]
    for dataset, pair in pairs:
        gen = DECONTAM_ROOT / dataset / pair / "gen"
        for role in ("source_seed1.csv", "target_seed1.csv"):
            texts = _read_responses(gen / role)
            X, names = reasoning_matrix(texts)
            mean = X.mean(axis=0)
            rec = {"dataset": dataset, "pair": pair, "role": role.split("_")[0]}
            rec.update({n: round(float(v), 3) for n, v in zip(names, mean)})
            out_rows.append(rec)
    df = pd.DataFrame(out_rows)
    out = Path("data/results/reasoning/feature_smoke.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("=== reasoning-structure smoke (per-response feature means) ===")
    print(df.to_string(index=False))
    print(f"\nwrote -> {out}")
    print(
        "\nsanity: nemotron source rows should show higher rs_step_count / "
        "rs_step_markers than gpt-oss; gpt-oss higher rs_equation_density."
    )


if __name__ == "__main__":
    _smoke()
