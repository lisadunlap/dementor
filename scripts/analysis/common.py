from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


CONDITIONS = ("source", "disguised", "target")


BIG5 = {
    "EXT": [
        "talkative", "bold", "assertive", "extraverted", "energetic",
        "outgoing", "sociable", "lively", "adventurous", "enthusiastic",
        "withdrawn", "quiet", "reserved", "timid", "inhibited",
        "shy", "silent", "introverted", "submissive", "unadventurous",
    ],
    "AGR": [
        "kind", "cooperative", "sympathetic", "warm", "helpful",
        "friendly", "trustful", "generous", "agreeable", "gentle",
        "harsh", "uncooperative", "unsympathetic", "cold", "unhelpful",
        "unfriendly", "distrustful", "stingy", "disagreeable", "unkind",
    ],
    "CON": [
        "organized", "efficient", "systematic", "thorough", "careful",
        "reliable", "dependable", "precise", "diligent", "orderly",
        "careless", "disorganized", "inefficient", "haphazard", "sloppy",
        "unreliable", "undependable", "imprecise", "negligent", "disorderly",
    ],
    "NEU": [
        "anxious", "nervous", "tense", "moody", "temperamental",
        "insecure", "unstable", "fearful", "emotional", "worrying",
        "relaxed", "calm", "stable", "secure", "confident",
        "easygoing", "untroubled", "composed", "serene", "balanced",
    ],
    "OPN": [
        "creative", "imaginative", "insightful", "artistic", "curious",
        "intellectual", "inventive", "perceptive", "thoughtful", "philosophical",
        "unimaginative", "uncreative", "incurious", "conventional", "uninventive",
        "imperceptive", "shallow", "literal", "narrow", "rigid",
    ],
}


STYLE = [
    "verbose", "concise", "terse", "elaborate", "succinct",
    "hedging", "definitive", "tentative", "cautious", "direct",
    "formal", "informal", "professional", "casual", "polished",
    "structured", "flowing", "fragmented", "methodical", "systematic",
    "confident", "uncertain", "authoritative", "hesitant", "decisive",
    "empathetic", "detached", "supportive", "clinical", "neutral",
    "analytical", "step-by-step", "didactic", "skeptical", "safety-conscious",
]


def descriptor_names(mode: str = "big5_style") -> list[str]:
    if mode == "style_only":
        return list(dict.fromkeys(STYLE))
    if mode == "big5_style":
        values = [adj for adjs in BIG5.values() for adj in adjs] + STYLE
        return list(dict.fromkeys(values))
    raise ValueError(f"Unsupported descriptor mode: {mode}")


def read_csv_robust(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        try:
            return pd.read_csv(path, on_bad_lines="skip", quoting=csv.QUOTE_ALL)
        except pd.errors.ParserError:
            return pd.read_csv(path, on_bad_lines="skip", quoting=csv.QUOTE_NONE, engine="python")


def slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "unknown"


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def normalize_comparison_df(
    comparison_csv: str | Path,
    *,
    source_responses: str | Path | None = None,
    source_col: str = "source_response",
    disguised_col: str = "model_response",
    target_col: str = "target_response",
) -> pd.DataFrame:
    df = read_csv_robust(comparison_csv).copy()
    if "prompt" not in df.columns:
        raise ValueError("comparison CSV must include a 'prompt' column")

    if disguised_col not in df.columns:
        if "disguised_response" in df.columns:
            disguised_col = "disguised_response"
        elif "model_response" in df.columns:
            disguised_col = "model_response"
        else:
            raise ValueError(f"Missing disguised response column: {disguised_col}")

    if target_col not in df.columns:
        if "target_response" in df.columns:
            target_col = "target_response"
        else:
            raise ValueError(f"Missing target response column: {target_col}")

    if source_col not in df.columns:
        if source_responses is None:
            raise ValueError(
                "Missing source response column and no --source-responses file was provided"
            )
        src = read_csv_robust(source_responses)
        if "prompt" not in src.columns:
            raise ValueError("source response CSV must include a 'prompt' column")
        if "model_response" in src.columns:
            src = src.rename(columns={"model_response": source_col})
        elif "target_response" in src.columns:
            src = src.rename(columns={"target_response": source_col})
        elif source_col not in src.columns:
            raise ValueError("source response CSV must include model_response, target_response, or source_response")
        df = pd.merge(df, src[["prompt", source_col]], on="prompt", how="inner")

    out = pd.DataFrame(
        {
            "prompt": df["prompt"].astype(str),
            "source_response": df[source_col].fillna("").astype(str),
            "disguised_response": df[disguised_col].fillna("").astype(str),
            "target_response": df[target_col].fillna("").astype(str),
        }
    )
    for col in ("source_model", "target_model", "method"):
        if col in df.columns:
            out[col] = df[col]
    return out


def condition_texts(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "source": df["source_response"].fillna("").astype(str).tolist(),
        "disguised": df["disguised_response"].fillna("").astype(str).tolist(),
        "target": df["target_response"].fillna("").astype(str).tolist(),
    }


def ensure_nonempty(items: Iterable, name: str) -> None:
    if len(list(items)) == 0:
        raise ValueError(f"{name} is empty")

