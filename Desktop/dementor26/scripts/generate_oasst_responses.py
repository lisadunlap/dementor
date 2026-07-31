"""
Generate model responses for the OpenAssistant oasst1 dataset.

Steps:
  1. Download oasst1 from HuggingFace and cache to data/datasets/openassistant/
  2. Extract root-level English prompter messages
  3. Sample 500 for train and 1000 for test (seed=42)
  4. For each model, call OpenRouter in parallel (workers=256) and stream rows
     to CSV in real-time as they complete

Outputs:
  data/model-responses/openasisstant/train/train_{MODEL_NAME}_500.csv
  data/model-responses/openasisstant/test/test_{MODEL_NAME}_1000.csv
"""
from __future__ import annotations

import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
DATASET_DIR = DATA_DIR / "datasets" / "openassistant"
TRAIN_OUT_DIR = DATA_DIR / "model-responses" / "openasisstant" / "train"
TEST_OUT_DIR = DATA_DIR / "model-responses" / "openasisstant" / "test"

TRAIN_N = 500
TEST_N = 1000
SEED = 42
MAX_TOKENS = 512
TEMPERATURE = 0.7  # slight creativity for natural paragraph prose
WORKERS = 256

MODELS: List[str] = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3.6-27B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "openai/gpt-oss-20b",
]

# Map model IDs to their OpenRouter equivalents.
OPENROUTER_MODEL_MAP: Dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct":           "meta-llama/llama-3.1-8b-instruct",
    "Qwen/Qwen3.6-27B":                            "qwen/qwen3-30b-a3b",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-20b":                          "openai/gpt-oss-20b",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Always respond in 1-2 well-structured paragraphs. "
    "Be thorough but concise — do not use bullet points or headers, write in flowing prose."
)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def download_oasst1() -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets library not found. pip install datasets")

    print("Downloading OpenAssistant/oasst1 from HuggingFace …")
    ds = load_dataset("OpenAssistant/oasst1")
    frames = []
    for split_name, split_data in ds.items():
        df = split_data.to_pandas()
        df["hf_split"] = split_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def extract_prompts(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["role"] == "prompter")
        & (df["parent_id"].isna())
        & (df["deleted"] == False)  # noqa: E712
        & (df["lang"] == "en")
    )
    prompts = df.loc[mask, ["message_id", "text", "hf_split"]].copy()
    prompts = prompts.rename(columns={"text": "prompt"})
    prompts = prompts.drop_duplicates(subset="prompt").reset_index(drop=True)
    print(f"Extracted {len(prompts):,} unique English root prompts.")
    return prompts


def split_prompts(prompts: pd.DataFrame, train_n: int, test_n: int, seed: int):
    need = train_n + test_n
    if len(prompts) < need:
        raise ValueError(f"Need {need} prompts but only have {len(prompts)}.")
    sampled = prompts.sample(n=need, random_state=seed).reset_index(drop=True)
    return sampled.iloc[:train_n].reset_index(drop=True), sampled.iloc[train_n:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Real-time CSV writer (thread-safe)
# ---------------------------------------------------------------------------

class StreamingCSVWriter:
    """Opens a CSV and appends rows immediately as they arrive."""

    FIELDS = ["prompt", "model_response", "model"]

    def __init__(self, path: Path, existing_prompts: set):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._existing = existing_prompts
        write_header = not path.exists() or path.stat().st_size == 0
        self._handle = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=self.FIELDS, quoting=csv.QUOTE_ALL
        )
        if write_header:
            self._writer.writeheader()
            self._handle.flush()

    def write(self, row: dict) -> None:
        with self._lock:
            self._writer.writerow(row)
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _load_existing(path: Path) -> set:
    """Return the set of prompts already in the CSV."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, quoting=csv.QUOTE_ALL)
        if "prompt" in df.columns:
            return set(df["prompt"].astype(str).tolist())
    except Exception:
        pass
    return set()


# ---------------------------------------------------------------------------
# Per-prompt generation (called inside thread pool)
# ---------------------------------------------------------------------------

def _call_openrouter(prompt: str, or_model: str, model_id: str, api_key: str) -> dict:
    from openai import OpenAI, APIError, RateLimitError

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/dementor26",
            "X-Title": "dementor26-oasst",
        },
    )
    # Disable reasoning/thinking for reasoning models to avoid wasting tokens
    _is_reasoning_model = "nemotron" in or_model.lower() or "thinking" in or_model.lower()
    extra = {"reasoning": {"enabled": False}} if _is_reasoning_model else {}

    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=or_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                extra_body=extra if extra else None,
            )
            return {
                "prompt": prompt,
                "model_response": resp.choices[0].message.content or "",
                "model": model_id,
            }
        except RateLimitError:
            time.sleep(min(2 ** attempt, 60))
        except APIError as exc:
            if attempt == 5:
                return {"prompt": prompt, "model_response": f"ERROR: {exc}", "model": model_id}
            time.sleep(5)
    return {"prompt": prompt, "model_response": "ERROR: max retries", "model": model_id}


def _call_tinker(prompt: str, model_id: str, api_key: str, base_url: str) -> dict:
    from openai import OpenAI, APIError, RateLimitError

    client = OpenAI(base_url=base_url, api_key=api_key)
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return {
                "prompt": prompt,
                "model_response": resp.choices[0].message.content or "",
                "model": model_id,
            }
        except RateLimitError:
            time.sleep(min(2 ** attempt, 60))
        except APIError as exc:
            if attempt == 5:
                return {"prompt": prompt, "model_response": f"ERROR: {exc}", "model": model_id}
            time.sleep(5)
    return {"prompt": prompt, "model_response": "ERROR: max retries", "model": model_id}


# ---------------------------------------------------------------------------
# Parallel generation for one model × split
# ---------------------------------------------------------------------------

def generate_parallel(
    prompts: List[str],
    model_id: str,
    out_path: Path,
    openrouter_key: Optional[str],
    tinker_key: Optional[str],
    tinker_base_url: str = "https://api.thinkingmachines.ai/v1",
    workers: int = WORKERS,
) -> int:
    existing = _load_existing(out_path)
    todo = [p for p in prompts if p not in existing]
    done_count = len(existing)

    print(f"  {done_count}/{len(prompts)} already done  ({len(todo)} remaining)")
    if not todo:
        return done_count

    or_model = OPENROUTER_MODEL_MAP.get(model_id, model_id.lower())
    csv_writer = StreamingCSVWriter(out_path, existing)

    # Write already-existing rows so the file is self-consistent
    # (they're already in the file from a previous run)

    completed = done_count
    pbar = tqdm(total=len(prompts), initial=done_count, desc=model_id, unit="resp")

    def _task(prompt: str) -> dict:
        if openrouter_key:
            try:
                return _call_openrouter(prompt, or_model, model_id, openrouter_key)
            except Exception:
                pass
        if tinker_key:
            return _call_tinker(prompt, model_id, tinker_key, tinker_base_url)
        return {"prompt": prompt, "model_response": "ERROR: no API key", "model": model_id}

    actual_workers = min(workers, len(todo))
    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = {pool.submit(_task, p): p for p in todo}
        for fut in as_completed(futures):
            row = fut.result()
            csv_writer.write(row)
            completed += 1
            pbar.update(1)

    pbar.close()
    csv_writer.close()
    return completed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    tinker_key = os.environ.get("TINKER_API_KEY")

    if not openrouter_key and not tinker_key:
        sys.exit(
            "Set at least one of OPENROUTER_API_KEY or TINKER_API_KEY.\n"
            "  export OPENROUTER_API_KEY='sk-or-v1-...'\n"
            "  export TINKER_API_KEY='tml-...'"
        )

    # 1. Dataset
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    raw_parquet = DATASET_DIR / "oasst1_all.parquet"
    if raw_parquet.exists():
        print(f"Loading cached dataset → {raw_parquet}")
        df = pd.read_parquet(raw_parquet)
    else:
        df = download_oasst1()
        df.to_parquet(raw_parquet, index=False)
        print(f"Saved raw dataset ({len(df):,} rows) → {raw_parquet}")

    prompts_df = extract_prompts(df)

    train_csv = DATASET_DIR / f"oasst1_train_{TRAIN_N}_seed{SEED}.csv"
    test_csv = DATASET_DIR / f"oasst1_test_{TEST_N}_seed{SEED}.csv"

    if train_csv.exists() and test_csv.exists():
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        print(f"Loaded existing splits: {len(train_df)} train / {len(test_df)} test")
    else:
        train_df, test_df = split_prompts(prompts_df, TRAIN_N, TEST_N, SEED)
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        train_df[["prompt"]].to_csv(train_csv, index=False)
        test_df[["prompt"]].to_csv(test_csv, index=False)
        print(f"Saved splits → {train_csv.name}  /  {test_csv.name}")

    train_prompts = train_df["prompt"].astype(str).tolist()
    test_prompts = test_df["prompt"].astype(str).tolist()

    # 2. Generate responses – all 4 models × 2 splits
    for split, prompts, n, out_dir in [
        ("train", train_prompts, TRAIN_N, TRAIN_OUT_DIR),
        ("test", test_prompts, TEST_N, TEST_OUT_DIR),
    ]:
        out_dir.mkdir(parents=True, exist_ok=True)
        for model_id in MODELS:
            slug = model_id.replace("/", "_")
            out_path = out_dir / f"{split}_{slug}_{n}.csv"
            print(f"\n[{split.upper()}] {model_id}")
            print(f"  → {out_path}")
            generate_parallel(
                prompts, model_id, out_path, openrouter_key, tinker_key
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
