"""
Generate model responses for GSM8K, Chatbot Arena, and WritingPrompts
using OpenRouter API with 256 parallel workers and real-time CSV writes.

Usage:
  export OPENROUTER_API_KEY='sk-or-v1-...'
  python scripts/generate_dataset_responses.py
  python scripts/generate_dataset_responses.py --datasets gsm8k chatbot_arena
  python scripts/generate_dataset_responses.py --splits eval   # only eval splits
"""
from __future__ import annotations

import argparse
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
DATA_DIR  = REPO_ROOT / "data"

WORKERS  = 256
MAX_TOKENS = 512
TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Always respond in 1-2 well-structured paragraphs. "
    "Be thorough but concise — do not use bullet points or headers, write in flowing prose."
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TINKER_BASE_URL     = "https://api.thinkingmachines.ai/v1"

# OpenRouter model IDs for non-Nemotron models
MODELS: Dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct":           "meta-llama/llama-3.1-8b-instruct",
    "Qwen/Qwen3.6-27B":                            "qwen/qwen3-30b-a3b",
    "openai/gpt-oss-20b":                          "openai/gpt-oss-20b",
}

# Nemotron goes through Tinker (rate-limited on OpenRouter free tier)
TINKER_MODELS: Dict[str, str] = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16":  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
}

ALL_MODELS = {**MODELS, **TINKER_MODELS}

# Reasoning models — disable chain-of-thought to save tokens
REASONING_MODELS: set = set()

DATASETS = {
    "gsm8k": {
        "eval":  DATA_DIR / "datasets/gsm8k/gsm8k_prompts_eval_1000_seed42.csv",
        "train": DATA_DIR / "datasets/gsm8k/gsm8k_prompts_train_500_seed42.csv",
    },
    "chatbot_arena": {
        "eval":  DATA_DIR / "datasets/chatbot_arena/chatbot_arena_prompts_eval_1000_seed42.csv",
        "train": DATA_DIR / "datasets/chatbot_arena/chatbot_arena_prompts_train_500_seed42.csv",
    },
    "writingprompts": {
        "eval":  DATA_DIR / "datasets/writingprompts/writingprompts_eval_500_seed42.csv",
        "train": DATA_DIR / "datasets/writingprompts/writingprompts_train_500_seed42.csv",
    },
}


# ---------------------------------------------------------------------------
# Thread-safe streaming CSV writer
# ---------------------------------------------------------------------------

class StreamingCSVWriter:
    FIELDS = ["prompt", "model_response", "model"]

    def __init__(self, path: Path, existing: set):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        write_header = not path.exists() or path.stat().st_size == 0
        self._handle = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDS, quoting=csv.QUOTE_ALL)
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
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, quoting=csv.QUOTE_ALL, on_bad_lines="skip")
        if "prompt" in df.columns:
            return set(df["prompt"].astype(str).tolist())
    except Exception:
        pass
    return set()


# ---------------------------------------------------------------------------
# Single-prompt API call
# ---------------------------------------------------------------------------

def _call(prompt: str, or_model: str, model_id: str, api_key: str,
          base_url: str = OPENROUTER_BASE_URL) -> dict:
    from openai import OpenAI, APIError, RateLimitError

    headers = {} if base_url == TINKER_BASE_URL else {
        "HTTP-Referer": "https://github.com/dementor26", "X-Title": "dementor26"
    }
    client = OpenAI(base_url=base_url, api_key=api_key, default_headers=headers)
    extra = {}

    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=or_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                extra_body=extra if extra else None,
            )
            if not resp.choices:
                if attempt == 5:
                    return {"prompt": prompt, "model_response": "ERROR: empty choices", "model": model_id}
                time.sleep(3)
                continue
            return {"prompt": prompt, "model_response": resp.choices[0].message.content or "", "model": model_id}
        except RateLimitError:
            time.sleep(min(2 ** attempt, 60))
        except APIError as exc:
            if attempt == 5:
                return {"prompt": prompt, "model_response": f"ERROR: {exc}", "model": model_id}
            time.sleep(5)
    return {"prompt": prompt, "model_response": "ERROR: max retries", "model": model_id}


# ---------------------------------------------------------------------------
# Generate one (dataset, split, model) job
# ---------------------------------------------------------------------------

def generate(
    prompts: List[str],
    model_id: str,
    or_model: str,
    out_path: Path,
    api_key: str,
    base_url: str = OPENROUTER_BASE_URL,
) -> int:
    existing = _load_existing(out_path)
    todo = [p for p in prompts if p not in existing]
    if not todo:
        tqdm.write(f"  [skip] {out_path.name} — all {len(prompts)} already done")
        return len(prompts)

    tqdm.write(f"  [gen]  {out_path.name} — {len(todo)} remaining")
    writer = StreamingCSVWriter(out_path, existing)
    pbar = tqdm(total=len(prompts), initial=len(existing), desc=out_path.stem[:55], unit="resp", leave=False)

    with ThreadPoolExecutor(max_workers=min(WORKERS, len(todo))) as pool:
        futures = {pool.submit(_call, p, or_model, model_id, api_key, base_url): p for p in todo}
        for fut in as_completed(futures):
            row = fut.result()
            writer.write(row)
            pbar.update(1)

    pbar.close()
    writer.close()
    return len(prompts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--splits",   nargs="+", default=["eval", "train"], choices=["eval", "train"])
    p.add_argument("--models",   nargs="+", default=list(MODELS))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    or_key = os.environ.get("OPENROUTER_API_KEY")
    tinker_key = os.environ.get("TINKER_API_KEY")
    if not or_key and not tinker_key:
        sys.exit("Set OPENROUTER_API_KEY or TINKER_API_KEY.")

    models_to_run = args.models if args.models != list(MODELS) else list(ALL_MODELS)

    jobs = []
    for ds in args.datasets:
        for split in args.splits:
            prompt_csv = DATASETS[ds][split]
            if not prompt_csv.exists():
                print(f"  [missing] {prompt_csv}")
                continue
            prompts_df = pd.read_csv(prompt_csv)
            prompt_col = "prompt" if "prompt" in prompts_df.columns else prompts_df.columns[0]
            prompts = prompts_df[prompt_col].astype(str).tolist()
            n = len(prompts)

            for model_id in models_to_run:
                if model_id in TINKER_MODELS:
                    if not tinker_key: continue
                    api_key, base_url, or_model = tinker_key, TINKER_BASE_URL, TINKER_MODELS[model_id]
                elif model_id in MODELS:
                    if not or_key: continue
                    api_key, base_url, or_model = or_key, OPENROUTER_BASE_URL, MODELS[model_id]
                else:
                    print(f"  [unknown model] {model_id}"); continue
                slug = model_id.replace("/", "_")
                out_dir = DATA_DIR / "model-responses" / ds / split
                out_path = out_dir / f"{split}_{slug}_{n}.csv"
                jobs.append((prompts, model_id, or_model, out_path, api_key, base_url))

    total = sum(len(j[0]) for j in jobs)
    print(f"\n{len(jobs)} jobs  ·  ~{total:,} total responses  ·  workers={WORKERS}\n")

    for prompts, model_id, or_model, out_path, key, base_url in jobs:
        print(f"\n[{model_id}] {out_path.parent.name}/{out_path.name}")
        generate(prompts, model_id, or_model, out_path, key, base_url=base_url)

    print("\nAll done.")


if __name__ == "__main__":
    main()
