#!/usr/bin/env python3

import os
import re
import time
import json
import requests
import pandas as pd

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8010/v1/chat/completions")
MODEL = os.environ.get("VLLM_MODEL", "microsoft/Phi-4-mini-instruct")
TEMPERATURE = 0.0

BASE = "disguising/scores/hierarchical_math_disguise"
MODEL_DIRS = [
    ("source_meta-llama/Llama-3-8B-Instruct/target_gpt/gpt-5",),
    ("source_microsoft/Phi-4-mini-instruct/target_gpt/gpt-5",),
]

SYSTEM_PROMPT = (
    "You are an expert evaluator of AI model responses. Rate similarity between two responses "
    "in two dimensions strictly as integers 1-4: (1) Semantic Meaning (2) Stylistic Similarity. "
    "At the end, output ONLY this block exactly:\n"
    "Breakdown:\n- Semantic Meaning: X/4\n- Stylistic Similarity: Y/4\n"
)

USER_TEMPLATE = (
    "Here are the two responses to compare:\nResponse 1:\n{resp1}\n\nResponse 2:\n{resp2}\n\n"
    "Think through your response and end with the exact Breakdown block."
)

RE_SCORE = re.compile(
    r"Semantic Meaning:\s*([0-9]+(?:\.[0-9]+)?)\s*\/\s*([0-9]+)[\s\S]*?"
    r"Stylistic Similarity:\s*([0-9]+(?:\.[0-9]+)?)\s*\/\s*([0-9]+)",
    re.I,
)

def _normalize_score(score: float, denom: int) -> int:
    # Normalize to 1-4 scale. If model returns 5/5, scale to 4-scale.
    if denom and denom != 4:
        scaled = round(float(score) * 4.0 / float(denom))
    else:
        scaled = float(score)
    if scaled < 1:
        scaled = 1
    if scaled > 4:
        scaled = 4
    return int(scaled)

def judge_pair(resp1: str, resp2: str) -> tuple[int, int]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(resp1=resp1, resp2=resp2)},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": 512,
    }
    r = requests.post(VLLM_URL, json=payload, timeout=120)
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    m = RE_SCORE.search(out)
    if not m:
        raise ValueError(f"Could not parse scores from output: {out[:200]}")
    sem_raw, sem_den, sty_raw, sty_den = m.groups()
    sem = _normalize_score(float(sem_raw), int(sem_den))
    sty = _normalize_score(float(sty_raw), int(sty_den))
    return sem, sty

def compute_baseline_for_dir(rel_dir: str):
    dir_path = os.path.join(BASE, rel_dir)
    comp_path = os.path.join(dir_path, "comparison_results.csv")
    out_path = os.path.join(dir_path, "comparison_results_original.csv")
    if not os.path.exists(comp_path):
        print(f"Missing {comp_path}; skipping")
        return
    df = pd.read_csv(comp_path)
    # Resume support: if output exists, skip already processed prompts
    processed = set()
    if os.path.exists(out_path):
        try:
            done_df = pd.read_csv(out_path)
            processed = set(done_df.get("prompt", pd.Series()).dropna().astype(str).tolist())
        except Exception:
            processed = set()
    rows = []
    for idx, row in df.iterrows():
        target_resp = str(row.get("target_response", "") or "")
        source_resp = str(row.get("source_response", "") or "")
        prompt = str(row.get("prompt", "") or "")
        if prompt in processed:
            continue
        if not target_resp or not source_resp:
            sem, sty = None, None
        else:
            # target vs source (original baseline)
            for attempt in range(3):
                try:
                    sem, sty = judge_pair(target_resp, source_resp)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(1.0)
        rows.append({
            "prompt": prompt,
            "target_model": row.get("target_model", ""),
            "target_response": target_resp,
            "target_response_token_length": row.get("target_response_token_length", None),
            "source_model": row.get("source_model", ""),
            "source_response": source_resp,
            "source_response_token_length": row.get("source_response_token_length", None),
            "model": row.get("model", ""),
            "method": row.get("method", ""),
            "semantic_score": sem,
            "stylistic_score": sty,
        })
        if (idx + 1) % 25 == 0:
            print(f"Scored {idx+1}/{len(df)} for {rel_dir}")
    out_df = pd.DataFrame(rows)
    if os.path.exists(out_path) and not out_df.empty:
        try:
            prev = pd.read_csv(out_path)
            out_df = pd.concat([prev, out_df], ignore_index=True)
            out_df = out_df.drop_duplicates(subset=["prompt"], keep="last")
        except Exception:
            pass
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

def main():
    only_rel = os.environ.get("ONLY_REL", "").strip()
    for (rel,) in MODEL_DIRS:
        if only_rel and only_rel not in rel:
            continue
        compute_baseline_for_dir(rel)

if __name__ == "__main__":
    main()


