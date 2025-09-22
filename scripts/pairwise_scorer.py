#!/usr/bin/env python3
"""
Pairwise scoring for disguised vs target CSVs (per-prompt similarity):
- semantic_score, stylistic_score via LLM judge
- heuristic_match_score via structural heuristics

Usage:
  python scripts/pairwise_scorer.py --input <pairs.csv> --output <scored.csv> \
    --judge-model openai/gpt-4.1-mini [--heuristics-only]

Note: For single-file model scoring (no comparisons), use scripts/scorer.py.
"""
from __future__ import annotations

import argparse
import os
import pandas as pd
import csv as _csv
import logging
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        try:
            return pd.read_csv(path, on_bad_lines='skip', quoting=_csv.QUOTE_ALL)
        except pd.errors.ParserError:
            return pd.read_csv(path, on_bad_lines='skip', quoting=_csv.QUOTE_NONE, engine='python')


def _compute_heuristics(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    try:
        # Prefer full stylistic analysis util
        from scripts.methods.utils import stylistic_analysis as _style
    except Exception:
        try:
            import scripts.stylistic_analysis as _style  # type: ignore
        except Exception:
            _style = None

    def simple_feats(text: str) -> Dict[str, bool]:
        t = str(text or "")
        return {
            'markdown': ('#' in t) or ('```' in t) or ('* ' in t),
            'list': any(s in t for s in ['\n- ', '\n1. ', '\n* ']),
            'code': ('```' in t) or ('`' in t),
            'bullets': ('\n- ' in t) or ('\n* ' in t),
        }

    feats_a: List[Dict[str, bool]] = []
    feats_b: List[Dict[str, bool]] = []
    for _, row in df.iterrows():
        a = str(row[col_a] or "")
        b = str(row[col_b] or "")
        if _style is not None:
            feats_a.append({
                'markdown': _style.has_markdown(a),
                'list': _style.contains_list(a),
                'code': _style.contains_code(a),
                'bullets': _style.contains_bullets(a),
            })
            feats_b.append({
                'markdown': _style.has_markdown(b),
                'list': _style.contains_list(b),
                'code': _style.contains_code(b),
                'bullets': _style.contains_bullets(b),
            })
        else:
            feats_a.append(simple_feats(a))
            feats_b.append(simple_feats(b))
    A = pd.DataFrame(feats_a)
    B = pd.DataFrame(feats_b)
    match = (A == B).mean(axis=1)
    out = pd.DataFrame({'heuristic_match_score': match})
    # Attach individual flags for transparency
    for col in A.columns:
        out[f'heuristic_{col}'] = A[col]
    out['heuristic_heuristic_match_score'] = out['heuristic_match_score']
    return out


def _judge_backend() -> str:
    # We use LiteLLM by default; you can add vLLM/HF if needed.
    return 'litellm'


def _litellm_generate(messages: List[dict], model: str, temperature: float, max_tokens: int) -> str:
    try:
        from scripts.cached_llm import cached_completion
        resp = cached_completion(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content
    except Exception:
        from litellm import completion
        import litellm
        if not hasattr(litellm, 'cache') or litellm.cache is None:
            litellm.cache = litellm.Cache()
        resp = completion(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content


def _build_prompt(a: str, b: str, prompt: Optional[str] = None) -> str:
    system = """You are an expert evaluator. Rate similarity between two responses.

1) Semantic (1-4)
2) Stylistic (1-4)
Return both scores and a brief justification."""
    context = f"Original prompt: {prompt}\n\n" if prompt else ""
    return f"{system}\n\n{context}Response A:\n{a}\n\nResponse B:\n{b}\n\nProvide semantic and stylistic scores."


def _parse_scores(output: str) -> Dict[str, float]:
    import re
    sem = 0.0; sty = 0.0
    for pat in [r'[Ss]emantic[:\s]+(\d(?:\.\d)?)', r'(\d(?:\.\d)?)[^0-9]*semantic']:
        m = re.search(pat, output)
        if m: sem = float(m.group(1)); break
    for pat in [r'[Ss]tylistic[:\s]+(\d(?:\.\d)?)', r'(\d(?:\.\d)?)[^0-9]*stylistic']:
        m = re.search(pat, output)
        if m: sty = float(m.group(1)); break
    return {'semantic_score': sem, 'stylistic_score': sty}


def score_pairwise(df: pd.DataFrame, judge_model: str, heuristics_only: bool) -> pd.DataFrame:
    result = df.copy()
    # Heuristics
    heur = _compute_heuristics(result, 'target_response', 'model_response')
    for c in heur.columns:
        result[c] = heur[c]
    if heuristics_only:
        return result
    # LLM judge
    scores: List[Dict[str, float]] = []
    for _, row in result.iterrows():
        prompt_text = _build_prompt(str(row.get('target_response','')), str(row.get('model_response','')), row.get('prompt'))
        messages = [{"role":"user","content": prompt_text}]
        raw = _litellm_generate(messages, model=judge_model, temperature=0.0, max_tokens=512)
        sc = _parse_scores(raw)
        sc['raw_explanation'] = raw
        scores.append(sc)
    for k in ['semantic_score','stylistic_score','raw_explanation']:
        result[k] = [s.get(k) for s in scores]
    return result


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if 'semantic_score' in df.columns:
        s = df['semantic_score'].dropna()
        if len(s):
            out['semantic_score_mean'] = float(s.mean()); out['semantic_score_std'] = float(s.std()); out['semantic_score_count'] = int(s.shape[0])
    if 'stylistic_score' in df.columns:
        s = df['stylistic_score'].dropna()
        if len(s):
            out['stylistic_score_mean'] = float(s.mean()); out['stylistic_score_std'] = float(s.std()); out['stylistic_score_count'] = int(s.shape[0])
    if 'heuristic_match_score' in df.columns:
        s = df['heuristic_match_score'].dropna()
        if len(s):
            out['heuristic_match_mean'] = float(s.mean()); out['heuristic_match_std'] = float(s.std()); out['heuristic_match_count'] = int(s.shape[0])
    return out


def main():
    ap = argparse.ArgumentParser(description='Pairwise scorer (disguised vs target)')
    ap.add_argument('--input', required=True, help='CSV with columns: prompt, model_response, target_response')
    ap.add_argument('--output', required=True)
    ap.add_argument('--heuristics-only', action='store_true')
    ap.add_argument('--judge-model', default='openai/gpt-4.1-mini')
    args = ap.parse_args()

    df = _read_csv_robust(args.input)
    for c in ('model_response','target_response'):
        if c not in df.columns:
            raise ValueError('Input must contain columns: model_response and target_response')

    scored = score_pairwise(df, judge_model=args.judge_model, heuristics_only=args.heuristics_only)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")

    # Metrics
    metrics = summarize(scored)
    base, _ = os.path.splitext(args.output)
    mpath = f"{base}_metrics.csv"
    with open(mpath, 'w', newline='') as f:
        w = _csv.writer(f); w.writerow(['metric','value']);
        for k,v in metrics.items(): w.writerow([k,v])
    print(f"Metrics CSV: {mpath}")


if __name__ == '__main__':
    main()

