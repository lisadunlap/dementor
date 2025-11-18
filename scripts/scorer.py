"""Unified scoring utilities for single responses and pairwise comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    from .methods.utils import stylistic_analysis as _style  # type: ignore
except Exception:
    try:
        import scripts.methods.utils.stylistic_analysis as _style  # type: ignore
    except Exception:
        try:
            import scripts.stylistic_analysis as _style  # type: ignore
        except Exception:
            _style = None


def _read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        try:
            return pd.read_csv(path, on_bad_lines='skip', quoting=csv.QUOTE_ALL)
        except pd.errors.ParserError:
            return pd.read_csv(path, on_bad_lines='skip', quoting=csv.QUOTE_NONE, engine='python')


def _llm_generate(messages: List[dict], model: str, temperature: float, max_tokens: int) -> str:
    try:
        from scripts.cache_llm import cached_completion

        resp = cached_completion(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content
    except Exception:
        from litellm import completion
        import litellm

        if not hasattr(litellm, 'cache') or litellm.cache is None:
            litellm.cache = litellm.Cache()
        resp = completion(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content


def _build_quality_prompt(response: str, prompt: Optional[str] = None) -> str:
    system = (
        "You are an expert evaluator. Rate this response on a scale of 1-5 for the following criteria:\n\n"
        "1) Coherence (1-5): How logically consistent and well-structured is the response?\n"
        "2) Helpfulness (1-5): How useful and informative is the response for the given task?\n"
        "3) Accuracy (1-5): How factually correct and precise is the response?\n\n"
        "Return scores in the format: Coherence: X, Helpfulness: Y, Accuracy: Z\n"
        "Followed by a brief justification."
    )
    context = f"Original prompt: {prompt}\n\n" if prompt else ""
    return f"{system}\n\n{context}Response to evaluate:\n{response}\n\nProvide coherence, helpfulness, and accuracy scores."


def _parse_quality_scores(output: str) -> Dict[str, float]:
    scores = {
        'coherence_score': 0.0,
        'helpfulness_score': 0.0,
        'accuracy_score': 0.0,
        'quality_raw_explanation': output,
    }
    patterns = {
        'coherence_score': [r'[Cc]oherence[:\s]+(\d(?:\.\d)?)', r'(\d(?:\.\d)?)[^0-9]*coherence'],
        'helpfulness_score': [r'[Hh]elpfulness[:\s]+(\d(?:\.\d)?)', r'(\d(?:\.\d)?)[^0-9]*helpfulness'],
        'accuracy_score': [r'[Aa]ccuracy[:\s]+(\d(?:\.\d)?)', r'(\d(?:\.\d)?)[^0-9]*accuracy'],
    }
    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, output, re.IGNORECASE)
            if m:
                scores[key] = float(m.group(1))
                break
    return scores


def _compute_single_style_features(responses: List[str]) -> pd.DataFrame:
    feats: List[Dict[str, object]] = []
    for text in responses:
        value = str(text or "")
        row: Dict[str, object] = {'response_length': len(value)}
        if _style is not None:
            row.update({
                'markdown': _style.has_markdown(value),
                'list': _style.contains_list(value),
                'header': _style.contains_header(value),
                'code': _style.contains_code(value),
                'links': _style.contains_link(value),
                'greeting': _style.starts_with_greeting(value),
                'signoff': _style.ends_with_signoff(value),
                'emojis': _style.contains_emoji(value),
                'bullets': _style.contains_bullets(value),
                'questions': _style.contains_question(value),
                'parentheses': _style.uses_parentheses(value),
                'exclamations': _style.contains_exclamation(value),
                'long_sentences': _style.has_long_sentences(value),
                'starts_list': _style.starts_with_list(value),
                'math_symbols': _style.contains_math_symbols(value),
                'blockquotes': _style.contains_blockquote(value),
                'repetition': _style.contains_repetition(value),
                'numbered_steps': _style.contains_numbered_steps(value),
                'all_caps': _style.contains_all_caps(value),
                'first_person': _style.uses_first_person(value),
            })
        feats.append(row)
    return pd.DataFrame(feats)


def score_model_single(
    input_file: str,
    output_file: Optional[str] = None,
    *,
    llm_quality: bool = False,
    judge_model: str = "openai/gpt-4.1-mini",
) -> pd.DataFrame:
    df = _read_csv_robust(input_file)
    if 'model_response' not in df.columns:
        raise ValueError("Input CSV must contain a 'model_response' column")

    features = _compute_single_style_features(df['model_response'].astype(str).tolist())
    result_df = pd.concat([df.reset_index(drop=True), features], axis=1)

    if llm_quality:
        logging.info("Evaluating %s responses with LLM quality assessment", len(result_df))
        result_df = _evaluate_quality_llm(result_df, judge_model=judge_model)

    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        result_df.to_csv(output_file, index=False)
        _write_single_metrics(result_df, output_file)

    return result_df


def _evaluate_quality_llm(df: pd.DataFrame, *, judge_model: str, response_col: str = 'model_response') -> pd.DataFrame:
    result = df.copy()
    scored_rows: List[Dict[str, object]] = []
    for _, row in result.iterrows():
        response_text = str(row.get(response_col, ''))
        prompt_text = str(row.get('prompt', ''))
        if not response_text.strip():
            scored_rows.append({
                'coherence_score': 0.0,
                'helpfulness_score': 0.0,
                'accuracy_score': 0.0,
                'quality_raw_explanation': 'Empty response',
            })
            continue

        eval_prompt = _build_quality_prompt(response_text, prompt_text)
        messages = [{"role": "user", "content": eval_prompt}]
        try:
            raw = _llm_generate(messages, model=judge_model, temperature=0.0, max_tokens=512)
            scored_rows.append(_parse_quality_scores(raw))
        except Exception as exc:  # pragma: no cover - network errors handled upstream
            scored_rows.append({
                'coherence_score': 0.0,
                'helpfulness_score': 0.0,
                'accuracy_score': 0.0,
                'quality_raw_explanation': f'Error: {exc}',
            })

    for key in ['coherence_score', 'helpfulness_score', 'accuracy_score', 'quality_raw_explanation']:
        result[key] = [row.get(key, 0.0 if key != 'quality_raw_explanation' else '') for row in scored_rows]
    return result


def _write_single_metrics(df: pd.DataFrame, output_file: str) -> None:
    metrics: Dict[str, float] = {}
    for column in df.columns:
        series = df[column].dropna()
        if column == 'response_length' and not series.empty:
            metrics[f'{column}_mean'] = float(series.mean())
            metrics[f'{column}_std'] = float(series.std())
        elif column in {'coherence_score', 'helpfulness_score', 'accuracy_score'} and not series.empty:
            metrics[f'{column}_mean'] = float(series.mean())
            metrics[f'{column}_std'] = float(series.std())
            metrics[f'{column}_count'] = int(series.shape[0])
        elif series.isin([True, False]).all():
            metrics[f'{column}_rate'] = float(series.mean())

    metrics_path = Path(output_file).with_suffix('').as_posix() + '_metrics.csv'
    try:
        with open(metrics_path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['metric', 'value'])
            for metric, value in metrics.items():
                writer.writerow([metric, value])
        logging.info("Metrics saved to %s", metrics_path)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        logging.warning("Could not write metrics CSV: %s", exc)


def _compute_pairwise_heuristics(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    def simple_features(text: str) -> Dict[str, bool]:
        value = str(text or "")
        return {
            'markdown': ('#' in value) or ('```' in value) or ('* ' in value),
            'list': any(s in value for s in ['\n- ', '\n1. ', '\n* ']),
            'code': ('```' in value) or ('`' in value),
            'bullets': ('\n- ' in value) or ('\n* ' in value),
        }

    feats_a: List[Dict[str, bool]] = []
    feats_b: List[Dict[str, bool]] = []
    for _, row in df.iterrows():
        text_a = str(row.get(col_a, '') or '')
        text_b = str(row.get(col_b, '') or '')
        if _style is not None:
            feats_a.append({
                'markdown': _style.has_markdown(text_a),
                'list': _style.contains_list(text_a),
                'code': _style.contains_code(text_a),
                'bullets': _style.contains_bullets(text_a),
            })
            feats_b.append({
                'markdown': _style.has_markdown(text_b),
                'list': _style.contains_list(text_b),
                'code': _style.contains_code(text_b),
                'bullets': _style.contains_bullets(text_b),
            })
        else:
            feats_a.append(simple_features(text_a))
            feats_b.append(simple_features(text_b))

    df_a = pd.DataFrame(feats_a)
    df_b = pd.DataFrame(feats_b)
    match = (df_a == df_b).mean(axis=1)
    heuristics = pd.DataFrame({'heuristic_match_score': match})
    for column in df_a.columns:
        heuristics[f'heuristic_{column}'] = df_a[column]
    heuristics['heuristic_heuristic_match_score'] = heuristics['heuristic_match_score']
    return heuristics


def _build_pairwise_prompt(target_response: str, model_response: str, prompt: Optional[str] = None) -> str:
    _ = prompt  # prompt is intentionally ignored; similarity score should focus on responses only
    system = (
        "You are an expert evaluator comparing two responses that were generated for the same prompt.\n"
        "Treat Response A as the reference answer and judge how faithfully Response B reproduces it.\n"
        "Provide scores between 1.0 and 4.0, and you may use decimals (e.g., 3.5) when appropriate.\n\n"
        "Semantic fidelity (meaning/content)\n"
        "  • Focus on whether Response B preserves Response A’s claims, quantitative results, assumptions, and step-by-step reasoning.\n"
        "  • Paraphrasing or re-ordering is fine if all obligations, constraints, and final answers remain equivalent.\n"
        "  • Deduct when Response B changes numbers, omits necessary steps, adds new unsupported claims, contradicts conclusions, refuses, or answers a different task.\n"
        "  4 — All key facts and commitments match Response A exactly.\n"
        "  3 — Same solution with only minor wording compression or harmless omissions.\n"
        "  2 — Partial alignment; important details are missing or altered.\n"
        "  1 — Meaningfully different answer, contradiction, refusal, or off-topic response.\n\n"
        "Stylistic similarity (tone/presentation)\n"
        "  • Judge tone, persona, formality, length, structure, formatting (lists vs. prose, equations, bulleting), and use of disclaimers or emojis.\n"
        "  • Ignore factual accuracy here—style is about how the content is delivered.\n"
        "  • Penalize when Response B shifts to a refusal style, rewrites as dialogue, changes from terse to verbose (or vice versa), or otherwise feels like a different persona.\n"
        "  4 — Tone and formatting mirror Response A.\n"
        "  3 — Mostly similar style with small deviations.\n"
        "  2 — Noticeable stylistic drift that changes presentation.\n"
        "  1 — Major tone/format mismatch or refusal.\n\n"
        "Edge cases: If Response A is empty, irrelevant, or incorrect, still grade how closely Response B follows it. Penalize hallucinated new tasks in Response B.\n"
        "Output format: `Semantic: <int>, Stylistic: <int>`.\n"
        "After the scores, include one or two concise sentences citing evidence for each axis."
    )
    return (
        f"{system}\n\nResponse A:\n{target_response}\n\n"
        f"Response B:\n{model_response}\n"
    )


def _parse_pairwise_scores(output: str) -> Dict[str, float]:
    semantic = 0.0
    stylistic = 0.0
    for pattern in [
        r'[Ss]emantic(?:\s+similarity)?[:\s]+(\d+(?:\.\d)?)',
        r'(\d+(?:\.\d)?)[^0-9]*semantic',
    ]:
        match = re.search(pattern, output)
        if match:
            semantic = float(match.group(1))
            break
    for pattern in [
        r'[Ss]tylistic(?:\s+similarity)?[:\s]+(\d+(?:\.\d)?)',
        r'(\d+(?:\.\d)?)[^0-9]*stylistic',
    ]:
        match = re.search(pattern, output)
        if match:
            stylistic = float(match.group(1))
            break
    semantic = max(0.0, min(semantic, 4.0))
    stylistic = max(0.0, min(stylistic, 4.0))
    return {
        'semantic_score': semantic,
        'stylistic_score': stylistic,
    }


def score_pairwise_dataframe(
    df: pd.DataFrame,
    *,
    judge_model: str,
) -> pd.DataFrame:
    result = df.copy()
    heuristics = _compute_pairwise_heuristics(result, 'target_response', 'model_response')
    for column in heuristics.columns:
        result[column] = heuristics[column]

    scores: List[Dict[str, object]] = []
    for _, row in result.iterrows():
        prompt_text = _build_pairwise_prompt(
            str(row.get('target_response', '')),
            str(row.get('model_response', '')),
            row.get('prompt'),
        )
        messages = [{"role": "user", "content": prompt_text}]
        raw = _llm_generate(messages, model=judge_model, temperature=0.0, max_tokens=512)
        parsed = _parse_pairwise_scores(raw)
        parsed['raw_explanation'] = raw
        scores.append(parsed)

    for key in ['semantic_score', 'stylistic_score', 'raw_explanation']:
        result[key] = [row.get(key) for row in scores]
    return result


def summarize_scores(df: pd.DataFrame) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for column in ['semantic_score', 'stylistic_score', 'heuristic_match_score']:
        if column in df.columns:
            series = df[column].dropna()
            if not series.empty:
                summary[f'{column}_mean'] = float(series.mean())
                summary[f'{column}_std'] = float(series.std())
                summary[f'{column}_count'] = int(series.shape[0])
    return summary


def score_pairwise(
    input_file: str,
    output_path: str,
    *,
    judge_model: str = "openai/gpt-4.1-mini",
) -> pd.DataFrame:
    df = _read_csv_robust(input_file)
    for column in ('model_response', 'target_response'):
        if column not in df.columns:
            raise ValueError("Input must contain columns: model_response and target_response")

    scored = score_pairwise_dataframe(df, judge_model=judge_model)

    out_path = Path(output_path)
    out_dir = out_path.parent or Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_path
    scored.to_csv(scored_path, index=False)

    metrics = summarize_scores(scored)
    metrics_path = out_dir / 'scored_metrics.csv'
    with open(metrics_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'value'])
        for metric, value in metrics.items():
            writer.writerow([metric, value])

    summary = {
        'input_file': input_file,
        'num_samples': len(scored),
        'judge_model': judge_model,
        'metrics': metrics,
    }
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as handle:
        json.dump(summary, handle, indent=2)

    logging.info("Wrote pairwise scoring artifacts to %s", out_dir)
    return scored


def score_model_comparison(
    source_file: str,
    target_file: str,
    output_file: Optional[str] = None,
    *,
    judge_model: str = "openai/gpt-4.1-mini",
    openai_api_base: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> pd.DataFrame:
    df_a = _read_csv_robust(source_file)
    df_b = _read_csv_robust(target_file)
    for column in ('prompt', 'model_response'):
        if column not in df_a.columns or column not in df_b.columns:
            raise ValueError("Both input CSVs must contain 'prompt' and 'model_response' columns")

    merged = pd.merge(
        df_a[['prompt', 'model_response']],
        df_b[['prompt', 'model_response']],
        on='prompt',
        suffixes=('_a', '_b'),
    )
    merged = merged.rename(columns={'model_response_a': 'model_response', 'model_response_b': 'target_response'})

    if output_file is None:
        base = Path(source_file).with_suffix('').name
        target = Path(target_file).with_suffix('').name
        output_file = str(Path(source_file).parent / f"{base}_vs_{target}.csv")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    original_base = os.environ.get('OPENAI_API_BASE')
    original_key = os.environ.get('OPENAI_API_KEY')
    try:
        if openai_api_base is not None:
            os.environ['OPENAI_API_BASE'] = openai_api_base
        elif judge_model.lower().startswith('openai/'):
            os.environ['OPENAI_API_BASE'] = 'https://api.openai.com/v1'
        if openai_api_key is not None:
            os.environ['OPENAI_API_KEY'] = openai_api_key

        score_dir = out_path.parent / 'scores' / out_path.stem
        score_path = score_dir / 'scored.csv'
        scored = score_pairwise(
            str(out_path),
            str(score_path),
            judge_model=judge_model,
        )
    finally:
        if original_base is not None:
            os.environ['OPENAI_API_BASE'] = original_base
        elif 'OPENAI_API_BASE' in os.environ:
            del os.environ['OPENAI_API_BASE']
        if original_key is not None:
            os.environ['OPENAI_API_KEY'] = original_key
        elif openai_api_key is not None and 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']

    return scored


def merge_scoring_files(score_dir: str, clean_old: bool = False) -> None:
    root = Path(score_dir)
    full_scored = root / 'pairwise_full_scored.csv'
    heuristic_scored = root / 'pairwise_heuristic_scored.csv'
    full_metrics = root / 'pairwise_full_scored_metrics.csv'
    heuristic_metrics = root / 'pairwise_heuristic_scored_metrics.csv'

    output_scored = root / 'scored.csv'
    output_metrics = root / 'scored_metrics.csv'
    output_summary = root / 'summary.json'

    if full_scored.exists():
        df = pd.read_csv(full_scored)
    elif heuristic_scored.exists():
        df = pd.read_csv(heuristic_scored)
    else:
        logging.info("No pairwise_* files found under %s", root)
        return

    df.to_csv(output_scored, index=False)

    combined_metrics: Dict[str, float] = {}
    for metrics_path in [full_metrics, heuristic_metrics]:
        if metrics_path.exists():
            with open(metrics_path, 'r') as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2 and row[0] not in combined_metrics:
                        combined_metrics[row[0]] = float(row[1])

    if combined_metrics:
        with open(output_metrics, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['metric', 'value'])
            for metric, value in combined_metrics.items():
                writer.writerow([metric, value])

    summary = {
        'merged_from': [p.name for p in [full_scored, heuristic_scored] if p.exists()],
        'num_samples': len(df),
        'metrics': combined_metrics,
    }
    with open(output_summary, 'w') as handle:
        json.dump(summary, handle, indent=2)

    if clean_old:
        for path in [full_scored, heuristic_scored, full_metrics, heuristic_metrics]:
            if path.exists():
                path.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified scoring CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    single = subparsers.add_parser('single', help='Score a single-file response CSV.')
    single.add_argument('input', help="CSV input with 'model_response' column.")
    single.add_argument('--output', help='Output CSV path.')
    single.add_argument('--judge-model', default='openai/gpt-4.1-mini', help='LLM judge for optional quality scoring.')
    single.add_argument('--llm-quality', action='store_true', help='Include LLM-based quality evaluation.')

    pairwise = subparsers.add_parser('pairwise', help='Score disguised vs target responses.')
    pairwise.add_argument('--input', required=True, help='CSV with prompt, model_response, target_response columns.')
    pairwise.add_argument('--output', required=True, help='Destination file path (directory inferred).')
    pairwise.add_argument('--judge-model', default='openai/gpt-4.1-mini', help='LLM judge model.')

    compare = subparsers.add_parser('compare', help='Merge two response files and score pairwise.')
    compare.add_argument('--a', required=True, help='Source model CSV (prompt, model_response).')
    compare.add_argument('--b', required=True, help='Target model CSV (prompt, model_response).')
    compare.add_argument('--output', required=True, help='Merged comparison CSV path.')
    compare.add_argument('--judge-model', default='openai/gpt-4.1-mini', help='LLM judge model.')
    compare.add_argument('--openai-api-base', default=None, help='Override OPENAI_API_BASE for judge routing.')
    compare.add_argument('--openai-api-key', default=None, help='Override OPENAI_API_KEY for judge routing.')

    merge = subparsers.add_parser('merge', help='Merge legacy pairwise_* files into standard outputs.')
    merge.add_argument('score_dir', help='Directory containing legacy pairwise files.')
    merge.add_argument('--clean-old', action='store_true', help='Remove legacy files after merging.')

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == 'single':
        output_path = args.output
        if output_path is None:
            base, ext = os.path.splitext(args.input)
            output_path = f"{base}_single_scored{ext}"
        score_model_single(
            input_file=args.input,
            output_file=output_path,
            llm_quality=args.llm_quality,
            judge_model=args.judge_model,
        )
    elif args.command == 'pairwise':
        score_pairwise(
            input_file=args.input,
            output_path=args.output,
            judge_model=args.judge_model,
        )
    elif args.command == 'compare':
        score_model_comparison(
            source_file=args.a,
            target_file=args.b,
            output_file=args.output,
            judge_model=args.judge_model,
            openai_api_base=args.openai_api_base,
            openai_api_key=args.openai_api_key,
        )
    elif args.command == 'merge':
        merge_scoring_files(args.score_dir, clean_old=args.clean_old)
    else:  # pragma: no cover
        parser.error('Unknown command')


if __name__ == '__main__':
    main()
