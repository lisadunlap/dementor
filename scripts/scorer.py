"""Scoring utilities for model comparisons - pairwise only."""
from __future__ import annotations

import argparse
import csv
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
                'header': _style.contains_header(text_a),
                'code': _style.contains_code(text_a),
                'links': _style.contains_link(text_a),
                'greeting': _style.starts_with_greeting(text_a),
                'signoff': _style.ends_with_signoff(text_a),
                'emojis': _style.contains_emoji(text_a),
                'bullets': _style.contains_bullets(text_a),
                'questions': _style.contains_question(text_a),
                'parentheses': _style.uses_parentheses(text_a),
                'exclamations': _style.contains_exclamation(text_a),
                'long_sentences': _style.has_long_sentences(text_a),
                'starts_list': _style.starts_with_list(text_a),
                'math_symbols': _style.contains_math_symbols(text_a),
                'blockquotes': _style.contains_blockquote(text_a),
                'repetition': _style.contains_repetition(text_a),
                'numbered_steps': _style.contains_numbered_steps(text_a),
                'all_caps': _style.contains_all_caps(text_a),
                'first_person': _style.uses_first_person(text_a),
                'second_person': _style.uses_second_person(text_a),
                'third_person': _style.uses_third_person(text_a),
                'past_tense': _style.uses_past_tense(text_a),
                'present_tense': _style.uses_present_tense(text_a),
                'future_tense': _style.uses_future_tense(text_a),
                'conditional': _style.uses_conditional(text_a),
                'questions_rhetorical': _style.contains_rhetorical_question(text_a),
                'imperatives': _style.uses_imperative(text_a),
                'contractions': _style.uses_contractions(text_a),
                'formal_language': _style.uses_formal_language(text_a),
                'informal_language': _style.uses_informal_language(text_a),
                'technical_jargon': _style.uses_technical_jargon(text_a),
                'emotional_language': _style.uses_emotional_language(text_a),
                'transition_words': _style.uses_transition_words(text_a),
                'numbers_stats': _style.contains_numbers_stats(text_a),
                'numbers_currency': _style.contains_currency(text_a),
                'numbers_dates': _style.contains_dates(text_a),
                'punctuation_variety': _style.punctuation_variety(text_a),
                'sentence_complexity': _style.sentence_complexity(text_a),
                'avg_sentence_length': _style.avg_sentence_length(text_a),
                'readability_score': _style.readability_score(text_a),
            })
            feats_b.append({
                'markdown': _style.has_markdown(text_b),
                'list': _style.contains_list(text_b),
                'header': _style.contains_header(text_b),
                'code': _style.contains_code(text_b),
                'links': _style.contains_link(text_b),
                'greeting': _style.starts_with_greeting(text_b),
                'signoff': _style.ends_with_signoff(text_b),
                'emojis': _style.contains_emoji(text_b),
                'bullets': _style.contains_bullets(text_b),
                'questions': _style.contains_question(text_b),
                'parentheses': _style.uses_parentheses(text_b),
                'exclamations': _style.contains_exclamation(text_b),
                'long_sentences': _style.has_long_sentences(text_b),
                'starts_list': _style.starts_with_list(text_b),
                'math_symbols': _style.contains_math_symbols(text_b),
                'blockquotes': _style.contains_blockquote(text_b),
                'repetition': _style.contains_repetition(text_b),
                'numbered_steps': _style.contains_numbered_steps(text_b),
                'all_caps': _style.contains_all_caps(text_b),
                'first_person': _style.uses_first_person(text_b),
                'second_person': _style.uses_second_person(text_b),
                'third_person': _style.uses_third_person(text_b),
                'past_tense': _style.uses_past_tense(text_b),
                'present_tense': _style.uses_present_tense(text_b),
                'future_tense': _style.uses_future_tense(text_b),
                'conditional': _style.uses_conditional(text_b),
                'questions_rhetorical': _style.contains_rhetorical_question(text_b),
                'imperatives': _style.uses_imperative(text_b),
                'contractions': _style.uses_contractions(text_b),
                'formal_language': _style.uses_formal_language(text_b),
                'informal_language': _style.uses_informal_language(text_b),
                'technical_jargon': _style.uses_technical_jargon(text_b),
                'emotional_language': _style.uses_emotional_language(text_b),
                'transition_words': _style.uses_transition_words(text_b),
                'numbers_stats': _style.contains_numbers_stats(text_b),
                'numbers_currency': _style.contains_currency(text_b),
                'numbers_dates': _style.contains_dates(text_b),
                'punctuation_variety': _style.punctuation_variety(text_b),
                'sentence_complexity': _style.sentence_complexity(text_b),
                'avg_sentence_length': _style.avg_sentence_length(text_b),
                'readability_score': _style.readability_score(text_b),
            })
        else:
            feats_a.append(simple_features(text_a))
            feats_b.append(simple_features(text_b))

    # Compute match scores
    heuristic_scores = []
    for i in range(len(feats_a)):
        matches = sum(1 for key in feats_a[i] if feats_a[i][key] == feats_b[i][key])
        total_features = len(feats_a[i])
        match_score = matches / total_features if total_features > 0 else 0.0
        heuristic_scores.append({'heuristic_match_score': match_score})

    return pd.DataFrame(heuristic_scores)


def _build_pairwise_prompt(target_response: str, model_response: str, prompt: Optional[str] = None) -> str:
    system_message = (
        "You are evaluating how well a disguised model response mimics a target model's style and content. "
        "Compare the two responses and provide scores for semantic similarity and stylistic similarity.\n\n"
        "Rate each on a scale of 1-4 (decimals allowed):\n"
        "1. Semantic Score (1-4): How similar is the meaning and content?\n"
        "2. Stylistic Score (1-4): How similar is the writing style, tone, and format?\n\n"
        "Provide your response in the format:\n"
        "Semantic Score: X/4\n"
        "Stylistic Score: Y/4\n\n"
        "Then provide a brief explanation for your scores."
    )

    if prompt:
        return f"""{system_message}

Original Prompt: {prompt}

Target Response (the style to mimic):
{target_response}

Model Response (disguised):
{model_response}

Evaluate how well the model response mimics the target response."""
    else:
        return f"""{system_message}

Target Response (the style to mimic):
{target_response}

Model Response (disguised):
{model_response}

Evaluate how well the model response mimics the target response."""


def _parse_pairwise_scores(output: str) -> Dict[str, float]:
    scores = {
        'semantic_score': 0.0,
        'stylistic_score': 0.0,
    }

    # Parse semantic score
    semantic_patterns = [
        r'[Ss]emantic[_\s]*Score[:\s]+(\d+(?:\.\d+)?)(?:\s*/\s*4)?',
        r'(\d+(?:\.\d+)?)(?:\s*/\s*4)?[^0-9]*semantic',
    ]
    for pattern in semantic_patterns:
        match = re.search(pattern, output)
        if match:
            scores['semantic_score'] = float(match.group(1))
            break

    # Parse stylistic score
    stylistic_patterns = [
        r'[Ss]tylistic[_\s]*Score[:\s]+(\d+(?:\.\d+)?)(?:\s*/\s*4)?',
        r'(\d+(?:\.\d+)?)(?:\s*/\s*4)?[^0-9]*stylistic',
    ]
    for pattern in stylistic_patterns:
        match = re.search(pattern, output)
        if match:
            scores['stylistic_score'] = float(match.group(1))
            break

    for key in ['semantic_score', 'stylistic_score']:
        scores[key] = max(0.0, min(scores[key], 4.0))
    return scores


def score_pairwise_dataframe(
    df: pd.DataFrame,
    *,
    judge_model: str,
    include_heuristics: bool = False,
) -> pd.DataFrame:
    result = df.copy()
    if include_heuristics:
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
    include_heuristics: bool = False,
) -> pd.DataFrame:
    df = _read_csv_robust(input_file)
    for column in ('model_response', 'target_response'):
        if column not in df.columns:
            raise ValueError("Input must contain columns: model_response and target_response")

    scored = score_pairwise_dataframe(df, judge_model=judge_model, include_heuristics=include_heuristics)

    out_path = Path(output_path)
    out_dir = out_path.parent or Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_path
    scored.to_csv(scored_path, index=False)

    # Write summary metrics
    summary = summarize_scores(scored)
    summary_path = out_path.parent / f"{out_path.stem}_metrics.csv"
    summary_data = [{'metric': k, 'value': v} for k, v in summary.items()]
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)

    logging.info("Saved scored responses to %s", scored_path)
    logging.info("Saved metrics to %s", summary_path)

    return scored

def score_model_comparison(
    source_file: str,
    target_file: str,
    output_file: str,
    *,
    judge_model: str = "openai/gpt-4.1-mini",
    openai_api_base: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> pd.DataFrame:
    source_df = _read_csv_robust(source_file)
    target_df = _read_csv_robust(target_file)

    if 'prompt' not in source_df.columns or 'model_response' not in source_df.columns:
        raise ValueError("Source file must contain 'prompt' and 'model_response' columns")
    if 'prompt' not in target_df.columns or 'model_response' not in target_df.columns:
        raise ValueError("Target file must contain 'prompt' and 'model_response' columns")

    # Merge on prompt
    merged = pd.merge(source_df, target_df, on='prompt', suffixes=('_source', '_target'))

    # Rename columns for scoring
    merged = merged.rename(columns={
        'model_response_source': 'model_response',
        'model_response_target': 'target_response',
    })

    # Preserve original columns
    for col in merged.columns:
        if col.endswith('_source') or col.endswith('_target'):
            merged = merged.rename(columns={col: col.replace('_response_', '_response_')})

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    # Set up environment variables
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
        score_dir.mkdir(parents=True, exist_ok=True)
        score_path = score_dir / 'scored.csv'
        scored = score_pairwise(
            str(out_path),
            str(score_path),
            judge_model=judge_model,
            include_heuristics=False,  # Default to LLM judge only
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
    """Legacy function for merging old scoring files."""
    import glob

    pattern = os.path.join(score_dir, "pairwise_*.csv")
    files = glob.glob(pattern)

    if not files:
        logging.warning("No pairwise_*.csv files found in %s", score_dir)
        return

    dfs = []
    for file in files:
        df = _read_csv_robust(file)
        filename = os.path.basename(file)
        method = filename.replace("pairwise_", "").replace(".csv", "")
        df['method'] = method
        dfs.append(df)

    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(score_dir, "merged_scores.csv")
        merged.to_csv(output_path, index=False)
        logging.info("Merged %d files into %s", len(dfs), output_path)

        if clean_old:
            for file in files:
                os.remove(file)
            logging.info("Cleaned up original files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise scoring CLI - LLM judge compares disguised vs target responses")
    parser.add_argument('--input', help='CSV with prompt, model_response, target_response columns.')
    parser.add_argument('--output', required=True, help='Destination file path (directory inferred).')
    parser.add_argument('--judge-model', default='openai/gpt-4.1-mini', help='LLM judge model.')
    parser.add_argument('--heuristics', action='store_true', help='Include heuristic scoring (LLM judge only by default).')
    parser.add_argument('--compare', action='store_true', help='Compare mode: merge two model CSVs and score.')
    parser.add_argument('--a', help='Source model CSV for compare mode (prompt, model_response).')
    parser.add_argument('--b', help='Target model CSV for compare mode (prompt, model_response).')
    parser.add_argument('--openai-api-base', default=None, help='Override OPENAI_API_BASE for judge routing.')
    parser.add_argument('--openai-api-key', default=None, help='Override OPENAI_API_KEY for judge routing.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.compare:
        if not args.a or not args.b:
            parser.error("Compare mode requires --a and --b arguments")
        score_model_comparison(
            source_file=args.a,
            target_file=args.b,
            output_file=args.output,
            judge_model=args.judge_model,
            openai_api_base=args.openai_api_base,
            openai_api_key=args.openai_api_key,
        )
    else:
        if not args.input:
            parser.error("--input is required unless --compare is passed")
        score_pairwise(
            input_file=args.input,
            output_path=args.output,
            judge_model=args.judge_model,
            include_heuristics=args.heuristics,
        )


if __name__ == "__main__":
    main()
