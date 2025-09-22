"""
Single-file scoring: stylistic features + simple aggregates for a model CSV.

Pairwise scoring has moved to scripts/pairwise_scorer.py.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Try to reuse stylistic feature helpers if available
try:
    from .methods.utils import stylistic_analysis as _style
except Exception:
    try:
        import scripts.methods.utils.stylistic_analysis as _style  # type: ignore
    except Exception:
        try:
            import scripts.stylistic_analysis as _style  # type: ignore
        except Exception:
            _style = None


def _compute_single_style_features(responses: List[str]) -> pd.DataFrame:
    feats: List[Dict[str, object]] = []
    for r in responses:
        r = str(r or "")
        row: Dict[str, object] = {'response_length': len(r)}
        if _style is not None:
            row.update({
                'markdown': _style.has_markdown(r),
                'list': _style.contains_list(r),
                'header': _style.contains_header(r),
                'code': _style.contains_code(r),
                'links': _style.contains_link(r),
                'greeting': _style.starts_with_greeting(r),
                'signoff': _style.ends_with_signoff(r),
                'emojis': _style.contains_emoji(r),
                'bullets': _style.contains_bullets(r),
                'questions': _style.contains_question(r),
                'parentheses': _style.uses_parentheses(r),
                'exclamations': _style.contains_exclamation(r),
                'long_sentences': _style.has_long_sentences(r),
                'starts_list': _style.starts_with_list(r),
                'math_symbols': _style.contains_math_symbols(r),
                'blockquotes': _style.contains_blockquote(r),
                'repetition': _style.contains_repetition(r),
                'numbered_steps': _style.contains_numbered_steps(r),
                'all_caps': _style.contains_all_caps(r),
                'first_person': _style.uses_first_person(r),
            })
        feats.append(row)
    return pd.DataFrame(feats)


def score_model_single(input_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
    def _read_csv_robust(path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except pd.errors.ParserError:
            try:
                return pd.read_csv(path, on_bad_lines='skip', quoting=csv.QUOTE_ALL)
            except pd.errors.ParserError:
                return pd.read_csv(path, on_bad_lines='skip', quoting=csv.QUOTE_NONE, engine='python')

    df = _read_csv_robust(input_file)
    if 'model_response' not in df.columns:
        raise ValueError("Input CSV must contain a 'model_response' column")

    features = _compute_single_style_features(df['model_response'].astype(str).tolist())
    result_df = pd.concat([df.reset_index(drop=True), features], axis=1)

    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        result_df.to_csv(output_file, index=False)

        # Aggregate simple metrics
        metrics: Dict[str, float] = {}
        for col in result_df.columns:
            if col == 'response_length':
                metrics[f'{col}_mean'] = float(result_df[col].dropna().mean())
                metrics[f'{col}_std'] = float(result_df[col].dropna().std())
            elif result_df[col].dropna().isin([True, False]).all():
                metrics[f'{col}_rate'] = float(result_df[col].mean())
        try:
            base, _ = os.path.splitext(output_file)
            metrics_csv = f"{base}_metrics.csv"
            with open(metrics_csv, 'w', newline='') as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(['metric', 'value'])
                for k, v in metrics.items():
                    writer.writerow([k, v])
            logging.info(f"Metrics saved to {metrics_csv}")
        except Exception as e:
            logging.warning(f"Could not write metrics.csv: {e}")

    return result_df


def _cli():
    parser = argparse.ArgumentParser(description="Single-file scoring (pairwise moved to scripts/pairwise_scorer.py)")
    parser.add_argument("input", help="CSV input with 'model_response' (no pairwise).")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--single", action="store_true", help="No-op; single-file mode only.")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_single_scored{ext}"

    df = score_model_single(
        input_file=args.input,
        output_file=args.output,
    )
    print(f"Scored {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    _cli()

