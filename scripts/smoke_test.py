#!/usr/bin/env python3
"""
Offline smoke test that does not require API keys.

Checks:
- Instantiates stylistic and random_sampling methods
- Calls .forward() to ensure message formatting
- Creates a tiny dummy results CSV and runs scorer in heuristics-only mode
"""
import pandas as pd
from pathlib import Path
from disguising.methods.get_method import get_method
from scripts.scorer import score_model_comparison, summarize_scores


def main():
    # Build tiny target dataset (no external calls required)
    target_df = pd.DataFrame({
        'prompt': [
            'Explain quicksort in one paragraph.',
            'List three benefits of unit testing.',
            'Write a short welcome message.'
        ],
        'target_response': [
            'Quicksort is a divide-and-conquer algorithm using pivots...\n- Steps:\n1) Choose pivot...\n2) Partition...\n3) Recurse...',
            '- Prevents regressions\n- Documents behavior\n- Enables refactoring',
            'Welcome! Glad you are here. Let me know how I can help.'
        ]
    })

    # Stylistic method
    stylistic = get_method('stylistic', model='dummy', disguise_as='target', disguise_df=target_df)
    msgs = stylistic.forward('Provide a short system overview.')
    assert isinstance(msgs, list) and len(msgs) >= 1

    # Random sampling method
    rnd = get_method('random_sampling', model='dummy', disguise_as='target', disguise_df=target_df)
    msgs2 = rnd.forward('Provide a short system overview.')
    assert isinstance(msgs2, list) and len(msgs2) >= 1

    # Create dummy results CSV
    out_dir = Path('results/validation')
    out_dir.mkdir(parents=True, exist_ok=True)
    dummy = pd.DataFrame({
        'prompt': ['A', 'B'],
        'model_response': ['Hello world!', 'Here are three points:\n- One\n- Two\n- Three'],
        'target_response': ['Hello world!!', 'Three points:\n- One\n- Two\n- Three']
    })
    csv_path = out_dir / 'dummy_results.csv'
    dummy.to_csv(csv_path, index=False)

    # Score heuristics-only
    df = score_model_comparison(str(csv_path), str(csv_path).replace('.csv', '_scored.csv'), heuristics_only=True)
    summary = summarize_scores(df)
    print('Smoke test summary:', summary)


if __name__ == '__main__':
    main()
