from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path('data/results/gsm8k/eval200')
EXCLUDE = {'plots', 'finetuned', 'self_comparisons', 'base'}
TARGET_MAP = {
    'openai/gpt-4.1-mini': Path('data/model-responses/gsm8k/splits/seed42/eval_200/openai_gpt-4.1-mini_responses_eval200_seed42.csv'),
    'meta-llama/Meta-Llama-3.1-8B-Instruct': Path('data/model-responses/gsm8k/splits/seed42/eval_200/meta-llama_Meta-Llama-3.1-8B-Instruct_responses_eval200_seed42.csv'),
}

for method_dir in sorted(ROOT.iterdir()):
    if not method_dir.is_dir() or method_dir.name in EXCLUDE:
        continue
    for csv_path in sorted(method_dir.glob('*.csv')):
        df = pd.read_csv(csv_path)
        if 'target_model' not in df.columns or 'prompt' not in df.columns:
            continue
        target_model = str(df['target_model'].iloc[0]).strip()
        mapping_path = TARGET_MAP.get(target_model)
        if not mapping_path or not mapping_path.exists():
            print(f"[skip] {csv_path}: no mapping for {target_model}")
            continue
        base = pd.read_csv(mapping_path)[['prompt', 'model_response']].rename(columns={'model_response': 'target_response'})
        base = base.drop_duplicates('prompt', keep='first')
        merged = df.drop(columns=['target_response'], errors='ignore').merge(base, on='prompt', how='left', validate='many_to_one')
        missing = merged['target_response'].isna().sum()
        if missing:
            print(f"[warn] {csv_path}: {missing} prompts missing in base file")
        merged.to_csv(csv_path, index=False)
        print(f"[fix] Overwrote {csv_path} with aligned target responses")
