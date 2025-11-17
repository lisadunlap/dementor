from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path('data/results/gsm8k/eval200')
PLOT_DIR = ROOT / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
EXCLUDE = {'plots', 'self_comparisons', 'finetuned', 'base'}
IGNORE_METHODS = {'just_name_it'}
BASE_ADDITIONS = {
    'meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini': (
        'Base GPT vs Llama',
        ROOT / 'base' / 'scores' / 'gpt4.1_vs_llama' / 'scored.csv',
    ),
    'openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct': (
        'Base Llama vs GPT',
        ROOT / 'base' / 'scores' / 'llama_vs_gpt' / 'scored.csv',
    ),
}

def gather(direction_keyword: str):
    entries = []
    for method_dir in sorted(ROOT.iterdir()):
        if method_dir.is_dir() and method_dir.name not in EXCLUDE and method_dir.name not in IGNORE_METHODS:
            for csv_path in method_dir.glob('*.csv'):
                if direction_keyword in csv_path.name:
                    score_dir = method_dir / 'scores' / csv_path.stem / 'scored.csv'
                    if score_dir.exists():
                        entries.append((method_dir.name, pd.read_csv(score_dir)))
    return entries

def build_entries(direction_keyword: str):
    entries = gather(direction_keyword)
    base_info = BASE_ADDITIONS.get(direction_keyword)
    if base_info:
        label, path = base_info
        if path.exists():
            entries.append((label, pd.read_csv(path)))
    return entries

def plot(entries, title: str, out_name: str):
    metrics = ['semantic_score', 'stylistic_score']
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for ax, metric in zip(axes, metrics):
        labels, means, cis = [], [], []
        for method, df in entries:
            series = df[metric].dropna()
            mean = series.mean() if not series.empty else 0.0
            std = series.std(ddof=1) if len(series) > 1 else 0.0
            ci = 1.96 * std / math.sqrt(len(series)) if len(series) > 1 else 0.0
            labels.append(method.replace('_', '\n'))
            means.append(mean)
            cis.append(ci)
        ax.barh(labels, means, xerr=cis, capsize=6)
        ax.set_xlim(0, 4)
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(axis='x', linestyle='--', alpha=0.4)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = PLOT_DIR / out_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved {out_path}")

plot(
    build_entries('meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini'),
    'Llama-3.1-8B → GPT-4.1-mini (eval-200)',
    'llama_to_gpt_ci.png',
)
plot(
    build_entries('openai_gpt-4.1-mini_as_meta-llama_Meta-Llama-3.1-8B-Instruct'),
    'GPT-4.1-mini → Llama-3.1-8B (eval-200)',
    'gpt_to_llama_ci.png',
)
