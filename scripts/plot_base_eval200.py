from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path('data/results/gsm8k/eval200')
PLOT_DIR = ROOT / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

cases = [
    ('gpt-4.1-mini run1 vs run2', ROOT / 'self_comparisons' / 'scores' / 'gpt-4.1-mini_run1_vs_run2_eval200' / 'scored.csv'),
    ('llama-3.1-8B run1 vs run2', ROOT / 'self_comparisons' / 'scores' / 'llama-3.1-8b_run1_vs_run2_eval200' / 'scored.csv'),
]

metrics = ['semantic_score', 'stylistic_score']
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, metric in zip(axes, metrics):
    labels, means, cis = [], [], []
    for label, scored_csv in cases:
        df = pd.read_csv(scored_csv)
        series = df[metric].dropna()
        mean = series.mean() if not series.empty else 0.0
        std = series.std(ddof=1) if len(series) > 1 else 0.0
        ci = 1.96 * std / math.sqrt(len(series)) if len(series) > 1 else 0.0
        labels.append(label)
        means.append(mean)
        cis.append(ci)
    ax.barh(labels, means, xerr=cis, capsize=6, color=['#4c72b0', '#dd8452'])
    ax.set_xlim(0, 4)
    ax.set_title(metric.replace('_', ' ').title())
    ax.grid(axis='x', linestyle='--', alpha=0.4)
fig.suptitle('Self Baselines (eval-200, 95% CI)')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOT_DIR / 'base_l1_vs_l2_ci.png', dpi=200)
plt.close(fig)
print("Saved data/results/gsm8k/eval200/plots/base_l1_vs_l2_ci.png")
