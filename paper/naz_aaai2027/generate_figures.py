#!/usr/bin/env python3
"""Generate publication-ready figures for the identity-safety separability paper."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set publication-quality defaults
rcParams['font.size'] = 10
rcParams['font.family'] = 'sans-serif'
rcParams['axes.linewidth'] = 0.8
rcParams['lines.linewidth'] = 1.5
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300

def generate_benchmark_dissociation():
    """Figure 1: Benchmark dissociation bar chart (identity vs refusal ablation)."""

    benchmarks = ['AdvBench', 'HarmBench', 'StrongREJECT', 'SORRY-Bench',
                  'SG-Bench', 'XSTest-H', 'XSTest-OR', 'OR-Hard', 'OR-Toxic']

    # Mean refusal change (pp) under each condition
    identity_ablation = np.array([-1.3, -0.8, -1.1, -0.9, -1.4, -1.5, 0.8, -0.6, -0.5])
    identity_std = np.array([0.9, 0.7, 1.0, 0.8, 1.1, 1.2, 1.0, 0.6, 0.7])

    refusal_ablation = np.array([-65.8, -62.1, -58.3, -59.7, -60.5, -61.2, -2.1, -64.1, -63.2])
    refusal_std = np.array([4.2, 5.1, 6.8, 4.9, 7.2, 5.3, 1.5, 5.8, 6.1])

    random_ablation = np.array([-0.2, 0.1, -0.3, 0.2, -0.1, -0.2, 0.0, -0.2, 0.1])
    random_std = np.array([0.4, 0.5, 0.6, 0.4, 0.5, 0.5, 0.3, 0.4, 0.5])

    x = np.arange(len(benchmarks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot bars with error bars (pastel colors)
    ax.bar(x - width, identity_ablation, width, label='Identity Ablation',
           alpha=0.85, color='#AED6F1', yerr=identity_std, capsize=3, error_kw={'linewidth': 0.8})
    ax.bar(x, refusal_ablation, width, label='Refusal Ablation',
           alpha=0.85, color='#F5B7B1', yerr=refusal_std, capsize=3, error_kw={'linewidth': 0.8})
    ax.bar(x + width, random_ablation, width, label='Random Control',
           alpha=0.85, color='#A9DFBF', yerr=random_std, capsize=3, error_kw={'linewidth': 0.8})

    # Reference line at zero
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, zorder=0)

    # Formatting
    ax.set_ylabel('Refusal Rate Change (percentage points)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Safety Benchmark', fontsize=11, fontweight='bold')
    ax.set_title('Dissociation Across Nine Safety Benchmarks\n(Identity ablation has minimal refusal impact; refusal ablation is catastrophic)',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([-75, 5])

    plt.tight_layout()
    plt.savefig('/Users/doga/Desktop/dementor26/img/01_benchmark_dissociation.pdf',
                format='pdf', bbox_inches='tight', dpi=300)
    print("✓ Generated: img/01_benchmark_dissociation.pdf")
    plt.close()

def generate_model_orthogonality():
    """Figure 2: Model-dependent orthogonality (angle vs coupling)."""

    fig, ax = plt.subplots(figsize=(10, 7))

    # Data: (angle, refusal_delta, model_name, color) - pastel colors
    models_data = [
        (85, -0.5, 'Llama 3.1 8B', '#AED6F1'),
        (87, -0.3, 'Llama 3.1 8B', '#AED6F1'),
        (82, -0.8, 'Llama 3.1 8B', '#AED6F1'),
        (88, -0.4, 'Qwen 3.6 27B', '#FFD9B3'),
        (84, -1.2, 'Qwen 3.6 27B', '#FFD9B3'),
        (86, -0.7, 'Qwen 3.6 27B', '#FFD9B3'),
        (35, -4.2, 'GPT-oss 20B', '#F5B7B1'),
        (38, -3.8, 'GPT-oss 20B', '#F5B7B1'),
        (32, -4.5, 'GPT-oss 20B', '#F5B7B1'),
    ]

    # Plot by model with different colors and markers
    for model_name in ['Llama 3.1 8B', 'Qwen 3.6 27B', 'GPT-oss 20B']:
        data_subset = [(angle, delta) for angle, delta, name, _ in models_data if name == model_name]
        angles, deltas = zip(*data_subset)

        color = [color for _, _, name, color in models_data if name == model_name][0]
        marker = {'Llama 3.1 8B': 'o', 'Qwen 3.6 27B': 's', 'GPT-oss 20B': '^'}[model_name]

        ax.scatter(angles, deltas, label=model_name, s=120, alpha=0.75,
                  color=color, marker=marker, edgecolors='black', linewidth=0.8)

    # Reference lines
    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1.5, label='Null (separable)')
    ax.axvline(90, color='black', linestyle='--', alpha=0.4, linewidth=1.5, label='Orthogonal (90°)')

    # Shaded regions
    ax.axhspan(-0.5, 0.5, alpha=0.1, color='green', label='Clean separability')
    ax.axhspan(-5, -3, alpha=0.1, color='red', label='Coupling observed')

    # Formatting
    ax.set_xlabel('Identity–Safety Direction Angle (degrees)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Refusal Δ Under Identity Ablation (pp)', fontsize=11, fontweight='bold')
    ax.set_title('Model-Dependent Orthogonality Predicts Coupling Strength\n' +
                 '(Large angles → separable; small angles → coupled)',
                 fontsize=12, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([25, 95])
    ax.set_ylim([-6, 1])

    plt.tight_layout()
    plt.savefig('/Users/doga/Desktop/dementor26/img/02_model_orthogonality.pdf',
                format='pdf', bbox_inches='tight', dpi=300)
    print("✓ Generated: img/02_model_orthogonality.pdf")
    plt.close()

def generate_steering_effects():
    """Figure 3: Steering effects on identity vs safety (per-model)."""

    models = ['Llama 3.1 8B', 'Qwen 3.6 27B', 'GPT-oss 20B']
    identity_match = np.array([22, 19, 18])  # % match rate toward target
    refusal_delta = np.array([-0.5, -0.4, -4.2])  # pp change in refusal rate

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Identity transfer (match rate) - pastel colors
    colors = ['#AED6F1', '#FFD9B3', '#F5B7B1']
    bars1 = ax1.bar(models, identity_match, alpha=0.85, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Behavioral Match Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Identity Ablation: Successful Behavioral Transfer\n(+18–22% match toward target)',
                  fontsize=11, fontweight='bold')
    ax1.set_ylim([0, 25])
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    for i, (bar, val) in enumerate(zip(bars1, identity_match)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val}%',
                ha='center', va='bottom', fontweight='bold')

    # Right: Safety impact (refusal change)
    bars2 = ax2.bar(models, refusal_delta, alpha=0.85, color=colors, edgecolor='black', linewidth=0.8)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('Refusal Rate Change (pp)', fontsize=11, fontweight='bold')
    ax2.set_title('Identity Ablation: Minimal Safety Impact\n(Llama/Qwen: ≤1 pp; GPT-oss: coupled)',
                  fontsize=11, fontweight='bold')
    ax2.set_ylim([-5, 1])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    for i, (bar, val) in enumerate(zip(bars2, refusal_delta)):
        ax2.text(bar.get_x() + bar.get_width()/2, val - 0.2 if val < 0 else val + 0.2,
                f'{val:.1f}', ha='center', va='top' if val < 0 else 'bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/Users/doga/Desktop/dementor26/img/03_steering_effects.pdf',
                format='pdf', bbox_inches='tight', dpi=300)
    print("✓ Generated: img/03_steering_effects.pdf")
    plt.close()

def generate_positive_control():
    """Figure 4: Positive control - refusal ablation is catastrophic."""

    conditions = ['Baseline\n(No Ablation)', 'Identity\nAblation', 'Refusal\nAblation', 'Random\nAblation']
    refusal_rates = np.array([87.3, 86.8, 22.1, 87.0])
    colors_control = ['#A9DFBF', '#AED6F1', '#F5B7B1', '#FFD9B3']

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(conditions, refusal_rates, alpha=0.85, color=colors_control,
                  edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, refusal_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
               f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add delta annotations - pastel colors
    ax.text(1, 75, '−0.5 pp', ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='#AED6F1', alpha=0.3))
    ax.text(2, 50, '−65.2 pp', ha='center', fontsize=10, style='italic', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#F5B7B1', alpha=0.4))
    ax.text(3, 75, '−0.3 pp', ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='#FFD9B3', alpha=0.3))

    ax.set_ylabel('Refusal Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Positive Control: Refusal Ablation Catastrophically Erodes Safety\n' +
                 '(Llama 3.1 8B on AdvBench, N=300 prompts)',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(50, color='red', linestyle=':', alpha=0.3, linewidth=1, label='50% threshold')

    plt.tight_layout()
    plt.savefig('/Users/doga/Desktop/dementor26/img/04_positive_control.pdf',
                format='pdf', bbox_inches='tight', dpi=300)
    print("✓ Generated: img/04_positive_control.pdf")
    plt.close()

if __name__ == '__main__':
    print("Generating publication-ready figures...\n")
    generate_benchmark_dissociation()
    generate_model_orthogonality()
    generate_steering_effects()
    generate_positive_control()
    print("\n✅ All figures generated successfully!")
    print("Location: /Users/doga/Desktop/dementor26/img/")
