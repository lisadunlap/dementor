#!/usr/bin/env python3
"""
Generate stylometric ensemble visualization plots for bidirectional disguise evaluation.
Creates two plots showing results for both GPT->Llama and Llama->GPT disguises.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats


def calculate_stats(values):
    """Calculate mean and 95% confidence interval."""
    mean = np.mean(values)
    n = len(values)
    stderr = stats.sem(values)
    ci = stderr * stats.t.ppf((1 + 0.95) / 2., n-1)
    return mean, ci


def load_and_analyze_scores(base_dir, methods, directions):
    """Load scores for all methods and both directions."""
    results = []
    
    for method in methods:
        for direction in directions:
            if direction == 'gpt_as_llama':
                score_file = Path(base_dir) / method / "scores" / "stylometric_ensemble_gpt_as_llama.csv"
                label = f"GPT -> LLAMA ({method.replace('_', ' ').title()})"
            else:  # llama_as_gpt
                score_file = Path(base_dir) / method / "scores" / "stylometric_ensemble_llama_as_gpt.csv"
                label = f"LLAMA -> GPT ({method.replace('_', ' ').title()})"
            
            if not score_file.exists():
                print(f"Warning: {score_file} not found, skipping")
                continue
            
            print(f"Loading {method} - {direction}...")
            df = pd.read_csv(score_file)
            
            # Extract probabilities
            prob_gpt = df['ensemble_prob_openai/gpt-4.1-mini'].values
            prob_llama = df['ensemble_prob_meta-llama/Meta-Llama-3.1-8B-Instruct'].values
            
            # Calculate unanimous agreement
            unanimous = (df['ensemble_agreement_count'] == 3).astype(int).values
            
            # Calculate statistics
            gpt_mean, gpt_ci = calculate_stats(prob_gpt)
            llama_mean, llama_ci = calculate_stats(prob_llama)
            unanimous_mean, unanimous_ci = calculate_stats(unanimous)
            
            results.append({
                'method': method,
                'direction': direction,
                'label': label,
                'gpt_mean': gpt_mean,
                'gpt_ci': gpt_ci,
                'llama_mean': llama_mean,
                'llama_ci': llama_ci,
                'unanimous_mean': unanimous_mean,
                'unanimous_ci': unanimous_ci,
                'n': len(df)
            })
    
    return pd.DataFrame(results)


def create_probability_plot(stats_df, output_path, title="MMLU-Pro - Stylometric Ensemble Avg Probabilities"):
    """Create the average probability plot with error bars for both directions."""
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by direction first (GPT->Llama on top), then by Llama probability descending
    stats_df['dir_order'] = stats_df['direction'].map({'gpt_as_llama': 0, 'llama_as_gpt': 1})
    stats_df = stats_df.sort_values(['dir_order', 'llama_mean'], ascending=[True, True])
    
    y_positions = np.arange(len(stats_df))
    bar_height = 0.35
    
    # Plot GPT probabilities (blue bars)
    ax.barh(y_positions - bar_height/2, stats_df['gpt_mean'], 
            bar_height, xerr=stats_df['gpt_ci'],
            label='GPT-4.1-mini', color='#5DA5DA', 
            error_kw={'linewidth': 2, 'ecolor': 'black', 'capsize': 4})
    
    # Plot Llama probabilities (red bars)
    ax.barh(y_positions + bar_height/2, stats_df['llama_mean'], 
            bar_height, xerr=stats_df['llama_ci'],
            label='Llama-3.1-8B', color='#F15854',
            error_kw={'linewidth': 2, 'ecolor': 'black', 'capsize': 4})
    
    # Add vertical line at 0.5
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Customize
    ax.set_yticks(y_positions)
    ax.set_yticklabels(stats_df['label'])
    ax.set_xlabel('Average probability (mean ± 95% CI)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    # Add n values as text on the right
    for i, (idx, row) in enumerate(stats_df.iterrows()):
        ax.text(1.02, y_positions[i], 
                f"GPT {row['gpt_mean']:.3f} (n={row['n']})\nLlama {row['llama_mean']:.3f} (n={row['n']})",
                va='center', fontsize=8, transform=ax.get_yaxis_transform())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved probability plot to {output_path}")
    plt.close()


def create_unanimous_plot(stats_df, output_path, title="MMLU-Pro - Stylometric Ensemble Unanimous Agreement Rate"):
    """Create the unanimous agreement rate plot for both directions."""
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by direction first, then by unanimous rate descending
    stats_df['dir_order'] = stats_df['direction'].map({'gpt_as_llama': 0, 'llama_as_gpt': 1})
    stats_df = stats_df.sort_values(['dir_order', 'unanimous_mean'], ascending=[True, True])
    
    y_positions = np.arange(len(stats_df))
    
    # Plot unanimous agreement rates (green bars)
    ax.barh(y_positions, stats_df['unanimous_mean'], 
            xerr=stats_df['unanimous_ci'],
            color='#60BD68', 
            error_kw={'linewidth': 2, 'ecolor': 'black', 'capsize': 4})
    
    # Add vertical line at 0.5
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Customize
    ax.set_yticks(y_positions)
    ax.set_yticklabels(stats_df['label'])
    ax.set_xlabel('Unanimous rate (mean ± 95% CI)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    
    # Add values as text on the right
    for i, (idx, row) in enumerate(stats_df.iterrows()):
        ax.text(1.02, y_positions[i], 
                f"{row['unanimous_mean']:.3f} (n={row['n']})",
                va='center', fontsize=9, transform=ax.get_yaxis_transform())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved unanimous agreement plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate stylometric ensemble plots for both directions')
    parser.add_argument('--base-dir', default='data/results/generic/eval200',
                        help='Base directory containing method subdirectories')
    parser.add_argument('--methods', nargs='+', 
                        default=['contrastive', 'behavioral_based', 'stylistic', 'random_sampling'],
                        help='Methods to include in plots')
    parser.add_argument('--directions', nargs='+',
                        default=['gpt_as_llama', 'llama_as_gpt'],
                        help='Directions to include')
    parser.add_argument('--output-dir', default='data/results/plots',
                        help='Output directory for plots')
    parser.add_argument('--title-prefix', default='MMLU-Pro',
                        help='Prefix for plot titles')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and analyze scores
    print("Loading and analyzing scores...")
    stats_df = load_and_analyze_scores(args.base_dir, args.methods, args.directions)
    
    if stats_df.empty:
        print("Error: No data loaded. Check that score files exist.")
        return
    
    print("\n=== Summary Statistics ===")
    print(stats_df[['label', 'gpt_mean', 'llama_mean', 'unanimous_mean', 'n']].to_string(index=False))
    print()
    
    # Create plots
    prob_plot_path = output_dir / 'stylometric_ensemble_probabilities_bidirectional.png'
    unanimous_plot_path = output_dir / 'stylometric_ensemble_unanimous_bidirectional.png'
    
    create_probability_plot(
        stats_df, 
        prob_plot_path, 
        title=f"{args.title_prefix} - Stylometric Ensemble Avg Probabilities"
    )
    
    create_unanimous_plot(
        stats_df, 
        unanimous_plot_path,
        title=f"{args.title_prefix} - Stylometric Ensemble Unanimous Agreement Rate"
    )
    
    print("\n=== Done! ===")
    print(f"Plots saved to {output_dir}/")


if __name__ == '__main__':
    main()

