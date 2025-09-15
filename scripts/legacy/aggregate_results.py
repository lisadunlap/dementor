#!/usr/bin/env python3

import pandas as pd
import os
import numpy as np
from pathlib import Path

def load_comparison_results(model_path):
    """Load comparison results for a model"""
    comparison_file = os.path.join(model_path, "comparison_results.csv")
    if os.path.exists(comparison_file):
        df = pd.read_csv(comparison_file)
        return df
    return None

def load_heuristic_table(model_path):
    """Load heuristic table for a model"""
    heuristic_file = os.path.join(model_path, "heuristic_table.csv")
    if os.path.exists(heuristic_file):
        df = pd.read_csv(heuristic_file)
        return df
    return None

def compute_statistics(df, model_name):
    """Compute comprehensive statistics for a model"""
    if df is None or len(df) == 0:
        return {}
    
    stats = {
        'model': model_name,
        'total_responses': len(df),
        'semantic_score_mean': df['semantic_score'].mean(),
        'semantic_score_std': df['semantic_score'].std(),
        'semantic_score_min': df['semantic_score'].min(),
        'semantic_score_max': df['semantic_score'].max(),
        'stylistic_score_mean': df['stylistic_score'].mean(),
        'stylistic_score_std': df['stylistic_score'].std(),
        'stylistic_score_min': df['stylistic_score'].min(),
        'stylistic_score_max': df['stylistic_score'].max(),
    }
    
    # Add percentage of high scores
    stats['semantic_high_score_pct'] = (df['semantic_score'] >= 3).mean() * 100
    stats['stylistic_high_score_pct'] = (df['stylistic_score'] >= 3).mean() * 100
    
    return stats

def main():
    # Define paths
    base_path = "disguising/scores/hierarchical_math_disguise"
    
    # Meta Llama results (standardized structure)
    meta_llama_path = os.path.join(base_path, "source_meta-llama/Llama-32-8B-Instrct/target_gpt/gpt-5")
    
    # Microsoft Phi-4 results (standardized structure)
    microsoft_phi_path = os.path.join(base_path, "source_microsoft/Phi-4-mini-instruct/target_gpt/gpt-5")
    
    print("=== HIERARCHICAL MATH DISGUISE - COMPREHENSIVE RESULTS AGGREGATION ===\n")
    
    # Load data
    print("Loading Meta Llama results...")
    meta_llama_comparison = load_comparison_results(meta_llama_path)
    meta_llama_heuristics = load_heuristic_table(meta_llama_path)
    
    print("Loading Microsoft Phi-4 results...")
    microsoft_phi_comparison = load_comparison_results(microsoft_phi_path)
    microsoft_phi_heuristics = load_heuristic_table(microsoft_phi_path)
    
    # Compute statistics
    print("\nComputing statistics...")
    meta_llama_stats = compute_statistics(meta_llama_comparison, "Meta-Llama-3-8B-Instruct")
    microsoft_phi_stats = compute_statistics(microsoft_phi_comparison, "Microsoft-Phi-4-mini-instruct")
    
    # Create comparison table
    print("\n=== DETAILED COMPARISON TABLE ===")
    comparison_data = [meta_llama_stats, microsoft_phi_stats]
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format the table nicely
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(comparison_df.to_string(index=False, float_format='%.3f'))
    
    # Create summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Meta Llama - Total Responses: {meta_llama_stats.get('total_responses', 'N/A')}")
    print(f"Microsoft Phi-4 - Total Responses: {microsoft_phi_stats.get('total_responses', 'N/A')}")
    
    print(f"\n--- SEMANTIC SIMILARITY ---")
    print(f"Meta Llama: {meta_llama_stats.get('semantic_score_mean', 0):.3f} ± {meta_llama_stats.get('semantic_score_std', 0):.3f}")
    print(f"Microsoft Phi-4: {microsoft_phi_stats.get('semantic_score_mean', 0):.3f} ± {microsoft_phi_stats.get('semantic_score_std', 0):.3f}")
    
    print(f"\n--- STYLISTIC SIMILARITY ---")
    print(f"Meta Llama: {meta_llama_stats.get('stylistic_score_mean', 0):.3f} ± {meta_llama_stats.get('stylistic_score_std', 0):.3f}")
    print(f"Microsoft Phi-4: {microsoft_phi_stats.get('stylistic_score_mean', 0):.3f} ± {microsoft_phi_stats.get('stylistic_score_std', 0):.3f}")
    
    print(f"\n--- HIGH SCORE PERCENTAGES ---")
    print(f"Meta Llama - Semantic (≥3): {meta_llama_stats.get('semantic_high_score_pct', 0):.1f}%")
    print(f"Microsoft Phi-4 - Semantic (≥3): {microsoft_phi_stats.get('semantic_high_score_pct', 0):.1f}%")
    print(f"Meta Llama - Stylistic (≥3): {meta_llama_stats.get('stylistic_high_score_pct', 0):.1f}%")
    print(f"Microsoft Phi-4 - Stylistic (≥3): {microsoft_phi_stats.get('stylistic_high_score_pct', 0):.1f}%")
    
    # Create combined results table
    print("\n=== COMBINED RESULTS TABLE ===")
    combined_data = []
    
    if meta_llama_comparison is not None:
        meta_llama_comparison['model'] = 'Meta-Llama-3-8B-Instruct'
        combined_data.append(meta_llama_comparison)
    
    if microsoft_phi_comparison is not None:
        microsoft_phi_comparison['model'] = 'Microsoft-Phi-4-mini-instruct'
        combined_data.append(microsoft_phi_comparison)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Group by model and compute averages
        summary_df = combined_df.groupby('model').agg({
            'semantic_score': ['mean', 'std', 'min', 'max'],
            'stylistic_score': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        print(summary_df)
        
        # Save combined results
        output_file = os.path.join(base_path, "combined_hierarchical_math_results.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"\nCombined results saved to: {output_file}")
    
    # Create a simple comparison summary
    print("\n=== QUICK COMPARISON SUMMARY ===")
    if meta_llama_stats and microsoft_phi_stats:
        print(f"Semantic Similarity:")
        print(f"  Meta Llama:    {meta_llama_stats['semantic_score_mean']:.3f}")
        print(f"  Microsoft Phi-4: {microsoft_phi_stats['semantic_score_mean']:.3f}")
        print(f"  Difference:    {microsoft_phi_stats['semantic_score_mean'] - meta_llama_stats['semantic_score_mean']:+.3f}")
        
        print(f"\nStylistic Similarity:")
        print(f"  Meta Llama:    {meta_llama_stats['stylistic_score_mean']:.3f}")
        print(f"  Microsoft Phi-4: {microsoft_phi_stats['stylistic_score_mean']:.3f}")
        print(f"  Difference:    {microsoft_phi_stats['stylistic_score_mean'] - meta_llama_stats['stylistic_score_mean']:+.3f}")
    
    print("\n=== AGGREGATION COMPLETE ===")

if __name__ == "__main__":
    main()
