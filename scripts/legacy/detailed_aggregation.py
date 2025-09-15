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

def create_comprehensive_report():
    """Create a comprehensive report comparing both models"""
    
    # Define paths
    base_path = "disguising/scores/hierarchical_math_disguise"
    meta_llama_path = os.path.join(base_path, "source_meta-llama/Llama-32-8B-Instrct/target_gpt/gpt-5")
    microsoft_phi_path = os.path.join(base_path, "source_microsoft/Phi-4-mini-instruct/target_gpt/gpt-5")
    
    print("=" * 80)
    print("HIERARCHICAL MATH DISGUISE - COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 80)
    
    # Load all data
    print("\n📊 Loading data...")
    meta_llama_comparison = load_comparison_results(meta_llama_path)
    meta_llama_heuristics = load_heuristic_table(meta_llama_path)
    microsoft_phi_comparison = load_comparison_results(microsoft_phi_path)
    microsoft_phi_heuristics = load_heuristic_table(microsoft_phi_path)
    
    print(f"✅ Meta Llama: {len(meta_llama_comparison) if meta_llama_comparison is not None else 0} responses")
    print(f"✅ Microsoft Phi-4: {len(microsoft_phi_comparison) if microsoft_phi_comparison is not None else 0} responses")
    
    # 1. LLM-Based Scoring Comparison
    print("\n" + "=" * 50)
    print("1. LLM-BASED SCORING COMPARISON")
    print("=" * 50)
    
    if meta_llama_comparison is not None and microsoft_phi_comparison is not None:
        # Semantic scores
        print(f"\n🎯 SEMANTIC SIMILARITY (1-4 scale):")
        print(f"   Meta Llama:     {meta_llama_comparison['semantic_score'].mean():.3f} ± {meta_llama_comparison['semantic_score'].std():.3f}")
        print(f"   Microsoft Phi-4: {microsoft_phi_comparison['semantic_score'].mean():.3f} ± {microsoft_phi_comparison['semantic_score'].std():.3f}")
        print(f"   Difference:     {microsoft_phi_comparison['semantic_score'].mean() - meta_llama_comparison['semantic_score'].mean():+.3f}")
        
        # Stylistic scores
        print(f"\n🎨 STYLISTIC SIMILARITY (1-4 scale):")
        print(f"   Meta Llama:     {meta_llama_comparison['stylistic_score'].mean():.3f} ± {meta_llama_comparison['stylistic_score'].std():.3f}")
        print(f"   Microsoft Phi-4: {microsoft_phi_comparison['stylistic_score'].mean():.3f} ± {microsoft_phi_comparison['stylistic_score'].std():.3f}")
        print(f"   Difference:     {microsoft_phi_comparison['stylistic_score'].mean() - meta_llama_comparison['stylistic_score'].mean():+.3f}")
        
        # High score percentages
        print(f"\n📈 HIGH SCORE PERCENTAGES (≥3/4):")
        meta_semantic_high = (meta_llama_comparison['semantic_score'] >= 3).mean() * 100
        meta_stylistic_high = (meta_llama_comparison['stylistic_score'] >= 3).mean() * 100
        phi_semantic_high = (microsoft_phi_comparison['semantic_score'] >= 3).mean() * 100
        phi_stylistic_high = (microsoft_phi_comparison['stylistic_score'] >= 3).mean() * 100
        
        print(f"   Semantic (≥3):")
        print(f"     Meta Llama:     {meta_semantic_high:.1f}%")
        print(f"     Microsoft Phi-4: {phi_semantic_high:.1f}%")
        print(f"   Stylistic (≥3):")
        print(f"     Meta Llama:     {meta_stylistic_high:.1f}%")
        print(f"     Microsoft Phi-4: {phi_stylistic_high:.1f}%")
    
    # 2. Heuristic Analysis Comparison
    print("\n" + "=" * 50)
    print("2. HEURISTIC STYLE ANALYSIS")
    print("=" * 50)
    
    if meta_llama_heuristics is not None and microsoft_phi_heuristics is not None:
        print(f"\n📋 STYLE SIMILARITY HEURISTICS:")
        
        # Display heuristic tables side by side
        print(f"\nMeta Llama Heuristics:")
        print(meta_llama_heuristics.to_string(index=False))
        
        print(f"\nMicrosoft Phi-4 Heuristics:")
        print(microsoft_phi_heuristics.to_string(index=False))
        
        # Compare specific metrics
        if 'disguised_vs_target' in meta_llama_heuristics.columns and 'disguised_vs_target' in microsoft_phi_heuristics.columns:
            print(f"\n🔍 KEY METRICS COMPARISON:")
            print(f"   Disguised vs Target Style Match:")
            print(f"     Meta Llama:     {meta_llama_heuristics['disguised_vs_target'].iloc[0]:.1f}%")
            print(f"     Microsoft Phi-4: {microsoft_phi_heuristics['disguised_vs_target'].iloc[0]:.1f}%")
    
    # 3. Overall Performance Summary
    print("\n" + "=" * 50)
    print("3. OVERALL PERFORMANCE SUMMARY")
    print("=" * 50)
    
    if meta_llama_comparison is not None and microsoft_phi_comparison is not None:
        # Calculate overall scores
        meta_overall = (meta_llama_comparison['semantic_score'].mean() + meta_llama_comparison['stylistic_score'].mean()) / 2
        phi_overall = (microsoft_phi_comparison['semantic_score'].mean() + microsoft_phi_comparison['stylistic_score'].mean()) / 2
        
        print(f"\n🏆 OVERALL DISGUISE QUALITY (Average of Semantic + Stylistic):")
        print(f"   Meta Llama:     {meta_overall:.3f}/4.0")
        print(f"   Microsoft Phi-4: {phi_overall:.3f}/4.0")
        print(f"   Winner:         {'Microsoft Phi-4' if phi_overall > meta_overall else 'Meta Llama'} (+{abs(phi_overall - meta_overall):.3f})")
        
        # Consistency analysis
        meta_consistency = meta_llama_comparison['semantic_score'].std() + meta_llama_comparison['stylistic_score'].std()
        phi_consistency = microsoft_phi_comparison['semantic_score'].std() + microsoft_phi_comparison['stylistic_score'].std()
        
        print(f"\n📊 CONSISTENCY (Lower is Better):")
        print(f"   Meta Llama:     {meta_consistency:.3f}")
        print(f"   Microsoft Phi-4: {phi_consistency:.3f}")
        print(f"   More Consistent: {'Microsoft Phi-4' if phi_consistency < meta_consistency else 'Meta Llama'}")
    
    # 4. Key Insights
    print("\n" + "=" * 50)
    print("4. KEY INSIGHTS")
    print("=" * 50)
    
    if meta_llama_comparison is not None and microsoft_phi_comparison is not None:
        print(f"\n💡 ANALYSIS:")
        
        # Semantic analysis
        if microsoft_phi_comparison['semantic_score'].mean() > meta_llama_comparison['semantic_score'].mean():
            print(f"   • Microsoft Phi-4 shows better semantic preservation (+{microsoft_phi_comparison['semantic_score'].mean() - meta_llama_comparison['semantic_score'].mean():.3f})")
        else:
            print(f"   • Meta Llama shows better semantic preservation (+{meta_llama_comparison['semantic_score'].mean() - microsoft_phi_comparison['semantic_score'].mean():.3f})")
        
        # Stylistic analysis
        if microsoft_phi_comparison['stylistic_score'].mean() > meta_llama_comparison['stylistic_score'].mean():
            print(f"   • Microsoft Phi-4 shows better stylistic mimicry (+{microsoft_phi_comparison['stylistic_score'].mean() - meta_llama_comparison['stylistic_score'].mean():.3f})")
        else:
            print(f"   • Meta Llama shows better stylistic mimicry (+{meta_llama_comparison['stylistic_score'].mean() - microsoft_phi_comparison['stylistic_score'].mean():.3f})")
        
        # Consistency analysis
        if phi_consistency < meta_consistency:
            print(f"   • Microsoft Phi-4 is more consistent in its disguise quality")
        else:
            print(f"   • Meta Llama is more consistent in its disguise quality")
        
        # High score analysis
        if phi_semantic_high > meta_semantic_high:
            print(f"   • Microsoft Phi-4 achieves higher semantic scores more frequently")
        if phi_stylistic_high > meta_stylistic_high:
            print(f"   • Microsoft Phi-4 achieves higher stylistic scores more frequently")
    
    print("\n" + "=" * 80)
    print("REPORT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    create_comprehensive_report()
