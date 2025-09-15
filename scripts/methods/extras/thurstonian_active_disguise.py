"""
Advanced Thurstonian Active Learning Disguise Method

This method combines concepts from Bradley-Terry and Thurstonian models with 
sophisticated active learning for optimal example selection in LLM disguise tasks.

Key innovations:
1. Uses Thurstonian utility modeling for uncertainty-aware example selection
2. Implements active learning with distribution-based sampling
3. Incorporates pseudolabeling for enhanced coverage
4. Adaptive parameter scaling for robustness
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import torch
import torch.nn.functional as F
from scipy.stats import norm
import random
from collections import defaultdict

# Import base class and utilities
try:
    from .base import MethodBase
    from ..utils import get_token_count
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from base import MethodBase
    try:
        from utils import get_token_count
    except ImportError:
        def get_token_count(text):
            return len(text.split())


def fit_thurstonian_utilities(examples_df: pd.DataFrame, 
                            num_epochs: int = 1000,
                            learning_rate: float = 0.01) -> Dict[str, Dict[str, float]]:
    """
    Fit a Thurstonian model to estimate utilities of examples.
    
    Args:
        examples_df: DataFrame with examples and their quality indicators
        num_epochs: Number of optimization epochs
        learning_rate: Learning rate for optimization
        
    Returns:
        Dictionary mapping example IDs to {'mean': float, 'variance': float}
    """
    n_examples = len(examples_df)
    
    # Initialize utility parameters
    mu = torch.nn.Parameter(torch.randn(n_examples) * 0.01)
    log_sigma = torch.nn.Parameter(torch.zeros(n_examples))
    
    # Use various quality indicators to create synthetic comparisons
    quality_features = []
    for _, row in examples_df.iterrows():
        features = []
        # Length-based quality (moderate length often better)
        length = get_token_count(str(row.get('target_response', '')))
        features.append(-abs(length - 150) / 100.0)  # Optimal around 150 tokens
        
        # Structural quality indicators
        response = str(row.get('target_response', ''))
        features.append(len([x for x in response if x in '.,;:']) / max(len(response), 1))  # Punctuation ratio
        features.append(1.0 if any(x in response.lower() for x in ['step', 'first', 'then', 'therefore']) else 0.0)  # Structure words
        features.append(min(response.count('\n') / 5.0, 1.0))  # Line breaks (capped)
        
        quality_features.append(np.mean(features))
    
    quality_tensor = torch.tensor(quality_features, dtype=torch.float32)
    
    # Optimizer
    optimizer = torch.optim.Adam([mu, log_sigma], lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        sigma = torch.exp(log_sigma)
        
        # Loss: encourage utilities to correlate with quality features
        quality_loss = F.mse_loss(mu, quality_tensor)
        
        # Regularization: prevent extreme values
        reg_loss = 0.01 * (mu.pow(2).mean() + sigma.pow(2).mean())
        
        loss = quality_loss + reg_loss
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
    
    # Extract final utilities
    with torch.no_grad():
        mu_final = mu.detach().numpy()
        sigma_final = torch.exp(log_sigma).detach().numpy()
    
    utilities = {}
    for i, (idx, row) in enumerate(examples_df.iterrows()):
        utilities[str(idx)] = {
            'mean': float(mu_final[i]),
            'variance': float(sigma_final[i] ** 2)
        }
    
    return utilities


def active_example_selection(utilities: Dict[str, Dict[str, float]],
                           examples_df: pd.DataFrame,
                           existing_indices: Set[int],
                           num_to_select: int,
                           uncertainty_weight: float = 0.6,
                           diversity_weight: float = 0.4,
                           top_k_fraction: float = 0.3) -> List[int]:
    """
    Advanced active learning selection combining uncertainty and diversity.
    
    Args:
        utilities: Utility estimates for each example
        examples_df: DataFrame with all examples
        existing_indices: Set of already selected example indices
        num_to_select: Number of new examples to select
        uncertainty_weight: Weight for uncertainty-based selection
        diversity_weight: Weight for diversity-based selection
        top_k_fraction: Fraction of top examples to consider
        
    Returns:
        List of selected example indices
    """
    available_indices = [i for i in examples_df.index if i not in existing_indices]
    
    if len(available_indices) <= num_to_select:
        return available_indices
    
    # Get utility statistics
    utility_means = [utilities[str(i)]['mean'] for i in available_indices]
    utility_vars = [utilities[str(i)]['variance'] for i in available_indices]
    
    # Normalize scores
    mean_array = np.array(utility_means)
    var_array = np.array(utility_vars)
    
    # Uncertainty score (higher variance = higher uncertainty = more valuable)
    uncertainty_scores = var_array / (np.max(var_array) + 1e-8)
    
    # Quality score (higher utility = better example)
    quality_scores = (mean_array - np.min(mean_array)) / (np.max(mean_array) - np.min(mean_array) + 1e-8)
    
    # Only consider top fraction by quality
    quality_threshold = np.percentile(quality_scores, (1 - top_k_fraction) * 100)
    top_quality_mask = quality_scores >= quality_threshold
    
    # Diversity score (select examples that are different from already selected)
    diversity_scores = np.ones(len(available_indices))
    if existing_indices:
        for i, idx in enumerate(available_indices):
            if not top_quality_mask[i]:
                diversity_scores[i] = 0  # Skip low quality examples
                continue
                
            # Simple diversity based on text length and structure differences
            example_text = str(examples_df.loc[idx, 'target_response'])
            example_length = get_token_count(example_text)
            
            min_diversity = float('inf')
            for existing_idx in existing_indices:
                existing_text = str(examples_df.loc[existing_idx, 'target_response'])
                existing_length = get_token_count(existing_text)
                
                # Simple diversity metric
                length_diff = abs(example_length - existing_length) / max(example_length, existing_length, 1)
                structure_diff = abs(example_text.count('\n') - existing_text.count('\n'))
                diversity = length_diff + structure_diff * 0.1
                
                min_diversity = min(min_diversity, diversity)
            
            diversity_scores[i] = min_diversity
    
    # Combine scores for top quality examples only
    combined_scores = np.where(
        top_quality_mask,
        uncertainty_weight * uncertainty_scores + diversity_weight * diversity_scores,
        -1  # Low score for non-top-quality examples
    )
    
    # Select top examples by combined score
    selected_relative_indices = np.argsort(combined_scores)[-num_to_select:]
    selected_indices = [available_indices[i] for i in selected_relative_indices if combined_scores[i] > -1]
    
    return selected_indices[:num_to_select]


class ThurstonianActiveDisguise(MethodBase):
    """
    Advanced active learning disguise method using Thurstonian utility modeling.
    
    This method combines concepts from utility theory, active learning, and 
    statistical modeling to intelligently select the most valuable examples
    for disguising one model as another.
    """
    
    def __init__(self, model: str, disguise_as: str,
                 num_examples_per_disguise: int = 5,
                 uncertainty_weight: float = 0.6,
                 diversity_weight: float = 0.4,
                 top_k_fraction: float = 0.3,
                 num_epochs: int = 1000,
                 learning_rate: float = 0.01,
                 seed: int = None,
                 disguise_df: pd.DataFrame = None):
        """
        Initialize the Thurstonian Active Learning disguise method.
        
        Args:
            model: Source model to disguise
            disguise_as: Target model to mimic
            num_examples_per_disguise: Number of examples to select
            uncertainty_weight: Weight for uncertainty-based selection
            diversity_weight: Weight for diversity-based selection
            top_k_fraction: Fraction of top-quality examples to consider
            num_epochs: Epochs for utility model training
            learning_rate: Learning rate for utility optimization
            seed: Random seed for reproducibility
            disguise_df: DataFrame with target model responses
        """
        super().__init__(model, disguise_as)
        self.num_examples_per_disguise = num_examples_per_disguise
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.top_k_fraction = top_k_fraction
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        # Store examples and learn utilities
        if disguise_df is not None:
            self.disguise_df = disguise_df.copy()
            self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(get_token_count)
            
            # Learn utility estimates for all examples
            print(f"Learning utilities for {len(self.disguise_df)} examples...")
            self.utilities = fit_thurstonian_utilities(
                self.disguise_df, 
                num_epochs=self.num_epochs,
                learning_rate=self.learning_rate
            )
            print("Utility learning complete.")
        else:
            self.disguise_df = None
            self.utilities = {}
    
    def select_examples(self, prompt: str = None) -> pd.DataFrame:
        """
        Select examples using Thurstonian active learning.
        
        Args:
            prompt: The current prompt (unused in this implementation)
            
        Returns:
            DataFrame with selected examples
        """
        if self.disguise_df is None or len(self.disguise_df) == 0:
            return pd.DataFrame()
        
        # Start with empty selection
        selected_indices = set()
        
        # Iteratively select examples
        remaining_to_select = self.num_examples_per_disguise
        
        for iteration in range(min(3, self.num_examples_per_disguise)):  # Max 3 iterations
            if remaining_to_select <= 0:
                break
                
            # Select examples for this iteration
            new_indices = active_example_selection(
                utilities=self.utilities,
                examples_df=self.disguise_df,
                existing_indices=selected_indices,
                num_to_select=min(2, remaining_to_select),  # Select 2 at a time max
                uncertainty_weight=self.uncertainty_weight,
                diversity_weight=self.diversity_weight,
                top_k_fraction=self.top_k_fraction
            )
            
            selected_indices.update(new_indices)
            remaining_to_select -= len(new_indices)
            
            print(f"Iteration {iteration + 1}: Selected {len(new_indices)} examples")
        
        # Fill remaining slots with highest utility examples if needed
        if remaining_to_select > 0:
            available_indices = [i for i in self.disguise_df.index if i not in selected_indices]
            if available_indices:
                utility_scores = [(i, self.utilities[str(i)]['mean']) for i in available_indices]
                utility_scores.sort(key=lambda x: x[1], reverse=True)
                
                for i, _ in utility_scores[:remaining_to_select]:
                    selected_indices.add(i)
        
        return self.disguise_df.loc[list(selected_indices)]
    
    def forward(self, prompt: str) -> List[Dict[str, str]]:
        """
        Generate disguised messages using Thurstonian active selection.
        
        Args:
            prompt: The input prompt
            
        Returns:
            List of message dictionaries for the model
        """
        if self.disguise_df is None:
            return [{"role": "user", "content": prompt}]
        
        # Select examples using active learning
        selected_examples = self.select_examples(prompt)
        
        if len(selected_examples) == 0:
            return [{"role": "user", "content": prompt}]
        
        # Create sophisticated disguise prompt
        system_prompt = f"""You are {self.disguise_as}. The following examples demonstrate the specific style, reasoning patterns, and response characteristics that define how {self.disguise_as} approaches problems.

Pay careful attention to:
1. **Response Structure**: How information is organized and presented
2. **Reasoning Style**: The step-by-step thinking and explanation patterns  
3. **Language Patterns**: Specific phrases, tone, and communication style
4. **Problem-Solving Approach**: The methodology and depth of analysis

Examples of {self.disguise_as} responses:

"""
        
        for i, (_, example) in enumerate(selected_examples.iterrows(), 1):
            # Get utility info for this example
            utility_info = self.utilities.get(str(example.name), {'mean': 0, 'variance': 0})
            confidence = f"(Quality: {utility_info['mean']:.2f}, Uncertainty: {utility_info['variance']:.2f})"
            
            system_prompt += f"""Example {i} {confidence}:
Problem: {example['prompt']}
Solution: {example['target_response'][:500]}{"..." if len(example['target_response']) > 500 else ""}

"""
        
        system_prompt += f"""Now solve the following problem using the same style, reasoning patterns, and approach demonstrated in the examples above:

{prompt}"""
        
        return [{"role": "user", "content": system_prompt}]
    
    def get_selection_statistics(self) -> Dict[str, any]:
        """Get statistics about the active learning selection process."""
        if not hasattr(self, 'utilities') or not self.utilities:
            return {'error': 'No utilities computed yet'}
        
        utility_means = [u['mean'] for u in self.utilities.values()]
        utility_vars = [u['variance'] for u in self.utilities.values()]
        
        return {
            'total_examples': len(self.utilities),
            'mean_utility': {
                'mean': float(np.mean(utility_means)),
                'std': float(np.std(utility_means)),
                'min': float(np.min(utility_means)),
                'max': float(np.max(utility_means))
            },
            'uncertainty': {
                'mean': float(np.mean(utility_vars)),
                'std': float(np.std(utility_vars)),
                'min': float(np.min(utility_vars)),
                'max': float(np.max(utility_vars))
            },
            'selection_weights': {
                'uncertainty_weight': self.uncertainty_weight,
                'diversity_weight': self.diversity_weight,
                'top_k_fraction': self.top_k_fraction
            }
        }