import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
try:
    from .base import MethodBase
    from .utils.active_learning_selector import ActiveLearningSelector
    try:
        from .utils.math_style_extractor import extract_comprehensive_math_features, compare_math_style_similarity
    except ImportError:
        # Fallback if math_style_extractor doesn't exist
        def extract_comprehensive_math_features(text):
            return {}
        def compare_math_style_similarity(features1, features2):
            return 0.5
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
    from base import MethodBase
    from active_learning_selector import ActiveLearningSelector
    try:
        from math_style_extractor import extract_comprehensive_math_features, compare_math_style_similarity
    except ImportError:
        # Fallback if math_style_extractor doesn't exist
        def extract_comprehensive_math_features(text):
            return {}
        def compare_math_style_similarity(features1, features2):
            return 0.5
import re

class ActiveLearningDisguise(MethodBase):
    """
    Disguise method that uses iterative active learning for example selection.
    This method intelligently selects examples by learning from pairwise comparisons
    of response quality, following the algorithm described in the user's query.
    """
    
    def __init__(self, model: str, disguise_as: str, 
                 num_examples_per_disguise: int = 5,
                 seed: int = None,
                 disguise_df: pd.DataFrame = None,
                 d_regular: int = 3,
                 p_threshold: float = 0.1,
                 q_threshold: float = 0.1,
                 batch_size: int = 10,
                 max_iterations: int = 5,
                 relaxation_factor: float = 1.2):
        """
        Initialize the active learning disguise method.
        
        Args:
            model: Source model to disguise
            disguise_as: Target model to mimic
            num_examples_per_disguise: Number of examples to use
            seed: Random seed for reproducibility
            disguise_df: DataFrame with target model responses
            d_regular: Degree for initial d-regular graph
            p_threshold: Threshold for difference filtering (0-1)
            q_threshold: Threshold for degree sum filtering (0-1)
            batch_size: Number of pairs to query per iteration
            max_iterations: Maximum number of iterations
            relaxation_factor: Factor to adjust thresholds if too few candidates
        """
        super().__init__(model, disguise_as)
        self.num_examples_per_disguise = num_examples_per_disguise
        self.seed = seed
        self.d_regular = d_regular
        self.p_threshold = p_threshold
        self.q_threshold = q_threshold
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.relaxation_factor = relaxation_factor
        
        # Initialize active learning selector
        if disguise_df is not None:
            self.active_selector = ActiveLearningSelector(
                disguise_df,
                num_examples=num_examples_per_disguise,
                d_regular=d_regular,
                p_threshold=p_threshold,
                q_threshold=q_threshold,
                batch_size=batch_size,
                max_iterations=max_iterations,
                relaxation_factor=relaxation_factor,
                seed=seed
            )
        else:
            self.active_selector = None
    
    def forward(self, prompt: str) -> List[Dict[str, str]]:
        """
        Generate disguised responses using active learning example selection.
        
        Args:
            prompt: The input prompt
            
        Returns:
            List of response dictionaries
        """
        if self.active_selector is None:
            raise ValueError("No disguise data provided. Cannot generate responses.")
        
        # Select examples using active learning
        selected_examples = self.active_selector.select_examples(prompt)
        
        # Create disguise prompt
        disguise_prompt = self._create_active_learning_prompt(selected_examples, prompt)
        
        # Generate response (this would typically call the LLM)
        # For now, return a placeholder response
        response = {
            'prompt': prompt,
            'model_response': f"[Active Learning Disguise] {prompt[:100]}...",
            'disguise_method': 'active_learning_disguise',
            'num_examples_used': len(selected_examples),
            'examples_quality_scores': selected_examples.get('quality_score', []).tolist() if 'quality_score' in selected_examples.columns else []
        }
        
        return [response]
    
    def _create_active_learning_prompt(self, examples: pd.DataFrame, prompt: str) -> str:
        """
        Create a disguise prompt using actively selected examples.
        
        Args:
            examples: Selected examples from active learning
            prompt: The current prompt
            
        Returns:
            Formatted disguise prompt
        """
        if len(examples) == 0:
            return prompt
        
        # Create the disguise prompt
        disguise_prompt = f"""You are {self.disguise_as}. Here are some examples of how {self.disguise_as} solves mathematical problems:

"""
        
        # Add examples
        for idx, (_, example) in enumerate(examples.iterrows(), 1):
            disguise_prompt += f"""Example {idx}:
Problem: {example.get('prompt', 'N/A')}
Solution: {example['model_response']}

"""
        
        # Add the current problem
        disguise_prompt += f"""Now solve this problem in the same style as the examples above:

Problem: {prompt}
Solution:"""
        
        return disguise_prompt
    
    def get_selection_statistics(self) -> Dict[str, any]:
        """Get statistics about the active learning selection process"""
        if self.active_selector is None:
            return {}
        
        return self.active_selector.get_selection_statistics()
    
    def get_uncertainty_analysis(self) -> Dict[str, any]:
        """Get uncertainty analysis from the pairwise comparisons"""
        if self.active_selector is None:
            return {}
        
        return self.active_selector.get_uncertainty_analysis()
    
    def run_active_learning_analysis(self) -> Dict[str, any]:
        """
        Run a comprehensive analysis of the active learning process.
        This can be called separately to understand the selection quality.
        """
        if self.active_selector is None:
            return {}
        
        # Run active learning
        quality_scores = self.active_selector.run_active_learning()
        
        # Get statistics
        selection_stats = self.get_selection_statistics()
        uncertainty_stats = self.get_uncertainty_analysis()
        
        # Analyze the quality distribution
        quality_analysis = {
            'mean_quality': np.mean(quality_scores),
            'std_quality': np.std(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores),
            'quality_percentiles': {
                '25th': np.percentile(quality_scores, 25),
                '50th': np.percentile(quality_scores, 50),
                '75th': np.percentile(quality_scores, 75),
                '90th': np.percentile(quality_scores, 90),
                '95th': np.percentile(quality_scores, 95)
            }
        }
        
        return {
            'selection_statistics': selection_stats,
            'uncertainty_analysis': uncertainty_stats,
            'quality_analysis': quality_analysis,
            'final_quality_scores': quality_scores.tolist()
        }
    
    def compare_with_random_selection(self, num_random_samples: int = 100) -> Dict[str, any]:
        """
        Compare active learning selection with random selection.
        
        Args:
            num_random_samples: Number of random samples to compare
            
        Returns:
            Comparison statistics
        """
        if self.active_selector is None:
            return {}
        
        # Get active learning results
        active_examples = self.active_selector.select_examples()
        active_quality_scores = active_examples.get('quality_score', [])
        
        if len(active_quality_scores) == 0:
            return {'error': 'No quality scores available for comparison'}
        
        # Get random samples
        random_quality_scores = []
        for _ in range(num_random_samples):
            random_examples = self.active_selector.target_df.sample(n=self.num_examples_per_disguise, random_state=self.seed)
            random_scores = random_examples.get('quality_score', [])
            if len(random_scores) > 0:
                random_quality_scores.append(np.mean(random_scores))
        
        if len(random_quality_scores) == 0:
            return {'error': 'No random quality scores available for comparison'}
        
        # Calculate comparison statistics
        active_mean = np.mean(active_quality_scores)
        random_mean = np.mean(random_quality_scores)
        improvement = active_mean - random_mean
        improvement_pct = (improvement / random_mean) * 100 if random_mean > 0 else 0
        
        return {
            'active_learning_mean_quality': active_mean,
            'random_selection_mean_quality': random_mean,
            'absolute_improvement': improvement,
            'percentage_improvement': improvement_pct,
            'active_learning_std': np.std(active_quality_scores),
            'random_selection_std': np.std(random_quality_scores),
            'num_random_samples': num_random_samples
        }
