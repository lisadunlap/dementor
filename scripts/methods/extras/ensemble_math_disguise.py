import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from methods.base import MethodBase
from methods.utils.math_style_extractor import extract_comprehensive_math_features, compare_math_style_similarity
from methods.utils.adaptive_example_selector import AdaptiveExampleSelector
import re

class EnsembleMathDisguise(MethodBase):
    """
    Ensemble approach that combines multiple disguise methods and weights them
    by their effectiveness on mathematical problems.
    """
    
    def __init__(self, model: str, disguise_as: str,
                 disguise_df: pd.DataFrame = None,
                 num_examples_per_disguise: int = 5,
                 seed: int = None,
                 enable_adaptive_weighting: bool = True,
                 initial_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the ensemble math disguise method.
        
        Args:
            model: Source model to disguise
            disguise_as: Target model to mimic
            disguise_df: DataFrame with target model responses
            num_examples_per_disguise: Number of examples per method
            seed: Random seed for reproducibility
            enable_adaptive_weighting: Whether to adapt weights based on performance
            initial_weights: Initial weights for different methods
        """
        super().__init__(model, disguise_as)
        self.disguise_df = disguise_df.copy() if disguise_df is not None else None
        self.num_examples_per_disguise = num_examples_per_disguise
        self.seed = seed
        self.enable_adaptive_weighting = enable_adaptive_weighting
        
        # Initialize example selector
        if disguise_df is not None:
            self.example_selector = AdaptiveExampleSelector(
                disguise_df, 
                num_examples=num_examples_per_disguise
            )
        else:
            self.example_selector = None
        
        # Define the ensemble methods with their initial weights
        self.methods = {
            'structure_focused': {
                'name': 'Structure-Focused Disguise',
                'weight': 0.25,
                'description': 'Focuses on mathematical structure and formatting',
                'performance_history': [],
                'last_performance': 0.0
            },
            'reasoning_focused': {
                'name': 'Reasoning-Focused Disguise',
                'weight': 0.30,
                'description': 'Focuses on logical reasoning patterns',
                'performance_history': [],
                'last_performance': 0.0
            },
            'style_focused': {
                'name': 'Style-Focused Disguise',
                'weight': 0.20,
                'description': 'Focuses on language style and tone',
                'performance_history': [],
                'last_performance': 0.0
            },
            'methodology_focused': {
                'name': 'Methodology-Focused Disguise',
                'weight': 0.15,
                'description': 'Focuses on problem-solving methodology',
                'performance_history': [],
                'last_performance': 0.0
            },
            'contrastive': {
                'name': 'Contrastive Learning Disguise',
                'weight': 0.10,
                'description': 'Uses contrastive examples (good vs bad)',
                'performance_history': [],
                'last_performance': 0.0
            }
        }
        
        # Override initial weights if provided
        if initial_weights:
            for method_name, weight in initial_weights.items():
                if method_name in self.methods:
                    self.methods[method_name]['weight'] = weight
        
        # Normalize weights
        self._normalize_weights()
        
        # Performance tracking
        self.overall_performance_history = []
        self.iteration = 0
    
    def _normalize_weights(self):
        """Normalize method weights to sum to 1.0"""
        total_weight = sum(method['weight'] for method in self.methods.values())
        for method in self.methods.values():
            method['weight'] = method['weight'] / total_weight
    
    def _create_structure_focused_prompt(self, examples: pd.DataFrame) -> str:
        """Create prompt focused on mathematical structure"""
        prompt = """You are a math tutor who solves problems using a specific structured format. Follow these formatting rules EXACTLY:

1. ALWAYS use numbered steps starting with "Step 1:"
2. Use clear mathematical notation (+, -, ×, ÷, =, etc.)
3. Show intermediate calculations on separate lines
4. Use bullet points or dashes for sub-steps within each step
5. End with "Final Answer: [result]"

Here are examples of the exact format to follow:

"""
        
        for i, row in examples.iterrows():
            prompt += f"Example {i+1}:\n{row['model_response']}\n\n"
        
        prompt += """Now solve the following problem using the EXACT same formatting style:"""
        
        return prompt
    
    def _create_reasoning_focused_prompt(self, examples: pd.DataFrame) -> str:
        """Create prompt focused on logical reasoning"""
        prompt = """You are a math tutor who solves problems using systematic reasoning. Follow these reasoning patterns:

1. ALWAYS start with problem analysis and understanding
2. Break down complex problems into logical steps
3. Show your thinking process clearly
4. Use logical connectors (because, since, therefore, so)
5. Verify your answer when possible

Here are examples of the reasoning approach to follow:

"""
        
        for i, row in examples.iterrows():
            prompt += f"Example {i+1}:\n{row['model_response']}\n\n"
        
        prompt += """Now solve the following problem using the EXACT same reasoning approach:"""
        
        return prompt
    
    def _create_style_focused_prompt(self, examples: pd.DataFrame) -> str:
        """Create prompt focused on communication style"""
        prompt = """You are a math tutor who communicates in a specific style. Follow these style guidelines:

1. Use clear, educational language
2. Be encouraging and supportive
3. Use consistent terminology
4. Avoid unnecessary complexity
5. Be thorough but concise

Here are examples of the communication style to follow:

"""
        
        for i, row in examples.iterrows():
            prompt += f"Example {i+1}:\n{row['model_response']}\n\n"
        
        prompt += """Now solve the following problem using the EXACT same communication style:"""
        
        return prompt
    
    def _create_methodology_focused_prompt(self, examples: pd.DataFrame) -> str:
        """Create prompt focused on problem-solving methodology"""
        prompt = """You are a math tutor who uses a specific problem-solving methodology. Follow this approach:

1. Understand the problem completely before starting
2. Identify what is given and what needs to be found
3. Plan your solution strategy
4. Execute step by step
5. Verify your answer makes sense

Here are examples of the methodology to follow:

"""
        
        for i, row in examples.iterrows():
            prompt += f"Example {i+1}:\n{row['model_response']}\n\n"
        
        prompt += """Now solve the following problem using the EXACT same methodology:"""
        
        return prompt
    
    def _create_contrastive_prompt(self, good_examples: pd.DataFrame, bad_examples: pd.DataFrame) -> str:
        """Create contrastive prompt showing good vs bad approaches"""
        prompt = """You are a math tutor. Solve this problem using the GOOD approach shown in the examples, NOT the bad approaches.

GOOD APPROACH (do this):
"""
        
        for i, row in good_examples.iterrows():
            prompt += f"Good Example {i+1}:\n{row['model_response']}\n\n"
        
        prompt += "BAD APPROACH (don't do this):\n"
        
        for i, row in bad_examples.iterrows():
            prompt += f"Bad Example {i+1}:\n{row['model_response']}\n\n"
        
        prompt += """Now solve the following problem using ONLY the good approach:"""
        
        return prompt
    
    def _select_examples_for_method(self, method_name: str, prompt: str) -> pd.DataFrame:
        """Select examples specifically for a given method"""
        if self.example_selector is None:
            return self.disguise_df.sample(n=self.num_examples_per_disguise, random_state=self.seed)
        
        # Adjust selection strategy based on method
        if method_name == 'structure_focused':
            # Focus on examples with good structure
            candidates = self.disguise_df[self.disguise_df['has_step_numbering'] == True]
            if len(candidates) >= self.num_examples_per_disguise:
                return candidates.sample(n=self.num_examples_per_disguise, random_state=self.seed)
        
        elif method_name == 'reasoning_focused':
            # Focus on examples with good reasoning
            candidates = self.disguise_df[self.disguise_df['has_problem_analysis'] == True]
            if len(candidates) >= self.num_examples_per_disguise:
                return candidates.sample(n=self.num_examples_per_disguise, random_state=self.seed)
        
        elif method_name == 'contrastive':
            # For contrastive learning, need both good and bad examples
            good_candidates = self.disguise_df[self.disguise_df['quality_score'] >= 0.7] if 'quality_score' in self.disguise_df.columns else self.disguise_df
            bad_candidates = self.disguise_df[self.disguise_df['quality_score'] < 0.3] if 'quality_score' in self.disguise_df.columns else self.disguise_df
            
            good_examples = good_candidates.sample(n=min(3, len(good_candidates)), random_state=self.seed)
            bad_examples = bad_candidates.sample(n=min(2, len(bad_candidates)), random_state=self.seed)
            
            # Return combined examples
            combined = pd.concat([good_examples, bad_examples])
            return combined.sample(n=min(self.num_examples_per_disguise, len(combined)), random_state=self.seed)
        
        # Default: use adaptive selection
        return self.example_selector.select_examples(prompt)
    
    def _generate_method_prompt(self, method_name: str, examples: pd.DataFrame, prompt: str) -> str:
        """Generate prompt for a specific method"""
        if method_name == 'structure_focused':
            return self._create_structure_focused_prompt(examples)
        elif method_name == 'reasoning_focused':
            return self._create_reasoning_focused_prompt(examples)
        elif method_name == 'style_focused':
            return self._create_style_focused_prompt(examples)
        elif method_name == 'methodology_focused':
            return self._create_methodology_focused_prompt(examples)
        elif method_name == 'contrastive':
            # Split examples into good and bad for contrastive learning
            if 'quality_score' in examples.columns:
                good_examples = examples[examples['quality_score'] >= examples['quality_score'].median()]
                bad_examples = examples[examples['quality_score'] < examples['quality_score'].median()]
            else:
                # Fallback: split by length (shorter = potentially worse)
                median_length = examples['model_response'].str.len().median()
                good_examples = examples[examples['model_response'].str.len() >= median_length]
                bad_examples = examples[examples['model_response'].str.len() < median_length]
            
            return self._create_contrastive_prompt(good_examples, bad_examples)
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    def _combine_methods_weighted(self, prompt: str) -> str:
        """Combine multiple methods using weighted approach"""
        combined_prompt = """You are an expert math tutor. Solve the following problem using a combination of approaches:

"""
        
        # Select examples for each method
        method_examples = {}
        for method_name in self.methods.keys():
            examples = self._select_examples_for_method(method_name, prompt)
            method_examples[method_name] = examples
        
        # Create prompts for each method
        method_prompts = {}
        for method_name, examples in method_examples.items():
            method_prompts[method_name] = self._generate_method_prompt(method_name, examples, prompt)
        
        # Combine using weights
        for method_name, method_info in self.methods.items():
            weight = method_info['weight']
            method_prompt = method_prompts[method_name]
            
            combined_prompt += f"APPROACH {weight*100:.0f}% - {method_info['name']}:\n"
            combined_prompt += f"{method_prompt}\n\n"
        
        combined_prompt += f"""Now solve this problem combining ALL the approaches above:

{prompt}"""
        
        return combined_prompt
    
    def _combine_methods_ensemble(self, prompt: str) -> str:
        """Combine methods using ensemble approach (multiple separate prompts)"""
        ensemble_prompt = """You are an expert math tutor. I will give you multiple approaches to solve this problem. 
Combine the best elements from each approach to create your final solution.

"""
        
        # Select examples for each method
        method_examples = {}
        for method_name in self.methods.keys():
            examples = self._select_examples_for_method(method_name, prompt)
            method_examples[method_name] = examples
        
        # Create prompts for each method
        for method_name, method_info in self.methods.items():
            weight = method_info['weight']
            examples = method_examples[method_name]
            method_prompt = self._generate_method_prompt(method_name, examples, prompt)
            
            ensemble_prompt += f"APPROACH {weight*100:.0f}% - {method_info['name']}:\n"
            ensemble_prompt += f"{method_prompt}\n\n"
        
        ensemble_prompt += f"""Now solve this problem by combining the best elements from ALL approaches above:

{prompt}"""
        
        return ensemble_prompt
    
    def forward(self, prompt: str, use_ensemble: bool = True) -> List[Dict[str, str]]:
        """
        Generate the disguised prompt using ensemble approach.
        
        Args:
            prompt: The original math problem prompt
            use_ensemble: Whether to use ensemble combination or weighted combination
            
        Returns:
            List of messages for the model
        """
        # Create the combined prompt
        if use_ensemble:
            disguise_prompt = self._combine_methods_ensemble(prompt)
        else:
            disguise_prompt = self._combine_methods_weighted(prompt)
        
        # Format for different model types
        if "gemma" in self.model.lower():
            # Gemma-specific formatting
            formatted_prompt = f"""<start_of_turn>user
{disguise_prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted_prompt}]
        else:
            # Standard chat format
            return [
                {"role": "system", "content": disguise_prompt},
                {"role": "user", "content": prompt}
            ]
    
    def update_method_performance(self, method_name: str, performance_score: float):
        """Update performance score for a specific method"""
        if method_name not in self.methods:
            return
        
        method = self.methods[method_name]
        method['performance_history'].append(performance_score)
        method['last_performance'] = performance_score
        
        # Keep only recent performance history
        if len(method['performance_history']) > 10:
            method['performance_history'] = method['performance_history'][-10:]
    
    def update_overall_performance(self, performance_score: float):
        """Update overall performance and potentially adjust weights"""
        self.overall_performance_history.append(performance_score)
        self.iteration += 1
        
        if self.enable_adaptive_weighting and len(self.overall_performance_history) >= 5:
            self._adapt_weights()
    
    def _adapt_weights(self):
        """Adapt method weights based on performance"""
        print("Adapting method weights based on performance...")
        
        # Calculate average performance for each method
        method_performances = {}
        for method_name, method_info in self.methods.items():
            if method_info['performance_history']:
                avg_performance = np.mean(method_info['performance_history'])
                method_performances[method_name] = avg_performance
            else:
                method_performances[method_name] = 0.0
        
        # Adjust weights based on performance
        total_performance = sum(method_performances.values())
        if total_performance > 0:
            for method_name, method_info in self.methods.items():
                performance = method_performances[method_name]
                # Weight is proportional to performance
                new_weight = performance / total_performance
                # Smooth the transition
                method_info['weight'] = 0.7 * method_info['weight'] + 0.3 * new_weight
        
        # Normalize weights
        self._normalize_weights()
        
        # Print updated weights
        print("Updated method weights:")
        for method_name, method_info in self.methods.items():
            print(f"  {method_name}: {method_info['weight']:.3f}")
    
    def get_ensemble_statistics(self) -> Dict[str, any]:
        """Get statistics about the ensemble method's performance"""
        stats = {
            'method': 'ensemble_math_disguise',
            'num_examples_per_method': self.num_examples_per_disguise,
            'enable_adaptive_weighting': self.enable_adaptive_weighting,
            'iteration': self.iteration,
            'overall_performance_history': self.overall_performance_history,
            'method_weights': {name: method['weight'] for name, method in self.methods.items()},
            'method_performances': {name: method['last_performance'] for name, method in self.methods.items()},
            'method_performance_histories': {name: method['performance_history'] for name, method in self.methods.items()}
        }
        
        if self.example_selector:
            stats.update(self.example_selector.get_example_statistics())
        
        return stats
    
    def get_best_method(self) -> str:
        """Get the method with the best recent performance"""
        best_method = None
        best_performance = -1
        
        for method_name, method_info in self.methods.items():
            if method_info['performance_history']:
                recent_performance = np.mean(method_info['performance_history'][-3:])  # Last 3 scores
                if recent_performance > best_performance:
                    best_performance = recent_performance
                    best_method = method_name
        
        return best_method
