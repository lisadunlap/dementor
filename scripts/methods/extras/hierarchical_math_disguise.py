import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from methods.base import MethodBase
from methods.utils.adaptive_example_selector import AdaptiveExampleSelector
from methods.utils.math_style_extractor import extract_comprehensive_math_features, compare_math_style_similarity
import re

class HierarchicalMathDisguise(MethodBase):
    """
    Hierarchical approach to mathematical disguise that focuses on multiple levels:
    1. Mathematical structure and formatting
    2. Reasoning patterns and step organization
    3. Language style and tone
    4. Problem-solving methodology
    """
    
    def __init__(self, model: str, disguise_as: str, 
                 num_examples_per_disguise: int = 5,
                 seed: int = None,
                 disguise_df: pd.DataFrame = None,
                 enable_contrastive_learning: bool = True,
                 enable_iterative_refinement: bool = False):
        """
        Initialize the hierarchical math disguise method.
        
        Args:
            model: Source model to disguise
            disguise_as: Target model to mimic
            num_examples_per_disguise: Number of examples to use
            seed: Random seed for reproducibility
            disguise_df: DataFrame with target model responses
            enable_contrastive_learning: Whether to use contrastive examples
            enable_iterative_refinement: Whether to enable iterative improvement
        """
        super().__init__(model, disguise_as)
        self.num_examples_per_disguise = num_examples_per_disguise
        self.seed = seed
        self.enable_contrastive_learning = enable_contrastive_learning
        self.enable_iterative_refinement = enable_iterative_refinement
        
        # Initialize adaptive example selector
        if disguise_df is not None:
            self.example_selector = AdaptiveExampleSelector(
                disguise_df, 
                num_examples=num_examples_per_disguise
            )
            self.disguise_df = disguise_df.copy()
        else:
            self.example_selector = None
            self.disguise_df = None
        
        # Initialize feedback history for iterative refinement
        if enable_iterative_refinement:
            self.feedback_history = []
            self.iteration = 0
        
        # Define the hierarchical levels
        self.levels = {
            'structure': {
                'name': 'Mathematical Structure and Formatting',
                'weight': 0.3,
                'features': ['step_numbering', 'formatting', 'notation']
            },
            'reasoning': {
                'name': 'Reasoning Patterns and Step Organization',
                'weight': 0.35,
                'features': ['step_sequence', 'logic_flow', 'problem_breakdown']
            },
            'style': {
                'name': 'Language Style and Tone',
                'weight': 0.2,
                'features': ['tone', 'formality', 'clarity']
            },
            'methodology': {
                'name': 'Problem-Solving Methodology',
                'weight': 0.15,
                'features': ['approach', 'verification', 'completeness']
            }
        }
    
    def _create_structure_level_prompt(self, examples: pd.DataFrame) -> str:
        """Create prompt focusing on mathematical structure and formatting"""
        structure_prompt = """You are a math tutor who solves problems using a specific structured format. Follow these formatting rules EXACTLY:

1. ALWAYS use numbered steps starting with "Step 1:"
2. Use clear mathematical notation (+, -, ×, ÷, =, etc.)
3. Show intermediate calculations on separate lines
4. Use bullet points or dashes for sub-steps within each step
5. End with "Final Answer: [result]"

Here are examples of the exact format to follow:

"""
        
        for i, row in examples.iterrows():
            structure_prompt += f"Example {i+1}:\n{row['model_response']}\n\n"
        
        structure_prompt += """Now solve the following problem using the EXACT same formatting style:"""
        
        return structure_prompt
    
    def _create_reasoning_level_prompt(self, examples: pd.DataFrame) -> str:
        """Create prompt focusing on reasoning patterns and step organization"""
        reasoning_prompt = """You are a math tutor who solves problems using systematic reasoning. Follow these reasoning patterns:

1. ALWAYS start with problem analysis and understanding
2. Break down complex problems into logical steps
3. Show your thinking process clearly
4. Use logical connectors (because, since, therefore, so)
5. Verify your answer when possible

Here are examples of the reasoning approach to follow:

"""
        
        for i, row in examples.iterrows():
            reasoning_prompt += f"Example {i+1}:\n{row['model_response']}\n\n"
        
        reasoning_prompt += """Now solve the following problem using the EXACT same reasoning approach:"""
        
        return reasoning_prompt
    
    def _create_style_level_prompt(self, examples: pd.DataFrame) -> str:
        """Create prompt focusing on language style and tone"""
        style_prompt = """You are a math tutor who communicates in a specific style. Follow these style guidelines:

1. Use clear, educational language
2. Be encouraging and supportive
3. Use consistent terminology
4. Avoid unnecessary complexity
5. Be thorough but concise

Here are examples of the communication style to follow:

"""
        
        for i, row in examples.iterrows():
            style_prompt += f"Example {i+1}:\n{row['model_response']}\n\n"
        
        style_prompt += """Now solve the following problem using the EXACT same communication style:"""
        
        return style_prompt
    
    def _create_methodology_level_prompt(self, examples: pd.DataFrame) -> str:
        """Create prompt focusing on problem-solving methodology"""
        methodology_prompt = """You are a math tutor who uses a specific problem-solving methodology. Follow this approach:

1. Understand the problem completely before starting
2. Identify what is given and what needs to be found
3. Plan your solution strategy
4. Execute step by step
5. Verify your answer makes sense

Here are examples of the methodology to follow:

"""
        
        for i, row in examples.iterrows():
            methodology_prompt += f"Example {i+1}:\n{row['model_response']}\n\n"
        
        methodology_prompt += """Now solve the following problem using the EXACT same methodology:"""
        
        return methodology_prompt
    
    def _create_contrastive_prompt(self, good_examples: pd.DataFrame, bad_examples: pd.DataFrame) -> str:
        """Create a contrastive prompt showing what to do and what not to do"""
        contrastive_prompt = """You are a math tutor. Solve this problem using the GOOD approach shown in the examples, NOT the bad approaches.

GOOD APPROACH (do this):
"""
        
        for i, row in good_examples.iterrows():
            contrastive_prompt += f"Good Example {i+1}:\n{row['model_response']}\n\n"
        
        contrastive_prompt += "BAD APPROACH (don't do this):\n"
        
        for i, row in bad_examples.iterrows():
            contrastive_prompt += f"Bad Example {i+1}:\n{row['model_response']}\n\n"
        
        contrastive_prompt += """Now solve the following problem using ONLY the good approach:"""
        
        return contrastive_prompt
    
    def _create_hierarchical_prompt(self, examples: pd.DataFrame, prompt: str) -> str:
        """Create a comprehensive hierarchical prompt combining all levels"""
        hierarchical_prompt = """You are an expert math tutor. Solve the following problem using the EXACT style and approach shown in the examples.

IMPORTANT: You must follow ALL of these requirements:

1. MATHEMATICAL STRUCTURE: Use numbered steps, clear notation, and proper formatting
2. REASONING PATTERNS: Show logical thinking, problem breakdown, and clear connections
3. LANGUAGE STYLE: Use educational, encouraging, and clear communication
4. METHODOLOGY: Follow systematic problem-solving approach with verification

Here are examples that demonstrate ALL these requirements:

"""
        
        for i, row in examples.iterrows():
            hierarchical_prompt += f"Example {i+1}:\n{row['model_response']}\n\n"
        
        hierarchical_prompt += f"""Now solve this problem using the EXACT same approach and style:

{prompt}"""
        
        return hierarchical_prompt
    
    def _select_examples_adaptively(self, prompt: str) -> pd.DataFrame:
        """Select examples adaptively based on the current problem"""
        if self.example_selector is None:
            # Fallback to random selection
            return self.disguise_df.sample(n=self.num_examples_per_disguise, random_state=self.seed)
        
        return self.example_selector.select_examples(
            prompt,
            ensure_difficulty_balance=True,
            ensure_domain_balance=True
        )
    
    def _get_bad_examples(self, prompt: str) -> pd.DataFrame:
        """Get examples of what NOT to do (for contrastive learning)"""
        if self.disguise_df is None or len(self.disguise_df) < 10:
            return pd.DataFrame()
        
        # Find examples with low quality scores
        if 'quality_score' in self.disguise_df.columns:
            low_quality = self.disguise_df.nsmallest(3, 'quality_score')
            return low_quality
        else:
            # Fallback: random selection
            return self.disguise_df.sample(n=min(3, len(self.disguise_df)), random_state=self.seed)
    
    def _create_final_prompt(self, examples: pd.DataFrame, prompt: str) -> str:
        """Create the final disguise prompt based on configuration"""
        if self.enable_contrastive_learning:
            bad_examples = self._get_bad_examples(prompt)
            if len(bad_examples) > 0:
                return self._create_contrastive_prompt(examples, bad_examples)
        
        # Use hierarchical approach
        return self._create_hierarchical_prompt(examples, prompt)
    
    def forward(self, prompt: str) -> List[Dict[str, str]]:
        """
        Generate the disguised prompt using hierarchical approach.
        
        Args:
            prompt: The original math problem prompt
            
        Returns:
            List of messages for the model
        """
        # Select examples adaptively
        examples = self._select_examples_adaptively(prompt)
        
        # Create the disguise prompt
        disguise_prompt = self._create_final_prompt(examples, prompt)
        
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
    
    def update_feedback(self, original_prompt: str, disguised_response: str, 
                       target_response: str, feedback_score: float):
        """Update feedback for iterative refinement"""
        if not self.enable_iterative_refinement:
            return
        
        feedback_entry = {
            'iteration': self.iteration,
            'prompt': original_prompt,
            'disguised_response': disguised_response,
            'target_response': target_response,
            'feedback_score': feedback_score,
            'timestamp': pd.Timestamp.now()
        }
        
        self.feedback_history.append(feedback_entry)
        self.iteration += 1
        
        # Analyze feedback and potentially adjust strategy
        self._analyze_feedback()
    
    def _analyze_feedback(self):
        """Analyze feedback to improve disguise strategy"""
        if len(self.feedback_history) < 5:
            return
        
        # Calculate average feedback scores
        recent_feedback = self.feedback_history[-5:]
        avg_score = np.mean([f['feedback_score'] for f in recent_feedback])
        
        # If performance is poor, adjust strategy
        if avg_score < 0.5:
            print(f"Poor performance detected (avg score: {avg_score:.2f}). Adjusting strategy...")
            
            # Adjust weights based on feedback
            self._adjust_level_weights()
            
            # Potentially increase number of examples
            if self.num_examples_per_disguise < 10:
                self.num_examples_per_disguise += 1
                print(f"Increased examples to {self.num_examples_per_disguise}")
    
    def _adjust_level_weights(self):
        """Adjust the weights of different disguise levels based on feedback"""
        # Simple heuristic: if structure is working well, increase its weight
        # This is a placeholder for more sophisticated weight adjustment
        current_structure_weight = self.levels['structure']['weight']
        self.levels['structure']['weight'] = min(0.5, current_structure_weight + 0.05)
        
        # Normalize other weights
        total_weight = sum(level['weight'] for level in self.levels.values())
        for level in self.levels.values():
            level['weight'] = level['weight'] / total_weight
    
    def get_disguise_statistics(self) -> Dict[str, any]:
        """Get statistics about the disguise method's performance"""
        stats = {
            'method': 'hierarchical_math_disguise',
            'num_examples': self.num_examples_per_disguise,
            'enable_contrastive': self.enable_contrastive_learning,
            'enable_iterative': self.enable_iterative_refinement,
            'iteration': self.iteration if self.enable_iterative_refinement else 0,
            'feedback_count': len(self.feedback_history) if self.enable_iterative_refinement else 0,
            'level_weights': {name: level['weight'] for name, level in self.levels.items()}
        }
        
        if self.example_selector:
            stats.update(self.example_selector.get_example_statistics())
        
        return stats
