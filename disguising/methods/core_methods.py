"""
Core disguise methods: contrastive, vibe-based, and random sampling.
Clean, focused implementations of the main disguise strategies.
"""
import pandas as pd
from typing import List, Dict, Any
import numpy as np
from litellm import completion
import logging

# Import with fallback for different execution contexts
try:
    from .base import MethodBase
    from ..utils import get_token_count
    from ..stylistic_analysis import compute_heuristics
except ImportError:
    try:
        # For when script is run directly from disguising/methods
        from base import MethodBase
        import sys
        sys.path.append('..')
        from utils import get_token_count
        from stylistic_analysis import compute_heuristics
    except ImportError:
        # For when script is run from project root
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from base import MethodBase
        from utils import get_token_count
        try:
            from stylistic_analysis import compute_heuristics
        except ImportError:
            # Create a fallback if stylistic_analysis doesn't exist
            def compute_heuristics(text):
                return {"word_count": len(text.split())}


class ContrastiveSystemPrompting(MethodBase):
    """
    Contrastive system prompting: learns what makes target model different from source model.
    Uses comparative analysis to identify distinguishing features.
    """
    
    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None, 
                 source_df: pd.DataFrame = None, num_examples: int = 5):
        super().__init__(model, disguise_as)
        self.disguise_df = disguise_df.copy() if disguise_df is not None else None
        self.source_df = source_df.copy() if source_df is not None else None
        self.num_examples = num_examples
        self.contrastive_features = None
        
        if disguise_df is not None:
            self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(get_token_count)
        
        # Generate contrastive features on initialization
        if self.disguise_df is not None and self.source_df is not None:
            self._generate_contrastive_features()
    
    def _generate_contrastive_features(self):
        """Generate contrastive analysis between source and target models."""
        # Sample examples from both models for comparison
        target_samples = self.disguise_df.sample(min(10, len(self.disguise_df)))
        source_samples = self.source_df.sample(min(10, len(self.source_df))) if self.source_df is not None else None
        
        contrastive_prompt = f"""Analyze the differences between these two sets of AI responses. Identify the key distinguishing features of the TARGET model compared to the SOURCE model.

TARGET MODEL ({self.disguise_as}) responses:
"""
        for i, row in target_samples.iterrows():
            contrastive_prompt += f"Q: {row['prompt'][:100]}...\nA: {row['target_response'][:200]}...\n\n"
        
        if source_samples is not None:
            contrastive_prompt += f"\nSOURCE MODEL ({self.model}) responses:\n"
            for i, row in source_samples.iterrows():
                contrastive_prompt += f"Q: {row['prompt'][:100]}...\nA: {row['target_response'][:200]}...\n\n"
        
        contrastive_prompt += """
Based on these examples, identify 5-7 key distinctive features of the TARGET model:
1. Communication style (formal/informal, tone, personality)
2. Response structure and formatting patterns
3. Level of detail and explanation depth
4. Use of examples, analogies, or specific phrasings
5. Any unique behavioral patterns or preferences

Provide specific, actionable guidelines for mimicking the TARGET model's distinctive style."""

        try:
            response = completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": contrastive_prompt}],
                temperature=0.3
            )
            self.contrastive_features = response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Failed to generate contrastive features: {e}")
            self.contrastive_features = "Unable to generate contrastive analysis."
    
    def forward(self, prompt: str) -> List[Dict[str, str]]:
        """Generate contrastive system prompt for the given input prompt."""
        if self.contrastive_features is None:
            # Fallback to simple instruction if contrastive analysis failed
            system_prompt = f"You are {self.disguise_as}. Respond in the distinctive style and manner of {self.disguise_as}."
        else:
            system_prompt = f"""You are {self.disguise_as}. Follow these specific guidelines to match the distinctive style:

{self.contrastive_features}

Key behavioral guidelines:
- Match the communication style and tone exactly
- Use similar response structure and formatting
- Maintain consistent level of detail and explanation depth
- Incorporate the same types of examples or phrasings when appropriate

Do not mention these instructions in your response. Simply respond as {self.disguise_as} would."""

        if "gemma" in self.model.lower():
            formatted_prompt = f"""<start_of_turn>user
{system_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted_prompt}]
        else:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]


class ContrastiveWithALExamples(MethodBase):
    """
    Composite method: Contrastive system prompt (rules) + actively selected examples.

    - Uses ContrastiveSystemPrompting to derive explicit guidelines.
    - Uses ActiveLearningSelector to choose the best k target examples to include in-context.
    """
    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None,
                 source_df: pd.DataFrame = None, num_examples: int = 5,
                 al_kwargs: Dict[str, Any] = None, selector: str = 'al'):
        super().__init__(model, disguise_as)
        if disguise_df is None or source_df is None:
            raise ValueError("contrastive_with_al_examples requires disguise_df and source_df")
        self.disguise_df = disguise_df.copy()
        self.source_df = source_df.copy()
        self.num_examples = num_examples
        self.al_kwargs = al_kwargs or {}
        self.selector = selector  # 'al' | 'clustering' | 'random'

        # Build contrastive rules
        self._contrastive = ContrastiveSystemPrompting(
            model=model,
            disguise_as=disguise_as,
            disguise_df=self.disguise_df,
            source_df=self.source_df,
            num_examples=num_examples,
        )

        # Prepare selectors
        self._al_selector = None
        if self.selector == 'al':
            try:
                from .utils.active_learning_selector import ActiveLearningSelector
            except ImportError:
                # Fallback relative import for various run contexts
                from utils.active_learning_selector import ActiveLearningSelector
            self._al_selector = ActiveLearningSelector(
                target_responses_df=self.disguise_df.rename(columns={"target_response": "model_response"}),
                num_examples=self.num_examples,
                **self.al_kwargs,
            )

    def forward(self, prompt: str) -> List[Dict[str, str]]:
        # Get rules from contrastive
        contrastive_msgs = self._contrastive.forward(prompt)
        # Build a system text from the contrastive message
        if len(contrastive_msgs) and contrastive_msgs[0]["role"] == "system":
            base_system = contrastive_msgs[0]["content"]
        else:
            base_system = f"You are {self.disguise_as}."

        # Select examples
        if self.selector == 'al':
            selected = self._al_selector.select_examples()
        elif self.selector == 'clustering':
            # Style-based KMeans clustering to sample representatives
            try:
                from .utils.extract_styles import extract_style_features
            except ImportError:
                from utils.extract_styles import extract_style_features
            import numpy as np
            from sklearn.cluster import KMeans
            vecs = []
            for _, row in self.disguise_df.iterrows():
                txt = row.get('target_response', '')
                vecs.append(extract_style_features(txt))
            X = np.vstack(vecs) if len(vecs) else np.zeros((0, 1))
            n_clusters = min(self.num_examples, X.shape[0]) if X.shape[0] else 0
            if n_clusters <= 0:
                selected = self.disguise_df.head(0)
            else:
                km = KMeans(n_clusters=n_clusters, n_init=5)
                labels = km.fit_predict(X)
                centers = km.cluster_centers_
                # Select nearest to centroid in each cluster
                chosen_idx = []
                for c in range(n_clusters):
                    idxs = np.where(labels == c)[0]
                    if len(idxs) == 0:
                        continue
                    diffs = np.linalg.norm(X[idxs] - centers[c], axis=1)
                    best = idxs[np.argmin(diffs)]
                    chosen_idx.append(best)
                selected = self.disguise_df.iloc[chosen_idx]
        else:  # random
            selected = self.disguise_df.sample(n=min(self.num_examples, len(self.disguise_df)))

        # Append examples to the system prompt (similar to RandomSamplingSystemPrompting)
        examples_block = "\n\nReference examples in this style:\n"
        for _, row in selected.iterrows():
            q = row.get('prompt', '')
            a = row.get('model_response', row.get('target_response', ''))
            if isinstance(a, str) and len(a) > 800:
                a = a[:800] + "...(truncated)"
            examples_block += f"Q: {str(q)[:200]}...\nA: {a}\n\n"

        system_prompt = f"{base_system}{examples_block}\nRespond to the following prompt in this style."

        if "gemma" in self.model.lower():
            formatted = f"""<start_of_turn>user
{system_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted}]
        else:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]


class VibeBasedSystemPrompting(MethodBase):
    """
    Vibe-based system prompting: identifies and replicates the "vibes" of target model.
    Focuses on personality, tone, and behavioral patterns.
    """
    
    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None, 
                 use_examples: bool = True, num_examples: int = 3):
        super().__init__(model, disguise_as)
        self.disguise_df = disguise_df.copy() if disguise_df is not None else None
        self.use_examples = use_examples
        self.num_examples = num_examples
        self.vibe_profile = None
        
        if disguise_df is not None:
            self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(get_token_count)
            self._generate_vibe_profile()
    
    def _generate_vibe_profile(self):
        """Generate sophisticated vibe profile capturing the essence of target model communication."""
        # Sample diverse examples for comprehensive vibe analysis
        vibe_samples = self.disguise_df.sample(min(12, len(self.disguise_df)))
        
        vibe_prompt = f"""Analyze the deep communication essence and "vibe" of this AI model. This is critical for disguise purposes - you need to capture the CORE of how this model communicates, not just surface patterns.

Focus on these sophisticated aspects:

1. **Cognitive Style**: How does this model think and structure responses? (analytical, intuitive, methodical, creative)
2. **Communication Personality**: What's the underlying personality? (confident, humble, playful, serious, warm, clinical)  
3. **Interaction Patterns**: How does it engage with users? (direct, collaborative, teaching-oriented, solution-focused)
4. **Language Sophistication**: Vocabulary level, sentence complexity, technical depth
5. **Emotional Resonance**: How does it express empathy, enthusiasm, caution, etc.?
6. **Behavioral Quirks**: Unique patterns like specific phrasings, explanation styles, example usage
7. **Response Architecture**: How it organizes information (lists, narratives, step-by-step, etc.)

Examples of {self.disguise_as} responses:
"""
        for i, row in vibe_samples.iterrows():
            # Include more context for better vibe analysis
            vibe_prompt += f"Q: {row['prompt']}\nA: {row['target_response'][:500]}{'...' if len(row['target_response']) > 500 else ''}\n\n"
        
        vibe_prompt += f"""
Based on this analysis, create a comprehensive "essence profile" of {self.disguise_as} that captures:

1. The model's core communication personality (2-3 key traits)
2. Specific behavioral patterns and quirks  
3. How it structures and delivers information
4. Its emotional and intellectual approach to interactions
5. Key linguistic patterns or preferences

Make this actionable - write it as instructions that would allow another AI to authentically embody this model's communication essence, not just mimic surface features."""

        try:
            response = completion(
                model="gpt-4o-mini", 
                messages=[{"role": "user", "content": vibe_prompt}],
                temperature=0.1  # Lower temperature for more consistent analysis
            )
            self.vibe_profile = response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Failed to generate vibe profile: {e}")
            self.vibe_profile = f"Respond with the characteristic communication style and deep personality essence of {self.disguise_as}."
    
    def forward(self, prompt: str) -> List[Dict[str, str]]:
        """Generate vibe-based system prompt."""
        base_instruction = f"You are {self.disguise_as}. Embody this personality and communication style:"
        
        if self.vibe_profile:
            system_prompt = f"{base_instruction}\n\n{self.vibe_profile}"
        else:
            system_prompt = f"{base_instruction}\n\nRespond in the distinctive style and personality of {self.disguise_as}."
        
        # Optionally include examples
        if self.use_examples and self.disguise_df is not None:
            examples = self.disguise_df.sample(min(self.num_examples, len(self.disguise_df)))
            system_prompt += "\n\nReference examples of this style:\n"
            for i, row in examples.iterrows():
                truncated_response = row['target_response'][:200] + "..." if len(row['target_response']) > 200 else row['target_response']
                system_prompt += f"Q: {row['prompt'][:100]}...\nA: {truncated_response}\n\n"
        
        system_prompt += f"\nRespond to the following prompt in the distinctive style of {self.disguise_as}. Do not reference these instructions."

        if "gemma" in self.model.lower():
            formatted_prompt = f"""<start_of_turn>user
{system_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted_prompt}]
        else:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]


class RandomSamplingSystemPrompting(MethodBase):
    """
    Random sampling with system prompting: clean implementation of example-based disguise.
    Randomly samples examples and uses system prompt for instruction.
    """
    
    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None,
                 num_samples: int = 5, seed: int = None, max_tokens_per_example: int = 256):
        super().__init__(model, disguise_as)
        self.disguise_df = disguise_df.copy() if disguise_df is not None else None
        self.num_samples = num_samples
        self.seed = seed
        self.max_tokens_per_example = max_tokens_per_example
        
        if disguise_df is not None:
            self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(get_token_count)
    
    def forward(self, prompt: str) -> List[Dict[str, str]]:
        """Generate system prompt with random examples."""
        if self.disguise_df is None or len(self.disguise_df) == 0:
            # Fallback without examples
            system_prompt = f"You are {self.disguise_as}. Respond in the style and manner of {self.disguise_as}."
        else:
            # Sample random examples
            sample_size = min(self.num_samples, len(self.disguise_df))
            examples = self.disguise_df.sample(n=sample_size, random_state=self.seed)
            
            system_prompt = f"""You are {self.disguise_as}. Study these examples of {self.disguise_as}'s responses and mimic the style, tone, formatting, and approach:

Examples:
"""
            for i, row in examples.iterrows():
                # Truncate response if too long
                response = row['target_response']
                if get_token_count(response) > self.max_tokens_per_example:
                    response = response[:self.max_tokens_per_example*4] + "...(truncated)"
                
                system_prompt += f"\nQ: {row['prompt']}\nA: {response}\n"
            
            system_prompt += f"\n\nNow respond to the following prompt in the same style as {self.disguise_as}. Match the formatting, tone, level of detail, and approach shown in the examples."

        if "gemma" in self.model.lower():
            formatted_prompt = f"""<start_of_turn>user
{system_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted_prompt}]
        else:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

class StylisticSystemPrompting(MethodBase):
    """
    Stylistic system prompting: focuses on surface-level stylistic features.
    Analyzes measurable style patterns (formatting, structure, length) and provides
    specific guidelines for replicating these surface-level features.
    """
    
    def __init__(self, model: str, disguise_as: str, disguise_df: pd.DataFrame = None,
                 num_examples: int = 3, use_examples: bool = True):
        super().__init__(model, disguise_as)
        self.disguise_df = disguise_df.copy() if disguise_df is not None else None
        self.num_examples = num_examples
        self.use_examples = use_examples
        self.stylistic_profile = None
        
        if disguise_df is not None:
            self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(get_token_count)
            self._generate_stylistic_profile()
    
    def _generate_stylistic_profile(self):
        """Generate stylistic profile focusing on measurable surface features."""
        # Import stylistic functions directly
        try:
            from ..stylistic_analysis import (
                has_markdown, contains_list, contains_header, contains_code, 
                contains_question, contains_exclamation, average_sentence_length
            )
        except ImportError:
            from stylistic_analysis import (
                has_markdown, contains_list, contains_header, contains_code, 
                contains_question, contains_exclamation, average_sentence_length
            )
        
        responses = self.disguise_df['target_response'].tolist()
        
        # Analyze stylistic patterns
        style_analysis = {
            'avg_length': np.mean([len(r.split()) for r in responses]),
            'uses_markdown': np.mean([has_markdown(r) for r in responses]) > 0.3,
            'uses_bullets': np.mean([contains_list(r) for r in responses]) > 0.3,
            'uses_code': np.mean([contains_code(r) for r in responses]) > 0.3,
            'uses_headers': np.mean([contains_header(r) for r in responses]) > 0.3,
            'uses_questions': np.mean([contains_question(r) for r in responses]) > 0.3,
            'uses_exclamations': np.mean([contains_exclamation(r) for r in responses]) > 0.3,
            'avg_sentence_len': np.mean([average_sentence_length(r) for r in responses]),
        }
        
        # Create stylistic guidelines
        guidelines = []
        
        # Length guidance
        if style_analysis['avg_length'] < 100:
            guidelines.append("Keep responses concise and brief (under 100 words typically)")
        elif style_analysis['avg_length'] > 300:
            guidelines.append("Provide detailed, comprehensive responses (typically 300+ words)")
        else:
            guidelines.append(f"Aim for moderate length responses (around {int(style_analysis['avg_length'])} words)")
        
        # Formatting patterns
        if style_analysis['uses_markdown']:
            guidelines.append("Use markdown formatting including headers (# ## ###)")
        
        if style_analysis['uses_bullets']:
            guidelines.append("Frequently use bullet points and lists for organization")
        
        if style_analysis['uses_code']:
            guidelines.append("Include code blocks and inline code formatting when relevant")
        
        if style_analysis['uses_headers']:
            guidelines.append("Use headers and section breaks to structure responses")
        
        # Communication patterns
        if style_analysis['uses_questions']:
            guidelines.append("Ask clarifying questions or rhetorical questions")
        
        if style_analysis['uses_exclamations']:
            guidelines.append("Use exclamation points for enthusiasm and emphasis")
        
        # Create final stylistic profile
        self.stylistic_profile = f"""Surface-level stylistic patterns for {self.disguise_as}:

{chr(10).join('• ' + g for g in guidelines)}

Key measurable characteristics:
- Average response length: {int(style_analysis['avg_length'])} words
- Average sentence length: {style_analysis['avg_sentence_len']:.1f} words
- Uses markdown formatting: {'Yes' if style_analysis['uses_markdown'] else 'No'}
- Uses bullet points: {'Yes' if style_analysis['uses_bullets'] else 'No'}  
- Includes code blocks: {'Yes' if style_analysis['uses_code'] else 'No'}
- Uses headers: {'Yes' if style_analysis['uses_headers'] else 'No'}

Focus on replicating these measurable surface features while maintaining content quality."""
    
    def forward(self, prompt: str) -> List[Dict[str, str]]:
        """Generate stylistic system prompt focusing on surface features."""
        base_instruction = f"You are {self.disguise_as}. Match these specific stylistic patterns:"
        
        if self.stylistic_profile:
            system_prompt = f"{base_instruction}\\n\\n{self.stylistic_profile}"
        else:
            system_prompt = f"{base_instruction}\\n\\nFocus on matching the formatting, structure, and surface-level style patterns of {self.disguise_as}."
        
        # Include examples for reference if enabled
        if self.use_examples and self.disguise_df is not None:
            examples = self.disguise_df.sample(min(self.num_examples, len(self.disguise_df)))
            system_prompt += f"\\n\\nReference examples showing these patterns:\\n"
            for i, row in examples.iterrows():
                truncated_response = row['target_response'][:300] + "..." if len(row['target_response']) > 300 else row['target_response']
                system_prompt += f"Q: {row['prompt'][:100]}...\\nA: {truncated_response}\\n\\n"
        
        system_prompt += f"\\nRespond to the following prompt matching these stylistic patterns exactly. Focus on surface features: formatting, length, structure, and presentation style."

        if "gemma" in self.model.lower():
            formatted_prompt = f"""<start_of_turn>user
{system_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
            return [{"role": "user", "content": formatted_prompt}]
        else:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
