import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import re
from methods.utils.math_style_extractor import extract_comprehensive_math_features

class AdaptiveExampleSelector:
    """
    Intelligently selects examples for disguise based on problem characteristics,
    difficulty, domain, and quality metrics.
    """
    
    def __init__(self, target_responses_df: pd.DataFrame, 
                 num_examples: int = 5,
                 diversity_weight: float = 0.3,
                 quality_weight: float = 0.4,
                 relevance_weight: float = 0.3):
        """
        Initialize the adaptive example selector.
        
        Args:
            target_responses_df: DataFrame with target model responses
            num_examples: Number of examples to select
            diversity_weight: Weight for diversity in selection
            quality_weight: Weight for response quality
            relevance_weight: Weight for relevance to current problem
        """
        self.target_df = target_responses_df.copy()
        self.num_examples = num_examples
        self.diversity_weight = diversity_weight
        self.quality_weight = quality_weight
        self.relevance_weight = relevance_weight
        
        # Extract features for all responses
        self._extract_all_features()
        
        # Calculate quality scores
        self._calculate_quality_scores()
        
        # Create difficulty clusters
        self._create_difficulty_clusters()
        
        # Create domain clusters
        self._create_domain_clusters()
    
    def _extract_all_features(self):
        """Extract mathematical features for all responses"""
        print("Extracting mathematical features...")
        features_list = []
        
        for idx, row in self.target_df.iterrows():
            features = extract_comprehensive_math_features(row['model_response'])
            features['response_id'] = idx
            features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        self.target_df = pd.concat([self.target_df, self.features_df], axis=1)
    
    def _calculate_quality_scores(self):
        """Calculate quality scores for each response"""
        # Quality indicators
        quality_indicators = [
            'has_step_numbering',
            'has_intermediate_calculations', 
            'has_problem_analysis',
            'math_symbol_count',
            'total_steps',
            'avg_step_length'
        ]
        
        # Normalize each indicator
        for indicator in quality_indicators:
            if indicator in self.target_df.columns:
                # Coerce to numeric (booleans -> 0/1, strings -> numeric if possible)
                col = pd.to_numeric(self.target_df[indicator], errors='coerce')
                col = col.fillna(0)
                max_val = col.max()
                if pd.isna(max_val) or max_val == 0:
                    self.target_df[f'{indicator}_normalized'] = 0.0
                else:
                    self.target_df[f'{indicator}_normalized'] = col / float(max_val)
        
        # Calculate overall quality score
        quality_cols = [col for col in self.target_df.columns if col.endswith('_normalized')]
        self.target_df['quality_score'] = self.target_df[quality_cols].mean(axis=1)
    
    def _create_difficulty_clusters(self):
        """Create clusters based on problem difficulty"""
        difficulty_features = [
            'total_steps', 'math_symbol_count', 'word_count', 'line_count',
            'fractions', 'equations', 'inequalities'
        ]
        
        # Filter to only include features that exist
        available_features = [f for f in difficulty_features if f in self.target_df.columns]
        
        if len(available_features) >= 2:
            # Standardize features
            scaler = StandardScaler()
            difficulty_data = scaler.fit_transform(self.target_df[available_features])
            
            # Create 3 difficulty clusters (easy, medium, hard)
            kmeans = KMeans(n_clusters=3, random_state=42)
            self.target_df['difficulty_cluster'] = kmeans.fit_predict(difficulty_data)
            
            # Sort clusters by difficulty (assuming cluster 0 is easiest)
            cluster_means = self.target_df.groupby('difficulty_cluster')[available_features].mean()
            difficulty_order = cluster_means.sum(axis=1).sort_values().index
            cluster_mapping = {old: new for new, old in enumerate(difficulty_order)}
            self.target_df['difficulty_cluster'] = self.target_df['difficulty_cluster'].map(cluster_mapping)
        else:
            self.target_df['difficulty_cluster'] = 0
    
    def _create_domain_clusters(self):
        """Create clusters based on mathematical domain"""
        domain_features = [
            'fractions', 'decimals', 'percentages', 'equations', 'inequalities',
            'units', 'conditional_logic', 'causal_connections'
        ]
        
        # Filter to only include features that exist
        available_features = [f for f in domain_features if f in self.target_df.columns]
        
        if len(available_features) >= 2:
            # Standardize features
            scaler = StandardScaler()
            domain_data = scaler.fit_transform(self.target_df[available_features])
            
            # Create domain clusters
            n_clusters = min(5, len(available_features))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.target_df['domain_cluster'] = kmeans.fit_predict(domain_data)
        else:
            self.target_df['domain_cluster'] = 0
    
    def _classify_problem(self, prompt: str) -> Dict[str, any]:
        """Classify the current problem to determine relevant examples"""
        # Simple keyword-based classification
        prompt_lower = prompt.lower()
        
        classification = {
            'has_fractions': any(word in prompt_lower for word in ['fraction', 'half', 'third', 'quarter']),
            'has_percentages': '%' in prompt or any(word in prompt_lower for word in ['percent', 'percentage']),
            'has_units': any(word in prompt_lower for word in ['miles', 'hours', 'dollars', 'pounds', 'meters', 'seconds']),
            'has_equations': '=' in prompt,
            'has_inequalities': any(symbol in prompt for symbol in ['<', '>', '≤', '≥']),
            'word_problem': len(prompt.split()) > 10,
            'has_geometry': any(word in prompt_lower for word in ['area', 'perimeter', 'volume', 'circle', 'square', 'triangle']),
            'has_algebra': any(word in prompt_lower for word in ['solve', 'equation', 'variable', 'unknown']),
        }
        
        return classification
    
    def _calculate_relevance_scores(self, prompt: str, candidate_indices: List[int]) -> np.ndarray:
        """Calculate relevance scores for candidate examples"""
        problem_classification = self._classify_problem(prompt)
        
        relevance_scores = []
        for idx in candidate_indices:
            response = self.target_df.loc[idx, 'model_response']
            response_classification = self._classify_problem(response)
            
            # Calculate similarity between problem and response classifications
            similarity = 0
            for key in problem_classification:
                if key in response_classification:
                    if problem_classification[key] == response_classification[key]:
                        similarity += 1
            
            # Normalize by number of classification features
            similarity = similarity / len(problem_classification)
            relevance_scores.append(similarity)
        
        return np.array(relevance_scores)
    
    def _calculate_diversity_scores(self, candidate_indices: List[int]) -> np.ndarray:
        """Calculate diversity scores to ensure variety in selected examples"""
        if len(candidate_indices) <= 1:
            return np.ones(len(candidate_indices))
        
        # Get features for candidates
        candidate_features = self.target_df.loc[candidate_indices, 
                                              [col for col in self.target_df.columns if not col.endswith(('_normalized', '_cluster', '_score'))]]
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(candidate_features)
        
        # Diversity score is inverse of average similarity to other candidates
        diversity_scores = []
        for i in range(len(candidate_indices)):
            # Exclude self-similarity
            other_similarities = [similarities[i][j] for j in range(len(candidate_indices)) if j != i]
            if other_similarities:
                avg_similarity = np.mean(other_similarities)
                diversity_scores.append(1 - avg_similarity)
            else:
                diversity_scores.append(1.0)
        
        return np.array(diversity_scores)
    
    def select_examples(self, prompt: str, 
                       ensure_difficulty_balance: bool = True,
                       ensure_domain_balance: bool = True) -> pd.DataFrame:
        """
        Select examples adaptively based on the current prompt.
        
        Args:
            prompt: The current problem prompt
            ensure_difficulty_balance: Whether to ensure examples cover different difficulty levels
            ensure_domain_balance: Whether to ensure examples cover different mathematical domains
            
        Returns:
            DataFrame with selected examples
        """
        # Initial candidate selection based on quality
        quality_threshold = self.target_df['quality_score'].quantile(0.7)
        high_quality_candidates = self.target_df[self.target_df['quality_score'] >= quality_threshold].index.tolist()
        
        if len(high_quality_candidates) < self.num_examples:
            # If not enough high-quality examples, use all
            high_quality_candidates = self.target_df.index.tolist()
        
        # If we need to ensure balance, implement stratified sampling
        if ensure_difficulty_balance and 'difficulty_cluster' in self.target_df.columns:
            selected_indices = self._stratified_selection(high_quality_candidates, 'difficulty_cluster')
        elif ensure_domain_balance and 'domain_cluster' in self.target_df.columns:
            selected_indices = self._stratified_selection(high_quality_candidates, 'domain_cluster')
        else:
            # Use weighted selection
            selected_indices = self._weighted_selection(high_quality_candidates, prompt)
        
        return self.target_df.loc[selected_indices]
    
    def _stratified_selection(self, candidates: List[int], cluster_column: str) -> List[int]:
        """Select examples ensuring representation from each cluster"""
        selected_indices = []
        clusters = self.target_df.loc[candidates, cluster_column].unique()
        
        # Calculate how many examples per cluster
        examples_per_cluster = self.num_examples // len(clusters)
        remaining = self.num_examples % len(clusters)
        
        for cluster in clusters:
            cluster_candidates = [idx for idx in candidates if self.target_df.loc[idx, cluster_column] == cluster]
            
            # Sort by quality score
            cluster_candidates.sort(key=lambda x: self.target_df.loc[x, 'quality_score'], reverse=True)
            
            # Select examples for this cluster
            n_select = examples_per_cluster + (1 if remaining > 0 else 0)
            selected_indices.extend(cluster_candidates[:n_select])
            remaining -= 1
        
        # If we still need more examples, add the highest quality remaining ones
        if len(selected_indices) < self.num_examples:
            remaining_candidates = [idx for idx in candidates if idx not in selected_indices]
            remaining_candidates.sort(key=lambda x: self.target_df.loc[x, 'quality_score'], reverse=True)
            selected_indices.extend(remaining_candidates[:self.num_examples - len(selected_indices)])
        
        return selected_indices[:self.num_examples]
    
    def _weighted_selection(self, candidates: List[int], prompt: str) -> List[int]:
        """Select examples using weighted scoring"""
        # Calculate scores for each candidate
        relevance_scores = self._calculate_relevance_scores(prompt, candidates)
        diversity_scores = self._calculate_diversity_scores(candidates)
        quality_scores = self.target_df.loc[candidates, 'quality_score'].values
        
        # Normalize scores
        relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min() + 1e-8)
        diversity_scores = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min() + 1e-8)
        quality_scores = (quality_scores - quality_scores.min()) / (quality_scores.max() - quality_scores.min() + 1e-8)
        
        # Calculate weighted scores
        weighted_scores = (self.relevance_weight * relevance_scores + 
                          self.diversity_weight * diversity_scores + 
                          self.quality_weight * quality_scores)
        
        # Select top examples
        top_indices = np.argsort(weighted_scores)[-self.num_examples:]
        return [candidates[i] for i in top_indices]
    
    def get_example_statistics(self) -> Dict[str, any]:
        """Get statistics about the available examples"""
        stats = {
            'total_examples': len(self.target_df),
            'quality_score_mean': self.target_df['quality_score'].mean(),
            'quality_score_std': self.target_df['quality_score'].std(),
            'difficulty_distribution': self.target_df['difficulty_cluster'].value_counts().to_dict() if 'difficulty_cluster' in self.target_df.columns else {},
            'domain_distribution': self.target_df['domain_cluster'].value_counts().to_dict() if 'domain_cluster' in self.target_df.columns else {},
        }
        return stats
