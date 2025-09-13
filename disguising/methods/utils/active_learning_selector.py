import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.optimize import minimize
import random
from methods.utils.math_style_extractor import extract_comprehensive_math_features, compare_math_style_similarity

class ThurstonianModel:
    """
    Thurstonian model for pairwise comparisons.
    Models the probability that response A is better than response B.
    """
    
    def __init__(self, n_responses: int):
        self.n_responses = n_responses
        self.mu = np.zeros(n_responses)  # Mean quality scores
        self.sigma_sq = np.ones(n_responses)  # Variance estimates
        self.comparison_matrix = np.zeros((n_responses, n_responses))
        self.comparison_count = np.zeros((n_responses, n_responses))
    
    def add_comparison(self, i: int, j: int, i_better: bool):
        """Add a pairwise comparison result"""
        self.comparison_count[i, j] += 1
        self.comparison_count[j, i] += 1
        
        if i_better:
            self.comparison_matrix[i, j] += 1
        else:
            self.comparison_matrix[j, i] += 1
    
    def fit(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """Fit the Thurstonian model using maximum likelihood estimation"""
        for iteration in range(max_iterations):
            old_mu = self.mu.copy()
            
            # Update mu using gradient descent
            for i in range(self.n_responses):
                gradient = 0
                for j in range(self.n_responses):
                    if self.comparison_count[i, j] > 0:
                        # Probability that i is better than j
                        prob_i_better = self._prob_i_better(i, j)
                        # Gradient contribution
                        gradient += (self.comparison_matrix[i, j] - 
                                   self.comparison_count[i, j] * prob_i_better) / np.sqrt(self.sigma_sq[i] + self.sigma_sq[j])
                
                # Update mu[i]
                self.mu[i] += 0.01 * gradient
            
            # Check convergence
            if np.max(np.abs(self.mu - old_mu)) < tolerance:
                break
    
    def _prob_i_better(self, i: int, j: int) -> float:
        """Calculate probability that response i is better than response j"""
        diff = self.mu[i] - self.mu[j]
        var_sum = self.sigma_sq[i] + self.sigma_sq[j]
        return 1 / (1 + np.exp(-diff / np.sqrt(var_sum)))
    
    def get_quality_scores(self) -> np.ndarray:
        """Get the estimated quality scores"""
        return self.mu
    
    def get_uncertainty(self, i: int, j: int) -> float:
        """Get uncertainty for a pairwise comparison"""
        diff = abs(self.mu[i] - self.mu[j])
        var_sum = self.sigma_sq[i] + self.sigma_sq[j]
        return diff / np.sqrt(var_sum)

class ActiveLearningSelector:
    """
    Iterative active learning for pairwise example selection.
    Uses the algorithm described in the user's query.
    """
    
    def __init__(self, target_responses_df: pd.DataFrame,
                 num_examples: int = 5,
                 d_regular: int = 3,
                 p_threshold: float = 0.1,  # Bottom P% of differences
                 q_threshold: float = 0.1,  # Bottom Q% of degree sums
                 batch_size: int = 10,
                 max_iterations: int = 5,
                 relaxation_factor: float = 1.2,
                 seed: int = None):
        """
        Initialize the active learning selector.
        
        Args:
            target_responses_df: DataFrame with target model responses
            num_examples: Number of examples to select
            d_regular: Degree for initial d-regular graph
            p_threshold: Threshold for difference filtering (0-1)
            q_threshold: Threshold for degree sum filtering (0-1)
            batch_size: Number of pairs to query per iteration
            max_iterations: Maximum number of iterations
            relaxation_factor: Factor to adjust thresholds if too few candidates
            seed: Random seed for reproducibility
        """
        self.target_df = target_responses_df.copy()
        self.num_examples = num_examples
        self.d_regular = d_regular
        self.p_threshold = p_threshold
        self.q_threshold = q_threshold
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.relaxation_factor = relaxation_factor
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize response indices
        self.response_indices = list(range(len(self.target_df)))
        self.n_responses = len(self.response_indices)
        
        # Initialize Thurstonian model
        self.thurstonian = ThurstonianModel(self.n_responses)
        
        # Track queried pairs
        self.queried_pairs: Set[Tuple[int, int]] = set()
        
        # Extract features for all responses
        self._extract_all_features()
        
        # Initialize with random d-regular graph
        self._initialize_d_regular_graph()
    
    def _extract_all_features(self):
        """Extract mathematical features for all responses"""
        print("Extracting mathematical features for active learning...")
        features_list = []
        
        for idx, row in self.target_df.iterrows():
            features = extract_comprehensive_math_features(row['model_response'])
            features['response_id'] = idx
            features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        self.target_df = pd.concat([self.target_df, self.features_df], axis=1)
    
    def _initialize_d_regular_graph(self):
        """Generate initial d-regular graph and query all pairs"""
        print(f"Initializing d-regular graph with degree {self.d_regular}...")
        
        # Create d-regular graph using NetworkX
        G = nx.random_regular_graph(self.d_regular, self.n_responses, seed=self.seed)
        
        # Query all initial pairs
        initial_pairs = list(G.edges())
        print(f"Querying {len(initial_pairs)} initial pairs...")
        
        for i, j in initial_pairs:
            self._query_pair(i, j)
            self.queried_pairs.add((i, j))
            self.queried_pairs.add((j, i))  # Add both directions
        
        # Fit initial Thurstonian model
        self.thurstonian.fit()
        print("Initial Thurstonian model fitted.")
    
    def _query_pair(self, i: int, j: int) -> bool:
        """
        Query which response is better between i and j.
        Returns True if i is better, False if j is better.
        """
        response_i = self.target_df.iloc[i]['model_response']
        response_j = self.target_df.iloc[j]['model_response']
        
        # Use mathematical style comparison
        similarity_i = self._calculate_quality_score(response_i)
        similarity_j = self._calculate_quality_score(response_j)
        
        # Add some noise to make it more realistic
        noise_i = np.random.normal(0, 0.1)
        noise_j = np.random.normal(0, 0.1)
        
        return (similarity_i + noise_i) > (similarity_j + noise_j)
    
    def _calculate_quality_score(self, response: str) -> float:
        """Calculate a quality score for a response"""
        features = extract_comprehensive_math_features(response)
        
        # Weighted quality indicators
        quality_indicators = [
            'has_step_numbering',
            'has_intermediate_calculations',
            'has_problem_analysis',
            'has_verification_steps',
            'math_symbol_count',
            'total_steps'
        ]
        
        score = 0
        weights = [0.2, 0.2, 0.2, 0.1, 0.15, 0.15]  # Corresponding weights
        
        for indicator, weight in zip(quality_indicators, weights):
            if indicator in features:
                value = features[indicator]
                if isinstance(value, bool):
                    score += weight * (1.0 if value else 0.0)
                elif isinstance(value, (int, float)):
                    # Normalize by typical ranges
                    if indicator == 'math_symbol_count':
                        score += weight * min(value / 20.0, 1.0)  # Cap at 20 symbols
                    elif indicator == 'total_steps':
                        score += weight * min(value / 10.0, 1.0)  # Cap at 10 steps
                    else:
                        score += weight * min(value, 1.0)
        
        return score
    
    def _get_candidate_pairs(self) -> List[Tuple[int, int]]:
        """Get all unsampled pairs"""
        candidates = []
        for i in range(self.n_responses):
            for j in range(i + 1, self.n_responses):
                if (i, j) not in self.queried_pairs:
                    candidates.append((i, j))
        return candidates
    
    def _calculate_differences_and_degrees(self, candidate_pairs: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate differences and degree sums for candidate pairs"""
        differences = []
        degree_sums = []
        
        for i, j in candidate_pairs:
            # Calculate difference in estimated quality
            diff = abs(self.thurstonian.mu[i] - self.thurstonian.mu[j])
            differences.append(diff)
            
            # Calculate sum of degrees (number of comparisons each has been in)
            degree_i = sum(1 for k in range(self.n_responses) if (i, k) in self.queried_pairs or (k, i) in self.queried_pairs)
            degree_j = sum(1 for k in range(self.n_responses) if (j, k) in self.queried_pairs or (k, j) in self.queried_pairs)
            degree_sums.append(degree_i + degree_j)
        
        return np.array(differences), np.array(degree_sums)
    
    def _select_pairs_for_query(self, candidate_pairs: List[Tuple[int, int]], 
                               differences: np.ndarray, degree_sums: np.ndarray) -> List[Tuple[int, int]]:
        """Select pairs to query based on the active learning criteria"""
        if len(candidate_pairs) == 0:
            return []
        
        # Calculate thresholds
        p_threshold_val = np.percentile(differences, self.p_threshold * 100)
        q_threshold_val = np.percentile(degree_sums, self.q_threshold * 100)
        
        # Filter pairs
        filtered_pairs = []
        for idx, (i, j) in enumerate(candidate_pairs):
            if (differences[idx] <= p_threshold_val and 
                degree_sums[idx] <= q_threshold_val):
                filtered_pairs.append((i, j))
        
        # If too few pairs, relax thresholds
        if len(filtered_pairs) < self.batch_size:
            print(f"Only {len(filtered_pairs)} pairs meet criteria, relaxing thresholds...")
            self.p_threshold *= self.relaxation_factor
            self.q_threshold *= self.relaxation_factor
            
            # Recalculate with relaxed thresholds
            p_threshold_val = np.percentile(differences, self.p_threshold * 100)
            q_threshold_val = np.percentile(degree_sums, self.q_threshold * 100)
            
            filtered_pairs = []
            for idx, (i, j) in enumerate(candidate_pairs):
                if (differences[idx] <= p_threshold_val and 
                    degree_sums[idx] <= q_threshold_val):
                    filtered_pairs.append((i, j))
        
        # Select up to batch_size pairs
        if len(filtered_pairs) >= self.batch_size:
            selected_pairs = random.sample(filtered_pairs, self.batch_size)
        else:
            selected_pairs = filtered_pairs
            # If still not enough, add random pairs from remaining candidates
            remaining = [pair for pair in candidate_pairs if pair not in selected_pairs]
            needed = self.batch_size - len(selected_pairs)
            if len(remaining) >= needed:
                selected_pairs.extend(random.sample(remaining, needed))
            else:
                selected_pairs.extend(remaining)
        
        return selected_pairs
    
    def run_active_learning(self) -> np.ndarray:
        """Run the iterative active learning algorithm"""
        print(f"Starting active learning with {self.max_iterations} iterations...")
        
        for iteration in range(self.max_iterations):
            print(f"\nIteration {iteration + 1}/{self.max_iterations}")
            
            # Get candidate pairs
            candidate_pairs = self._get_candidate_pairs()
            print(f"Found {len(candidate_pairs)} candidate pairs")
            
            if len(candidate_pairs) == 0:
                print("No more candidate pairs, stopping.")
                break
            
            # Calculate differences and degree sums
            differences, degree_sums = self._calculate_differences_and_degrees(candidate_pairs)
            
            # Select pairs to query
            selected_pairs = self._select_pairs_for_query(candidate_pairs, differences, degree_sums)
            print(f"Selected {len(selected_pairs)} pairs to query")
            
            if len(selected_pairs) == 0:
                print("No pairs selected, stopping.")
                break
            
            # Query selected pairs
            for i, j in selected_pairs:
                i_better = self._query_pair(i, j)
                self.thurstonian.add_comparison(i, j, i_better)
                self.queried_pairs.add((i, j))
                self.queried_pairs.add((j, i))
            
            # Refit Thurstonian model
            print("Refitting Thurstonian model...")
            self.thurstonian.fit()
            
            # Print current quality scores
            quality_scores = self.thurstonian.get_quality_scores()
            print(f"Quality score range: {quality_scores.min():.3f} - {quality_scores.max():.3f}")
        
        print("Active learning completed.")
        return self.thurstonian.get_quality_scores()
    
    def select_examples(self, prompt: str = None) -> pd.DataFrame:
        """
        Select examples using the active learning approach.
        
        Args:
            prompt: The current problem prompt (not used in this method)
            
        Returns:
            DataFrame with selected examples
        """
        # Run active learning if not already done
        if len(self.queried_pairs) == 0:
            self.run_active_learning()
        
        # Get quality scores
        quality_scores = self.thurstonian.get_quality_scores()
        
        # Select top examples
        top_indices = np.argsort(quality_scores)[-self.num_examples:]
        
        return self.target_df.iloc[top_indices]
    
    def get_selection_statistics(self) -> Dict[str, any]:
        """Get statistics about the selection process"""
        quality_scores = self.thurstonian.get_quality_scores()
        
        stats = {
            'total_responses': self.n_responses,
            'total_queries': len(self.queried_pairs) // 2,  # Divide by 2 since we store both directions
            'quality_score_mean': quality_scores.mean(),
            'quality_score_std': quality_scores.std(),
            'quality_score_range': (quality_scores.min(), quality_scores.max()),
            'selected_examples': self.num_examples,
            'p_threshold_final': self.p_threshold,
            'q_threshold_final': self.q_threshold
        }
        
        return stats
    
    def get_uncertainty_analysis(self) -> Dict[str, any]:
        """Analyze uncertainty in the pairwise comparisons"""
        uncertainties = []
        for i in range(self.n_responses):
            for j in range(i + 1, self.n_responses):
                if (i, j) in self.queried_pairs:
                    uncertainty = self.thurstonian.get_uncertainty(i, j)
                    uncertainties.append(uncertainty)
        
        if uncertainties:
            return {
                'mean_uncertainty': np.mean(uncertainties),
                'std_uncertainty': np.std(uncertainties),
                'min_uncertainty': np.min(uncertainties),
                'max_uncertainty': np.max(uncertainties)
            }
        else:
            return {'mean_uncertainty': 0, 'std_uncertainty': 0, 'min_uncertainty': 0, 'max_uncertainty': 0}
