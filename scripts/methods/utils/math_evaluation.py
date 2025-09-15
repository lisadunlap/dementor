import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from math_style_extractor import extract_comprehensive_math_features, compare_math_style_similarity
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class MathDisguiseEvaluator:
    """
    Comprehensive evaluation of mathematical disguise effectiveness,
    focusing on math-specific features and reasoning patterns.
    """
    
    def __init__(self, source_responses: List[str], 
                 target_responses: List[str], 
                 disguised_responses: List[str]):
        """
        Initialize the math disguise evaluator.
        
        Args:
            source_responses: Original responses from source model
            target_responses: Target responses to mimic
            disguised_responses: Disguised responses from source model
        """
        self.source_responses = source_responses
        self.target_responses = target_responses
        self.disguised_responses = disguised_responses
        
        # Extract features for all responses
        self._extract_all_features()
        
        # Calculate evaluation metrics
        self.evaluation_results = self._calculate_all_metrics()
    
    def _extract_all_features(self):
        """Extract mathematical features for all response sets"""
        print("Extracting mathematical features for evaluation...")
        
        # Extract features for each response set
        self.source_features = [extract_comprehensive_math_features(resp) for resp in self.source_responses]
        self.target_features = [extract_comprehensive_math_features(resp) for resp in self.target_responses]
        self.disguised_features = [extract_comprehensive_math_features(resp) for resp in self.disguised_responses]
        
        # Convert to DataFrames
        self.source_df = pd.DataFrame(self.source_features)
        self.target_df = pd.DataFrame(self.source_features)
        self.disguised_df = pd.DataFrame(self.disguised_features)
    
    def _calculate_all_metrics(self) -> Dict[str, any]:
        """Calculate all evaluation metrics"""
        metrics = {}
        
        # 1. Mathematical Structure Similarity
        metrics['structure_similarity'] = self._calculate_structure_similarity()
        
        # 2. Reasoning Pattern Similarity
        metrics['reasoning_similarity'] = self._calculate_reasoning_similarity()
        
        # 3. Mathematical Notation Similarity
        metrics['notation_similarity'] = self._calculate_notation_similarity()
        
        # 4. Problem Breakdown Similarity
        metrics['breakdown_similarity'] = self._calculate_breakdown_similarity()
        
        # 5. Overall Math Style Similarity
        metrics['overall_math_similarity'] = self._calculate_overall_math_similarity()
        
        # 6. Disguise Effectiveness Score
        metrics['disguise_effectiveness'] = self._calculate_disguise_effectiveness()
        
        # 7. Feature-by-feature Analysis
        metrics['feature_analysis'] = self._analyze_individual_features()
        
        # 8. Statistical Significance Tests
        metrics['statistical_significance'] = self._calculate_statistical_significance()
        
        return metrics
    
    def _calculate_structure_similarity(self) -> Dict[str, float]:
        """Calculate similarity in mathematical structure"""
        structure_features = [
            'has_step_numbering', 'total_steps', 'avg_step_length',
            'line_count', 'word_count'
        ]
        
        # Calculate similarities
        source_target_sim = self._calculate_feature_similarity(
            self.source_df[structure_features], 
            self.target_df[structure_features]
        )
        
        disguised_target_sim = self._calculate_feature_similarity(
            self.disguised_df[structure_features], 
            self.target_df[structure_features]
        )
        
        improvement = disguised_target_sim - source_target_sim
        
        return {
            'source_to_target': source_target_sim,
            'disguised_to_target': disguised_target_sim,
            'improvement': improvement,
            'improvement_ratio': improvement / (source_target_sim + 1e-8)
        }
    
    def _calculate_reasoning_similarity(self) -> Dict[str, float]:
        """Calculate similarity in reasoning patterns"""
        reasoning_features = [
            'has_problem_analysis', 'has_verification_steps',
            'conditional_logic', 'causal_connections', 'comparisons',
            'definitions', 'examples'
        ]
        
        # Filter to only include features that exist
        available_features = [f for f in reasoning_features if f in self.source_df.columns]
        
        if len(available_features) < 2:
            return {'error': 'Insufficient reasoning features available'}
        
        # Calculate similarities
        source_target_sim = self._calculate_feature_similarity(
            self.source_df[available_features], 
            self.target_df[available_features]
        )
        
        disguised_target_sim = self._calculate_feature_similarity(
            self.disguised_df[available_features], 
            self.target_df[available_features]
        )
        
        improvement = disguised_target_sim - source_target_sim
        
        return {
            'source_to_target': source_target_sim,
            'disguised_to_target': disguised_target_sim,
            'improvement': improvement,
            'improvement_ratio': improvement / (source_target_sim + 1e-8)
        }
    
    def _calculate_notation_similarity(self) -> Dict[str, float]:
        """Calculate similarity in mathematical notation usage"""
        notation_features = [
            'math_symbol_count', 'fractions', 'decimals', 'percentages',
            'equations', 'inequalities', 'parentheses', 'brackets'
        ]
        
        # Filter to only include features that exist
        available_features = [f for f in notation_features if f in self.source_df.columns]
        
        if len(available_features) < 2:
            return {'error': 'Insufficient notation features available'}
        
        # Calculate similarities
        source_target_sim = self._calculate_feature_similarity(
            self.source_df[available_features], 
            self.target_df[available_features]
        )
        
        disguised_target_sim = self._calculate_feature_similarity(
            self.disguised_df[available_features], 
            self.target_df[available_features]
        )
        
        improvement = disguised_target_sim - source_target_sim
        
        return {
            'source_to_target': source_target_sim,
            'disguised_to_target': disguised_target_sim,
            'improvement': improvement,
            'improvement_ratio': improvement / (source_target_sim + 1e-8)
        }
    
    def _calculate_breakdown_similarity(self) -> Dict[str, float]:
        """Calculate similarity in problem breakdown approach"""
        breakdown_features = [
            'given_info', 'what_to_find', 'assumptions', 'constraints', 'units'
        ]
        
        # Filter to only include features that exist
        available_features = [f for f in breakdown_features if f in self.source_df.columns]
        
        if len(available_features) < 2:
            return {'error': 'Insufficient breakdown features available'}
        
        # Calculate similarities
        source_target_sim = self._calculate_feature_similarity(
            self.source_df[available_features], 
            self.target_df[available_features]
        )
        
        disguised_target_sim = self._calculate_feature_similarity(
            self.disguised_df[available_features], 
            self.target_df[available_features]
        )
        
        improvement = disguised_target_sim - source_target_sim
        
        return {
            'source_to_target': source_target_sim,
            'disguised_to_target': disguised_target_sim,
            'improvement': improvement,
            'improvement_ratio': improvement / (source_target_sim + 1e-8)
        }
    
    def _calculate_overall_math_similarity(self) -> Dict[str, float]:
        """Calculate overall mathematical style similarity"""
        # Get all numerical features
        numerical_features = self.source_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_features) < 2:
            return {'error': 'Insufficient numerical features available'}
        
        # Calculate similarities
        source_target_sim = self._calculate_feature_similarity(
            self.source_df[numerical_features], 
            self.target_df[numerical_features]
        )
        
        disguised_target_sim = self._calculate_feature_similarity(
            self.disguised_df[numerical_features], 
            self.target_df[numerical_features]
        )
        
        improvement = disguised_target_sim - source_target_sim
        
        return {
            'source_to_target': source_target_sim,
            'disguised_to_target': disguised_target_sim,
            'improvement': improvement,
            'improvement_ratio': improvement / (source_target_sim + 1e-8)
        }
    
    def _calculate_disguise_effectiveness(self) -> Dict[str, float]:
        """Calculate overall disguise effectiveness score"""
        # Weight different aspects
        weights = {
            'structure': 0.25,
            'reasoning': 0.30,
            'notation': 0.20,
            'breakdown': 0.15,
            'overall': 0.10
        }
        
        effectiveness_score = 0
        component_scores = {}
        
        for component, weight in weights.items():
            if component in self.evaluation_results:
                result = self.evaluation_results[component]
                if 'improvement_ratio' in result and not isinstance(result, dict):
                    score = result['improvement_ratio']
                    component_scores[component] = score
                    effectiveness_score += weight * score
        
        return {
            'overall_effectiveness': effectiveness_score,
            'component_scores': component_scores,
            'weights_used': weights
        }
    
    def _analyze_individual_features(self) -> Dict[str, any]:
        """Analyze individual feature improvements"""
        feature_analysis = {}
        
        # Get all numerical features
        numerical_features = self.source_df.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in numerical_features:
            if feature in self.source_df.columns and feature in self.target_df.columns and feature in self.disguised_df.columns:
                source_vals = self.source_df[feature].values
                target_vals = self.target_df[feature].values
                disguised_vals = self.disguised_df[feature].values
                
                # Calculate distances
                source_target_dist = np.mean(np.abs(source_vals - target_vals))
                disguised_target_dist = np.mean(np.abs(disguised_vals - target_vals))
                
                improvement = source_target_dist - disguised_target_dist
                
                feature_analysis[feature] = {
                    'source_target_distance': source_target_dist,
                    'disguised_target_distance': disguised_target_dist,
                    'improvement': improvement,
                    'improvement_ratio': improvement / (source_target_dist + 1e-8)
                }
        
        return feature_analysis
    
    def _calculate_statistical_significance(self) -> Dict[str, any]:
        """Calculate statistical significance of improvements"""
        significance_results = {}
        
        # Get overall improvement scores
        improvement_scores = []
        for i in range(len(self.source_responses)):
            source_sim = compare_math_style_similarity(
                self.source_responses[i], 
                self.target_responses[i]
            )
            disguised_sim = compare_math_style_similarity(
                self.disguised_responses[i], 
                self.target_responses[i]
            )
            improvement = disguised_sim - source_sim
            improvement_scores.append(improvement)
        
        improvement_scores = np.array(improvement_scores)
        
        # Basic statistics
        mean_improvement = np.mean(improvement_scores)
        std_improvement = np.std(improvement_scores)
        
        # T-test for significance (assuming we want to test if improvement > 0)
        from scipy import stats
        try:
            t_stat, p_value = stats.ttest_1samp(improvement_scores, 0)
            significant = p_value < 0.05 and mean_improvement > 0
        except:
            t_stat, p_value = np.nan, np.nan
            significant = False
        
        # Effect size (Cohen's d)
        if std_improvement > 0:
            effect_size = mean_improvement / std_improvement
        else:
            effect_size = 0
        
        significance_results = {
            'mean_improvement': mean_improvement,
            'std_improvement': std_improvement,
            't_statistic': t_stat,
            'p_value': p_value,
            'statistically_significant': significant,
            'effect_size': effect_size,
            'effect_size_interpretation': self._interpret_effect_size(effect_size)
        }
        
        return significance_results
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size"""
        if abs(effect_size) < 0.2:
            return "negligible"
        elif abs(effect_size) < 0.5:
            return "small"
        elif abs(effect_size) < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_feature_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """Calculate cosine similarity between two feature matrices"""
        # Handle missing values
        df1_clean = df1.fillna(0)
        df2_clean = df2.fillna(0)
        
        # Ensure same columns
        common_cols = df1_clean.columns.intersection(df2_clean.columns)
        if len(common_cols) < 2:
            return 0.0
        
        df1_clean = df1_clean[common_cols]
        df2_clean = df2_clean[common_cols]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(df1_clean, df2_clean)
        
        # Return mean similarity (diagonal elements)
        return np.mean(np.diag(similarities))
    
    def get_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        report = "=== MATH DISGUISE EVALUATION REPORT ===\n\n"
        
        # Overall effectiveness
        effectiveness = self.evaluation_results['disguise_effectiveness']
        report += f"Overall Disguise Effectiveness: {effectiveness['overall_effectiveness']:.3f}\n\n"
        
        # Component breakdown
        report += "Component Scores:\n"
        for component, score in effectiveness['component_scores'].items():
            report += f"  {component.capitalize()}: {score:.3f}\n"
        
        report += "\n"
        
        # Statistical significance
        stats = self.evaluation_results['statistical_significance']
        report += f"Statistical Significance:\n"
        report += f"  Mean Improvement: {stats['mean_improvement']:.3f}\n"
        report += f"  P-value: {stats['p_value']:.4f}\n"
        report += f"  Statistically Significant: {stats['statistically_significant']}\n"
        report += f"  Effect Size: {stats['effect_size']:.3f} ({stats['effect_size_interpretation']})\n\n"
        
        # Top improvements
        feature_analysis = self.evaluation_results['feature_analysis']
        improvements = [(feature, data['improvement_ratio']) 
                       for feature, data in feature_analysis.items() 
                       if 'improvement_ratio' in data]
        improvements.sort(key=lambda x: x[1], reverse=True)
        
        report += "Top 5 Feature Improvements:\n"
        for i, (feature, improvement) in enumerate(improvements[:5]):
            report += f"  {i+1}. {feature}: {improvement:.3f}\n"
        
        return report
    
    def save_evaluation_results(self, filepath: str):
        """Save evaluation results to a file"""
        # Convert results to a more serializable format
        serializable_results = {}
        
        for key, value in self.evaluation_results.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.integer):
                serializable_results[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        # Save to JSON
        import json
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Evaluation results saved to {filepath}")
    
    def get_visualization_data(self) -> Dict[str, any]:
        """Get data formatted for visualization"""
        viz_data = {
            'component_scores': self.evaluation_results['disguise_effectiveness']['component_scores'],
            'feature_improvements': {},
            'statistical_summary': {
                'mean_improvement': self.evaluation_results['statistical_significance']['mean_improvement'],
                'p_value': self.evaluation_results['statistical_significance']['p_value'],
                'effect_size': self.evaluation_results['statistical_significance']['effect_size']
            }
        }
        
        # Add feature improvements
        feature_analysis = self.evaluation_results['feature_analysis']
        for feature, data in feature_analysis.items():
            if 'improvement_ratio' in data:
                viz_data['feature_improvements'][feature] = data['improvement_ratio']
        
        return viz_data
