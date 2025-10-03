#!/usr/bin/env python3
"""
LLM Response Classifier

This script implements a classification pipeline to identify which LLM generated a given response.
It uses various features extracted from the responses to train a classifier.

Features used:
- Response length statistics (chars, words, sentences)
- Vocabulary richness metrics
- Formatting patterns (markdown, code blocks, lists)
- Stylistic markers (formality, complexity)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import wandb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import spacy
from typing import List, Dict, Any
import re
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseFeatureExtractor:
    """Extract features from LLM responses for classification."""
    
    def __init__(self):
        """Initialize feature extractor with spacy model for NLP tasks."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spacy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_length_features(self, text: str) -> Dict[str, float]:
        """Extract length-based features from text."""
        doc = self.nlp(text)
        return {
            'char_length': len(text),
            'word_count': len(doc),
            'sentence_count': len(list(doc.sents)),
            'avg_word_length': np.mean([len(token.text) for token in doc]) if len(doc) > 0 else 0,
            'avg_sentence_length': np.mean([len(sent) for sent in doc.sents]) if len(list(doc.sents)) > 0 else 0
        }
    
    def extract_vocabulary_features(self, text: str) -> Dict[str, float]:
        """Extract vocabulary richness features."""
        doc = self.nlp(text)
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        unique_words = set(words)
        
        # Type-Token Ratio (vocabulary richness)
        ttr = len(unique_words) / len(words) if words else 0
        
        # POS tag distributions
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)
        
        return {
            'type_token_ratio': ttr,
            'unique_words_ratio': len(unique_words) / len(words) if words else 0,
            'noun_ratio': pos_counts['NOUN'] / total_tokens if total_tokens > 0 else 0,
            'verb_ratio': pos_counts['VERB'] / total_tokens if total_tokens > 0 else 0,
            'adj_ratio': pos_counts['ADJ'] / total_tokens if total_tokens > 0 else 0
        }
    
    def extract_formatting_features(self, text: str) -> Dict[str, float]:
        """Extract formatting pattern features."""
        return {
            'has_code_blocks': bool(re.search(r'```.*?```', text, re.DOTALL)),
            'has_inline_code': bool(re.search(r'`[^`]+`', text)),
            'has_bullet_points': bool(re.search(r'^\s*[-*]\s', text, re.MULTILINE)),
            'has_numbered_lists': bool(re.search(r'^\s*\d+\.\s', text, re.MULTILINE)),
            'has_markdown_headers': bool(re.search(r'^#+\s', text, re.MULTILINE)),
            'has_links': bool(re.search(r'\[.*?\]\(.*?\)', text))
        }
    
    def extract_stylistic_features(self, text: str) -> Dict[str, float]:
        """Extract stylistic markers."""
        doc = self.nlp(text)
        
        # Complexity markers
        complex_words = len([token for token in doc if len(token.text) > 6])
        sentences = list(doc.sents)
        words_per_sent = [len([token for token in sent if not token.is_punct]) for sent in sentences]
        
        return {
            'complex_words_ratio': complex_words / len(doc) if len(doc) > 0 else 0,
            'sentence_length_variance': np.var(words_per_sent) if words_per_sent else 0,
            'punctuation_ratio': len([token for token in doc if token.is_punct]) / len(doc) if len(doc) > 0 else 0
        }
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """Extract all features from text."""
        features = {}
        features.update(self.extract_length_features(text))
        features.update(self.extract_vocabulary_features(text))
        features.update(self.extract_formatting_features(text))
        features.update(self.extract_stylistic_features(text))
        return features

class LLMResponseClassifier:
    """Classifier for identifying source LLMs from responses."""
    
    def __init__(self, feature_extractor: ResponseFeatureExtractor):
        """Initialize classifier with feature extractor."""
        self.feature_extractor = feature_extractor
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.feature_names = None
        self.classes = None
    
    def prepare_features(self, responses: List[str]) -> pd.DataFrame:
        """Extract features from responses into DataFrame."""
        features = []
        for response in responses:
            features.append(self.feature_extractor.extract_all_features(response))
        return pd.DataFrame(features)
    
    def fit(self, responses: List[str], labels: List[str]):
        """Train classifier on responses and their source labels."""
        X = self.prepare_features(responses)
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier.fit(X_scaled, labels)
        self.classes = self.classifier.classes_
        
        # Get feature importances
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importances
    
    def predict(self, responses: List[str]) -> List[str]:
        """Predict LLM sources for responses."""
        X = self.prepare_features(responses)
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)
    
    def predict_proba(self, responses: List[str]) -> np.ndarray:
        """Get probability estimates for each class."""
        X = self.prepare_features(responses)
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

def load_data(data_dir: str) -> pd.DataFrame:
    """Load and combine response data from CSV files."""
    data_path = Path(data_dir)
    dfs = []
    
    for csv_file in data_path.glob("**/*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if 'model_response' in df.columns:
                model_name = csv_file.stem.split('-')[0]  # Extract model name from filename
                df['source_model'] = model_name
                dfs.append(df[['model_response', 'source_model']])
        except Exception as e:
            logger.warning(f"Error loading {csv_file}: {e}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate LLM response classifier")
    
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing response CSV files")
    parser.add_argument("--output_dir", type=str, default="results/classifier",
                       help="Output directory for results")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Fraction of data to use for testing")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading data...")
    df = load_data(args.data_dir)
    if df.empty:
        logger.error("No valid data found!")
        return
    
    # Initialize feature extractor and classifier
    feature_extractor = ResponseFeatureExtractor()
    classifier = LLMResponseClassifier(feature_extractor)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['model_response'].tolist(),
        df['source_model'].tolist(),
        test_size=args.test_size,
        random_state=42,
        stratify=df['source_model']
    )
    
    # Train classifier
    logger.info("Training classifier...")
    feature_importance = classifier.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating classifier...")
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    
    # Generate reports
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save feature importance
    feature_importance.to_csv(output_path / 'feature_importance.csv', index=False)
    
    # Save classification report
    with open(output_path / 'classification_report.txt', 'w') as f:
        f.write(classification_rep)
    
    # Save confusion matrix
    np.save(output_path / 'confusion_matrix.npy', conf_matrix)
    
    # Log to wandb
    if not args.no_wandb:
        wandb.init(project="llm-classifier", config=vars(args))
        
        # Log metrics
        wandb.log({
            "feature_importance": wandb.Table(dataframe=feature_importance),
            "classification_report": classification_rep,
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_test,
                preds=y_pred,
                class_names=classifier.classes
            )
        })
        
        wandb.finish()
    
    logger.info(f"Results saved to {output_path}")
    logger.info("\nClassification Report:\n" + classification_rep)

if __name__ == "__main__":
    main()
