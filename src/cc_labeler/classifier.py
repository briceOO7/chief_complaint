"""
Main CEDIS classifier.

Orchestrates preprocessing, rule-based matching, and text similarity matching
to classify chief complaints into CEDIS codes.
"""

import pandas as pd
from typing import List, Dict, Optional, Union, Tuple

from .preprocessing import ChiefComplaintPreprocessor
from .rule_matcher import RuleMatcher
from .matchers import EnsembleMatcher
from .cedis_codes import CEDIS_CODES, get_cedis_by_code


class CEDISClassifier:
    """
    Complete CEDIS chief complaint classifier.
    
    Uses hierarchical classification approach:
    1. Rule-based matching for high-confidence cases
    2. Text similarity matching (fuzzy + TF-IDF) for others
    3. Ensemble scoring to combine evidence
    """
    
    def __init__(self, 
                 rule_confidence_threshold: float = 0.75,
                 use_fuzzy: bool = True,
                 use_tfidf: bool = True,
                 fuzzy_weight: float = 0.4,
                 tfidf_weight: float = 0.6):
        """
        Initialize CEDIS classifier.
        
        Args:
            rule_confidence_threshold: Minimum confidence to accept rule-based match
            use_fuzzy: Enable fuzzy string matching
            use_tfidf: Enable TF-IDF semantic matching
            fuzzy_weight: Weight for fuzzy matching in ensemble
            tfidf_weight: Weight for TF-IDF matching in ensemble
        """
        self.rule_confidence_threshold = rule_confidence_threshold
        
        # Initialize components
        self.preprocessor = ChiefComplaintPreprocessor()
        self.rule_matcher = RuleMatcher()
        self.ensemble_matcher = EnsembleMatcher(
            use_fuzzy=use_fuzzy,
            use_tfidf=use_tfidf,
            fuzzy_weight=fuzzy_weight,
            tfidf_weight=tfidf_weight,
        )
    
    def classify(self, 
                 text: str, 
                 return_top_k: int = 3,
                 min_confidence: float = 0.1) -> List[Dict]:
        """
        Classify a single chief complaint.
        
        Args:
            text: Raw chief complaint text
            return_top_k: Number of top predictions to return
            min_confidence: Minimum confidence threshold for predictions
            
        Returns:
            List of prediction dictionaries with keys:
                - cedis_code: CEDIS code
                - description: CEDIS description
                - category: CEDIS category
                - confidence: Confidence score (0-1)
                - method: Classification method used
                - details: Additional method-specific details
        """
        if not text or pd.isna(text):
            return []
        
        # Preprocess
        processed = self.preprocessor.preprocess(text)
        normalized_text = processed['normalized']
        processed_text = processed['processed']
        
        # Try rule-based matching first
        rule_match = self.rule_matcher.get_best_match(
            normalized_text, 
            min_confidence=self.rule_confidence_threshold
        )
        
        results = []
        
        if rule_match:
            # High-confidence rule match found
            code, confidence, rule_name = rule_match
            entry = get_cedis_by_code(code)
            
            results.append({
                'cedis_code': code,
                'description': entry['description'],
                'category': entry['category'],
                'confidence': confidence,
                'method': 'rule',
                'details': {'rule_name': rule_name}
            })
            
            # If rule confidence is very high, can return early
            if confidence >= 0.90:
                return results[:return_top_k]
        
        # Apply ensemble matching for additional candidates
        ensemble_matches = self.ensemble_matcher.match(
            processed_text,
            top_k=return_top_k * 2  # Get extra to filter
        )
        
        for code, combined_score, component_scores in ensemble_matches:
            # Skip if we already have this code from rules
            if results and results[0]['cedis_code'] == code:
                continue
            
            entry = get_cedis_by_code(code)
            
            results.append({
                'cedis_code': code,
                'description': entry['description'],
                'category': entry['category'],
                'confidence': combined_score,
                'method': 'ensemble',
                'details': {
                    'fuzzy_score': component_scores.get('fuzzy'),
                    'tfidf_score': component_scores.get('tfidf'),
                }
            })
        
        # Filter by minimum confidence and return top-k
        results = [r for r in results if r['confidence'] >= min_confidence]
        results = results[:return_top_k]
        
        return results
    
    def classify_batch(self, 
                       texts: List[str],
                       return_top_k: int = 3,
                       min_confidence: float = 0.1,
                       show_progress: bool = True) -> List[List[Dict]]:
        """
        Classify a batch of chief complaints.
        
        Args:
            texts: List of raw chief complaint texts
            return_top_k: Number of top predictions per complaint
            min_confidence: Minimum confidence threshold
            show_progress: Whether to show progress bar (requires tqdm)
            
        Returns:
            List of prediction lists (one per input text)
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Classifying")
            except ImportError:
                iterator = texts
        else:
            iterator = texts
        
        results = []
        for text in iterator:
            predictions = self.classify(
                text,
                return_top_k=return_top_k,
                min_confidence=min_confidence
            )
            results.append(predictions)
        
        return results
    
    def classify_dataframe(self,
                          df: pd.DataFrame,
                          text_column: str,
                          return_top_k: int = 1,
                          add_columns: bool = True) -> pd.DataFrame:
        """
        Classify chief complaints in a DataFrame.
        
        Args:
            df: DataFrame with chief complaint column
            text_column: Name of column containing chief complaints
            return_top_k: Number of predictions to return
            add_columns: If True, add classification columns to df
            
        Returns:
            DataFrame with classification results
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Classify all texts
        all_predictions = self.classify_batch(
            df[text_column].tolist(),
            return_top_k=return_top_k,
            min_confidence=0.0,  # Include all predictions
            show_progress=True
        )
        
        if not add_columns:
            return df
        
        # Add columns for top prediction
        df = df.copy()
        
        for i in range(return_top_k):
            suffix = f"_{i+1}" if return_top_k > 1 else ""
            
            codes = []
            descriptions = []
            categories = []
            confidences = []
            methods = []
            
            for predictions in all_predictions:
                if len(predictions) > i:
                    pred = predictions[i]
                    codes.append(pred['cedis_code'])
                    descriptions.append(pred['description'])
                    categories.append(pred['category'])
                    confidences.append(pred['confidence'])
                    methods.append(pred['method'])
                else:
                    codes.append(None)
                    descriptions.append(None)
                    categories.append(None)
                    confidences.append(None)
                    methods.append(None)
            
            df[f'cedis_code{suffix}'] = codes
            df[f'cedis_description{suffix}'] = descriptions
            df[f'cedis_category{suffix}'] = categories
            df[f'cedis_confidence{suffix}'] = confidences
            df[f'cedis_method{suffix}'] = methods
        
        return df
    
    def evaluate_prediction(self, 
                           text: str, 
                           true_code: str,
                           return_top_k: int = 3) -> Dict:
        """
        Evaluate classification against ground truth.
        
        Args:
            text: Chief complaint text
            true_code: True CEDIS code
            return_top_k: Number of predictions to consider
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.classify(text, return_top_k=return_top_k)
        
        if not predictions:
            return {
                'text': text,
                'true_code': true_code,
                'predicted_codes': [],
                'correct': False,
                'rank': None,
                'confidence': None,
            }
        
        predicted_codes = [p['cedis_code'] for p in predictions]
        
        # Check if true code is in predictions
        correct = true_code in predicted_codes
        rank = predicted_codes.index(true_code) + 1 if correct else None
        
        # Get confidence of true code if predicted
        confidence = None
        if correct:
            for pred in predictions:
                if pred['cedis_code'] == true_code:
                    confidence = pred['confidence']
                    break
        
        return {
            'text': text,
            'true_code': true_code,
            'predicted_codes': predicted_codes,
            'top_prediction': predicted_codes[0] if predicted_codes else None,
            'correct': correct,
            'top1_correct': predicted_codes[0] == true_code if predicted_codes else False,
            'rank': rank,
            'confidence': confidence,
            'predictions': predictions,
        }
    
    def get_classification_summary(self, text: str) -> Dict:
        """
        Get detailed classification information for debugging/analysis.
        
        Args:
            text: Chief complaint text
            
        Returns:
            Dictionary with preprocessing results, rule matches, and predictions
        """
        # Preprocess
        processed = self.preprocessor.preprocess(
            text, 
            preserve_demographics=True,
            split_compounds=True
        )
        
        # Get rule matches (all, not just best)
        rule_matches = self.rule_matcher.match(processed['normalized'])
        
        # Get ensemble matches
        ensemble_matches = self.ensemble_matcher.match(processed['processed'], top_k=10)
        
        # Get final classification
        predictions = self.classify(text, return_top_k=5)
        
        return {
            'original_text': text,
            'preprocessing': processed,
            'rule_matches': [
                {
                    'code': code,
                    'confidence': conf,
                    'rule': rule_name,
                    'description': CEDIS_CODES[code]['description']
                }
                for code, conf, rule_name in rule_matches
            ],
            'ensemble_matches': [
                {
                    'code': code,
                    'score': score,
                    'details': details,
                    'description': CEDIS_CODES[code]['description']
                }
                for code, score, details in ensemble_matches[:10]
            ],
            'final_predictions': predictions,
        }
