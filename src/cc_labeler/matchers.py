"""
Text matching algorithms for CEDIS classification.

Implements:
- Fuzzy string matching using rapidfuzz
- TF-IDF semantic similarity using scikit-learn
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not available. Fuzzy matching will be disabled.")

from .cedis_codes import CEDIS_CODES


class FuzzyMatcher:
    """Fuzzy string matching for CEDIS codes."""
    
    def __init__(self):
        """Initialize fuzzy matcher."""
        if not RAPIDFUZZ_AVAILABLE:
            raise ImportError(
                "rapidfuzz is required for fuzzy matching. "
                "Install with: pip install rapidfuzz"
            )
        
        # Build searchable corpus from CEDIS descriptions
        self.corpus = self._build_corpus()
    
    def _build_corpus(self) -> Dict[str, str]:
        """
        Build searchable corpus of CEDIS descriptions.
        
        Returns:
            Dictionary mapping CEDIS codes to descriptions
        """
        return {
            code: entry['description'].lower()
            for code, entry in CEDIS_CODES.items()
        }
    
    def match(self, text: str, top_k: int = 5, min_score: float = 60.0) -> List[Tuple[str, float]]:
        """
        Find best fuzzy matches for text against CEDIS descriptions.
        
        Args:
            text: Query text (chief complaint)
            top_k: Number of top matches to return
            min_score: Minimum fuzzy match score (0-100)
            
        Returns:
            List of tuples (cedis_code, score) sorted by score descending
        """
        if not text:
            return []
        
        text_lower = text.lower()
        
        # Use rapidfuzz's process.extract for efficient matching
        results = process.extract(
            text_lower,
            self.corpus,
            scorer=fuzz.token_sort_ratio,
            limit=top_k
        )
        
        # Filter by minimum score and format results
        matches = [
            (cedis_code, score / 100.0)  # Normalize to 0-1
            for description, score, cedis_code in results
            if score >= min_score
        ]
        
        return matches
    
    def match_partial(self, text: str, top_k: int = 5, min_score: float = 50.0) -> List[Tuple[str, float]]:
        """
        Find best partial fuzzy matches (more lenient).
        
        Uses partial ratio matching which is more forgiving of extra text.
        
        Args:
            text: Query text (chief complaint)
            top_k: Number of top matches to return
            min_score: Minimum fuzzy match score (0-100)
            
        Returns:
            List of tuples (cedis_code, score) sorted by score descending
        """
        if not text:
            return []
        
        text_lower = text.lower()
        
        results = process.extract(
            text_lower,
            self.corpus,
            scorer=fuzz.partial_ratio,
            limit=top_k
        )
        
        matches = [
            (cedis_code, score / 100.0)
            for description, score, cedis_code in results
            if score >= min_score
        ]
        
        return matches


class TFIDFMatcher:
    """TF-IDF based semantic similarity matching for CEDIS codes."""
    
    def __init__(self, max_features: int = 500):
        """
        Initialize TF-IDF matcher.
        
        Args:
            max_features: Maximum number of features for TF-IDF
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
            min_df=1,
            stop_words=None,  # We handle stopwords in preprocessing
            lowercase=True,
        )
        
        # Build corpus and fit vectorizer
        self.cedis_codes, self.descriptions = self._build_corpus()
        self.description_vectors = self.vectorizer.fit_transform(self.descriptions)
    
    def _build_corpus(self) -> Tuple[List[str], List[str]]:
        """
        Build corpus of CEDIS descriptions for TF-IDF.
        
        Returns:
            Tuple of (codes list, descriptions list)
        """
        codes = []
        descriptions = []
        
        for code, entry in sorted(CEDIS_CODES.items()):
            codes.append(code)
            # Combine description with category for better context
            desc_text = f"{entry['category']} {entry['description']}"
            descriptions.append(desc_text.lower())
        
        return codes, descriptions
    
    def match(self, text: str, top_k: int = 5, min_similarity: float = 0.1) -> List[Tuple[str, float]]:
        """
        Find best TF-IDF similarity matches for text.
        
        Args:
            text: Query text (chief complaint)
            top_k: Number of top matches to return
            min_similarity: Minimum cosine similarity (0-1)
            
        Returns:
            List of tuples (cedis_code, similarity) sorted by similarity descending
        """
        if not text:
            return []
        
        # Vectorize query text
        query_vector = self.vectorizer.transform([text.lower()])
        
        # Compute cosine similarity with all CEDIS descriptions
        similarities = cosine_similarity(query_vector, self.description_vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by minimum similarity and format results
        matches = [
            (self.cedis_codes[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] >= min_similarity
        ]
        
        return matches
    
    def match_with_category_boost(self, 
                                   text: str, 
                                   category: Optional[str] = None,
                                   top_k: int = 5, 
                                   category_boost: float = 0.2) -> List[Tuple[str, float]]:
        """
        Match with optional category boost for scores.
        
        If category is provided, boost scores for codes in that category.
        
        Args:
            text: Query text (chief complaint)
            category: Optional CEDIS category to boost
            top_k: Number of top matches to return
            category_boost: Boost factor for matching category (0-1)
            
        Returns:
            List of tuples (cedis_code, boosted_similarity) sorted descending
        """
        # Get base matches
        base_matches = self.match(text, top_k=top_k * 2, min_similarity=0.0)
        
        if not category:
            return base_matches[:top_k]
        
        # Apply category boost
        boosted_matches = []
        for code, score in base_matches:
            entry = CEDIS_CODES[code]
            if entry['category'] == category:
                boosted_score = min(1.0, score + category_boost)
            else:
                boosted_score = score
            boosted_matches.append((code, boosted_score))
        
        # Re-sort and return top-k
        boosted_matches.sort(key=lambda x: x[1], reverse=True)
        return boosted_matches[:top_k]


class EnsembleMatcher:
    """Ensemble matcher combining fuzzy and TF-IDF methods."""
    
    def __init__(self, 
                 use_fuzzy: bool = True,
                 use_tfidf: bool = True,
                 fuzzy_weight: float = 0.4,
                 tfidf_weight: float = 0.6):
        """
        Initialize ensemble matcher.
        
        Args:
            use_fuzzy: Whether to use fuzzy matching
            use_tfidf: Whether to use TF-IDF matching
            fuzzy_weight: Weight for fuzzy matching scores (0-1)
            tfidf_weight: Weight for TF-IDF matching scores (0-1)
        """
        self.use_fuzzy = use_fuzzy and RAPIDFUZZ_AVAILABLE
        self.use_tfidf = use_tfidf
        
        # Normalize weights
        total_weight = fuzzy_weight + tfidf_weight
        self.fuzzy_weight = fuzzy_weight / total_weight
        self.tfidf_weight = tfidf_weight / total_weight
        
        # Initialize matchers
        if self.use_fuzzy:
            self.fuzzy_matcher = FuzzyMatcher()
        if self.use_tfidf:
            self.tfidf_matcher = TFIDFMatcher()
    
    def match(self, text: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Match using ensemble of methods.
        
        Args:
            text: Query text (chief complaint)
            top_k: Number of top matches to return
            
        Returns:
            List of tuples (cedis_code, combined_score, component_scores)
            where component_scores is dict with 'fuzzy' and 'tfidf' keys
        """
        if not text:
            return []
        
        # Collect scores from each method
        all_scores = {}  # code -> {method: score}
        
        if self.use_fuzzy:
            fuzzy_matches = self.fuzzy_matcher.match(text, top_k=top_k * 2)
            for code, score in fuzzy_matches:
                if code not in all_scores:
                    all_scores[code] = {}
                all_scores[code]['fuzzy'] = score
        
        if self.use_tfidf:
            tfidf_matches = self.tfidf_matcher.match(text, top_k=top_k * 2)
            for code, score in tfidf_matches:
                if code not in all_scores:
                    all_scores[code] = {}
                all_scores[code]['tfidf'] = score
        
        # Combine scores
        combined_matches = []
        for code, scores in all_scores.items():
            fuzzy_score = scores.get('fuzzy', 0.0)
            tfidf_score = scores.get('tfidf', 0.0)
            
            combined_score = (
                fuzzy_score * self.fuzzy_weight +
                tfidf_score * self.tfidf_weight
            )
            
            combined_matches.append((code, combined_score, scores))
        
        # Sort by combined score
        combined_matches.sort(key=lambda x: x[1], reverse=True)
        
        return combined_matches[:top_k]
