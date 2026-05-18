"""
Text preprocessing for chief complaints.

Handles cleaning, normalization, tokenization, and feature extraction
from free-text chief complaint descriptions.
"""

import re
import pandas as pd
from typing import List, Dict, Tuple, Optional

from .medical_terms import (
    expand_abbreviations,
    normalize_symptoms,
    AGE_MODIFIERS,
    GENDER_MODIFIERS,
    BODY_PARTS,
)


class ChiefComplaintPreprocessor:
    """Preprocess chief complaints for classification."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.stopwords = self._get_medical_stopwords()
    
    def _get_medical_stopwords(self):
        """Get stopwords relevant for chief complaints (minimal set)."""
        # Keep medical stopwords minimal - we want to preserve most terms
        return {
            'a', 'an', 'the', 'and', 'or', 'but',
            'to', 'from', 'in', 'on', 'at', 'by', 'for',
            'with', 'without', 'of', 'is', 'are', 'was', 'were',
            'has', 'have', 'had', 'be', 'been', 'being',
        }
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Raw chief complaint text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not text:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep hyphens and slashes (important for medical terms)
        text = re.sub(r'[^\w\s\-/]', ' ', text)
        
        # Clean up spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_demographics(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract demographic modifiers (age, gender) from text.
        
        Args:
            text: Chief complaint text
            
        Returns:
            Dictionary with 'age_group' and 'gender' keys
        """
        text_lower = text.lower()
        demographics = {'age_group': None, 'gender': None}
        
        # Extract age group
        for age_mod in AGE_MODIFIERS:
            if age_mod in text_lower:
                demographics['age_group'] = age_mod
                break
        
        # Extract gender
        for gender_mod in GENDER_MODIFIERS:
            if gender_mod in text_lower:
                demographics['gender'] = gender_mod
                break
        
        return demographics
    
    def extract_body_parts(self, text: str) -> List[str]:
        """
        Extract mentioned body parts from text.
        
        Args:
            text: Chief complaint text
            
        Returns:
            List of body parts mentioned
        """
        text_lower = text.lower()
        mentioned_parts = []
        
        all_body_parts = BODY_PARTS['anatomical'] + BODY_PARTS['organs']
        
        for part in all_body_parts:
            if re.search(r'\b' + re.escape(part) + r'\b', text_lower):
                mentioned_parts.append(part)
        
        return mentioned_parts
    
    def split_compound_complaints(self, text: str) -> List[str]:
        """
        Split compound complaints into individual components.
        
        Examples:
            "chest pain and shortness of breath" -> ["chest pain", "shortness of breath"]
            "abdominal pain - pediatric" -> ["abdominal pain", "pediatric"]
        
        Args:
            text: Chief complaint text
            
        Returns:
            List of individual complaint components
        """
        # Common separators in chief complaints
        separators = [' and ', ' & ', ' - ', ' / ', ', ', ';', '|']
        
        parts = [text]
        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend([p.strip() for p in part.split(sep) if p.strip()])
            parts = new_parts
        
        return parts
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords while preserving medical terms.
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
        """
        words = text.split()
        filtered_words = [w for w in words if w not in self.stopwords]
        return ' '.join(filtered_words)
    
    def preprocess(self, text: str, 
                   preserve_demographics: bool = False,
                   split_compounds: bool = False) -> Dict:
        """
        Complete preprocessing pipeline for a chief complaint.
        
        Args:
            text: Raw chief complaint text
            preserve_demographics: If True, extract and preserve demographic info
            split_compounds: If True, split compound complaints
            
        Returns:
            Dictionary with processed text and extracted features
        """
        if pd.isna(text) or not text:
            return {
                'original': '',
                'cleaned': '',
                'normalized': '',
                'processed': '',
                'demographics': {'age_group': None, 'gender': None},
                'body_parts': [],
                'components': [],
            }
        
        # Store original
        original = str(text)
        
        # Extract demographics before processing
        demographics = self.extract_demographics(original) if preserve_demographics else {}
        
        # Extract body parts
        body_parts = self.extract_body_parts(original)
        
        # Split compounds if requested
        components = self.split_compound_complaints(original) if split_compounds else [original]
        
        # Clean text
        cleaned = self.clean_text(original)
        
        # Expand abbreviations
        expanded = expand_abbreviations(cleaned)
        
        # Normalize symptoms
        normalized = normalize_symptoms(expanded)
        
        # Remove stopwords for matching (but keep in normalized)
        processed = self.remove_stopwords(normalized)
        
        return {
            'original': original,
            'cleaned': cleaned,
            'normalized': normalized,
            'processed': processed,
            'demographics': demographics,
            'body_parts': body_parts,
            'components': components,
        }
    
    def preprocess_batch(self, texts: List[str], **kwargs) -> List[Dict]:
        """
        Preprocess a batch of chief complaints.
        
        Args:
            texts: List of raw chief complaint texts
            **kwargs: Additional arguments to pass to preprocess()
            
        Returns:
            List of preprocessing results
        """
        return [self.preprocess(text, **kwargs) for text in texts]


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract key medical terms from text.
    
    Simple keyword extraction based on:
    - Length (prefer longer terms)
    - Medical relevance (body parts, symptoms)
    
    Args:
        text: Processed text
        top_n: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    if not text:
        return []
    
    words = text.split()
    
    # Filter out very short words
    words = [w for w in words if len(w) > 2]
    
    # Sort by length (longer = more specific)
    words = sorted(words, key=len, reverse=True)
    
    # Return top N unique words
    seen = set()
    keywords = []
    for word in words:
        if word not in seen:
            keywords.append(word)
            seen.add(word)
            if len(keywords) >= top_n:
                break
    
    return keywords
