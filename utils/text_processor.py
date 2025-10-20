"""
Text processing module for semantic analysis.
Handles tokenization, stopwords, and n-gram generation.
"""

import re
from typing import List, Set, Dict
from collections import Counter
import os


class TextProcessor:
    """Process text for semantic and TF-IDF analysis."""

    def __init__(self, stopwords_path: str = None):
        """
        Initialize text processor.

        Args:
            stopwords_path: Path to stopwords file (one word per line)
        """
        self.stopwords = self._load_stopwords(stopwords_path)

    def _load_stopwords(self, path: str = None) -> Set[str]:
        """Load French stopwords from file."""
        if not path:
            # Default path
            path = os.path.join(os.path.dirname(__file__), '..', 'data', 'stopwords_fr.txt')

        stopwords = set()

        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    stopwords = {line.strip().lower() for line in f if line.strip()}
            except Exception as e:
                print(f"Error loading stopwords from {path}: {e}")

        # Basic fallback stopwords if file not found
        if not stopwords:
            stopwords = {
                'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou',
                'mais', 'donc', 'car', 'ni', 'or', 'ce', 'cette', 'ces', 'mon',
                'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre',
                'votre', 'leur', 'leurs', 'je', 'tu', 'il', 'elle', 'nous', 'vous',
                'ils', 'elles', 'on', 'qui', 'que', 'quoi', 'dont', 'où', 'à',
                'au', 'aux', 'en', 'par', 'pour', 'dans', 'sur', 'sous', 'avec',
                'sans', 'plus', 'moins', 'très', 'aussi', 'si', 'ne', 'pas',
                'chez', 'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'savoir',
                'pouvoir', 'vouloir', 'devoir', 'falloir'
            }

        return stopwords

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Keep only letters, numbers, spaces and hyphens
        text = re.sub(r'[^a-zàâäæçéèêëïîôùûüÿœ0-9\s\-]', ' ', text)

        # Replace multiple spaces/hyphens with single space
        text = re.sub(r'[\s\-]+', ' ', text)

        return text.strip()

    def tokenize(self, text: str, remove_stopwords: bool = True, min_length: int = 2) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize
            remove_stopwords: Whether to remove stopwords
            min_length: Minimum word length to keep

        Returns:
            List of tokens
        """
        cleaned = self.clean_text(text)
        tokens = cleaned.split()

        # Filter by length
        tokens = [t for t in tokens if len(t) >= min_length]

        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        return tokens

    def generate_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """
        Generate n-grams from tokens.

        Args:
            tokens: List of tokens
            n: N-gram size (1, 2, 3, etc.)

        Returns:
            List of n-grams as strings
        """
        if n < 1 or not tokens:
            return []

        if n == 1:
            return tokens

        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            ngrams.append(ngram)

        return ngrams

    def extract_all_ngrams(self, text: str, remove_stopwords: bool = True) -> Dict[str, List[str]]:
        """
        Extract unigrams, bigrams, and trigrams from text.

        Args:
            text: Text to analyze
            remove_stopwords: Whether to remove stopwords

        Returns:
            Dictionary with 'unigrams', 'bigrams', 'trigrams' keys
        """
        tokens = self.tokenize(text, remove_stopwords=remove_stopwords)

        return {
            'unigrams': self.generate_ngrams(tokens, 1),
            'bigrams': self.generate_ngrams(tokens, 2),
            'trigrams': self.generate_ngrams(tokens, 3),
        }

    def calculate_term_frequency(self, terms: List[str]) -> Dict[str, int]:
        """
        Calculate term frequency.

        Args:
            terms: List of terms (can be unigrams, bigrams, etc.)

        Returns:
            Dictionary mapping term to frequency
        """
        return dict(Counter(terms))

    def calculate_density(self, terms: List[str], top_n: int = 50) -> Dict[str, float]:
        """
        Calculate keyword density (percentage).

        Args:
            terms: List of terms
            top_n: Number of top terms to return

        Returns:
            Dictionary mapping term to density percentage
        """
        if not terms:
            return {}

        total = len(terms)
        freq = Counter(terms)

        # Calculate density as percentage
        density = {
            term: (count / total) * 100
            for term, count in freq.most_common(top_n)
        }

        return density

    def get_top_terms(self, terms: List[str], top_n: int = 50) -> List[tuple]:
        """
        Get top N most frequent terms.

        Args:
            terms: List of terms
            top_n: Number of top terms to return

        Returns:
            List of (term, frequency) tuples
        """
        return Counter(terms).most_common(top_n)

    def find_unique_terms(self, terms_a: List[str], terms_b: List[str], top_n: int = 50) -> List[tuple]:
        """
        Find terms present in B but absent in A.

        Args:
            terms_a: Your terms
            terms_b: Competitor terms
            top_n: Number of top unique terms to return

        Returns:
            List of (term, frequency in B) tuples
        """
        set_a = set(terms_a)
        freq_b = Counter(terms_b)

        # Find terms in B but not in A
        unique_to_b = {
            term: count
            for term, count in freq_b.items()
            if term not in set_a
        }

        # Sort by frequency and return top N
        return sorted(unique_to_b.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def compare_multiple_sources(self, target_terms: List[str], competitor_terms_list: List[List[str]]) -> Dict:
        """
        Compare target terms with multiple competitor sources.

        Args:
            target_terms: Your terms
            competitor_terms_list: List of competitor term lists

        Returns:
            Dictionary with analysis results
        """
        # Combine all competitor terms
        all_competitor_terms = []
        for terms in competitor_terms_list:
            all_competitor_terms.extend(terms)

        # Find unique terms across all competitors
        unique_terms = self.find_unique_terms(target_terms, all_competitor_terms, top_n=100)

        # Calculate coverage
        target_set = set(target_terms)
        competitor_set = set(all_competitor_terms)

        common_terms = target_set.intersection(competitor_set)
        coverage = len(common_terms) / len(competitor_set) * 100 if competitor_set else 0

        return {
            'unique_to_competitors': unique_terms,
            'common_terms_count': len(common_terms),
            'coverage_percentage': coverage,
            'total_competitor_unique_terms': len(competitor_set),
            'total_target_terms': len(target_set)
        }
