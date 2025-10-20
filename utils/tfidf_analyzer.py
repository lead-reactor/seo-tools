"""
TF-IDF analysis module for semantic comparison.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple
import numpy as np


class TFIDFAnalyzer:
    """Analyze text using TF-IDF scores."""

    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None

    def analyze_single_document(self, text: str, max_features: int = 100) -> pd.DataFrame:
        """
        Analyze a single document with TF-IDF.
        Note: Single document TF-IDF is essentially just term frequency.

        Args:
            text: Document text
            max_features: Maximum number of features to extract

        Returns:
            DataFrame with terms and scores
        """
        if not text or not text.strip():
            return pd.DataFrame(columns=['term', 'score'])

        # For single document, we'll use CountVectorizer logic
        # since TF-IDF needs multiple documents for IDF calculation
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(max_features=max_features)
        counts = vectorizer.fit_transform([text])

        terms = vectorizer.get_feature_names_out()
        frequencies = counts.toarray()[0]

        # Calculate relative frequency (normalized)
        total = frequencies.sum()
        scores = frequencies / total if total > 0 else frequencies

        df = pd.DataFrame({
            'term': terms,
            'frequency': frequencies,
            'score': scores
        })

        return df.sort_values('frequency', ascending=False)

    def analyze_multiple_documents(self, documents: Dict[str, str], max_features: int = 200) -> Dict:
        """
        Analyze multiple documents with TF-IDF.

        Args:
            documents: Dictionary mapping document name/URL to text
            max_features: Maximum number of features

        Returns:
            Dictionary with analysis results
        """
        if not documents or len(documents) == 0:
            return {}

        # Filter out empty documents
        documents = {k: v for k, v in documents.items() if v and v.strip()}

        if len(documents) == 0:
            return {}

        doc_names = list(documents.keys())
        doc_texts = list(documents.values())

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b'  # Single word tokens
        )

        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(doc_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Create results dictionary
        results = {}

        for idx, doc_name in enumerate(doc_names):
            # Get TF-IDF scores for this document
            doc_vector = self.tfidf_matrix[idx].toarray()[0]

            # Create DataFrame
            df = pd.DataFrame({
                'term': self.feature_names,
                'tfidf_score': doc_vector
            })

            # Sort by score and filter non-zero
            df = df[df['tfidf_score'] > 0].sort_values('tfidf_score', ascending=False)

            results[doc_name] = df

        return results

    def analyze_ngrams(self, documents: Dict[str, str], ngram_range: Tuple[int, int],
                       max_features: int = 200) -> Dict:
        """
        Analyze documents with n-gram TF-IDF.

        Args:
            documents: Dictionary mapping document name to text
            ngram_range: Tuple (min_n, max_n) for n-grams, e.g., (1,1), (2,2), (3,3)
            max_features: Maximum features

        Returns:
            Dictionary with analysis results per document
        """
        if not documents or len(documents) == 0:
            return {}

        # Filter out empty documents
        documents = {k: v for k, v in documents.items() if v and v.strip()}

        if len(documents) == 0:
            return {}

        doc_names = list(documents.keys())
        doc_texts = list(documents.values())

        # Create TF-IDF vectorizer with n-grams
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(doc_texts)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            # No features found
            return {name: pd.DataFrame(columns=['term', 'tfidf_score']) for name in doc_names}

        # Create results
        results = {}

        for idx, doc_name in enumerate(doc_names):
            doc_vector = tfidf_matrix[idx].toarray()[0]

            df = pd.DataFrame({
                'term': feature_names,
                'tfidf_score': doc_vector
            })

            df = df[df['tfidf_score'] > 0].sort_values('tfidf_score', ascending=False)
            results[doc_name] = df

        return results

    def compare_documents(self, target_doc: str, competitor_docs: List[str],
                          target_name: str = "Target",
                          competitor_names: List[str] = None) -> Dict:
        """
        Compare target document with competitor documents.

        Args:
            target_doc: Your document text
            competitor_docs: List of competitor document texts
            target_name: Name for target document
            competitor_names: Names for competitor documents

        Returns:
            Dictionary with comparison results
        """
        if not competitor_names:
            competitor_names = [f"Competitor {i+1}" for i in range(len(competitor_docs))]

        # Build documents dictionary
        documents = {target_name: target_doc}
        for name, doc in zip(competitor_names, competitor_docs):
            documents[name] = doc

        # Analyze with TF-IDF
        results = self.analyze_multiple_documents(documents)

        # Extract unique terms
        if target_name in results:
            target_terms = set(results[target_name]['term'].tolist())

            # Collect all competitor terms
            all_competitor_terms = set()
            for name in competitor_names:
                if name in results:
                    all_competitor_terms.update(results[name]['term'].tolist())

            # Find gaps (terms in competitors but not in target)
            gaps = all_competitor_terms - target_terms

            # Build gap analysis with average TF-IDF scores
            gap_scores = {}
            for term in gaps:
                scores = []
                for name in competitor_names:
                    if name in results:
                        term_df = results[name][results[name]['term'] == term]
                        if not term_df.empty:
                            scores.append(term_df['tfidf_score'].values[0])

                if scores:
                    gap_scores[term] = np.mean(scores)

            # Sort by score
            gap_analysis = sorted(gap_scores.items(), key=lambda x: x[1], reverse=True)

            results['gap_analysis'] = pd.DataFrame(gap_analysis, columns=['term', 'avg_tfidf_score'])
        else:
            results['gap_analysis'] = pd.DataFrame(columns=['term', 'avg_tfidf_score'])

        results['target_name'] = target_name
        results['competitor_names'] = competitor_names

        return results

    def get_top_terms(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N terms from a TF-IDF result DataFrame.

        Args:
            df: DataFrame with 'term' and score columns
            top_n: Number of top terms

        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df

        score_col = 'tfidf_score' if 'tfidf_score' in df.columns else 'score'
        return df.nlargest(top_n, score_col)
