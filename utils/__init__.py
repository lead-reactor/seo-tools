"""
Utils package for SEO semantic analysis tools.
"""

from .scraper import WebScraper
from .text_processor import TextProcessor
from .tfidf_analyzer import TFIDFAnalyzer
from .visualizations import SemanticVisualizer

# Google APIs for competitor semantic analysis
from .google_config import GoogleAPIConfig
from .google_search import GoogleSearchClient, get_top_10_urls
from .google_nlp import CloudNLPClient, analyze_text, extract_entities
from .google_embeddings import VertexEmbeddingsClient, SemanticAnalyzer, embed_texts
from .competitor_analyzer import CompetitorAnalyzer, analyze_competitors

__all__ = [
    # Original modules
    'WebScraper',
    'TextProcessor',
    'TFIDFAnalyzer',
    'SemanticVisualizer',
    # Google APIs
    'GoogleAPIConfig',
    'GoogleSearchClient',
    'get_top_10_urls',
    'CloudNLPClient',
    'analyze_text',
    'extract_entities',
    'VertexEmbeddingsClient',
    'SemanticAnalyzer',
    'embed_texts',
    'CompetitorAnalyzer',
    'analyze_competitors',
]
