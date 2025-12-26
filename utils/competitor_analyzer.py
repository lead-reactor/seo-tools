"""
Competitor Semantic Analyzer - Main orchestrator.

Combines all Google APIs to perform comprehensive competitor semantic analysis:
1. Custom Search API -> Get Top 10 URLs for a bucket query
2. WebScraper -> Extract content from each URL
3. Cloud NLP API -> Extract entities and categories
4. Vertex AI Embeddings -> Semantic clustering and gap analysis
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from .google_config import GoogleAPIConfig
from .google_search import GoogleSearchClient, SearchResponse
from .google_nlp import CloudNLPClient, NLPAnalysisResult, Entity, Category
from .google_embeddings import VertexEmbeddingsClient, SemanticAnalyzer, EmbeddingBatch
from .scraper import WebScraper


@dataclass
class PageAnalysis:
    """Complete analysis for a single page."""
    url: str
    position: int
    title: str
    snippet: str

    # Scraped content
    page_title: str = ""
    meta_description: str = ""
    headings: list[str] = field(default_factory=list)
    body_text: str = ""
    word_count: int = 0

    # NLP analysis
    entities: list[Entity] = field(default_factory=list)
    categories: list[Category] = field(default_factory=list)

    # Embedding (stored as list for serialization)
    embedding: list[float] = field(default_factory=list)

    # Status
    scrape_success: bool = False
    nlp_success: bool = False
    embedding_success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "position": self.position,
            "title": self.title,
            "snippet": self.snippet,
            "page_title": self.page_title,
            "meta_description": self.meta_description,
            "headings": self.headings,
            "word_count": self.word_count,
            "entities": [e.to_dict() for e in self.entities],
            "categories": [c.to_dict() for c in self.categories],
            "scrape_success": self.scrape_success,
            "nlp_success": self.nlp_success,
            "embedding_success": self.embedding_success,
            "error": self.error,
        }


@dataclass
class EntityFrequency:
    """Entity with frequency across pages."""
    name: str
    type: str
    total_mentions: int
    pages_count: int  # Number of pages containing this entity
    avg_salience: float
    mid: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "total_mentions": self.total_mentions,
            "pages_count": self.pages_count,
            "avg_salience": self.avg_salience,
            "mid": self.mid,
        }


@dataclass
class SemanticCluster:
    """Cluster of semantically similar pages."""
    cluster_id: int
    urls: list[str]
    sample_headings: list[str]


@dataclass
class CompetitorAnalysisResult:
    """Complete competitor semantic analysis result."""
    bucket_query: str
    search_response: SearchResponse
    pages: list[PageAnalysis]

    # Aggregated analysis
    common_entities: list[EntityFrequency] = field(default_factory=list)
    category_distribution: dict[str, int] = field(default_factory=dict)
    content_clusters: list[SemanticCluster] = field(default_factory=list)
    semantic_gaps: list[tuple[str, float]] = field(default_factory=list)

    # Statistics
    avg_word_count: float = 0
    total_unique_entities: int = 0
    successful_pages: int = 0

    def to_dict(self) -> dict:
        return {
            "bucket_query": self.bucket_query,
            "search_response": self.search_response.to_dict(),
            "pages": [p.to_dict() for p in self.pages],
            "common_entities": [e.to_dict() for e in self.common_entities],
            "category_distribution": self.category_distribution,
            "content_clusters": [
                {"cluster_id": c.cluster_id, "urls": c.urls, "sample_headings": c.sample_headings}
                for c in self.content_clusters
            ],
            "semantic_gaps": self.semantic_gaps,
            "avg_word_count": self.avg_word_count,
            "total_unique_entities": self.total_unique_entities,
            "successful_pages": self.successful_pages,
        }

    def get_top_entities(self, n: int = 20) -> list[EntityFrequency]:
        """Get top N entities by page frequency."""
        return sorted(
            self.common_entities,
            key=lambda e: (e.pages_count, e.total_mentions),
            reverse=True
        )[:n]

    def get_entities_coverage(self, min_pages: int = 5) -> list[EntityFrequency]:
        """Get entities that appear in at least min_pages pages."""
        return [e for e in self.common_entities if e.pages_count >= min_pages]


class CompetitorAnalyzer:
    """
    Main orchestrator for competitor semantic analysis.

    Usage:
        config = GoogleAPIConfig.from_env()
        analyzer = CompetitorAnalyzer(config)

        # Full analysis
        result = analyzer.analyze_bucket("assurance auto pas cher")

        # Access results
        print(result.get_top_entities(10))
        print(result.category_distribution)
        print(result.content_clusters)
    """

    def __init__(
        self,
        config: Optional[GoogleAPIConfig] = None,
        enable_nlp: bool = True,
        enable_embeddings: bool = True,
    ):
        """
        Initialize the analyzer.

        Args:
            config: Google API configuration (uses env if not provided)
            enable_nlp: Enable Cloud NLP analysis
            enable_embeddings: Enable Vertex AI embeddings
        """
        self.config = config or GoogleAPIConfig.from_env()
        self.enable_nlp = enable_nlp
        self.enable_embeddings = enable_embeddings

        # Initialize clients
        self.scraper = WebScraper()
        self.search_client = None
        self.nlp_client = None
        self.embeddings_client = None
        self.semantic_analyzer = None

        # Lazy initialization
        if self.config.validate_custom_search():
            self.search_client = GoogleSearchClient(self.config)

        if enable_nlp and self.config.validate_cloud_nlp():
            self.nlp_client = CloudNLPClient(self.config)

        if enable_embeddings and self.config.validate_vertex_ai():
            self.embeddings_client = VertexEmbeddingsClient(self.config)
            self.semantic_analyzer = SemanticAnalyzer(self.embeddings_client)

    def analyze_bucket(
        self,
        query: str,
        num_results: int = 10,
        language: str = "fr",
        target_url: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> CompetitorAnalysisResult:
        """
        Perform complete competitor semantic analysis for a bucket query.

        Args:
            query: The bucket/keyword query (e.g., "assurance auto pas cher")
            num_results: Number of results to analyze (max 10)
            language: Language code
            target_url: Optional reference URL for gap analysis
            progress_callback: Optional callback(step, total, message)

        Returns:
            CompetitorAnalysisResult with all analysis data
        """
        def progress(step: int, total: int, msg: str):
            if progress_callback:
                progress_callback(step, total, msg)

        total_steps = 4 + num_results  # search + scrape each + aggregate + cluster + gaps

        # Step 1: Get Top 10 URLs
        progress(1, total_steps, "Recherche des URLs Top 10...")

        if not self.search_client:
            raise ValueError("Custom Search API not configured")

        search_response = self.search_client.search(query, num_results=num_results, language=language)
        pages = []

        # Step 2: Scrape and analyze each page
        for idx, result in enumerate(search_response.results):
            progress(2 + idx, total_steps, f"Analyse de {result.display_link}...")

            page = PageAnalysis(
                url=result.url,
                position=result.position,
                title=result.title,
                snippet=result.snippet,
            )

            # Scrape content
            try:
                content = self.scraper.scrape(result.url)
                page.page_title = content.get("title", "")
                page.meta_description = content.get("meta_description", "")
                page.headings = content.get("headings", [])
                page.body_text = content.get("all_text", "")
                page.word_count = len(page.body_text.split())
                page.scrape_success = True
            except Exception as e:
                page.error = f"Scrape failed: {str(e)}"
                pages.append(page)
                continue

            # NLP analysis
            if self.nlp_client and page.body_text:
                try:
                    nlp_result = self.nlp_client.analyze(page.body_text, language=language)
                    page.entities = nlp_result.entities
                    page.categories = nlp_result.categories
                    page.nlp_success = True
                except Exception as e:
                    page.error = f"NLP failed: {str(e)}"

            # Generate embedding
            if self.embeddings_client and page.body_text:
                try:
                    # Use title + headings + first 1000 chars of body for embedding
                    embed_text = f"{page.page_title}. {' '.join(page.headings[:5])}. {page.body_text[:1000]}"
                    embedding = self.embeddings_client.embed_text(embed_text)
                    page.embedding = embedding.embedding
                    page.embedding_success = True
                except Exception as e:
                    if not page.error:
                        page.error = f"Embedding failed: {str(e)}"

            pages.append(page)

        # Step 3: Aggregate analysis
        progress(2 + num_results, total_steps, "Agrégation des résultats...")

        result = CompetitorAnalysisResult(
            bucket_query=query,
            search_response=search_response,
            pages=pages,
        )

        # Calculate statistics
        successful_pages = [p for p in pages if p.scrape_success]
        result.successful_pages = len(successful_pages)

        if successful_pages:
            result.avg_word_count = sum(p.word_count for p in successful_pages) / len(successful_pages)

        # Aggregate entities
        result.common_entities = self._aggregate_entities(pages)
        result.total_unique_entities = len(result.common_entities)

        # Aggregate categories
        result.category_distribution = self._aggregate_categories(pages)

        # Step 4: Semantic clustering
        progress(3 + num_results, total_steps, "Clustering sémantique...")

        if self.semantic_analyzer:
            result.content_clusters = self._cluster_pages(pages)

            # Gap analysis if target URL provided
            if target_url:
                progress(4 + num_results, total_steps, "Analyse des gaps sémantiques...")
                result.semantic_gaps = self._find_gaps(pages, target_url, language)

        return result

    def _aggregate_entities(self, pages: list[PageAnalysis]) -> list[EntityFrequency]:
        """Aggregate entities across all pages."""
        entity_data = {}  # name -> {type, mentions, pages, saliences, mid}

        for page in pages:
            if not page.nlp_success:
                continue

            seen_in_page = set()
            for entity in page.entities:
                key = entity.name.lower()

                if key not in entity_data:
                    entity_data[key] = {
                        "name": entity.name,
                        "type": entity.type,
                        "mentions": 0,
                        "pages": 0,
                        "saliences": [],
                        "mid": entity.mid,
                    }

                entity_data[key]["mentions"] += entity.mentions_count
                entity_data[key]["saliences"].append(entity.salience)

                if key not in seen_in_page:
                    entity_data[key]["pages"] += 1
                    seen_in_page.add(key)

        return [
            EntityFrequency(
                name=data["name"],
                type=data["type"],
                total_mentions=data["mentions"],
                pages_count=data["pages"],
                avg_salience=sum(data["saliences"]) / len(data["saliences"]),
                mid=data["mid"],
            )
            for data in entity_data.values()
        ]

    def _aggregate_categories(self, pages: list[PageAnalysis]) -> dict[str, int]:
        """Aggregate categories across all pages."""
        category_counts = Counter()

        for page in pages:
            if not page.nlp_success:
                continue

            for category in page.categories:
                # Use top-level category
                top_category = category.name.split("/")[1] if "/" in category.name else category.name
                category_counts[top_category] += 1

        return dict(category_counts)

    def _cluster_pages(self, pages: list[PageAnalysis], n_clusters: int = 3) -> list[SemanticCluster]:
        """Cluster pages by semantic similarity."""
        pages_with_embeddings = [p for p in pages if p.embedding_success and p.embedding]

        if len(pages_with_embeddings) < n_clusters:
            return []

        try:
            from sklearn.cluster import KMeans
            import numpy as np

            vectors = np.array([p.embedding for p in pages_with_embeddings])

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)

            clusters = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = {"urls": [], "headings": []}

                page = pages_with_embeddings[idx]
                clusters[label]["urls"].append(page.url)
                clusters[label]["headings"].extend(page.headings[:3])

            return [
                SemanticCluster(
                    cluster_id=cluster_id,
                    urls=data["urls"],
                    sample_headings=data["headings"][:5],
                )
                for cluster_id, data in clusters.items()
            ]
        except Exception:
            return []

    def _find_gaps(
        self,
        competitor_pages: list[PageAnalysis],
        target_url: str,
        language: str,
    ) -> list[tuple[str, float]]:
        """Find semantic gaps between competitors and target."""
        # Scrape target
        try:
            target_content = self.scraper.scrape(target_url)
            target_text = target_content.get("all_text", "")
        except Exception:
            return []

        if not target_text or not self.semantic_analyzer:
            return []

        # Get competitor headings as reference topics
        competitor_topics = []
        for page in competitor_pages:
            if page.scrape_success:
                competitor_topics.extend(page.headings)

        if not competitor_topics:
            return []

        # Split target into paragraphs
        target_paragraphs = [p.strip() for p in target_text.split("\n\n") if len(p.strip()) > 50]

        if not target_paragraphs:
            return []

        try:
            return self.semantic_analyzer.find_semantic_gaps(
                reference_texts=competitor_topics[:50],  # Limit for API costs
                target_texts=target_paragraphs[:20],
                threshold=0.6,
            )
        except Exception:
            return []


def analyze_competitors(
    query: str,
    config: Optional[GoogleAPIConfig] = None,
    num_results: int = 10,
    language: str = "fr",
) -> CompetitorAnalysisResult:
    """
    Convenience function for competitor analysis.

    Args:
        query: Bucket/keyword query
        config: Optional config
        num_results: Number of results
        language: Language code

    Returns:
        CompetitorAnalysisResult
    """
    if config is None:
        config = GoogleAPIConfig.from_env()

    analyzer = CompetitorAnalyzer(config)
    return analyzer.analyze_bucket(query, num_results=num_results, language=language)
