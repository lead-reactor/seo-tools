"""
Google Cloud Natural Language API integration.

Extracts entities, categories, and syntax from text content.

Documentation: https://cloud.google.com/natural-language/docs
Client library: google-cloud-language
"""

from dataclasses import dataclass, field
from typing import Optional
from .google_config import GoogleAPIConfig


@dataclass
class Entity:
    """Named entity extracted from text."""
    name: str
    type: str  # PERSON, LOCATION, ORGANIZATION, EVENT, WORK_OF_ART, CONSUMER_GOOD, OTHER, etc.
    salience: float  # 0.0 to 1.0, importance in text
    mid: Optional[str] = None  # Knowledge Graph MID (e.g., /m/0d6lp)
    wikipedia_url: Optional[str] = None
    mentions_count: int = 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "salience": self.salience,
            "mid": self.mid,
            "wikipedia_url": self.wikipedia_url,
            "mentions_count": self.mentions_count,
        }


@dataclass
class Category:
    """Content category classification."""
    name: str  # e.g., "/Finance/Insurance", "/Autos & Vehicles"
    confidence: float  # 0.0 to 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "confidence": self.confidence,
        }


@dataclass
class NLPAnalysisResult:
    """Complete NLP analysis result for a text."""
    text_length: int
    language: str
    entities: list[Entity] = field(default_factory=list)
    categories: list[Category] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text_length": self.text_length,
            "language": self.language,
            "entities": [e.to_dict() for e in self.entities],
            "categories": [c.to_dict() for c in self.categories],
        }

    def get_top_entities(self, n: int = 10) -> list[Entity]:
        """Get top N entities by salience."""
        return sorted(self.entities, key=lambda e: e.salience, reverse=True)[:n]

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Filter entities by type."""
        return [e for e in self.entities if e.type == entity_type]


class CloudNLPClient:
    """
    Client for Google Cloud Natural Language API.

    Usage:
        config = GoogleAPIConfig.from_env()
        client = CloudNLPClient(config)
        result = client.analyze("Votre texte à analyser...")
        print(result.entities)
        print(result.categories)
    """

    def __init__(self, config: GoogleAPIConfig):
        self.config = config
        self._client = None
        self._language_module = None

    def _get_client(self):
        """Lazy initialization of the NLP client."""
        if self._client is None:
            try:
                from google.cloud import language_v1
                self._language_module = language_v1
                self._client = language_v1.LanguageServiceClient()
            except ImportError:
                raise ImportError(
                    "google-cloud-language not installed. "
                    "Run: pip install google-cloud-language"
                )
        return self._client

    def analyze(
        self,
        text: str,
        language: str = "fr",
        extract_entities: bool = True,
        classify_content: bool = True,
    ) -> NLPAnalysisResult:
        """
        Perform full NLP analysis on text.

        Args:
            text: Text content to analyze
            language: Language code (fr, en, etc.)
            extract_entities: Whether to extract named entities
            classify_content: Whether to classify content categories

        Returns:
            NLPAnalysisResult with entities and categories

        Note:
            - classify_content requires at least 20 words
            - Uses GOOGLE_APPLICATION_CREDENTIALS for authentication
        """
        client = self._get_client()
        language_v1 = self._language_module

        document = language_v1.Document(
            content=text,
            type_=language_v1.Document.Type.PLAIN_TEXT,
            language=language,
        )

        entities = []
        categories = []

        # Entity extraction
        if extract_entities:
            entity_response = client.analyze_entities(
                document=document,
                encoding_type=language_v1.EncodingType.UTF8,
            )
            entities = self._parse_entities(entity_response)

        # Content classification (requires minimum text length)
        if classify_content and len(text.split()) >= 20:
            try:
                classification_response = client.classify_text(document=document)
                categories = self._parse_categories(classification_response)
            except Exception:
                # Classification may fail for short/unsuitable content
                pass

        return NLPAnalysisResult(
            text_length=len(text),
            language=language,
            entities=entities,
            categories=categories,
        )

    def analyze_entities_only(self, text: str, language: str = "fr") -> list[Entity]:
        """Extract only named entities from text."""
        result = self.analyze(text, language, extract_entities=True, classify_content=False)
        return result.entities

    def classify_only(self, text: str, language: str = "fr") -> list[Category]:
        """Classify text content only."""
        result = self.analyze(text, language, extract_entities=False, classify_content=True)
        return result.categories

    def _parse_entities(self, response) -> list[Entity]:
        """Parse entity analysis response."""
        entities = []

        for entity in response.entities:
            # Extract Knowledge Graph metadata if available
            mid = None
            wikipedia_url = None
            if entity.metadata:
                mid = entity.metadata.get("mid")
                wikipedia_url = entity.metadata.get("wikipedia_url")

            entities.append(Entity(
                name=entity.name,
                type=self._language_module.Entity.Type(entity.type_).name,
                salience=entity.salience,
                mid=mid,
                wikipedia_url=wikipedia_url,
                mentions_count=len(entity.mentions),
            ))

        return entities

    def _parse_categories(self, response) -> list[Category]:
        """Parse classification response."""
        return [
            Category(name=cat.name, confidence=cat.confidence)
            for cat in response.categories
        ]


def analyze_text(
    text: str,
    config: Optional[GoogleAPIConfig] = None,
    language: str = "fr",
) -> NLPAnalysisResult:
    """
    Convenience function to analyze text with Cloud NLP.

    Args:
        text: Text to analyze
        config: Optional config (uses env if not provided)
        language: Text language

    Returns:
        NLPAnalysisResult with entities and categories
    """
    if config is None:
        config = GoogleAPIConfig.from_env()

    client = CloudNLPClient(config)
    return client.analyze(text, language=language)


def extract_entities(
    text: str,
    config: Optional[GoogleAPIConfig] = None,
    language: str = "fr",
    min_salience: float = 0.01,
) -> list[Entity]:
    """
    Extract named entities from text.

    Args:
        text: Text to analyze
        config: Optional config
        language: Text language
        min_salience: Minimum salience threshold

    Returns:
        List of entities above salience threshold
    """
    if config is None:
        config = GoogleAPIConfig.from_env()

    client = CloudNLPClient(config)
    entities = client.analyze_entities_only(text, language=language)

    return [e for e in entities if e.salience >= min_salience]
