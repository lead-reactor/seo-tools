"""
Google APIs configuration module.

Centralized configuration for all Google Cloud APIs used in competitor semantic analysis.
APIs used:
- Custom Search JSON API (Top 10 URLs retrieval)
- Cloud Natural Language API (entities/categories extraction)
- Vertex AI Text Embeddings (semantic similarity/clustering)
- Knowledge Graph Search API (entity disambiguation) - optional
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GoogleAPIConfig:
    """Configuration for Google APIs."""

    # Custom Search API
    custom_search_api_key: Optional[str] = None
    custom_search_engine_id: Optional[str] = None  # cx parameter

    # Google Cloud Project (for NLP + Vertex AI)
    google_cloud_project: Optional[str] = None
    google_cloud_location: str = "us-central1"  # Default region for Vertex AI

    # Vertex AI Text Embeddings
    embedding_model: str = "text-multilingual-embedding-002"  # Best for French/multilingual
    embedding_dimensions: int = 768

    # Knowledge Graph API (optional)
    knowledge_graph_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GoogleAPIConfig":
        """Load configuration from environment variables."""
        return cls(
            custom_search_api_key=os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY"),
            custom_search_engine_id=os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID"),
            google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            embedding_model=os.getenv("VERTEX_EMBEDDING_MODEL", "text-multilingual-embedding-002"),
            knowledge_graph_api_key=os.getenv("GOOGLE_KNOWLEDGE_GRAPH_API_KEY"),
        )

    @classmethod
    def from_streamlit_secrets(cls) -> "GoogleAPIConfig":
        """Load configuration from Streamlit secrets (for deployment)."""
        try:
            import streamlit as st
            secrets = st.secrets.get("google", {})
            return cls(
                custom_search_api_key=secrets.get("custom_search_api_key"),
                custom_search_engine_id=secrets.get("custom_search_engine_id"),
                google_cloud_project=secrets.get("cloud_project"),
                google_cloud_location=secrets.get("cloud_location", "us-central1"),
                embedding_model=secrets.get("embedding_model", "text-multilingual-embedding-002"),
                knowledge_graph_api_key=secrets.get("knowledge_graph_api_key"),
            )
        except Exception:
            return cls.from_env()

    def validate_custom_search(self) -> bool:
        """Check if Custom Search API is configured."""
        return bool(self.custom_search_api_key and self.custom_search_engine_id)

    def validate_cloud_nlp(self) -> bool:
        """Check if Cloud NLP API is configured (uses default credentials)."""
        return bool(self.google_cloud_project) or bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

    def validate_vertex_ai(self) -> bool:
        """Check if Vertex AI is configured."""
        return bool(self.google_cloud_project)

    def get_missing_configs(self) -> list[str]:
        """Return list of missing required configurations."""
        missing = []
        if not self.validate_custom_search():
            missing.append("Custom Search API (GOOGLE_CUSTOM_SEARCH_API_KEY, GOOGLE_CUSTOM_SEARCH_ENGINE_ID)")
        if not self.validate_cloud_nlp():
            missing.append("Cloud NLP API (GOOGLE_CLOUD_PROJECT or GOOGLE_APPLICATION_CREDENTIALS)")
        if not self.validate_vertex_ai():
            missing.append("Vertex AI (GOOGLE_CLOUD_PROJECT)")
        return missing


# Endpoints reference
ENDPOINTS = {
    "custom_search": "https://customsearch.googleapis.com/customsearch/v1",
    "knowledge_graph": "https://kgsearch.googleapis.com/v1/entities:search",
    "cloud_nlp": "language.googleapis.com",  # via google-cloud-language client
    "vertex_ai": "aiplatform.googleapis.com",  # via vertexai SDK
}

# Model references
EMBEDDING_MODELS = {
    "multilingual": "text-multilingual-embedding-002",  # Best for French/multi-language
    "english": "text-embedding-004",  # Best for English-only
    "gecko": "textembedding-gecko@003",  # Legacy, still supported
}

# Cloud NLP supported languages
NLP_SUPPORTED_LANGUAGES = ["fr", "en", "de", "es", "it", "pt", "nl", "ja", "ko", "zh"]
