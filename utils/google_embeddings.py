"""
Vertex AI Text Embeddings API integration.

Generates text embeddings for semantic similarity, clustering, and gap analysis.

Documentation: https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings
Models:
- text-multilingual-embedding-002: Best for French/multilingual (768 dimensions)
- text-embedding-004: Best for English (768 dimensions)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .google_config import GoogleAPIConfig, EMBEDDING_MODELS


@dataclass
class TextEmbedding:
    """Single text embedding result."""
    text: str
    embedding: list[float]
    model: str

    @property
    def vector(self) -> np.ndarray:
        """Get embedding as numpy array."""
        return np.array(self.embedding)

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return len(self.embedding)


@dataclass
class EmbeddingBatch:
    """Batch of text embeddings."""
    embeddings: list[TextEmbedding]
    model: str

    def get_vectors(self) -> np.ndarray:
        """Get all embeddings as a 2D numpy array."""
        return np.array([e.embedding for e in self.embeddings])

    def get_texts(self) -> list[str]:
        """Get all original texts."""
        return [e.text for e in self.embeddings]


class VertexEmbeddingsClient:
    """
    Client for Vertex AI Text Embeddings API.

    Usage:
        config = GoogleAPIConfig.from_env()
        client = VertexEmbeddingsClient(config)

        # Single text
        embedding = client.embed_text("Votre texte ici")

        # Batch of texts
        batch = client.embed_batch(["Text 1", "Text 2", "Text 3"])

        # Similarity
        similarity = client.cosine_similarity(embedding1, embedding2)
    """

    def __init__(self, config: GoogleAPIConfig):
        self.config = config
        self.model_name = config.embedding_model
        self._model = None

    def _get_model(self):
        """Lazy initialization of the embedding model."""
        if self._model is None:
            try:
                import vertexai
                from vertexai.language_models import TextEmbeddingModel

                vertexai.init(
                    project=self.config.google_cloud_project,
                    location=self.config.google_cloud_location,
                )
                self._model = TextEmbeddingModel.from_pretrained(self.model_name)
            except ImportError:
                raise ImportError(
                    "vertexai not installed. "
                    "Run: pip install google-cloud-aiplatform"
                )
        return self._model

    def embed_text(self, text: str) -> TextEmbedding:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            TextEmbedding with vector
        """
        model = self._get_model()
        embeddings = model.get_embeddings([text])

        return TextEmbedding(
            text=text,
            embedding=embeddings[0].values,
            model=self.model_name,
        )

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 250,  # Vertex AI limit
    ) -> EmbeddingBatch:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Maximum batch size per API call (250 max)

        Returns:
            EmbeddingBatch with all vectors
        """
        model = self._get_model()
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = model.get_embeddings(batch)

            for text, emb in zip(batch, embeddings):
                all_embeddings.append(TextEmbedding(
                    text=text,
                    embedding=emb.values,
                    model=self.model_name,
                ))

        return EmbeddingBatch(embeddings=all_embeddings, model=self.model_name)

    @staticmethod
    def cosine_similarity(embedding1: TextEmbedding, embedding2: TextEmbedding) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (-1 to 1, higher = more similar)
        """
        v1 = embedding1.vector
        v2 = embedding2.vector
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    @staticmethod
    def similarity_matrix(batch: EmbeddingBatch) -> np.ndarray:
        """
        Calculate pairwise similarity matrix for all embeddings.

        Args:
            batch: Batch of embeddings

        Returns:
            NxN similarity matrix
        """
        vectors = batch.get_vectors()
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / norms
        # Compute similarity matrix
        return np.dot(normalized, normalized.T)

    def find_most_similar(
        self,
        query: TextEmbedding,
        candidates: EmbeddingBatch,
        top_k: int = 5,
    ) -> list[tuple[int, float, str]]:
        """
        Find most similar texts to a query.

        Args:
            query: Query embedding
            candidates: Candidate embeddings to search
            top_k: Number of results to return

        Returns:
            List of (index, similarity, text) tuples
        """
        query_vec = query.vector
        candidate_vecs = candidates.get_vectors()

        # Compute similarities
        query_norm = query_vec / np.linalg.norm(query_vec)
        candidate_norms = candidate_vecs / np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
        similarities = np.dot(candidate_norms, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (int(idx), float(similarities[idx]), candidates.embeddings[idx].text)
            for idx in top_indices
        ]


class SemanticAnalyzer:
    """
    High-level semantic analysis using embeddings.

    Provides clustering, gap detection, and similarity analysis.
    """

    def __init__(self, client: VertexEmbeddingsClient):
        self.client = client

    def cluster_texts(
        self,
        texts: list[str],
        n_clusters: int = 3,
    ) -> dict[int, list[str]]:
        """
        Cluster texts by semantic similarity using K-means.

        Args:
            texts: Texts to cluster
            n_clusters: Number of clusters

        Returns:
            Dict mapping cluster_id to list of texts
        """
        from sklearn.cluster import KMeans

        # Get embeddings
        batch = self.client.embed_batch(texts)
        vectors = batch.get_vectors()

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)

        # Group texts by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[idx])

        return clusters

    def find_semantic_gaps(
        self,
        reference_texts: list[str],
        target_texts: list[str],
        threshold: float = 0.7,
    ) -> list[tuple[str, float]]:
        """
        Find topics in reference that are not well covered in target.

        Args:
            reference_texts: Competitor/reference content (e.g., paragraphs)
            target_texts: Your content to compare against
            threshold: Similarity threshold (lower = stricter gap detection)

        Returns:
            List of (reference_text, max_similarity) for poorly covered topics
        """
        ref_batch = self.client.embed_batch(reference_texts)
        target_batch = self.client.embed_batch(target_texts)

        ref_vecs = ref_batch.get_vectors()
        target_vecs = target_batch.get_vectors()

        gaps = []

        for i, ref_vec in enumerate(ref_vecs):
            # Find max similarity to any target text
            ref_norm = ref_vec / np.linalg.norm(ref_vec)
            target_norms = target_vecs / np.linalg.norm(target_vecs, axis=1, keepdims=True)
            similarities = np.dot(target_norms, ref_norm)
            max_sim = float(np.max(similarities))

            if max_sim < threshold:
                gaps.append((reference_texts[i], max_sim))

        # Sort by similarity (lowest first = biggest gaps)
        return sorted(gaps, key=lambda x: x[1])

    def average_similarity(
        self,
        texts1: list[str],
        texts2: list[str],
    ) -> float:
        """
        Calculate average semantic similarity between two text sets.

        Args:
            texts1: First set of texts
            texts2: Second set of texts

        Returns:
            Average similarity score
        """
        batch1 = self.client.embed_batch(texts1)
        batch2 = self.client.embed_batch(texts2)

        vecs1 = batch1.get_vectors()
        vecs2 = batch2.get_vectors()

        # Normalize
        vecs1_norm = vecs1 / np.linalg.norm(vecs1, axis=1, keepdims=True)
        vecs2_norm = vecs2 / np.linalg.norm(vecs2, axis=1, keepdims=True)

        # Calculate all pairwise similarities and average
        similarity_matrix = np.dot(vecs1_norm, vecs2_norm.T)
        return float(np.mean(similarity_matrix))


def embed_texts(
    texts: list[str],
    config: Optional[GoogleAPIConfig] = None,
) -> EmbeddingBatch:
    """
    Convenience function to embed multiple texts.

    Args:
        texts: Texts to embed
        config: Optional config (uses env if not provided)

    Returns:
        EmbeddingBatch with all vectors
    """
    if config is None:
        config = GoogleAPIConfig.from_env()

    client = VertexEmbeddingsClient(config)
    return client.embed_batch(texts)
