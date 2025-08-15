# pmm/semantic_analysis.py
"""
Semantic analysis module for Phase 3C reflection quality and novelty detection.
Provides embedding-based similarity analysis for reflection deduplication,
evidence-commitment matching, and behavioral pattern clustering.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pmm.semantic_providers import EmbeddingProvider, get_default_provider


class SemanticAnalyzer:
    """Semantic analysis using embeddings for reflection quality assessment."""

    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None):
        self.provider = embedding_provider or get_default_provider()
        self._embedding_cache: Dict[str, List[float]] = {}

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching."""
        # Simple cache key (first 100 chars)
        cache_key = text[:100]
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        embedding = self.provider.embed_text(text)
        self._embedding_cache[cache_key] = embedding
        return embedding

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        if not text1.strip() or not text2.strip():
            return 0.0

        try:
            emb1 = np.array(self._get_embedding(text1))
            emb2 = np.array(self._get_embedding(text2))

            # Handle zero vectors
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(np.dot(emb1, emb2) / (norm1 * norm2))
        except Exception as e:
            print(f"Warning: Cosine similarity calculation failed: {e}")
            return 0.0

    def semantic_novelty_score(
        self, new_text: str, reference_texts: List[str], threshold: float = 0.8
    ) -> float:
        """
        Calculate semantic novelty score for new text against reference texts.
        Returns 1.0 for completely novel, 0.0 for duplicate.
        """
        if not new_text.strip() or not reference_texts:
            return 1.0

        max_similarity = 0.0
        for ref_text in reference_texts:
            similarity = self.cosine_similarity(new_text, ref_text)
            max_similarity = max(max_similarity, similarity)

        # Convert similarity to novelty score
        novelty = 1.0 - max_similarity
        return max(0.0, novelty)

    def is_semantic_duplicate(
        self, new_text: str, reference_texts: List[str], threshold: float = 0.8
    ) -> bool:
        """Check if new text is a semantic duplicate of any reference text."""
        novelty = self.semantic_novelty_score(new_text, reference_texts, threshold)
        return novelty < (1.0 - threshold)

    def find_best_match(
        self, query_text: str, candidate_texts: List[str]
    ) -> Tuple[int, float]:
        """
        Find the best semantic match for query text among candidates.
        Returns (index, similarity_score) of best match.
        """
        if not candidate_texts:
            return -1, 0.0

        best_idx = -1
        best_score = 0.0

        for i, candidate in enumerate(candidate_texts):
            score = self.cosine_similarity(query_text, candidate)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx, best_score

    def semantic_commitment_evidence_match(
        self, commitment_text: str, evidence_text: str, threshold: float = 0.6
    ) -> bool:
        """
        Check if evidence text semantically matches a commitment.
        More lenient threshold since evidence may be phrased differently.
        """
        similarity = self.cosine_similarity(commitment_text, evidence_text)
        return similarity >= threshold

    def cluster_similar_texts(
        self, texts: List[str], similarity_threshold: float = 0.7
    ) -> List[List[int]]:
        """
        Group texts into clusters based on semantic similarity.
        Returns list of clusters, where each cluster is a list of text indices.
        """
        if not texts:
            return []

        clusters = []
        assigned = set()

        for i, text in enumerate(texts):
            if i in assigned:
                continue

            # Start new cluster
            cluster = [i]
            assigned.add(i)

            # Find similar texts
            for j, other_text in enumerate(texts):
                if j <= i or j in assigned:
                    continue

                if self.cosine_similarity(text, other_text) >= similarity_threshold:
                    cluster.append(j)
                    assigned.add(j)

            clusters.append(cluster)

        return clusters

    def reflection_quality_score(
        self,
        reflection_text: str,
        recent_reflections: List[str],
        commitment_texts: List[str] = None,
    ) -> Dict[str, float]:
        """
        Comprehensive reflection quality assessment.
        Returns scores for novelty, salience, and overall quality.
        """
        scores = {"novelty": 1.0, "salience": 0.5, "overall": 0.5}  # baseline

        # Novelty score against recent reflections
        if recent_reflections:
            scores["novelty"] = self.semantic_novelty_score(
                reflection_text, recent_reflections, threshold=0.8
            )

        # Salience score based on commitment relevance
        if commitment_texts:
            max_commitment_similarity = 0.0
            for commitment in commitment_texts:
                similarity = self.cosine_similarity(reflection_text, commitment)
                max_commitment_similarity = max(max_commitment_similarity, similarity)
            scores["salience"] = max_commitment_similarity

        # Overall quality combines novelty and salience
        scores["overall"] = scores["novelty"] * 0.6 + scores["salience"] * 0.4

        return scores

    def clear_cache(self):
        """Clear embedding cache to free memory."""
        self._embedding_cache.clear()


# Global analyzer instance for reuse
_semantic_analyzer = None


def get_semantic_analyzer() -> SemanticAnalyzer:
    """Get global semantic analyzer instance."""
    global _semantic_analyzer
    if _semantic_analyzer is None:
        _semantic_analyzer = SemanticAnalyzer()
    return _semantic_analyzer
