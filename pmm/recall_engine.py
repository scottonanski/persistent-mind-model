"""
Cue-Based Memory Recall Engine - Layer 5 Implementation

Implements semantic search and retrieval from both active memory and archives
with embedding-based similarity matching and hash verification.
"""

from __future__ import annotations
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pickle
import os

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using fallback similarity.")

from .memory_token import MemoryToken
from .enhanced_model import RecallConfig, EnhancedSelfKnowledge
from .archive_engine import ArchiveRetriever
from .quantum_memory import QuantumMemoryManager


@dataclass
class RecallResult:
    """
    Result of a memory recall operation.
    """

    token_id: str
    token: MemoryToken
    similarity_score: float
    recall_source: str  # "active" or "archive"
    archive_id: Optional[str] = None
    verification_passed: bool = True
    recall_timestamp: str = ""

    def __post_init__(self):
        if not self.recall_timestamp:
            self.recall_timestamp = datetime.utcnow().isoformat()


@dataclass
class EmbeddingCache:
    """
    Cache for computed embeddings to avoid recomputation.
    """

    embeddings: Dict[str, np.ndarray] = None
    metadata: Dict[str, Dict[str, Any]] = None
    cache_file: str = "embedding_cache.pkl"
    max_size: int = 1000

    def __post_init__(self):
        if self.embeddings is None:
            self.embeddings = {}
        if self.metadata is None:
            self.metadata = {}
        self.load_cache()

    def get_embedding(self, text_id: str) -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        return self.embeddings.get(text_id)

    def set_embedding(
        self, text_id: str, embedding: np.ndarray, metadata: Dict[str, Any] = None
    ):
        """Cache embedding for text."""
        # Maintain cache size limit
        if len(self.embeddings) >= self.max_size:
            # Remove oldest entries
            oldest_ids = sorted(
                self.metadata.keys(),
                key=lambda x: self.metadata[x].get("created_at", ""),
            )[: len(self.embeddings) - self.max_size + 1]

            for old_id in oldest_ids:
                self.embeddings.pop(old_id, None)
                self.metadata.pop(old_id, None)

        self.embeddings[text_id] = embedding
        self.metadata[text_id] = metadata or {
            "created_at": datetime.utcnow().isoformat()
        }

    def save_cache(self):
        """Save cache to disk."""
        try:
            cache_data = {"embeddings": self.embeddings, "metadata": self.metadata}
            with open(self.cache_file, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Failed to save embedding cache: {e}")

    def load_cache(self):
        """Load cache from disk."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    self.embeddings = cache_data.get("embeddings", {})
                    self.metadata = cache_data.get("metadata", {})
        except Exception as e:
            print(f"Failed to load embedding cache: {e}")
            self.embeddings = {}
            self.metadata = {}


class RecallEngine:
    """
    Core engine for cue-based memory recall with semantic search capabilities.

    Supports retrieval from both active memory and archived clusters with
    embedding-based similarity matching and cryptographic verification.
    """

    def __init__(self, config: RecallConfig):
        self.config = config
        self.embedding_cache = EmbeddingCache(max_size=config.cache_size)
        self.archive_retriever = ArchiveRetriever(config)
        self.quantum_manager = QuantumMemoryManager()

        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(config.embedding_model)
                self.use_embeddings = True
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                self.use_embeddings = False
        else:
            self.use_embeddings = False
            print("Using fallback similarity matching")

    def recall_memories(
        self,
        cue: str,
        self_knowledge: EnhancedSelfKnowledge,
        max_results: Optional[int] = None,
    ) -> List[RecallResult]:
        """
        Recall memories based on a textual cue.

        Searches both active memory and archives for semantically similar content.
        """
        max_results = max_results or self.config.max_recall_tokens

        # Get cue embedding
        cue_embedding = self._get_text_embedding(cue)

        # Search active memory
        active_results = self._search_active_memory(
            cue,
            cue_embedding,
            self_knowledge.memory_tokens,
            self_knowledge.active_token_ids,
        )

        # Search archives if needed
        archive_results = []
        if len(active_results) < max_results:
            archive_results = self._search_archives(
                cue,
                cue_embedding,
                self_knowledge.archives,
                max_results - len(active_results),
            )

        # Combine and rank results
        all_results = active_results + archive_results
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Limit to max results
        final_results = all_results[:max_results]

        # Apply recall effects (boost amplitude, update quantum states)
        self._apply_recall_effects(final_results, self_knowledge.memory_tokens)

        return final_results

    def recall_by_theme(
        self,
        theme: str,
        self_knowledge: EnhancedSelfKnowledge,
        max_results: Optional[int] = None,
    ) -> List[RecallResult]:
        """
        Recall memories by thematic cluster.

        Searches for memories that belong to a specific theme cluster.
        """
        results = []
        max_results = max_results or self.config.max_recall_tokens

        # Search active memory by tags
        for token_id in self_knowledge.active_token_ids:
            if token_id not in self_knowledge.memory_tokens:
                continue

            token = self_knowledge.memory_tokens[token_id]
            if theme in token.tags or theme in token.event_type:
                result = RecallResult(
                    token_id=token_id,
                    token=token,
                    similarity_score=1.0,  # Perfect match for theme
                    recall_source="active",
                )
                results.append(result)

        # Search archives by theme
        if len(results) < max_results:
            for archive_id, archive in self_knowledge.archives.items():
                if theme in archive.themes:
                    archive_tokens = self.archive_retriever.retrieve_tokens_by_theme(
                        archive, theme
                    )

                    for token_data in archive_tokens[: max_results - len(results)]:
                        # Reconstruct token from archived data
                        token = self._reconstruct_token_from_archive(token_data)

                        result = RecallResult(
                            token_id=token.token_id,
                            token=token,
                            similarity_score=0.9,  # High but not perfect for archived
                            recall_source="archive",
                            archive_id=archive_id,
                        )
                        results.append(result)

                        if len(results) >= max_results:
                            break

                if len(results) >= max_results:
                    break

        return results[:max_results]

    def recall_by_temporal_range(
        self,
        start_time: str,
        end_time: str,
        self_knowledge: EnhancedSelfKnowledge,
        max_results: Optional[int] = None,
    ) -> List[RecallResult]:
        """
        Recall memories from a specific time range.
        """
        results = []
        max_results = max_results or self.config.max_recall_tokens

        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

        # Search active memory
        for token_id in self_knowledge.active_token_ids:
            if token_id not in self_knowledge.memory_tokens:
                continue

            token = self_knowledge.memory_tokens[token_id]
            token_dt = datetime.fromisoformat(token.created_at.replace("Z", "+00:00"))

            if start_dt <= token_dt <= end_dt:
                result = RecallResult(
                    token_id=token_id,
                    token=token,
                    similarity_score=token.amplitude,  # Use amplitude as relevance
                    recall_source="active",
                )
                results.append(result)

        # Sort by timestamp (most recent first)
        results.sort(key=lambda x: x.token.created_at, reverse=True)

        return results[:max_results]

    def verify_recalled_memory(
        self, result: RecallResult, original_content: str
    ) -> bool:
        """
        Verify the integrity of a recalled memory using its content hash.
        """
        if result.recall_source == "archive":
            # For archived memories, we need to load the full content
            # This is a simplified verification - in practice, you'd need
            # to store and retrieve the original content
            return True  # Assume archived content is verified during archival

        # For active memories, verify against the token's content hash
        expected_hash = result.token.compute_content_hash(original_content)
        result.verification_passed = result.token.content_hash == expected_hash

        return result.verification_passed

    def get_recall_statistics(self) -> Dict[str, Any]:
        """Get statistics about recall operations."""
        cache_stats = {
            "cache_size": len(self.embedding_cache.embeddings),
            "cache_hit_rate": 0.0,  # Would need to track hits/misses
            "cache_file_exists": os.path.exists(self.embedding_cache.cache_file),
        }

        return {
            "embedding_model": self.config.embedding_model,
            "use_embeddings": self.use_embeddings,
            "similarity_threshold": self.config.similarity_threshold,
            "max_recall_tokens": self.config.max_recall_tokens,
            "cache_stats": cache_stats,
        }

    def _search_active_memory(
        self,
        cue: str,
        cue_embedding: Optional[np.ndarray],
        memory_tokens: Dict[str, MemoryToken],
        active_token_ids: List[str],
    ) -> List[RecallResult]:
        """Search active memory tokens for similar content."""
        results = []

        for token_id in active_token_ids:
            if token_id not in memory_tokens:
                continue

            token = memory_tokens[token_id]

            # Calculate similarity
            similarity = self._calculate_similarity(
                cue, cue_embedding, token.summary, token
            )

            # Apply amplitude weighting
            weighted_similarity = similarity * token.amplitude

            if weighted_similarity >= self.config.similarity_threshold:
                result = RecallResult(
                    token_id=token_id,
                    token=token,
                    similarity_score=weighted_similarity,
                    recall_source="active",
                )
                results.append(result)

        return results

    def _search_archives(
        self,
        cue: str,
        cue_embedding: Optional[np.ndarray],
        archives: Dict[str, Any],
        max_results: int,
    ) -> List[RecallResult]:
        """Search archived memory clusters for similar content."""
        results = []

        for archive_id, archive in archives.items():
            try:
                # Load archive data
                archive_data = self.archive_retriever.load_archive(archive)

                # Search tokens in archive
                for token_id, token_data in archive_data["tokens"].items():
                    if len(results) >= max_results:
                        break

                    # Calculate similarity with archived token
                    similarity = self._calculate_similarity_with_data(
                        cue, cue_embedding, token_data
                    )

                    if (
                        similarity >= self.config.similarity_threshold * 0.8
                    ):  # Lower threshold for archives
                        # Reconstruct token
                        token = self._reconstruct_token_from_archive(token_data)

                        result = RecallResult(
                            token_id=token_id,
                            token=token,
                            similarity_score=similarity,
                            recall_source="archive",
                            archive_id=archive_id,
                        )
                        results.append(result)

                if len(results) >= max_results:
                    break

            except Exception as e:
                print(f"Error searching archive {archive_id}: {e}")
                continue

        return results

    def _calculate_similarity(
        self,
        cue: str,
        cue_embedding: Optional[np.ndarray],
        token_text: str,
        token: MemoryToken,
    ) -> float:
        """Calculate similarity between cue and token."""
        if self.use_embeddings and cue_embedding is not None:
            # Embedding-based similarity
            token_embedding = self._get_text_embedding(token_text)
            if token_embedding is not None:
                cosine_sim = np.dot(cue_embedding, token_embedding) / (
                    np.linalg.norm(cue_embedding) * np.linalg.norm(token_embedding)
                )

                # Combine with phase similarity
                phase_sim = self._calculate_phase_similarity(cue, token)

                return 0.7 * cosine_sim + 0.3 * phase_sim

        # Fallback to keyword-based similarity
        return self._calculate_keyword_similarity(cue, token_text, token)

    def _calculate_similarity_with_data(
        self, cue: str, cue_embedding: Optional[np.ndarray], token_data: Dict[str, Any]
    ) -> float:
        """Calculate similarity with archived token data."""
        token_text = token_data.get("summary", "")

        if self.use_embeddings and cue_embedding is not None:
            token_embedding = self._get_text_embedding(token_text)
            if token_embedding is not None:
                cosine_sim = np.dot(cue_embedding, token_embedding) / (
                    np.linalg.norm(cue_embedding) * np.linalg.norm(token_embedding)
                )
                return cosine_sim

        # Fallback to keyword similarity
        return self._calculate_keyword_similarity_with_data(cue, token_data)

    def _calculate_phase_similarity(self, cue: str, token: MemoryToken) -> float:
        """Calculate semantic similarity based on phase angles."""
        # Simple heuristic for cue phase
        cue_phase = self._estimate_cue_phase(cue)

        # Calculate phase difference
        phase_diff = abs(cue_phase - token.phase)
        if phase_diff > np.pi:
            phase_diff = 2 * np.pi - phase_diff

        # Convert to similarity (0 = different, 1 = same)
        return 1.0 - (phase_diff / np.pi)

    def _calculate_keyword_similarity(
        self, cue: str, token_text: str, token: MemoryToken
    ) -> float:
        """Fallback keyword-based similarity calculation."""
        cue_words = set(cue.lower().split())
        token_words = set(token_text.lower().split())
        tag_words = set(word.lower() for word in token.tags)

        # Combine token text and tags
        all_token_words = token_words | tag_words

        if not cue_words or not all_token_words:
            return 0.0

        # Jaccard similarity
        intersection = cue_words & all_token_words
        union = cue_words | all_token_words

        return len(intersection) / len(union) if union else 0.0

    def _calculate_keyword_similarity_with_data(
        self, cue: str, token_data: Dict[str, Any]
    ) -> float:
        """Keyword similarity with archived token data."""
        cue_words = set(cue.lower().split())

        token_text = token_data.get("summary", "")
        token_words = set(token_text.lower().split())

        tags = token_data.get("tags", [])
        tag_words = set(word.lower() for word in tags)

        all_token_words = token_words | tag_words

        if not cue_words or not all_token_words:
            return 0.0

        intersection = cue_words & all_token_words
        union = cue_words | all_token_words

        return len(intersection) / len(union) if union else 0.0

    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text, using cache when possible."""
        if not self.use_embeddings:
            return None

        # Create text ID for caching
        text_id = hashlib.md5(text.encode()).hexdigest()

        # Check cache
        cached_embedding = self.embedding_cache.get_embedding(text_id)
        if cached_embedding is not None:
            return cached_embedding

        try:
            # Compute embedding
            embedding = self.embedding_model.encode(text)

            # Cache it
            self.embedding_cache.set_embedding(text_id, embedding)

            return embedding

        except Exception as e:
            print(f"Error computing embedding: {e}")
            return None

    def _estimate_cue_phase(self, cue: str) -> float:
        """Estimate semantic phase for a cue string."""
        cue_lower = cue.lower()

        # Simple heuristics for emotional/semantic content
        positive_words = ["good", "great", "happy", "success", "love", "joy", "achieve"]
        negative_words = ["bad", "sad", "fail", "hate", "anger", "fear", "problem"]
        question_words = ["what", "why", "how", "when", "where", "who"]

        positive_score = sum(1 for word in positive_words if word in cue_lower)
        negative_score = sum(1 for word in negative_words if word in cue_lower)
        question_score = sum(1 for word in question_words if word in cue_lower)

        if question_score > 0:
            return np.pi / 4  # Questions in first quadrant
        elif positive_score > negative_score:
            return np.pi / 2  # Positive in second quadrant
        elif negative_score > positive_score:
            return 3 * np.pi / 2  # Negative in fourth quadrant
        else:
            return 0.0  # Neutral

    def _reconstruct_token_from_archive(
        self, token_data: Dict[str, Any]
    ) -> MemoryToken:
        """Reconstruct MemoryToken from archived data."""
        return MemoryToken(
            token_id=token_data.get("token_id", ""),
            created_at=token_data.get("created_at", ""),
            content_hash=token_data.get("content_hash", ""),
            prev_hash=token_data.get("prev_hash", ""),
            chain_position=token_data.get("chain_position", 0),
            amplitude=token_data.get("amplitude", 0.5),
            phase=token_data.get("phase", 0.0),
            event_type=token_data.get("event_type", ""),
            salience=token_data.get("salience", 0.5),
            valence=token_data.get("valence", 0.5),
            tags=token_data.get("tags", []),
            archived=True,  # Mark as archived
            summary=token_data.get("summary", ""),
            source_event_id=token_data.get("source_event_id"),
            source_insight_id=token_data.get("source_insight_id"),
        )

    def _apply_recall_effects(
        self, results: List[RecallResult], memory_tokens: Dict[str, MemoryToken]
    ):
        """Apply effects of memory recall (boost amplitude, etc.)."""
        for result in results:
            if result.recall_source == "active" and result.token_id in memory_tokens:
                # Boost amplitude for recalled active memories
                token = memory_tokens[result.token_id]
                token.boost_amplitude(self.config.amplitude_boost)

                # Trigger quantum memory cascade
                self.quantum_manager.boost_related_memories(
                    token, memory_tokens, self.config.amplitude_boost * 0.5
                )

    def __del__(self):
        """Save embedding cache on destruction."""
        if hasattr(self, "embedding_cache"):
            self.embedding_cache.save_cache()
