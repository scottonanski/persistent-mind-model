# pmm/atomic_reflection.py
from __future__ import annotations
from typing import List, Dict, Any
import threading
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ReflectionCandidate:
    """A reflection candidate for validation and persistence."""

    content: str
    timestamp: datetime
    source_events: List[str]
    model_config: Dict[str, Any]
    epoch: int


class AtomicReflectionManager:
    """Manages atomic reflection validation → dedup → persist pipeline."""

    def __init__(self, pmm_manager, embedding_threshold: float = 0.85):
        self.pmm = pmm_manager
        self.embedding_threshold = embedding_threshold
        self._lock = threading.Lock()

        # Cache recent insights for fast text similarity
        self._recent_insights_cache = []
        self._max_cache_size = 10

    def add_insight(
        self, insight_content: str, model_config: Dict[str, Any], epoch: int
    ) -> bool:
        """
        Atomic insight addition: validate → dedup → persist.
        Returns True if insight was added, False if rejected.
        """
        with self._lock:
            # Step 1: Basic validation
            cleaned_content = self._clean_and_normalize(insight_content)
            if not self._passes_basic_validation(cleaned_content):
                print("🔍 DEBUG: Insight failed basic validation")
                return False

            # Step 2: Fast text similarity check
            if self._is_duplicate_text(cleaned_content):
                print("🔍 DEBUG: Insight rejected - duplicate text similarity")
                return False

            # Step 3: Embedding similarity check (more expensive)
            if self._is_duplicate_embedding(cleaned_content):
                print(
                    f"🔍 DEBUG: Insight rejected - embedding similarity > {self.embedding_threshold}"
                )
                return False

            # Step 4: Epoch validation (ensure model hasn't changed)
            from pmm.llm_factory import get_llm_factory

            current_epoch = get_llm_factory().get_current_epoch()
            if epoch != current_epoch:
                print(
                    f"🔍 DEBUG: Insight rejected - epoch mismatch {epoch} != {current_epoch}"
                )
                return False

            # Step 5: Atomic persistence
            try:
                success = self._persist_insight(cleaned_content, model_config)
                if success:
                    self._update_cache(cleaned_content)
                    print("🔍 DEBUG: Insight successfully persisted")
                    return True
                else:
                    print("🔍 DEBUG: Insight persistence failed")
                    return False
            except Exception as e:
                print(f"🔍 DEBUG: Insight persistence error: {e}")
                return False

    def _clean_and_normalize(self, content: str) -> str:
        """Clean and normalize insight content."""
        if not content:
            return ""

        # Apply stance filter
        from pmm.stance_filter import StanceFilter

        stance_filter = StanceFilter()
        filtered_content, _ = stance_filter.filter_response(content)

        # Normalize whitespace
        import re

        normalized = re.sub(r"\s+", " ", filtered_content.strip())

        return normalized

    def _passes_basic_validation(self, content: str) -> bool:
        """Basic validation checks."""
        if not content or len(content.strip()) < 10:
            return False

        if len(content) > 2000:  # Too long
            return False

        # Check for obvious spam patterns
        words = content.lower().split()
        if len(set(words)) < len(words) * 0.3:  # Too repetitive
            return False

        return True

    def _is_duplicate_text(self, content: str) -> bool:
        """Fast text similarity check using cached insights."""
        if not self._recent_insights_cache:
            return False

        # Simple token-based similarity
        content_tokens = set(content.lower().split())

        for cached_insight in self._recent_insights_cache:
            cached_tokens = set(cached_insight.lower().split())

            if not cached_tokens:
                continue

            # Jaccard similarity
            intersection = len(content_tokens & cached_tokens)
            union = len(content_tokens | cached_tokens)

            if union > 0:
                similarity = intersection / union
                if similarity > 0.7:  # Fast rejection threshold
                    return True

        return False

    def _is_duplicate_embedding(self, content: str) -> bool:
        """Check embedding similarity against recent insights."""
        try:
            # Get recent insights from PMM
            recent_insights = self.pmm.model.self_knowledge.insights[-8:]
            if not recent_insights:
                return False

            # Skip embedding check if OpenAI API not available or embeddings disabled
            try:
                from openai import OpenAI
                import os

                if not os.getenv("OPENAI_API_KEY"):
                    print(
                        "🔍 DEBUG: No OpenAI API key - skipping embedding deduplication"
                    )
                    return False

                client = OpenAI()
                new_embedding = (
                    client.embeddings.create(
                        input=content.strip(), model="text-embedding-ada-002"
                    )
                    .data[0]
                    .embedding
                )

                # Compare with recent insights
                for insight in recent_insights:
                    if not insight.content:
                        continue

                    existing_embedding = (
                        client.embeddings.create(
                            input=insight.content.strip(),
                            model="text-embedding-ada-002",
                        )
                        .data[0]
                        .embedding
                    )

                    # Calculate cosine similarity
                    import numpy as np

                    similarity = np.dot(new_embedding, existing_embedding) / (
                        np.linalg.norm(new_embedding)
                        * np.linalg.norm(existing_embedding)
                    )

                    if similarity > self.embedding_threshold:
                        print(f"🔍 DEBUG: High embedding similarity: {similarity:.3f}")
                        return True

                return False
            except Exception as e:
                print(f"🔍 DEBUG: Embedding check failed, allowing insight: {e}")
                return False

        except Exception as e:
            print(f"🔍 DEBUG: Embedding similarity check failed: {e}")
            # Fallback to text similarity
            return False

    def _persist_insight(self, content: str, model_config: Dict[str, Any]) -> bool:
        """Atomically persist insight to PMM."""
        try:
            # Create insight object
            from pmm.model import Insight

            insight = Insight(
                content=content,
                timestamp=datetime.now(timezone.utc).isoformat(),
                confidence=0.8,  # Default confidence
                source="reflection",
                tags=["atomic_reflection"],
                metadata={
                    "model_provider": model_config.get("provider", "unknown"),
                    "model_name": model_config.get("name", "unknown"),
                    "temperature": model_config.get("temperature", 0.3),
                },
            )

            # Add to PMM model
            self.pmm.model.self_knowledge.insights.append(insight)

            # Save to disk
            self.pmm.save_model()

            return True

        except Exception as e:
            print(f"🔍 DEBUG: Failed to persist insight: {e}")
            return False

    def _update_cache(self, content: str) -> None:
        """Update the recent insights cache."""
        self._recent_insights_cache.append(content)

        # Trim cache if too large
        if len(self._recent_insights_cache) > self._max_cache_size:
            self._recent_insights_cache = self._recent_insights_cache[
                -self._max_cache_size :
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about reflection processing."""
        return {
            "cache_size": len(self._recent_insights_cache),
            "embedding_threshold": self.embedding_threshold,
            "total_insights": len(self.pmm.model.self_knowledge.insights),
            "recent_insights_preview": (
                self._recent_insights_cache[-3:] if self._recent_insights_cache else []
            ),
        }
