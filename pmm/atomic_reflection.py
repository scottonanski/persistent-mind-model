# pmm/atomic_reflection.py
from __future__ import annotations
from typing import List, Dict, Any
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
import os


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

    def __init__(self, pmm_manager, embedding_threshold: float | None = None):
        self.pmm = pmm_manager
        # Allow override via env var; default to a slightly more permissive 0.92
        # so that only near-identical insights are rejected by embeddings.
        env_threshold = os.getenv("PMM_EMBEDDING_THRESHOLD")
        if env_threshold is not None:
            try:
                self.embedding_threshold = float(env_threshold)
            except ValueError:
                # Fallback if env var is malformed
                self.embedding_threshold = 0.92
        else:
            self.embedding_threshold = (
                float(embedding_threshold) if embedding_threshold is not None else 0.92
            )
        self._lock = threading.Lock()

        # Adaptive dedup configuration
        flag = os.getenv("PMM_ADAPTIVE_DEDUP", "").lower()
        self._adaptive_enabled = flag in ("1", "true", "yes", "on")
        # Start with configured threshold and adapt within bounds
        self._effective_threshold = self.embedding_threshold
        self._min_thresh = 0.80
        self._max_thresh = 0.95
        self._step = 0.02
        self._accept_streak = 0
        self._reject_streak = 0

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
                    f"🔍 DEBUG: Insight rejected - embedding similarity > {self._effective_threshold}"
                )
                # Adaptive: rejection event
                self._on_decision(accepted=False)
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
                    # Adaptive: acceptance event
                    self._on_decision(accepted=True)
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

                # Compute similarity against recent insights and pick the best match
                import numpy as np

                best_sim = -1.0
                best_insight = None
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

                    sim = np.dot(new_embedding, existing_embedding) / (
                        np.linalg.norm(new_embedding)
                        * np.linalg.norm(existing_embedding)
                    )

                    if sim > best_sim:
                        best_sim = sim
                        best_insight = insight

                if best_sim > self._effective_threshold and best_insight is not None:
                    print(f"🔍 DEBUG: High embedding similarity: {best_sim:.3f}")

                    # Structured override: allow near-duplicates if they reference NEW IDs
                    candidate_refs = self._extract_referenced_ids(content)
                    prior_refs = self._extract_referenced_ids(best_insight.content)
                    new_refs = candidate_refs - prior_refs

                    telemetry = os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on")

                    if new_refs:
                        if telemetry:
                            print(
                                f"[PMM_TELEMETRY] dedup_override: reason=new_references, sim={best_sim:.3f}, threshold={self._effective_threshold:.3f}, added_refs={sorted(list(new_refs))}"
                            )
                        # Bypass duplicate rejection since candidate adds new references
                        return False

                    # No new references; treat as duplicate
                    if telemetry:
                        print(
                            f"[PMM_TELEMETRY] dedup_reject: reason=high_similarity, sim={best_sim:.3f}, threshold={self._effective_threshold:.3f}, gates=embedding"
                        )
                    return True

                # Not similar enough to consider duplicate
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
            "embedding_threshold_configured": self.embedding_threshold,
            "embedding_threshold_effective": self._effective_threshold,
            "adaptive_enabled": self._adaptive_enabled,
            "total_insights": len(self.pmm.model.self_knowledge.insights),
            "recent_insights_preview": (
                self._recent_insights_cache[-3:] if self._recent_insights_cache else []
            ),
        }

    def _extract_referenced_ids(self, text: str) -> set[str]:
        """Extract referenced event IDs and short commitment hashes from text.

        - Event IDs: ev123 or forms that include ev followed by digits
        - Commitment hashes: 16-char lowercase hex substrings
        """
        if not text:
            return set()

        try:
            import re

            refs: set[str] = set()

            # Event IDs: match 'ev123' (case-insensitive), normalize to lowercase
            for m in re.findall(r"\bev\d+\b", text, flags=re.IGNORECASE):
                refs.add(m.lower())

            # 16-char hex (commitment short hashes)
            for m in re.findall(r"\b[a-f0-9]{16}\b", text.lower()):
                refs.add(m)

            return refs
        except Exception:
            return set()

    # --- Adaptive threshold helpers ---
    def _on_decision(self, accepted: bool) -> None:
        """Update adaptive state on accept/reject and adjust threshold if enabled."""
        if not self._adaptive_enabled:
            return

        telemetry = os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on")

        if accepted:
            self._accept_streak += 1
            self._reject_streak = 0
            # If we keep accepting, become a bit stricter to catch more dups
            if self._accept_streak >= 2:
                old = self._effective_threshold
                self._effective_threshold = max(self._min_thresh, round(self._effective_threshold - self._step, 4))
                if telemetry:
                    print(
                        f"[PMM_TELEMETRY] dedup_adapt: accepted streak={self._accept_streak}, threshold {old:.3f} -> {self._effective_threshold:.3f}"
                    )
        else:
            self._reject_streak += 1
            self._accept_streak = 0
            # If we keep rejecting as duplicates, relax slightly to allow more acceptance
            if self._reject_streak >= 2:
                old = self._effective_threshold
                self._effective_threshold = min(self._max_thresh, round(self._effective_threshold + self._step, 4))
                if telemetry:
                    print(
                        f"[PMM_TELEMETRY] dedup_adapt: rejected streak={self._reject_streak}, threshold {old:.3f} -> {self._effective_threshold:.3f}"
                    )
