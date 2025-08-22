# pmm/atomic_reflection.py
from __future__ import annotations
from typing import List, Dict, Any
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pmm.config.models import (
    get_min_embedding_threshold,
    get_threshold_cooldown_turns,
)


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
        # Allow override via env var; default to a more permissive 0.975
        # so that only near-identical insights are rejected by embeddings.
        env_threshold = os.getenv("PMM_EMBEDDING_THRESHOLD")
        if env_threshold is not None:
            try:
                self.embedding_threshold = float(env_threshold)
            except ValueError:
                # Fallback if env var is malformed
                self.embedding_threshold = 0.975
        else:
            self.embedding_threshold = (
                float(embedding_threshold) if embedding_threshold is not None else 0.975
            )
        self._lock = threading.Lock()

        # Adaptive dedup configuration
        flag = os.getenv("PMM_ADAPTIVE_DEDUP")
        # Enable adaptive dedup by default; allow env to explicitly disable
        if flag is None:
            self._adaptive_enabled = True
        else:
            self._adaptive_enabled = flag.lower() in ("1", "true", "yes", "on")
        # Start with configured threshold and adapt within bounds
        self._effective_threshold = self.embedding_threshold
        # Min threshold is env-driven to avoid over-aggressive dedup
        self._min_thresh = float(get_min_embedding_threshold())
        self._max_thresh = 0.95
        self._step = 0.02
        self._accept_streak = 0
        self._reject_streak = 0
        # Cooldown turns between adjustments to prevent oscillation
        self._cooldown_turns = int(get_threshold_cooldown_turns())
        self._cooldown_remaining = 0

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
            # Step 1: Basic validation (with normalization to reduce style boilerplate)
            cleaned_content = self._clean_and_normalize(insight_content)
            if not self._passes_basic_validation(cleaned_content):
                print("🔍 DEBUG: Insight failed basic validation")
                return False

            # Step 2: Fast text similarity check
            if self._is_duplicate_text(cleaned_content):
                print("🔍 DEBUG: Insight rejected - duplicate text similarity")
                return False

            # Step 3: Embedding similarity check (more expensive), with first-hit pass for borderline cases
            dup_check = self._is_duplicate_embedding(
                cleaned_content, return_best_sim=True
            )
            if isinstance(dup_check, tuple):
                is_dup, best_sim = dup_check
            else:
                is_dup, best_sim = bool(dup_check), None

            if is_dup:
                # Allow a single borderline near-duplicate if we've had no recent accepted insights
                if self._first_hit_pass_allowed(best_sim):
                    telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                    if telemetry:
                        print(
                            f"[PMM_TELEMETRY] dedup_override: reason=first_hit_pass, sim={(best_sim if best_sim is not None else -1):.3f}, threshold={self._effective_threshold:.3f}"
                        )
                else:
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

        # Light style normalization to reduce false dedup (remove stock openers/closers)
        lowered = normalized.lower()
        boilerplate_prefixes = (
            "reflection:",
            "micro-reflection:",
            "insight:",
            "i realize that ",
            "i've realized that ",
            "i am noticing that ",
            "i notice that ",
        )
        for pref in boilerplate_prefixes:
            if lowered.startswith(pref):
                normalized = normalized[len(pref) :].lstrip()
                lowered = normalized.lower()
                break

        boilerplate_suffixes = (
            " this suggests i should ",
            " therefore, i will ",
        )
        for suff in boilerplate_suffixes:
            if lowered.endswith(suff):
                normalized = normalized[: -len(suff)].rstrip()
                lowered = normalized.lower()
                break

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

    def _is_duplicate_embedding(
        self, content: str, return_best_sim: bool = False
    ) -> bool | tuple[bool, float]:
        """Check embedding similarity against recent insights.

        If return_best_sim=True, returns a tuple (is_duplicate, best_similarity).
        """
        try:
            # Get recent insights from PMM
            recent_insights = self.pmm.model.self_knowledge.insights[-8:]
            if not recent_insights:
                return (False, -1.0) if return_best_sim else False

            # Skip embedding check if OpenAI API not available or embeddings disabled
            try:
                from openai import OpenAI
                import os

                if not os.getenv("OPENAI_API_KEY"):
                    print(
                        "🔍 DEBUG: No OpenAI API key - skipping embedding deduplication"
                    )
                    return (False, -1.0) if return_best_sim else False

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

                    telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )

                    if new_refs:
                        if telemetry:
                            print(
                                f"[PMM_TELEMETRY] dedup_override: reason=new_references, sim={best_sim:.3f}, threshold={self._effective_threshold:.3f}, added_refs={sorted(list(new_refs))}"
                            )
                        # Bypass duplicate rejection since candidate adds new references
                        return (False, best_sim) if return_best_sim else False

                    # No new references; treat as duplicate
                    if telemetry:
                        print(
                            f"[PMM_TELEMETRY] dedup_reject: reason=high_similarity, sim={best_sim:.3f}, threshold={self._effective_threshold:.3f}, gates=embedding"
                        )
                    return (True, best_sim) if return_best_sim else True

                # Not similar enough to consider duplicate
                return (False, best_sim) if return_best_sim else False
            except Exception as e:
                print(f"🔍 DEBUG: Embedding check failed, allowing insight: {e}")
                return (False, -1.0) if return_best_sim else False

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
            "min_threshold_cap": self._min_thresh,
            "max_threshold_cap": self._max_thresh,
            "cooldown_turns": self._cooldown_turns,
            "cooldown_remaining": self._cooldown_remaining,
            "accept_streak": self._accept_streak,
            "reject_streak": self._reject_streak,
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

    def _first_hit_pass_allowed(self, best_sim: float | None) -> bool:
        """Allow one borderline acceptance when no recent insight was accepted.

        Policy:
        - If there have been zero insights in the last 10 minutes, allow a pass when
          similarity is within a narrow band above the threshold (<= threshold + 0.02).
        - Otherwise, do not allow.
        """
        try:
            # Require borderline range if best_sim provided
            if best_sim is not None and best_sim > (self._effective_threshold + 0.02):
                return False

            insights = self.pmm.model.self_knowledge.insights
            if not insights:
                return True  # cold start

            from datetime import datetime, timezone
            import dateutil.parser as dp  # lightweight, typically available; if not, fallback below

            now = datetime.now(timezone.utc)
            last_ts = (
                insights[-1].timestamp
                if getattr(insights[-1], "timestamp", None)
                else None
            )
            if last_ts:
                try:
                    last_dt = dp.isoparse(last_ts)
                except Exception:
                    # naive parse: assume UTC
                    last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                delta = (now - last_dt).total_seconds()
                return delta >= 600.0  # 10 minutes
            # If no timestamp, be permissive
            return True
        except Exception:
            # On any failure, do not block acceptance
            return True

    # --- Adaptive threshold helpers ---
    def _on_decision(self, accepted: bool) -> None:
        """Update adaptive state on accept/reject and adjust threshold if enabled."""
        if not self._adaptive_enabled:
            return

        telemetry = os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on")

        # decrement cooldown if active
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if accepted:
            self._accept_streak += 1
            self._reject_streak = 0
            # If we keep accepting, become a bit stricter to catch more dups
            if self._accept_streak >= 2 and self._cooldown_remaining == 0:
                old = self._effective_threshold
                self._effective_threshold = max(
                    self._min_thresh, round(self._effective_threshold - self._step, 4)
                )
                self._cooldown_remaining = self._cooldown_turns
                if telemetry:
                    print(
                        f"[PMM_TELEMETRY] dedup_adapt: accepted streak={self._accept_streak}, threshold {old:.3f} -> {self._effective_threshold:.3f}, cooldown={self._cooldown_turns}"
                    )
        else:
            self._reject_streak += 1
            self._accept_streak = 0
            # If we keep rejecting as duplicates, relax slightly to allow more acceptance
            if self._reject_streak >= 2 and self._cooldown_remaining == 0:
                old = self._effective_threshold
                self._effective_threshold = min(
                    self._max_thresh, round(self._effective_threshold + self._step, 4)
                )
                self._cooldown_remaining = self._cooldown_turns
                if telemetry:
                    print(
                        f"[PMM_TELEMETRY] dedup_adapt: rejected streak={self._reject_streak}, threshold {old:.3f} -> {self._effective_threshold:.3f}, cooldown={self._cooldown_turns}"
                    )
