# pmm/commitment_ttl.py
from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import threading
import re


@dataclass
class TTLCommitment:
    """Commitment with TTL and type classification."""

    text: str
    kind: str  # "ask_deeper", "summarize_user", "explore_topic", etc.
    created_at: float  # epoch seconds
    expires_at: float  # epoch seconds
    source: str = "reflection"
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def is_expired(self) -> bool:
        """Check if commitment has expired."""
        return time.time() > self.expires_at

    def time_remaining(self) -> float:
        """Get remaining time in seconds."""
        return max(0, self.expires_at - time.time())

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "kind": self.kind,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TTLCommitment":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            kind=data["kind"],
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            source=data.get("source", "reflection"),
            metadata=data.get("metadata", {}),
        )


class CommitmentTTLManager:
    """Manages commitments with TTL and type-based deduplication."""

    def __init__(self, default_ttl_hours: float = 24.0):
        self.default_ttl_hours = default_ttl_hours
        self.commitments: List[TTLCommitment] = []
        self._lock = threading.Lock()

        # Commitment type classifications
        self.commitment_types = {
            "ask_deeper": {
                "patterns": [r"ask.*deeper", r"explore.*further", r"dig.*into"],
                "ttl_hours": 12.0,
                "max_active": 2,
            },
            "summarize_user": {
                "patterns": [
                    r"summariz.*user",
                    r"recap.*conversation",
                    r"review.*discussion",
                ],
                "ttl_hours": 6.0,
                "max_active": 1,
            },
            "explore_topic": {
                "patterns": [
                    r"explore.*topic",
                    r"investigate.*subject",
                    r"research.*area",
                ],
                "ttl_hours": 48.0,
                "max_active": 3,
            },
            "follow_up": {
                "patterns": [r"follow.*up", r"check.*back", r"revisit.*later"],
                "ttl_hours": 72.0,
                "max_active": 2,
            },
            "clarify": {
                "patterns": [
                    r"clarif.*",
                    r"ask.*clarification",
                    r"seek.*understanding",
                ],
                "ttl_hours": 8.0,
                "max_active": 1,
            },
            "generic": {"patterns": [], "ttl_hours": 24.0, "max_active": 5},  # fallback
        }

    def classify_commitment(self, text: str) -> str:
        """Classify commitment by type based on text patterns."""
        text_lower = text.lower()

        for kind, config in self.commitment_types.items():
            if kind == "generic":
                continue

            for pattern in config["patterns"]:
                if re.search(pattern, text_lower):
                    return kind

        return "generic"

    def enqueue_commitment(
        self,
        text: str,
        kind: Optional[str] = None,
        ttl_hours: Optional[float] = None,
        source: str = "reflection",
    ) -> bool:
        """
        Add commitment with automatic deduplication and TTL management.

        Args:
            text: Commitment text
            kind: Optional commitment type (auto-classified if None)
            ttl_hours: Optional TTL in hours (uses type default if None)
            source: Source of commitment

        Returns:
            True if commitment was added, False if deduplicated/rejected
        """
        with self._lock:
            # Auto-classify if kind not provided
            if kind is None:
                kind = self.classify_commitment(text)

            # Get TTL for this commitment type
            if ttl_hours is None:
                ttl_hours = self.commitment_types.get(kind, {}).get(
                    "ttl_hours", self.default_ttl_hours
                )

            # Clean up expired commitments first
            self._cleanup_expired()

            # Check for existing commitment of same kind (replace older)
            existing_indices = []
            for i, commitment in enumerate(self.commitments):
                if commitment.kind == kind:
                    existing_indices.append(i)

            # Remove older commitments of same kind if at max capacity
            max_active = self.commitment_types.get(kind, {}).get("max_active", 5)
            if len(existing_indices) >= max_active:
                # Remove oldest commitments of this kind
                existing_indices.sort(key=lambda i: self.commitments[i].created_at)
                for i in existing_indices[
                    : -max_active + 1
                ]:  # Keep max_active-1, remove rest
                    del self.commitments[i]
                    # Adjust indices after deletion
                    existing_indices = [
                        idx - 1 if idx > i else idx
                        for idx in existing_indices
                        if idx != i
                    ]

            # Check for duplicate text (within same kind)
            for commitment in self.commitments:
                if commitment.kind == kind and self._is_similar_text(
                    text, commitment.text
                ):
                    print(
                        f"🔍 DEBUG: Commitment deduplicated (similar to existing {kind})"
                    )
                    return False

            # Create new commitment
            now = time.time()
            expires_at = now + (ttl_hours * 3600)

            new_commitment = TTLCommitment(
                text=text,
                kind=kind,
                created_at=now,
                expires_at=expires_at,
                source=source,
                metadata={"ttl_hours": ttl_hours},
            )

            self.commitments.append(new_commitment)
            print(f"🔍 DEBUG: Added {kind} commitment (TTL: {ttl_hours}h)")
            return True

    def _is_similar_text(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """Check if two commitment texts are similar."""
        # Simple token-based similarity
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return False

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        similarity = intersection / union if union > 0 else 0
        return similarity > threshold

    def _cleanup_expired(self) -> int:
        """Remove expired commitments. Returns count of removed commitments."""
        initial_count = len(self.commitments)
        self.commitments = [c for c in self.commitments if not c.is_expired()]
        removed_count = initial_count - len(self.commitments)

        if removed_count > 0:
            print(f"🔍 DEBUG: Cleaned up {removed_count} expired commitments")

        return removed_count

    def get_active_commitments(self, kind: Optional[str] = None) -> List[TTLCommitment]:
        """Get active (non-expired) commitments, optionally filtered by kind."""
        with self._lock:
            self._cleanup_expired()

            if kind is None:
                return self.commitments.copy()
            else:
                return [c for c in self.commitments if c.kind == kind]

    def mark_commitment_completed(self, commitment_text: str) -> bool:
        """Mark a commitment as completed (remove it)."""
        with self._lock:
            for i, commitment in enumerate(self.commitments):
                if self._is_similar_text(
                    commitment_text, commitment.text, threshold=0.6
                ):
                    del self.commitments[i]
                    print(f"🔍 DEBUG: Marked {commitment.kind} commitment as completed")
                    return True
            return False

    def get_stats(self) -> Dict:
        """Get statistics about active commitments."""
        with self._lock:
            self._cleanup_expired()

            stats = {
                "total_active": len(self.commitments),
                "by_kind": {},
                "expiring_soon": 0,  # within 1 hour
                "oldest_commitment_age_hours": 0,
                "newest_commitment_age_hours": 0,
            }

            # Count by kind
            for commitment in self.commitments:
                kind = commitment.kind
                if kind not in stats["by_kind"]:
                    stats["by_kind"][kind] = {
                        "count": 0,
                        "avg_ttl_hours": 0,
                        "time_remaining_hours": [],
                    }

                stats["by_kind"][kind]["count"] += 1
                stats["by_kind"][kind]["time_remaining_hours"].append(
                    commitment.time_remaining() / 3600
                )

                # Check if expiring soon
                if commitment.time_remaining() < 3600:  # 1 hour
                    stats["expiring_soon"] += 1

            # Calculate averages
            for kind_stats in stats["by_kind"].values():
                if kind_stats["time_remaining_hours"]:
                    kind_stats["avg_time_remaining_hours"] = sum(
                        kind_stats["time_remaining_hours"]
                    ) / len(kind_stats["time_remaining_hours"])
                    del kind_stats["time_remaining_hours"]  # Remove raw data

            # Age statistics
            if self.commitments:
                now = time.time()
                ages = [(now - c.created_at) / 3600 for c in self.commitments]
                stats["oldest_commitment_age_hours"] = max(ages)
                stats["newest_commitment_age_hours"] = min(ages)

            return stats

    def force_cleanup(self) -> Dict:
        """Force cleanup of all commitments and return cleanup stats."""
        with self._lock:
            initial_count = len(self.commitments)
            expired_count = self._cleanup_expired()

            return {
                "initial_count": initial_count,
                "expired_removed": expired_count,
                "remaining_active": len(self.commitments),
            }
