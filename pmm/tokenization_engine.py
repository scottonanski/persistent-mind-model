"""
Memory Tokenization Engine - Layer 2 Implementation

Core engine for converting PMM events/thoughts/insights into cryptographically
verifiable memory tokens with blockchain-style integrity.
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from .memory_token import MemoryToken, MemoryChain, IdentityLockpoint
from .enhanced_model import EnhancedSelfKnowledge
from .model import Event, Thought, Insight


class TokenizationEngine:
    """
    Core engine for memory tokenization with cryptographic integrity.

    Converts traditional PMM events into tamper-evident memory tokens
    while preserving all semantic information and enabling efficient recall.
    """

    def __init__(self, self_knowledge: EnhancedSelfKnowledge):
        self.self_knowledge = self_knowledge
        self.chain = self_knowledge.memory_chain

    def tokenize_event(
        self, event: Event, content: Optional[str] = None
    ) -> MemoryToken:
        """
        Convert PMM Event to cryptographically verifiable MemoryToken.

        Args:
            event: PMM Event object
            content: Full event content (if None, uses event.summary)

        Returns:
            MemoryToken with computed hash and chain linking
        """
        # Use provided content or fall back to event summary
        full_content = content or event.summary

        # Create memory token
        token = MemoryToken(
            token_id=str(uuid.uuid4()),
            created_at=event.t,
            event_type=event.type,
            salience=event.salience,
            valence=event.valence,
            tags=event.tags.copy(),
            source_event_id=event.id,
            summary=event.summary,
            amplitude=1.0,  # New events start fully active
            phase=self._compute_semantic_phase(event),
        )

        # Add to chain and compute hash
        _content_hash = self.chain.add_token(token, full_content)

        # Store in self_knowledge
        self.self_knowledge.memory_tokens[token.token_id] = token
        self.self_knowledge.active_token_ids.append(token.token_id)

        # Update metrics
        self._update_tokenization_metrics()

        return token

    def tokenize_thought(self, thought: Thought) -> MemoryToken:
        """Convert PMM Thought to MemoryToken."""
        token = MemoryToken(
            token_id=str(uuid.uuid4()),
            created_at=thought.t,
            event_type="thought",
            salience=0.6,  # Thoughts generally less salient than events
            valence=0.5,  # Neutral valence for thoughts
            tags=["thought", "internal"],
            summary=(
                thought.content[:200] + "..."
                if len(thought.content) > 200
                else thought.content
            ),
            amplitude=0.8,  # Thoughts start slightly less active than events
            phase=0.0,  # Neutral phase for thoughts
        )

        # Add to chain
        _content_hash = self.chain.add_token(token, thought.content)

        # Store
        self.self_knowledge.memory_tokens[token.token_id] = token
        self.self_knowledge.active_token_ids.append(token.token_id)

        self._update_tokenization_metrics()
        return token

    def tokenize_insight(self, insight: Insight) -> MemoryToken:
        """Convert PMM Insight to MemoryToken with high salience."""
        token = MemoryToken(
            token_id=str(uuid.uuid4()),
            created_at=insight.t,
            event_type="insight",
            salience=0.9,  # Insights are highly salient
            valence=0.7,  # Insights generally positive
            tags=["insight", "reflection", "meta-cognitive"],
            summary=(
                insight.content[:200] + "..."
                if len(insight.content) > 200
                else insight.content
            ),
            amplitude=1.0,  # Insights start fully active
            phase=1.57,  # π/2 radians for "enlightenment" semantic position
        )

        # Add to chain
        _content_hash = self.chain.add_token(token, insight.content)

        # Store with insight reference
        token.source_insight_id = insight.id
        self.self_knowledge.memory_tokens[token.token_id] = token
        self.self_knowledge.active_token_ids.append(token.token_id)

        self._update_tokenization_metrics()
        return token

    def tokenize_custom_content(
        self,
        content: str,
        event_type: str = "custom",
        salience: float = 0.5,
        valence: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> MemoryToken:
        """
        Tokenize arbitrary content with custom parameters.

        Useful for importing external memories or creating synthetic tokens.
        """
        token = MemoryToken(
            token_id=str(uuid.uuid4()),
            created_at=datetime.utcnow().isoformat(),
            event_type=event_type,
            salience=salience,
            valence=valence,
            tags=tags or [],
            summary=content[:200] + "..." if len(content) > 200 else content,
            amplitude=1.0,
            phase=self._compute_content_phase(content),
        )

        # Add to chain
        _content_hash = self.chain.add_token(token, content)

        # Store
        self.self_knowledge.memory_tokens[token.token_id] = token
        self.self_knowledge.active_token_ids.append(token.token_id)

        self._update_tokenization_metrics()
        return token

    def batch_tokenize_legacy_events(self) -> List[MemoryToken]:
        """
        Convert all existing PMM events/thoughts/insights to tokens.

        Used for upgrading legacy PMM models to tokenized format.
        """
        tokens = []

        # Tokenize events
        for event in self.self_knowledge.autobiographical_events:
            if not self._already_tokenized(event.id, "event"):
                token = self.tokenize_event(event)
                tokens.append(token)

        # Tokenize thoughts
        for thought in self.self_knowledge.thoughts:
            if not self._already_tokenized(thought.id, "thought"):
                token = self.tokenize_thought(thought)
                tokens.append(token)

        # Tokenize insights
        for insight in self.self_knowledge.insights:
            if not self._already_tokenized(insight.id, "insight"):
                token = self.tokenize_insight(insight)
                tokens.append(token)

        return tokens

    def verify_token_integrity(self, token_id: str, full_content: str) -> bool:
        """
        Verify a token's content hash matches the provided content.

        Args:
            token_id: Token to verify
            full_content: Original content that was tokenized

        Returns:
            True if hash matches, False if tampered
        """
        if token_id not in self.self_knowledge.memory_tokens:
            return False

        token = self.self_knowledge.memory_tokens[token_id]
        expected_hash = token.compute_content_hash(full_content)

        return token.content_hash == expected_hash

    def get_chain_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the memory chain."""
        chain_stats = self.chain.get_chain_summary()

        # Add tokenization-specific stats
        token_types = {}
        avg_salience = 0.0
        avg_amplitude = 0.0

        if self.self_knowledge.memory_tokens:
            for token in self.self_knowledge.memory_tokens.values():
                token_types[token.event_type] = token_types.get(token.event_type, 0) + 1
                avg_salience += token.salience
                avg_amplitude += token.amplitude

            count = len(self.self_knowledge.memory_tokens)
            avg_salience /= count
            avg_amplitude /= count

        chain_stats.update(
            {
                "token_types": token_types,
                "avg_salience": avg_salience,
                "avg_amplitude": avg_amplitude,
                "active_tokens": len(self.self_knowledge.active_token_ids),
                "archived_tokens": len(
                    [
                        t
                        for t in self.self_knowledge.memory_tokens.values()
                        if t.archived
                    ]
                ),
            }
        )

        return chain_stats

    def create_identity_lockpoint(self, pmm_model: Any) -> IdentityLockpoint:
        """
        Create identity lockpoint for long-term coherence verification.

        Args:
            pmm_model: Full PMM model to snapshot

        Returns:
            IdentityLockpoint with integrity hash
        """
        lockpoint = IdentityLockpoint(
            chain_position=self.chain.current_position,
            chain_length=len(self.chain.tokens),
            chain_hash=(
                self.chain.tokens[-1].content_hash if self.chain.tokens else "empty"
            ),
        )

        # Create snapshot
        _integrity_hash = lockpoint.create_snapshot(pmm_model)

        # Store lockpoint
        self.self_knowledge.lockpoints.append(lockpoint)
        self.self_knowledge.last_lockpoint_at = lockpoint.created_at

        # Maintain lockpoint limit
        max_lockpoints = 50  # Keep only recent lockpoints
        if len(self.self_knowledge.lockpoints) > max_lockpoints:
            self.self_knowledge.lockpoints = self.self_knowledge.lockpoints[
                -max_lockpoints:
            ]

        return lockpoint

    def _compute_semantic_phase(self, event: Event) -> float:
        """
        Compute semantic phase angle from event characteristics.

        Maps event properties to a phase angle (0-2π) representing
        semantic/emotional position in memory space.
        """
        # Base phase from valence and arousal
        valence_component = (event.valence - 0.5) * 3.14159  # -π to π
        arousal_component = event.arousal * 1.57  # 0 to π/2

        # Adjust based on event type
        type_adjustments = {
            "experience": 0.0,
            "reflection": 1.57,  # π/2
            "interaction": 3.14,  # π
            "achievement": 4.71,  # 3π/2
            "insight": 1.57,  # π/2
            "thought": 0.0,
        }

        type_adjustment = type_adjustments.get(event.type, 0.0)

        # Combine components
        phase = (valence_component + arousal_component + type_adjustment) % (
            2 * 3.14159
        )

        return phase

    def _compute_content_phase(self, content: str) -> float:
        """Compute semantic phase from raw content using simple heuristics."""
        content_lower = content.lower()

        # Emotional indicators
        positive_words = [
            "good",
            "great",
            "excellent",
            "happy",
            "success",
            "achieve",
            "love",
            "joy",
        ]
        negative_words = [
            "bad",
            "terrible",
            "sad",
            "fail",
            "hate",
            "anger",
            "fear",
            "worry",
        ]

        positive_score = sum(1 for word in positive_words if word in content_lower)
        negative_score = sum(1 for word in negative_words if word in content_lower)

        # Map to phase
        if positive_score > negative_score:
            return 1.57  # π/2 (positive quadrant)
        elif negative_score > positive_score:
            return 4.71  # 3π/2 (negative quadrant)
        else:
            return 0.0  # Neutral

    def _already_tokenized(self, source_id: str, source_type: str) -> bool:
        """Check if content has already been tokenized."""
        for token in self.self_knowledge.memory_tokens.values():
            if source_type == "event" and token.source_event_id == source_id:
                return True
            elif source_type == "insight" and token.source_insight_id == source_id:
                return True
            # For thoughts, we'd need to add source_thought_id to MemoryToken
        return False

    def _update_tokenization_metrics(self):
        """Update metrics after tokenization operations."""
        # This would be called by the enhanced metrics system
        # Implementation depends on the metrics structure
        pass


class ChainVerifier:
    """
    Utility class for verifying memory chain integrity and detecting tampering.
    """

    @staticmethod
    def verify_full_chain(chain: MemoryChain) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Comprehensive chain verification with detailed diagnostics.

        Returns:
            (is_valid, error_messages, diagnostics)
        """
        is_valid, errors = chain.verify_chain_integrity()

        # Additional diagnostics
        diagnostics = {
            "chain_length": len(chain.tokens),
            "genesis_hash": chain.genesis_hash,
            "current_position": chain.current_position,
            "hash_distribution": {},
            "timestamp_gaps": [],
            "amplitude_stats": {},
        }

        if chain.tokens:
            # Hash distribution analysis
            hash_prefixes = [t.content_hash[:4] for t in chain.tokens]
            for prefix in set(hash_prefixes):
                diagnostics["hash_distribution"][prefix] = hash_prefixes.count(prefix)

            # Timestamp gap analysis
            timestamps = [
                datetime.fromisoformat(t.created_at.replace("Z", "+00:00"))
                for t in chain.tokens
            ]
            for i in range(1, len(timestamps)):
                gap = (timestamps[i] - timestamps[i - 1]).total_seconds()
                if gap > 86400:  # More than 24 hours
                    diagnostics["timestamp_gaps"].append(
                        {
                            "position": i,
                            "gap_hours": gap / 3600,
                            "before": timestamps[i - 1].isoformat(),
                            "after": timestamps[i].isoformat(),
                        }
                    )

            # Amplitude statistics
            amplitudes = [t.amplitude for t in chain.tokens]
            diagnostics["amplitude_stats"] = {
                "min": min(amplitudes),
                "max": max(amplitudes),
                "avg": sum(amplitudes) / len(amplitudes),
                "below_threshold": sum(1 for a in amplitudes if a < 0.1),
            }

        return is_valid, errors, diagnostics

    @staticmethod
    def detect_anomalies(chain: MemoryChain) -> List[Dict[str, Any]]:
        """
        Detect potential anomalies in the memory chain that might indicate
        tampering, corruption, or unusual patterns.
        """
        anomalies = []

        if not chain.tokens:
            return anomalies

        # Check for duplicate hashes (should be impossible)
        hashes = [t.content_hash for t in chain.tokens]
        duplicates = set([h for h in hashes if hashes.count(h) > 1])
        if duplicates:
            anomalies.append(
                {
                    "type": "duplicate_hashes",
                    "severity": "high",
                    "description": f"Found {len(duplicates)} duplicate content hashes",
                    "details": list(duplicates),
                }
            )

        # Check for amplitude anomalies
        amplitudes = [t.amplitude for t in chain.tokens]
        avg_amplitude = sum(amplitudes) / len(amplitudes)
        for i, token in enumerate(chain.tokens):
            if token.amplitude > avg_amplitude * 2:
                anomalies.append(
                    {
                        "type": "high_amplitude",
                        "severity": "medium",
                        "description": f"Token {token.token_id} has unusually high amplitude",
                        "details": {
                            "position": i,
                            "amplitude": token.amplitude,
                            "average": avg_amplitude,
                        },
                    }
                )

        # Check for timestamp ordering
        timestamps = [
            datetime.fromisoformat(t.created_at.replace("Z", "+00:00"))
            for t in chain.tokens
        ]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                anomalies.append(
                    {
                        "type": "timestamp_disorder",
                        "severity": "high",
                        "description": f"Token at position {i} has earlier timestamp than previous",
                        "details": {
                            "position": i,
                            "current": timestamps[i].isoformat(),
                            "previous": timestamps[i - 1].isoformat(),
                        },
                    }
                )

        return anomalies
