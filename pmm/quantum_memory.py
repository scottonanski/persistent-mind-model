"""
Quantum-Inspired Memory State Management - Layer 3 Implementation

Implements amplitude/phase state vectors for memory activation and semantic positioning,
mimicking quantum superposition for AI consciousness modeling.
"""

from __future__ import annotations
import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

from .memory_token import MemoryToken


@dataclass
class QuantumState:
    """
    Quantum-inspired state representation for memory tokens.

    Uses amplitude (activation probability) and phase (semantic position)
    to model memory accessibility and contextual relationships.
    """

    amplitude: float = 1.0  # Probability of activation (0.0-1.0)
    phase: float = 0.0  # Semantic angle in radians (0-2π)
    coherence: float = 1.0  # State coherence (1.0 = pure state)
    entanglement: Dict[str, float] = field(
        default_factory=dict
    )  # Links to other tokens

    def get_activation_probability(self) -> float:
        """Get probability of this state being activated."""
        return self.amplitude**2  # Quantum probability = |amplitude|²

    def get_semantic_vector(self) -> Tuple[float, float]:
        """Get 2D semantic position from phase."""
        return (
            self.amplitude * math.cos(self.phase),
            self.amplitude * math.sin(self.phase),
        )

    def measure_state(self) -> bool:
        """
        Quantum measurement - collapse state to activated/deactivated.
        Returns True if activated, updates amplitude accordingly.
        """
        probability = self.get_activation_probability()
        activated = np.random.random() < probability

        # Measurement collapse
        if activated:
            self.amplitude = min(1.0, self.amplitude + 0.1)  # Strengthen if measured
        else:
            self.amplitude *= 0.9  # Weaken if not measured

        return activated

    def entangle_with(self, other_token_id: str, strength: float = 0.5):
        """Create quantum entanglement with another token."""
        self.entanglement[other_token_id] = strength

    def apply_interference(self, other_state: "QuantumState", weight: float = 0.1):
        """Apply quantum interference from another state."""
        # Phase interference
        phase_diff = abs(self.phase - other_state.phase)
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff

        # Constructive/destructive interference
        if phase_diff < math.pi / 2:  # Constructive
            self.amplitude = min(1.0, self.amplitude + weight * other_state.amplitude)
        else:  # Destructive
            self.amplitude = max(0.0, self.amplitude - weight * other_state.amplitude)


class QuantumMemoryManager:
    """
    Manager for quantum-inspired memory state operations.

    Handles amplitude decay, phase evolution, entanglement relationships,
    and quantum-inspired memory dynamics.
    """

    def __init__(self):
        self.decay_rate = 0.01  # Daily amplitude decay rate
        self.phase_drift = 0.05  # Random phase drift rate
        self.entanglement_threshold = 0.7  # Similarity threshold for auto-entanglement

    def apply_temporal_decay(
        self, tokens: Dict[str, MemoryToken], days_elapsed: float = 1.0
    ):
        """
        Apply time-based amplitude decay to all tokens.

        Simulates natural memory fading over time.
        """
        decay_factor = 1.0 - (self.decay_rate * days_elapsed)

        for token in tokens.values():
            if not token.archived:  # Only decay active tokens
                token.amplitude *= decay_factor
                token.amplitude = max(0.0, token.amplitude)

    def apply_phase_evolution(self, tokens: Dict[str, MemoryToken]):
        """
        Apply gradual phase drift to simulate semantic drift over time.
        """
        for token in tokens.values():
            if not token.archived:
                # Random phase drift
                drift = np.random.normal(0, self.phase_drift)
                token.phase = (token.phase + drift) % (2 * math.pi)

    def boost_related_memories(
        self,
        activated_token: MemoryToken,
        all_tokens: Dict[str, MemoryToken],
        boost_strength: float = 0.1,
    ):
        """
        Boost amplitude of semantically related memories when one is recalled.

        Simulates associative memory activation.
        """
        activated_vector = activated_token.get_semantic_vector()

        for token_id, token in all_tokens.items():
            if token_id == activated_token.token_id or token.archived:
                continue

            # Calculate semantic similarity
            token_vector = token.get_semantic_vector()
            similarity = self._cosine_similarity(activated_vector, token_vector)

            if similarity > self.entanglement_threshold:
                # Boost similar memories
                boost = boost_strength * similarity
                token.amplitude = min(1.0, token.amplitude + boost)

                # Create/strengthen entanglement
                activated_token.entangle_with(token_id, similarity)
                token.entangle_with(activated_token.token_id, similarity)

    def compute_coherence_field(
        self, tokens: Dict[str, MemoryToken]
    ) -> Dict[str, float]:
        """
        Compute coherence field showing memory interconnectedness.

        Returns coherence score for each token based on its entanglements.
        """
        coherence_scores = {}

        for token_id, token in tokens.items():
            if token.archived:
                coherence_scores[token_id] = 0.0
                continue

            # Base coherence from amplitude
            base_coherence = token.amplitude

            # Entanglement contribution
            entanglement_boost = 0.0
            for other_id, strength in token.entanglement.items():
                if other_id in tokens and not tokens[other_id].archived:
                    entanglement_boost += strength * tokens[other_id].amplitude

            # Normalize entanglement boost
            if token.entanglement:
                entanglement_boost /= len(token.entanglement)

            coherence_scores[token_id] = min(
                1.0, base_coherence + 0.3 * entanglement_boost
            )

        return coherence_scores

    def identify_memory_clusters(
        self, tokens: Dict[str, MemoryToken]
    ) -> Dict[str, List[str]]:
        """
        Identify clusters of strongly entangled memories.

        Returns clusters as {cluster_id: [token_ids]}
        """
        clusters = {}
        visited = set()
        cluster_id = 0

        for token_id, token in tokens.items():
            if token_id in visited or token.archived:
                continue

            # Start new cluster
            current_cluster = []
            to_visit = [token_id]

            while to_visit:
                current_id = to_visit.pop()
                if current_id in visited:
                    continue

                visited.add(current_id)
                current_cluster.append(current_id)

                # Add strongly entangled neighbors
                current_token = tokens[current_id]
                for other_id, strength in current_token.entanglement.items():
                    if (
                        strength > self.entanglement_threshold
                        and other_id not in visited
                        and other_id in tokens
                        and not tokens[other_id].archived
                    ):
                        to_visit.append(other_id)

            if len(current_cluster) > 1:  # Only keep multi-token clusters
                clusters[f"cluster_{cluster_id}"] = current_cluster
                cluster_id += 1

        return clusters

    def simulate_quantum_measurement(
        self, token: MemoryToken
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Simulate quantum measurement of memory token.

        Returns (activated, measurement_data)
        """
        pre_amplitude = token.amplitude
        pre_phase = token.phase

        # Quantum measurement
        activated = np.random.random() < token.get_activation_probability()

        # Measurement effects
        if activated:
            token.boost_amplitude(0.1)
        else:
            token.decay_amplitude(0.05)

        measurement_data = {
            "pre_amplitude": pre_amplitude,
            "post_amplitude": token.amplitude,
            "pre_phase": pre_phase,
            "post_phase": token.phase,
            "activated": activated,
            "measurement_time": datetime.utcnow().isoformat(),
        }

        return activated, measurement_data

    def _cosine_similarity(
        self, vec1: Tuple[float, float], vec2: Tuple[float, float]
    ) -> float:
        """Compute cosine similarity between two 2D vectors."""
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        norm1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        norm2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class MemoryResonanceEngine:
    """
    Advanced quantum-inspired memory dynamics using resonance patterns.

    Models how memories can resonate with each other, creating
    cascading activation patterns similar to neural networks.
    """

    def __init__(self):
        self.resonance_threshold = 0.6
        self.cascade_depth = 3
        self.resonance_decay = 0.8

    def trigger_memory_cascade(
        self, initial_token: MemoryToken, all_tokens: Dict[str, MemoryToken]
    ) -> List[Dict[str, Any]]:
        """
        Trigger cascading memory activation starting from initial token.

        Returns list of activation events in cascade order.
        """
        cascade_events = []
        activated_tokens = {initial_token.token_id}
        current_wave = [initial_token]
        wave_strength = 1.0

        for depth in range(self.cascade_depth):
            if not current_wave:
                break

            next_wave = []

            for source_token in current_wave:
                # Find resonant memories
                resonant_tokens = self._find_resonant_memories(
                    source_token, all_tokens, activated_tokens
                )

                for target_token, resonance_strength in resonant_tokens:
                    activation_strength = wave_strength * resonance_strength

                    # Activate if above threshold
                    if activation_strength > self.resonance_threshold:
                        target_token.boost_amplitude(activation_strength * 0.1)
                        activated_tokens.add(target_token.token_id)
                        next_wave.append(target_token)

                        cascade_events.append(
                            {
                                "depth": depth + 1,
                                "source_token": source_token.token_id,
                                "target_token": target_token.token_id,
                                "resonance_strength": resonance_strength,
                                "activation_strength": activation_strength,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )

            current_wave = next_wave
            wave_strength *= self.resonance_decay

        return cascade_events

    def _find_resonant_memories(
        self,
        source_token: MemoryToken,
        all_tokens: Dict[str, MemoryToken],
        exclude_tokens: set,
    ) -> List[Tuple[MemoryToken, float]]:
        """Find memories that resonate with the source token."""
        resonant_memories = []
        _source_vector = source_token.get_semantic_vector()

        for token_id, token in all_tokens.items():
            if token_id in exclude_tokens or token.archived:
                continue

            # Calculate resonance strength
            resonance = self._calculate_resonance(source_token, token)

            if resonance > 0.1:  # Minimum resonance threshold
                resonant_memories.append((token, resonance))

        # Sort by resonance strength
        resonant_memories.sort(key=lambda x: x[1], reverse=True)

        return resonant_memories[:5]  # Top 5 resonant memories

    def _calculate_resonance(self, token1: MemoryToken, token2: MemoryToken) -> float:
        """
        Calculate resonance strength between two memory tokens.

        Combines semantic similarity, temporal proximity, and entanglement.
        """
        # Semantic similarity
        vec1 = token1.get_semantic_vector()
        vec2 = token2.get_semantic_vector()
        semantic_sim = self._cosine_similarity(vec1, vec2)

        # Temporal proximity (memories closer in time resonate more)
        time1 = datetime.fromisoformat(token1.created_at.replace("Z", "+00:00"))
        time2 = datetime.fromisoformat(token2.created_at.replace("Z", "+00:00"))
        time_diff_hours = abs((time1 - time2).total_seconds()) / 3600
        temporal_sim = 1.0 / (1.0 + time_diff_hours / 24)  # Decay over days

        # Entanglement strength
        entanglement_sim = token1.entanglement.get(token2.token_id, 0.0)

        # Tag similarity
        common_tags = set(token1.tags) & set(token2.tags)
        tag_sim = len(common_tags) / max(len(token1.tags), len(token2.tags), 1)

        # Combine factors
        resonance = (
            0.4 * semantic_sim
            + 0.2 * temporal_sim
            + 0.3 * entanglement_sim
            + 0.1 * tag_sim
        )

        return min(1.0, resonance)

    def _cosine_similarity(
        self, vec1: Tuple[float, float], vec2: Tuple[float, float]
    ) -> float:
        """Compute cosine similarity between two 2D vectors."""
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        norm1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        norm2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
