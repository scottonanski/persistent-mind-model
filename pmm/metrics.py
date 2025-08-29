"""
Metrics and telemetry for PMM system.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Track performance metrics for PMM operations."""

    reflection_time: float = 0.0
    commitment_extraction_time: float = 0.0
    drift_calculation_time: float = 0.0
    total_events: int = 0
    total_insights: int = 0
    total_commitments: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "reflection_time": self.reflection_time,
            "commitment_extraction_time": self.commitment_extraction_time,
            "drift_calculation_time": self.drift_calculation_time,
            "total_events": self.total_events,
            "total_insights": self.total_insights,
            "total_commitments": self.total_commitments,
        }


def compute_close_rate(events: List[Dict[str, Any]]) -> float:
    """
    Compute commitment close rate from actual stored events.

    Args:
        events: List of event dictionaries from storage

    Returns:
        float: Close rate (0.0-1.0) based on commit_open vs commit_close events
    """
    if not events:
        return 0.0

    opened = sum(1 for e in events if e.get("type") == "commit_open")
    closed = sum(1 for e in events if e.get("type") == "commit_close")

    if opened == 0:
        return 0.0

    return closed / opened


def compute_identity_coherence(m) -> float:
    """Simple, transparent metric.
    - Base increases with number of scenes (capped)
    - Penalty if conflicting behavioral flags present
    """
    n_scenes = len(m.narrative_identity.scenes)
    base = min(1.0, 0.5 + 0.05 * min(n_scenes, 10))
    patt = m.self_knowledge.behavioral_patterns or {}
    conflict_penalty = 0.0
    if patt.get("uncertainty", 0) and patt.get("confidence", 0):
        conflict_penalty += 0.1
    return max(0.0, min(1.0, base - conflict_penalty))


def compute_self_consistency(m) -> float:
    """Inverse of absolute drift magnitude from last apply cycle."""
    dv: Dict[str, float] = m.metrics.drift_velocity or {}
    drift_sum = sum(abs(v) for v in dv.values())
    return max(0.0, min(1.0, 1.0 - min(1.0, drift_sum * 5.0)))
