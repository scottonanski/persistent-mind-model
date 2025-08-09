from __future__ import annotations
from typing import Dict
from .model import PersistentMindModel


def compute_identity_coherence(m: PersistentMindModel) -> float:
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


def compute_self_consistency(m: PersistentMindModel) -> float:
    """Inverse of absolute drift magnitude from last apply cycle."""
    dv: Dict[str, float] = m.metrics.drift_velocity or {}
    drift_sum = sum(abs(v) for v in dv.values())
    return max(0.0, min(1.0, 1.0 - min(1.0, drift_sum * 5.0)))
