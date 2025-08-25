# pmm/s0_s1_tuning.py
"""
S0→S1 Transition Parameter Tuning Configuration

This module provides centralized configuration for parameters that affect
PMM's transition from S0 (Substrate) to S1 (Pattern Formation) stage.
"""
from __future__ import annotations
from typing import Dict, Any
import os


class S0S1TuningConfig:
    """Configuration for S0→S1 transition parameters."""

    def __init__(self):
        # Reflection cooldown parameters (extended for deeper consolidation)
        self.reflection_cooldown_seconds = int(os.getenv("PMM_S0S1_COOLDOWN", "180"))
        self.reflection_novelty_threshold = float(os.getenv("PMM_S0S1_NOVELTY", "0.85"))

        # Context weighting (increased historical weight for pattern recognition)
        self.semantic_context_results = int(os.getenv("PMM_S0S1_SEMANTIC", "8"))
        self.recent_events_limit = int(os.getenv("PMM_S0S1_RECENT", "45"))

        # Pattern reuse weighting (stronger emphasis on historical patterns)
        self.pattern_reuse_weight = float(os.getenv("PMM_S0S1_PATTERN_WEIGHT", "0.6"))
        self.novelty_decay_factor = float(os.getenv("PMM_S0S1_NOVELTY_DECAY", "0.85"))

        # Stage transition sensitivity (relaxed thresholds)
        self.dormant_exit_threshold = float(os.getenv("PMM_S0S1_DORMANT_EXIT", "-0.8"))
        self.awakening_entry_threshold = float(
            os.getenv("PMM_S0S1_AWAKENING_ENTRY", "-0.3")
        )

        # Multi-event continuity enforcement
        self.min_event_references = int(os.getenv("PMM_S0S1_MIN_REFS", "3"))
        self.continuity_boost_factor = float(
            os.getenv("PMM_S0S1_CONTINUITY_BOOST", "1.2")
        )

        # Duplicate detection (tightened for better pattern recognition)
        self.duplicate_threshold = float(os.getenv("PMM_S0S1_DUPLICATE_THRESH", "0.15"))

    def get_reflection_config(self) -> Dict[str, Any]:
        """Get reflection cooldown configuration."""
        return {
            "min_wall_time_seconds": self.reflection_cooldown_seconds,
            "novelty_threshold": self.reflection_novelty_threshold,
        }

    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory retrieval configuration."""
        return {
            "semantic_max_results": self.semantic_context_results,
            "recent_events_limit": self.recent_events_limit,
            "pattern_reuse_weight": self.pattern_reuse_weight,
        }

    def get_stage_config(self) -> Dict[str, Any]:
        """Get stage transition configuration."""
        return {
            "dormant_max": self.dormant_exit_threshold,
            "awakening_max": self.awakening_entry_threshold,
        }

    def get_pattern_config(self) -> Dict[str, Any]:
        """Get pattern recognition configuration."""
        return {
            "min_event_references": self.min_event_references,
            "continuity_boost_factor": self.continuity_boost_factor,
            "novelty_decay_factor": self.novelty_decay_factor,
            "duplicate_threshold": self.duplicate_threshold,
        }

    def apply_to_reflection_manager(self, manager):
        """Apply S0→S1 tuning to reflection cooldown manager."""
        config = self.get_reflection_config()
        manager.min_wall_time_seconds = config["min_wall_time_seconds"]
        manager.novelty_threshold = config["novelty_threshold"]
        return manager

    def get_summary(self) -> str:
        """Get human-readable summary of current tuning parameters."""
        return f"""
S0→S1 Transition Tuning Parameters:
=====================================
Reflection Cooldown: {self.reflection_cooldown_seconds}s (was 30s)
Novelty Threshold: {self.reflection_novelty_threshold} (was 0.78)
Semantic Context: {self.semantic_context_results} results (was 6)
Recent Events: {self.recent_events_limit} events (was 30)
Pattern Weight: {self.pattern_reuse_weight} (new)
Dormant Exit: {self.dormant_exit_threshold} (was -1.0)
Awakening Entry: {self.awakening_entry_threshold} (was -0.5)
Duplicate Threshold: {self.duplicate_threshold} (was 0.2)

Environment Variables (for runtime tuning):
PMM_S0S1_COOLDOWN={self.reflection_cooldown_seconds}
PMM_S0S1_NOVELTY={self.reflection_novelty_threshold}
PMM_S0S1_SEMANTIC={self.semantic_context_results}
PMM_S0S1_RECENT={self.recent_events_limit}
PMM_S0S1_PATTERN_WEIGHT={self.pattern_reuse_weight}
PMM_S0S1_DORMANT_EXIT={self.dormant_exit_threshold}
PMM_S0S1_AWAKENING_ENTRY={self.awakening_entry_threshold}
PMM_S0S1_DUPLICATE_THRESH={self.duplicate_threshold}
"""


# Global instance for easy access
s0_s1_config = S0S1TuningConfig()
