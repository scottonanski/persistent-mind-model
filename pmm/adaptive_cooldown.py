# pmm/adaptive_cooldown.py
"""
Stage-aware adaptive reflection cooldown system.

Adjusts cooldown periods based on emergence stage and context to balance
consolidation needs with responsive development.
"""
from __future__ import annotations
from typing import Optional
from .emergence_stages import EmergenceStage


class AdaptiveCooldownManager:
    """Manages stage-aware adaptive reflection cooldowns."""

    def __init__(self):
        # Stage-specific cooldown periods (seconds)
        self.stage_cooldowns = {
            EmergenceStage.DORMANT: 120,  # S0: Longer for consolidation
            EmergenceStage.AWAKENING: 90,  # S1: Moderate for pattern formation
            EmergenceStage.DEVELOPING: 60,  # S2: Shorter for active development
            EmergenceStage.MATURING: 45,  # S3: Responsive for refinement
            EmergenceStage.TRANSCENDENT: 30,  # S4: Highly responsive
        }

        # Context-based modifiers
        self.context_modifiers = {
            "user_question": 0.7,  # Reduce cooldown for direct questions
            "commitment_made": 0.8,  # Slight reduction after commitments
            "error_detected": 0.5,  # Fast response to errors
            "development_mode": 0.6,  # Faster iteration during development
            "consolidation_needed": 1.5,  # Increase for deep consolidation
        }

    def get_adaptive_cooldown(
        self,
        current_stage: EmergenceStage,
        context_hints: Optional[list] = None,
        base_cooldown: Optional[int] = None,
    ) -> int:
        """
        Calculate adaptive cooldown based on emergence stage and context.

        Args:
            current_stage: Current emergence stage
            context_hints: List of context hints for modifiers
            base_cooldown: Override base cooldown if provided

        Returns:
            Adaptive cooldown in seconds
        """
        # Get base cooldown for stage
        if base_cooldown is not None:
            cooldown = base_cooldown
        else:
            cooldown = self.stage_cooldowns.get(current_stage, 90)

        # Apply context modifiers
        if context_hints:
            modifier = 1.0
            for hint in context_hints:
                if hint in self.context_modifiers:
                    modifier *= self.context_modifiers[hint]

            cooldown = int(cooldown * modifier)

        # Ensure reasonable bounds (15s minimum, 300s maximum)
        return max(15, min(300, cooldown))

    def should_use_development_mode(self, recent_interactions: int = 0) -> bool:
        """
        Determine if development mode should be active.

        Development mode uses shorter cooldowns for faster iteration.
        """
        # Use development mode if there have been many recent interactions
        # (indicates active development/testing session)
        return recent_interactions > 5

    def get_context_hints(
        self,
        user_input: str = "",
        ai_output: str = "",
        recent_errors: int = 0,
        recent_interactions: int = 0,
    ) -> list:
        """Extract context hints from interaction data."""
        hints = []

        # User input analysis
        if user_input:
            if any(
                word in user_input.lower()
                for word in ["?", "how", "what", "why", "when", "where"]
            ):
                hints.append("user_question")

            if any(
                word in user_input.lower()
                for word in ["commit", "promise", "will do", "plan to"]
            ):
                hints.append("commitment_made")

        # AI output analysis
        if ai_output:
            if any(
                word in ai_output.lower()
                for word in ["commit", "will", "plan to", "intend"]
            ):
                hints.append("commitment_made")

        # Error detection
        if recent_errors > 0:
            hints.append("error_detected")

        # Development mode detection
        if self.should_use_development_mode(recent_interactions):
            hints.append("development_mode")

        return hints
