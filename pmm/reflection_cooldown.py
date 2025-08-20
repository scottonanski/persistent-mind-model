# pmm/reflection_cooldown.py
from __future__ import annotations
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass
import threading


@dataclass
class CooldownState:
    """State tracking for reflection cooldown."""

    last_reflection_time: Optional[datetime] = None
    turns_since_last_reflection: int = 0
    last_reflection_content: Optional[str] = None
    recent_contexts: List[str] = None

    def __post_init__(self):
        if self.recent_contexts is None:
            self.recent_contexts = []


class ReflectionCooldownManager:
    """Manages reflection cooldown with multiple gates."""

    def __init__(
        self,
        min_turns: int = 1,
        min_wall_time_seconds: int = 30,
        novelty_threshold: float = 0.82,
        context_window: int = 5,
    ):
        self.min_turns = min_turns
        self.min_wall_time_seconds = min_wall_time_seconds
        self.novelty_threshold = novelty_threshold
        self.context_window = context_window

        self.state = CooldownState()
        self._lock = threading.Lock()

    def should_reflect(
        self, current_context: str, force_reasons: Optional[List[str]] = None
    ) -> tuple[bool, str]:
        """
        Determine if reflection should be triggered based on multiple gates.

        Args:
            current_context: Current conversation/event context
            force_reasons: Optional list of reasons to force reflection (e.g., ["new_commitment"])

        Returns:
            (should_reflect: bool, reason: str)
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Check for force reasons first
            if force_reasons:
                reason = f"forced: {', '.join(force_reasons)}"
                self._update_state_on_reflection(now, current_context)
                return True, reason

            # Gate 1: Minimum turns
            if self.state.turns_since_last_reflection < self.min_turns:
                return (
                    False,
                    f"turns_gate: {self.state.turns_since_last_reflection}/{self.min_turns}",
                )

            # Gate 2: Minimum wall time
            time_since_last = 0.0
            if self.state.last_reflection_time:
                time_since_last = (
                    now - self.state.last_reflection_time
                ).total_seconds()
                if time_since_last < self.min_wall_time_seconds:
                    return (
                        False,
                        f"time_gate: {time_since_last:.0f}s/{self.min_wall_time_seconds}s",
                    )

            # Gate 3: Semantic novelty
            if not self._passes_novelty_gate(current_context):
                return False, f"novelty_gate: similarity > {self.novelty_threshold}"

            # All gates passed
            self._update_state_on_reflection(now, current_context)
            return (
                True,
                f"all_gates_passed: turns={self.state.turns_since_last_reflection}, time={time_since_last:.0f}s",
            )

    def increment_turn(self) -> None:
        """Increment turn counter."""
        with self._lock:
            self.state.turns_since_last_reflection += 1

    def add_context(self, context: str) -> None:
        """Add context to recent contexts for novelty checking."""
        with self._lock:
            self.state.recent_contexts.append(context)

            # Trim to window size
            if len(self.state.recent_contexts) > self.context_window:
                self.state.recent_contexts = self.state.recent_contexts[
                    -self.context_window :
                ]

    def _passes_novelty_gate(self, current_context: str) -> bool:
        """Check if current context is novel enough vs recent contexts."""
        if not self.state.recent_contexts:
            return True

        # Check similarity against recent contexts
        current_tokens = set(current_context.lower().split())

        for recent_context in self.state.recent_contexts[-self.context_window :]:
            recent_tokens = set(recent_context.lower().split())

            if not recent_tokens:
                continue

            # Jaccard similarity
            intersection = len(current_tokens & recent_tokens)
            union = len(current_tokens | recent_tokens)

            if union > 0:
                similarity = intersection / union
                if similarity > self.novelty_threshold:
                    return False

        return True

    def _update_state_on_reflection(self, timestamp: datetime, context: str) -> None:
        """Update state when reflection is triggered."""
        self.state.last_reflection_time = timestamp
        self.state.turns_since_last_reflection = 0
        self.state.last_reflection_content = context

        # Add to recent contexts
        self.state.recent_contexts.append(context)
        if len(self.state.recent_contexts) > self.context_window:
            self.state.recent_contexts = self.state.recent_contexts[
                -self.context_window :
            ]

    def reset_on_model_switch(self) -> None:
        """Reset cooldown state when model switches."""
        with self._lock:
            print("🔄 Reflection cooldown: Reset on model switch")
            self.state.turns_since_last_reflection = 0
            self.state.last_reflection_time = None
            self.state.recent_contexts.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get current cooldown status."""
        with self._lock:
            now = datetime.now(timezone.utc)
            time_since_last = None

            if self.state.last_reflection_time:
                time_since_last = (
                    now - self.state.last_reflection_time
                ).total_seconds()

            return {
                "turns_since_last": self.state.turns_since_last_reflection,
                "time_since_last_seconds": time_since_last,
                "min_turns_required": self.min_turns,
                "min_time_required_seconds": self.min_wall_time_seconds,
                "novelty_threshold": self.novelty_threshold,
                "recent_contexts_count": len(self.state.recent_contexts),
                "turns_gate_passed": self.state.turns_since_last_reflection
                >= self.min_turns,
                "time_gate_passed": time_since_last is None
                or time_since_last >= self.min_wall_time_seconds,
                "last_reflection_time": (
                    self.state.last_reflection_time.isoformat()
                    if self.state.last_reflection_time
                    else None
                ),
            }

    def simulate_reflection_decision(self, current_context: str) -> Dict[str, Any]:
        """Simulate reflection decision without updating state (for debugging)."""
        now = datetime.now(timezone.utc)

        # Check gates without updating state
        turns_passed = self.state.turns_since_last_reflection >= self.min_turns

        time_passed = True
        time_since_last = None
        if self.state.last_reflection_time:
            time_since_last = (now - self.state.last_reflection_time).total_seconds()
            time_passed = time_since_last >= self.min_wall_time_seconds

        novelty_passed = self._passes_novelty_gate(current_context)

        would_reflect = turns_passed and time_passed and novelty_passed

        return {
            "would_reflect": would_reflect,
            "turns_gate": {
                "passed": turns_passed,
                "current": self.state.turns_since_last_reflection,
                "required": self.min_turns,
            },
            "time_gate": {
                "passed": time_passed,
                "current_seconds": time_since_last,
                "required_seconds": self.min_wall_time_seconds,
            },
            "novelty_gate": {
                "passed": novelty_passed,
                "threshold": self.novelty_threshold,
            },
        }
