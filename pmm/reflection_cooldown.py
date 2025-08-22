# pmm/reflection_cooldown.py
from __future__ import annotations
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass
import threading
import os


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
        min_turns: int = 0,
        min_wall_time_seconds: int = 20,
        novelty_threshold: float = 0.78,
        context_window: int = 6,
    ):
        # Allow env var overrides for experimentation without code changes
        env_turns = os.getenv("PMM_REFLECTION_MIN_TURNS")
        env_time = os.getenv("PMM_REFLECTION_MIN_TIME_SECONDS")
        env_novelty = os.getenv("PMM_REFLECTION_NOVELTY_THRESHOLD")
        env_ctx = os.getenv("PMM_REFLECTION_CONTEXT_WINDOW")

        # Fallback to provided defaults if env not set or malformed
        try:
            self.min_turns = int(env_turns) if env_turns is not None else min_turns
        except ValueError:
            self.min_turns = min_turns

        try:
            self.min_wall_time_seconds = (
                int(env_time) if env_time is not None else min_wall_time_seconds
            )
        except ValueError:
            self.min_wall_time_seconds = min_wall_time_seconds

        try:
            self.novelty_threshold = (
                float(env_novelty) if env_novelty is not None else novelty_threshold
            )
        except ValueError:
            self.novelty_threshold = novelty_threshold

        try:
            self.context_window = (
                int(env_ctx) if env_ctx is not None else context_window
            )
        except ValueError:
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

            telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

            # Check for force reasons first
            if force_reasons:
                reason = f"forced: {', '.join(force_reasons)}"
                if telemetry:
                    self._telemetry_decision(
                        allow=True,
                        reason=reason,
                        time_since_last=(
                            (now - self.state.last_reflection_time).total_seconds()
                            if self.state.last_reflection_time
                            else None
                        ),
                    )
                self._update_state_on_reflection(now, current_context)
                return True, reason

            # Gate 1 (PRIMARY): Minimum wall time. Prioritize elapsed time for autonomous cadence.
            time_since_last = 0.0
            if self.state.last_reflection_time:
                time_since_last = (
                    now - self.state.last_reflection_time
                ).total_seconds()
                if time_since_last < self.min_wall_time_seconds:
                    reason = f"time_gate: {time_since_last:.0f}s/{self.min_wall_time_seconds}s"
                    if telemetry:
                        self._telemetry_decision(
                            allow=False, reason=reason, time_since_last=time_since_last
                        )
                    return False, reason

            # Gate 2 (SECONDARY): Minimum turns (optional). If min_turns <= 0, skip this gate.
            if (
                self.min_turns > 0
                and self.state.turns_since_last_reflection < self.min_turns
            ):
                reason = f"turns_gate: {self.state.turns_since_last_reflection}/{self.min_turns}"
                if telemetry:
                    self._telemetry_decision(
                        allow=False,
                        reason=reason,
                        time_since_last=time_since_last or None,
                    )
                return False, reason

            # Gate 3: Semantic novelty
            if not self._passes_novelty_gate(current_context):
                reason = f"novelty_gate: similarity > {self.novelty_threshold}"
                if telemetry:
                    self._telemetry_decision(
                        allow=False,
                        reason=reason,
                        time_since_last=time_since_last or None,
                    )
                return False, reason

            # All gates passed
            reason = f"all_gates_passed: turns={self.state.turns_since_last_reflection}, time={time_since_last:.0f}s"
            if telemetry:
                self._telemetry_decision(
                    allow=True, reason=reason, time_since_last=time_since_last or None
                )
            self._update_state_on_reflection(now, current_context)
            return True, reason

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

    def _telemetry_decision(
        self, allow: bool, reason: str, time_since_last: Optional[float]
    ) -> None:
        """Emit a single, structured telemetry line for cooldown decisions.

        Includes gate states and thresholds to aid debugging and analysis.
        """
        try:
            turns_current = self.state.turns_since_last_reflection
            turns_required = self.min_turns
            time_current = None if time_since_last is None else float(time_since_last)
            time_required = self.min_wall_time_seconds
            novelty_threshold = self.novelty_threshold
            ctx_count = len(self.state.recent_contexts)

            print(
                f"[PMM_TELEMETRY] cooldown_decision: decision={'allow' if allow else 'deny'}, reason={reason}, "
                f"turns={turns_current}/{turns_required}, time={time_current if time_current is not None else 'None'}/{time_required}s, "
                f"novelty_threshold={novelty_threshold:.2f}, recent_contexts={ctx_count}"
            )
        except Exception:
            # Never let telemetry break core logic
            pass

    def simulate_reflection_decision(self, current_context: str) -> Dict[str, Any]:
        """Simulate reflection decision without updating state (for debugging)."""
        now = datetime.now(timezone.utc)

        # Check gates without updating state
        turns_passed = (
            True
            if self.min_turns <= 0
            else (self.state.turns_since_last_reflection >= self.min_turns)
        )

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
