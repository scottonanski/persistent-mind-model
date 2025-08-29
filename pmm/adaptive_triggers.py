#!/usr/bin/env python3
"""
Adaptive triggers - intelligent reflection triggering based on emergence patterns.
Determines when PMM should reflect based on time, events, and emergence scores.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta


@dataclass
class TriggerConfig:
    """Configuration for adaptive trigger system."""

    time_threshold_seconds: float = 300.0
    event_threshold: int = 4
    emergence_threshold: float = 0.5
    cooldown_seconds: float = 60.0
    # Legacy parameters for backward compatibility
    cadence_days: Optional[float] = None
    events_min_gap: int = 4
    ias_low: float = 0.35
    gas_low: float = 0.35
    ias_high: float = 0.65
    gas_high: float = 0.65
    min_cooldown_minutes: float = 5.0
    max_skip_days: float = 7.0


@dataclass
class TriggerState:
    """Current state of trigger system."""

    last_trigger_time: Optional[datetime] = None
    event_count_since_trigger: int = 0
    cooldown_active: bool = False
    # Legacy parameters for backward compatibility
    last_reflection_at: Optional[datetime] = None
    events_since_reflection: int = 0


class AdaptiveTrigger:
    """
    Intelligent reflection triggering system for PMM.

    Determines when PMM should reflect based on:
    - Event accumulation (number of interactions)
    - Time gates (minimum/maximum intervals)
    - Emergence scores (IAS/GAS thresholds)
    - Adaptive cadence based on emergence patterns
    """

    def __init__(self, config: TriggerConfig, state: Optional[TriggerState] = None):
        self.config = config
        self.state = state or TriggerState()
        self.trigger_history = []

    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """
        Determine if a trigger should fire based on current context.

        Args:
            context: Current context including time, events, emergence scores

        Returns:
            True if trigger should fire
        """
        now = datetime.now(timezone.utc)

        # Check cooldown
        if self.state.last_trigger_time:
            time_since_last = (now - self.state.last_trigger_time).total_seconds()
            if time_since_last < self.config.cooldown_seconds:
                return False

        # Check event threshold
        if self.state.event_count_since_trigger >= self.config.event_threshold:
            return True

        # Check time threshold
        if self.state.last_trigger_time:
            time_since_last = (now - self.state.last_trigger_time).total_seconds()
            if time_since_last >= self.config.time_threshold_seconds:
                return True

        # Check emergence threshold
        emergence_score = context.get("emergence_score", 0.0)
        if emergence_score >= self.config.emergence_threshold:
            return True

        return False

    def decide(self, *args, **kwargs) -> Tuple[bool, str]:
        """
        Legacy method for backward compatibility with test suite.

        Supports both old and new calling patterns:
        - Old: decide(now, ias=0.5, gas=0.5, events_since_reflection=4)
        - New: decide(context_dict)
        """
        if len(args) >= 1 and isinstance(args[0], dict):
            # New pattern: decide(context_dict)
            should_trigger = self.should_trigger(args[0])
            return should_trigger, "adaptive_trigger"
        else:
            # Old pattern for test compatibility
            now = args[0] if args else datetime.now(timezone.utc)
            ias = kwargs.get("ias", 0.0)
            gas = kwargs.get("gas", 0.0)
            events_since_reflection = kwargs.get("events_since_reflection", 0)

            # Implement adaptive logic based on emergence
            emergence_score = (ias + gas) / 2.0

            # Time gate logic
            time_gate_passed = True
            if self.state.last_reflection_at and self.config.cadence_days:
                time_since_reflection = now - self.state.last_reflection_at
                min_interval = timedelta(days=self.config.cadence_days)
                time_gate_passed = time_since_reflection >= min_interval

            # Event accumulation trigger
            if (
                events_since_reflection >= self.config.events_min_gap
                and time_gate_passed
            ):
                return True, "event_accumulation"

            # Time gate trigger (forced reflection after max interval)
            if self.state.last_reflection_at and self.config.max_skip_days:
                time_since_reflection = now - self.state.last_reflection_at
                max_interval = timedelta(days=self.config.max_skip_days)
                if time_since_reflection >= max_interval:
                    return True, "time_gate"

            # Emergence-based adaptive triggers
            if (
                emergence_score <= self.config.ias_low
                or emergence_score <= self.config.gas_low
            ):
                # Low emergence - trigger sooner to boost engagement
                if events_since_reflection >= max(1, self.config.events_min_gap - 2):
                    return True, "low_emergence_boost"

            if (
                emergence_score >= self.config.ias_high
                and emergence_score >= self.config.gas_high
            ):
                # High emergence - can skip reflection to maintain momentum
                if (
                    not time_gate_passed
                    or events_since_reflection < self.config.events_min_gap + 2
                ):
                    return False, "high_emergence_skip"

            # Default emergence threshold
            if emergence_score >= self.config.emergence_threshold and time_gate_passed:
                return True, "emergence_threshold"

            return False, "no_trigger_conditions_met"

    def trigger(self) -> None:
        """Record that a trigger has fired."""
        now = datetime.now(timezone.utc)
        self.state.last_trigger_time = now
        self.state.last_reflection_at = now
        self.state.event_count_since_trigger = 0
        self.state.events_since_reflection = 0
        self.state.cooldown_active = True

        self.trigger_history.append(
            {"timestamp": now.isoformat(), "trigger_type": "adaptive"}
        )

    def record_event(self) -> None:
        """Record that an event has occurred."""
        self.state.event_count_since_trigger += 1
        self.state.events_since_reflection += 1

    def get_trigger_stats(self) -> Dict[str, Any]:
        """Get statistics about trigger performance."""
        return {
            "total_triggers": len(self.trigger_history),
            "events_since_last_trigger": self.state.event_count_since_trigger,
            "cooldown_active": self.state.cooldown_active,
            "config": {
                "time_threshold": self.config.time_threshold_seconds,
                "event_threshold": self.config.event_threshold,
                "emergence_threshold": self.config.emergence_threshold,
            },
        }
