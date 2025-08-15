# pmm/adaptive_triggers.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

@dataclass
class TriggerConfig:
    # minimum time between reflections if time-based is enabled
    cadence_days: Optional[float] = None   # e.g., 1.0 for daily; None = disable time gate
    # event-based thresholds
    events_min_gap: int = 4                # reflect after at least N new events
    # emergence-based adjustments
    ias_low: float = 0.35                  # if IAS < this, reflect sooner
    gas_low: float = 0.35
    ias_high: float = 0.65                 # if IAS/GAS high, stretch cadence (less frequent)
    gas_high: float = 0.65
    min_cooldown_minutes: int = 10         # prevent thrashing
    max_skip_days: float = 7.0             # force a reflection at least weekly

@dataclass
class TriggerState:
    last_reflection_at: Optional[datetime] = None
    last_event_id: Optional[int] = None
    events_since_reflection: int = 0

class AdaptiveTrigger:
    """
    Decides whether to trigger a reflection now, based on:
      - time-based cadence (respects reflection_cadence_days)
      - event accumulation
      - emergence signals (IAS/GAS)
    Returns (should_reflect: bool, reason: str)
    """

    def __init__(self, cfg: TriggerConfig, state: TriggerState):
        self.cfg = cfg
        self.state = state

    def _time_gate_passed(self, now: datetime) -> bool:
        # Force reflection if we've skipped too long
        if self.state.last_reflection_at:
            if (now - self.state.last_reflection_at).days >= self.cfg.max_skip_days:
                return True
        # Respect configured cadence if present
        if self.cfg.cadence_days is None or self.state.last_reflection_at is None:
            return True
        due_at = self.state.last_reflection_at + timedelta(days=self.cfg.cadence_days)
        # also enforce minimum cooldown in minutes
        cool_ok = True
        if self.state.last_reflection_at:
            cool_ok = (now - self.state.last_reflection_at) >= timedelta(
                minutes=self.cfg.min_cooldown_minutes
            )
        return (now >= due_at) and cool_ok

    def decide(
        self,
        now: datetime,
        ias: Optional[float],
        gas: Optional[float],
        events_since_reflection: Optional[int] = None,
    ) -> tuple[bool, str]:
        esr = events_since_reflection if events_since_reflection is not None else self.state.events_since_reflection

        # 1) Base: event accumulation
        if esr >= self.cfg.events_min_gap:
            # Sooner if low emergence (system is struggling to adopt identity/grow)
            if (ias is not None and ias < self.cfg.ias_low) or (gas is not None and gas < self.cfg.gas_low):
                return True, f"event-gap({esr}) + low-emergence"
            # Or pass time gate
            if self._time_gate_passed(now):
                return True, f"event-gap({esr}) + time-gate"

        # 2) Pure time-based cadence (if configured)
        if self._time_gate_passed(now):
            # If emergence high, we can skip to avoid over-reflecting
            if (ias is not None and ias > self.cfg.ias_high) and (gas is not None and gas > self.cfg.gas_high):
                return False, "time-gate hit but high-emergence (skip)"
            return True, "time-gate"

        return False, "not-due"
