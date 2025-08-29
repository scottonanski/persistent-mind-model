from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timezone, timedelta


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class EvalConfig:
    stagnation_turns: int = 10
    min_interval_hours: float = 12.0
    weekly_hours: float = 24 * 7
    improvement_eps: float = 0.05  # non‑trivial movement threshold


class PolicyEvolution:
    """Periodic evaluator that tunes autonomy/policy thresholds.

    Signals read via SQLite `autonomy_tick` events; writes JSON deltas to
    the persistent model and emits `policy_adjusted` events for audit.
    """

    def __init__(self, smm, config: Optional[EvalConfig] = None):
        self.smm = smm
        self.cfg = config or EvalConfig()
        self._last_check_at: Optional[datetime] = None

    # ---- public ----
    def maybe_adjust(self) -> Optional[Dict[str, Any]]:
        now = _now_utc()
        if self._last_check_at and (now - self._last_check_at) < timedelta(
            hours=self.cfg.min_interval_hours
        ):
            return None
        self._last_check_at = now

        # Load last N autonomy_tick events
        ticks = self._recent_ticks(limit=max(self.cfg.stagnation_turns, 12))
        if len(ticks) < self.cfg.stagnation_turns:
            return None

        # Compute movement over the window
        first, last = ticks[0], ticks[-1]
        di = float((last.get("ias") or 0.0) - (first.get("ias") or 0.0))
        dg = float((last.get("gas") or 0.0) - (first.get("gas") or 0.0))
        stagnating = (abs(di) < 1e-3) and (abs(dg) < 1e-3)

        if not stagnating:
            return None

        # Adjust a simple policy knob: evidence confidence threshold (0.60–0.85)
        # Keep it in the JSON model under meta_cognition as a pragmatic location.
        before = self._get_policy_snapshot()
        old = float(before.get("evidence_conf_threshold", 0.70))
        # If stagnating, slightly reduce threshold to capture more evidence
        new = max(0.60, round(old - 0.02, 2))
        self._set_policy_value("evidence_conf_threshold", new)

        after = self._get_policy_snapshot()
        self._emit_adjusted("evidence", before, after, reason="stagnation")
        return {"policy": "evidence_conf_threshold", "before": old, "after": new}

    # ---- helpers ----
    def _recent_ticks(self, limit: int = 20):
        rows = list(
            self.smm.sqlite_store.conn.execute(
                "SELECT ts, kind, evidence FROM events WHERE kind='autonomy_tick' ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
        )
        out = []
        import json

        for ts, kind, evidence in rows[::-1]:
            try:
                ev = json.loads(evidence) if isinstance(evidence, str) else (evidence or {})
            except Exception:
                ev = {}
            out.append(
                {
                    "ts": ts,
                    "ias": float(ev.get("ias") or 0.0),
                    "gas": float(ev.get("gas") or 0.0),
                    "stage": (ev.get("stage") or "Unknown"),
                }
            )
        return out

    def _get_policy_snapshot(self) -> Dict[str, Any]:
        snap = getattr(self.smm.model.meta_cognition, "policy", None) or {}
        # Default threshold if never set
        if "evidence_conf_threshold" not in snap:
            snap["evidence_conf_threshold"] = 0.70
        return dict(snap)

    def _set_policy_value(self, key: str, value: Any) -> None:
        try:
            mc = self.smm.model.meta_cognition
            current = getattr(mc, "policy", None)
            if current is None:
                current = {}
                setattr(mc, "policy", current)
            current[key] = value
            self.smm.save_model()
            # Also mirror into environment when recognized to affect runtime behavior
            if key == "evidence_conf_threshold":
                try:
                    import os

                    os.environ["PMM_EVIDENCE_CONFIDENCE_THRESHOLD"] = str(value)
                except Exception:
                    pass
        except Exception:
            pass

    def _emit_adjusted(self, policy_type: str, before: Dict[str, Any], after: Dict[str, Any], reason: str) -> None:
        import json

        try:
            self.smm.sqlite_store.append_event(
                kind="policy_adjusted",
                content=json.dumps(
                    {"policy_type": policy_type, "before": before, "after": after},
                    ensure_ascii=False,
                ),
                meta={"reason": reason},
            )
        except Exception:
            pass
