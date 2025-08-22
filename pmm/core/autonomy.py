# pmm/core/autonomy.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import time
import threading
from typing import Optional, Tuple, Dict, Any, List

# PMM internals
from pmm.self_model_manager import SelfModelManager
from pmm.integrated_directive_system import IntegratedDirectiveSystem
from pmm.atomic_reflection import AtomicReflectionManager
from pmm.reflection_cooldown import ReflectionCooldownManager
from pmm.model_baselines import ModelBaselineManager
from pmm.emergence_stages import EmergenceStageManager
from pmm.emergence import EmergenceAnalyzer, EmergenceEvent
from pmm.storage.sqlite_store import SQLiteStore
from pmm.llm_factory import get_llm_factory
from pmm.ngram_ban import NGramBanSystem

from pmm.adaptive_triggers import AdaptiveTrigger, TriggerConfig, TriggerState
from pmm.reflection import reflect_once


def _utcnow():
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _load_recent_events_for_emergence(
    store: SQLiteStore, limit: int = 15
) -> List[EmergenceEvent]:
    rows = []
    try:
        cur = store.conn.execute(
            """
            SELECT id, ts, kind, content, meta
            FROM events
            WHERE kind IN ('response','event','reflection','evidence','commitment')
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = list(cur.fetchall())
    except Exception:
        return []

    out: List[EmergenceEvent] = []
    import json

    for row in rows:
        meta = {}
        try:
            meta = json.loads(row[4]) if isinstance(row[4], str) else (row[4] or {})
        except Exception:
            meta = {}
        out.append(
            EmergenceEvent(
                id=row[0], timestamp=row[1], kind=row[2], content=row[3], meta=meta
            )
        )
    return list(reversed(out))  # chronological for novelty calc


@dataclass
class AutonomySnapshot:
    last_tick_at: Optional[str] = None
    last_reason: Optional[str] = None
    last_reflected: bool = False
    last_reflection_id: Optional[int] = None
    ias: Optional[float] = None
    gas: Optional[float] = None
    ias_delta: Optional[float] = None
    gas_delta: Optional[float] = None
    events_analyzed: int = 0
    stage: str = "Unknown"


class AutonomyLoop:
    """
    Background autonomy loop for PMM.
    - Computes IAS/GAS from recent events
    - Uses AdaptiveTrigger to decide reflection cadence
    - Runs atomic reflection, directive consolidation, drift, and auto-close
    - Emits an 'autonomy_tick' event for audit on every pass
    """

    def __init__(
        self,
        pmm: SelfModelManager,
        *,
        interval_seconds: Optional[int] = None,
        directive_system: Optional[IntegratedDirectiveSystem] = None,
    ):
        self.pmm = pmm
        self.db: SQLiteStore = pmm.sqlite_store
        self.directive_system = directive_system or IntegratedDirectiveSystem(
            storage_manager=self.db
        )
        self.atomic_reflection = AtomicReflectionManager(self.pmm)
        self.cooldown = ReflectionCooldownManager()
        self.baselines = ModelBaselineManager()
        self.stages = EmergenceStageManager(self.baselines)
        self.ngram_ban = NGramBanSystem()

        # Trigger state seeded from PMM model
        last_ref_ts = getattr(self.pmm.model.self_knowledge, "last_reflection_ts", None)
        last_dt = None
        if isinstance(last_ref_ts, str):
            try:
                last_dt = datetime.fromisoformat(last_ref_ts)
            except Exception:
                last_dt = None

        events_since = (
            getattr(self.pmm.model.self_knowledge, "events_since_reflection", 0) or 0
        )
        self.trigger_state = TriggerState(
            last_reflection_at=last_dt,
            last_event_id=None,
            events_since_reflection=int(events_since),
        )

        cadence_days = getattr(self.pmm.model.metrics, "reflection_cadence_days", 7.0)
        self.trigger_cfg = TriggerConfig(
            cadence_days=cadence_days,
            events_min_gap=4,
            ias_low=0.35,
            gas_low=0.35,
            ias_high=0.65,
            gas_high=0.65,
            min_cooldown_minutes=10,
            max_skip_days=7.0,
        )
        self.trigger = AdaptiveTrigger(self.trigger_cfg, self.trigger_state)

        # loop control
        self._interval = int(
            os.getenv("PMM_AUTONOMY_INTERVAL", str(interval_seconds or 300))
        )
        self._lock = threading.Lock()
        self._last_scores: Tuple[Optional[float], Optional[float]] = (None, None)
        self._snapshot = AutonomySnapshot()

    # ---------- public API ----------

    def status(self) -> AutonomySnapshot:
        return self._snapshot

    def tick(self) -> None:
        """Run one autonomy pass safely (no overlap)."""
        if not self._lock.acquire(blocking=False):
            return  # skip overlapping tick

        try:
            self._tick_inner()
        finally:
            self._lock.release()

    def run_forever(self, stop_event: Optional[threading.Event] = None) -> None:
        """Daemon loop. Caller should run this in a daemon thread."""
        while True:
            if stop_event and stop_event.is_set():
                return
            self.tick()
            time.sleep(self._interval)

    # ---------- internals ----------

    def _compute_emergence(self) -> Dict[str, Any]:
        analyzer = EmergenceAnalyzer(storage_manager=self.db)
        recent = _load_recent_events_for_emergence(self.db, limit=15)
        if not recent:
            # analyzer would return zeros; we still want events_analyzed=0
            scores = analyzer.compute_scores(window=15)
            scores["events_analyzed"] = 0
            return scores

        # Override retrieval to use our fetched rows
        analyzer.get_recent_events = lambda kind="response", limit=15: recent[-limit:]
        scores = analyzer.compute_scores(window=min(15, len(recent)))
        return scores

    def _reflect_once(self) -> Optional[str]:
        """Run atomic reflection pipeline, return content if accepted."""
        # Respect active LLM config
        llm_factory = get_llm_factory()
        cfg = llm_factory.get_active_config()
        if not cfg or not cfg.get("name") or not cfg.get("provider"):
            return None

        # cooldown heuristic uses event window summarized text
        try:
            recent_dicts = self.db.recent_events(limit=8)  # returns list[dict]
            window_texts = []
            for e in recent_dicts:
                kind = e.get("kind", "")
                content = e.get("content", "") or e.get("summary", "")
                if kind in ("prompt", "response", "event", "reflection"):
                    window_texts.append(str(content))
            current_ctx = " ".join(window_texts[-6:])
        except Exception:
            current_ctx = ""

        ok, _reason = self.cooldown.should_reflect(current_ctx)
        if not ok:
            return None

        # generate insight
        insight_obj = reflect_once(self.pmm, None, cfg)
        if not insight_obj or not getattr(insight_obj, "content", "").strip():
            return None

        content = insight_obj.content.strip()

        # style hygiene
        filtered, _replacements = self.ngram_ban.postprocess_style(
            content, cfg.get("name", "unknown")
        )
        content = filtered

        # atomic accept & persist (dedup etc.)
        accepted = self.atomic_reflection.add_insight(
            content,
            cfg,
            cfg.get("epoch", 0),
        )
        if not accepted:
            return None

        # record emergence baseline at acceptance
        try:
            ctx = self.pmm.get_emergence_context() or {}
            ias = float(ctx.get("ias", 0.0) or 0.0)
            gas = float(ctx.get("gas", 0.0) or 0.0)
            self.baselines.add_scores(cfg.get("name", "unknown"), ias, gas)
            profile = self.stages.calculate_emergence_profile(
                cfg.get("name", "unknown"), ias, gas
            )
            # soft log
            self.pmm.add_event(
                summary=f"Emergence stage now {profile.stage.value} (conf {profile.confidence:.2f})",
                etype="emergence_stage",
                effects=[],
            )
        except Exception:
            pass

        # auto-close + drift
        try:
            self.pmm.auto_close_commitments_from_reflection(content)
        except Exception:
            pass
        try:
            self.pmm.apply_drift_and_save()
        except Exception:
            pass

        # bookkeeping for next trigger decision
        now = _utcnow()
        self.pmm.model.self_knowledge.last_reflection_ts = _to_iso(now)
        self.pmm.model.self_knowledge.events_since_reflection = 0
        # directive consolidation
        try:
            self.directive_system.trigger_evolution_if_needed()
        except Exception:
            pass

        return content

    def _tick_inner(self) -> None:
        now = _utcnow()
        scores = self._compute_emergence()
        ias = scores.get("IAS")
        gas = scores.get("GAS")

        # Determine events since last reflection; keep heuristic if missing
        events_since = (
            getattr(self.pmm.model.self_knowledge, "events_since_reflection", 0) or 0
        )

        # Decide cadence
        should, reason = self.trigger.decide(now, ias, gas, events_since)
        reflected = False
        reflection_id = None

        if should:
            insight = self._reflect_once()
            reflected = bool(insight)
            if reflected:
                # Grab last reflection row id for audit (best-effort)
                try:
                    row = self.db.conn.execute(
                        "SELECT id FROM events WHERE kind='reflection' ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                    reflection_id = int(row[0]) if row else None
                except Exception:
                    reflection_id = None
            else:
                # even if rejected, count an “attempt” via events_since_reflection++
                try:
                    cur = (
                        getattr(
                            self.pmm.model.self_knowledge, "events_since_reflection", 0
                        )
                        or 0
                    )
                    self.pmm.model.self_knowledge.events_since_reflection = int(cur) + 1
                except Exception:
                    pass

        # deltas
        prev_ias, prev_gas = self._last_scores
        di = (ias - prev_ias) if (prev_ias is not None and ias is not None) else None
        dg = (gas - prev_gas) if (prev_gas is not None and gas is not None) else None
        self._last_scores = (ias, gas)

        # Persist audit event
        meta = {
            "reason": reason,
            "reflected": reflected,
            "reflection_id": reflection_id,
            "ias": ias,
            "gas": gas,
            "ias_delta": di,
            "gas_delta": dg,
            "events_analyzed": scores.get("events_analyzed", 0),
            "stage": scores.get("stage", "Unknown"),
        }
        try:
            self.pmm.add_event(
                summary=f"Autonomy tick: {reason} | IAS={ias} GAS={gas}",
                etype="autonomy_tick",
                effects=[],
                evidence=meta,
            )
        except Exception:
            pass

        # Update snapshot + persistence
        self._snapshot = AutonomySnapshot(
            last_tick_at=_to_iso(now),
            last_reason=reason,
            last_reflected=reflected,
            last_reflection_id=reflection_id,
            ias=ias,
            gas=gas,
            ias_delta=di,
            gas_delta=dg,
            events_analyzed=int(scores.get("events_analyzed", 0) or 0),
            stage=scores.get("stage", "Unknown"),
        )

        # increment ESR if we didn't reflect
        if not reflected:
            try:
                cur = (
                    getattr(self.pmm.model.self_knowledge, "events_since_reflection", 0)
                    or 0
                )
                self.pmm.model.self_knowledge.events_since_reflection = int(cur) + 1
            except Exception:
                pass

        # persist self-model
        try:
            self.pmm.save_model()
        except Exception:
            pass
