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
        self.cooldown = ReflectionCooldownManager(
            min_turns=2,
            min_wall_time_seconds=120,
            novelty_threshold=0.85,
        )
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
        # Adaptive metrics emission state
        self._last_metrics_at: Optional[datetime] = None
        self._last_metrics_event_id: Optional[int] = None

        # Plateau detection and dynamic drift tuning state
        try:
            self._plateau_eps = float(os.getenv("PMM_PLATEAU_EPS", "0.02"))
        except Exception:
            self._plateau_eps = 0.02
        try:
            self._plateau_ticks_req = int(os.getenv("PMM_PLATEAU_TICKS", "4"))
        except Exception:
            self._plateau_ticks_req = 4
        try:
            self._booster_cooldown_min = int(
                os.getenv("PMM_BOOSTER_COOLDOWN_MIN", "30")
            )
        except Exception:
            self._booster_cooldown_min = 30
        # Bounds and steps for drift adjustments
        self._inertia_min = 0.5
        self._inertia_max = 0.95
        self._inertia_step = 0.05
        self._max_delta_min = 0.01
        self._max_delta_max = 0.08
        self._max_delta_step = 0.01

        self._plateau_counter = 0
        self._last_booster_at: Optional[datetime] = None

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
            # Compute current IAS/GAS using EmergenceAnalyzer (safe fallback)
            analyzer = EmergenceAnalyzer(storage_manager=self.db)
            recent = _load_recent_events_for_emergence(self.db, limit=15)
            if recent:
                analyzer.get_recent_events = lambda kind="response", limit=15: recent[
                    -limit:
                ]
                scores = analyzer.compute_scores(window=min(15, len(recent)))
            else:
                scores = analyzer.compute_scores(window=15)

            ias = float(scores.get("IAS", 0.0) or 0.0)
            gas = float(scores.get("GAS", 0.0) or 0.0)

            model_name = cfg.get("name", "unknown")
            self.baselines.add_scores(model_name, ias, gas)
            profile = self.stages.calculate_emergence_profile(model_name, ias, gas)

            # soft log
            self.pmm.add_event(
                summary=f"Emergence stage now {profile.stage.value} (conf {profile.confidence:.2f})",
                etype="emergence_stage",
                effects=[],
            )
        except Exception:
            pass

        # auto-close + provisional hints + drift
        try:
            self.pmm.auto_close_commitments_from_reflection(content)
        except Exception:
            pass
        try:
            self.pmm.provisional_close_commitments_from_reflection(content)
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

    def _reflect_booster(self, reason: str) -> Optional[str]:
        """Run a booster reflection that bypasses cooldown/novelty gates.

        This uses ReflectionCooldownManager.force_reasons and can optionally
        one-shot force-accept the next insight via env flag.
        """
        # Respect active LLM config
        llm_factory = get_llm_factory()
        cfg = llm_factory.get_active_config()
        if not cfg or not cfg.get("name") or not cfg.get("provider"):
            return None

        # Prepare minimal current context window
        try:
            recent_dicts = self.db.recent_events(limit=8)
            window_texts = []
            for e in recent_dicts:
                kind = e.get("kind", "")
                content = e.get("content", "") or e.get("summary", "")
                if kind in ("prompt", "response", "event", "reflection"):
                    window_texts.append(str(content))
            current_ctx = " ".join(window_texts[-6:])
        except Exception:
            current_ctx = ""

        # Bypass cooldown with force reason
        ok, _reason = self.cooldown.should_reflect(
            current_ctx, force_reasons=["plateau_booster", reason]
        )
        if not ok:
            return None

        # One-shot dedup bypass for the booster to ensure progress
        prev_flag = os.getenv("PMM_FORCE_ACCEPT_NEXT_INSIGHT")
        os.environ["PMM_FORCE_ACCEPT_NEXT_INSIGHT"] = "1"
        try:
            insight_obj = reflect_once(self.pmm, None, cfg)
        finally:
            # Restore previous state
            if prev_flag is None:
                os.environ.pop("PMM_FORCE_ACCEPT_NEXT_INSIGHT", None)
            else:
                os.environ["PMM_FORCE_ACCEPT_NEXT_INSIGHT"] = prev_flag

        if not insight_obj or not getattr(insight_obj, "content", "").strip():
            return None

        content = insight_obj.content.strip()

        # Style hygiene
        filtered, _replacements = self.ngram_ban.postprocess_style(
            content, cfg.get("name", "unknown")
        )
        content = filtered

        # Persist insight atomically (AtomicReflectionManager handles validation/persist)
        accepted = self.atomic_reflection.add_insight(
            content,
            cfg,
            cfg.get("epoch", 0),
        )
        if not accepted:
            return None

        # Record emergence snapshot and apply drift as in regular reflection
        try:
            analyzer = EmergenceAnalyzer(storage_manager=self.db)
            recent = _load_recent_events_for_emergence(self.db, limit=15)
            if recent:
                analyzer.get_recent_events = lambda kind="response", limit=15: recent[
                    -limit:
                ]
                scores = analyzer.compute_scores(window=min(15, len(recent)))
            else:
                scores = analyzer.compute_scores(window=15)

            ias = float(scores.get("IAS", 0.0) or 0.0)
            gas = float(scores.get("GAS", 0.0) or 0.0)
            model_name = cfg.get("name", "unknown")
            self.baselines.add_scores(model_name, ias, gas)
            profile = self.stages.calculate_emergence_profile(model_name, ias, gas)
            self.pmm.add_event(
                summary=f"Booster reflection applied. Emergence stage {profile.stage.value} (conf {profile.confidence:.2f})",
                etype="emergence_stage",
                effects=[],
            )
        except Exception:
            pass

        # Auto-close + provisional hints + drift
        try:
            self.pmm.auto_close_commitments_from_reflection(content)
        except Exception:
            pass
        try:
            self.pmm.provisional_close_commitments_from_reflection(content)
        except Exception:
            pass
        try:
            self.pmm.apply_drift_and_save()
        except Exception:
            pass

        now = _utcnow()
        self.pmm.model.self_knowledge.last_reflection_ts = _to_iso(now)
        self.pmm.model.self_knowledge.events_since_reflection = 0
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

        # Decide cadence from AdaptiveTrigger (emergent autonomy)
        should, reason = self.trigger.decide(now, ias, gas, events_since)

        # Soft user override: suppress only if the last user explicitly blocked reflection
        try:
            last_user_text = self._get_last_user_prompt()
            if self._user_blocked_reflection(last_user_text):
                should = False
                reason = "suppressed:user-opt-out"
        except Exception:
            # If we can't read prompts, proceed with emergent decision
            pass
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
                # Extract directives/commitments from the reflection content
                try:
                    detected = self.directive_system.process_response(
                        user_message="",
                        ai_response=insight,
                        event_id="autonomy_reflection",
                    )
                    for directive in detected or []:
                        dcontent = getattr(directive, "content", None)
                        if dcontent:
                            try:
                                self.pmm.add_commitment(
                                    text=dcontent[:200],
                                    source_insight_id="autonomy_reflection",
                                )
                            except Exception:
                                pass
                except Exception:
                    pass
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

        # Plateau detection: increment counter if both deltas are tiny
        try:
            if di is not None and dg is not None:
                if abs(di) < self._plateau_eps and abs(dg) < self._plateau_eps:
                    self._plateau_counter += 1
                else:
                    self._plateau_counter = 0
        except Exception:
            pass

        # Dynamic drift tuning based on momentum vs plateau
        try:
            cfg = self.pmm.model.drift_config
            # Initialize safe values if missing
            inertia = float(getattr(cfg, "inertia", 0.9) or 0.9)
            max_step = float(getattr(cfg, "max_delta_per_reflection", 0.02) or 0.02)

            if di is not None and dg is not None:
                total_delta = abs(di) + abs(dg)
                # Plateau: reduce inertia (more responsive), increase max step (more change)
                if self._plateau_counter >= self._plateau_ticks_req:
                    new_inertia = max(
                        self._inertia_min, round(inertia - self._inertia_step, 3)
                    )
                    new_step = min(
                        self._max_delta_max, round(max_step + self._max_delta_step, 3)
                    )
                    if new_inertia != inertia or new_step != max_step:
                        self.pmm.model.drift_config.inertia = new_inertia
                        self.pmm.model.drift_config.max_delta_per_reflection = new_step
                        self.pmm.save_model()
                        try:
                            self.pmm.add_event(
                                summary=f"Drift tuned (plateau): inertia {inertia:.2f}->{new_inertia:.2f}, max_step {max_step:.3f}->{new_step:.3f}",
                                etype="drift_tuning",
                                effects=[],
                            )
                        except Exception:
                            pass
                # Momentum: if strong movement, gently restore inertia up and reduce max step
                elif total_delta > (self._plateau_eps * 4):
                    new_inertia = min(
                        self._inertia_max, round(inertia + self._inertia_step, 3)
                    )
                    new_step = max(
                        self._max_delta_min, round(max_step - self._max_delta_step, 3)
                    )
                    if new_inertia != inertia or new_step != max_step:
                        self.pmm.model.drift_config.inertia = new_inertia
                        self.pmm.model.drift_config.max_delta_per_reflection = new_step
                        self.pmm.save_model()
                        try:
                            self.pmm.add_event(
                                summary=f"Drift tuned (momentum): inertia {inertia:.2f}->{new_inertia:.2f}, max_step {max_step:.3f}->{new_step:.3f}",
                                etype="drift_tuning",
                                effects=[],
                            )
                        except Exception:
                            pass
        except Exception:
            pass

        # Booster reflection trigger on plateau with cooldown
        booster_triggered = False
        try:
            can_trigger = self._plateau_counter >= self._plateau_ticks_req
            if can_trigger:
                # Check booster cooldown
                if self._last_booster_at is None:
                    cooldown_passed = True
                else:
                    elapsed = (now - self._last_booster_at).total_seconds() / 60.0
                    cooldown_passed = elapsed >= self._booster_cooldown_min

                if cooldown_passed:
                    insight = self._reflect_booster("plateau")
                    booster_triggered = bool(insight)
                    if booster_triggered:
                        self._last_booster_at = now
                        self._plateau_counter = 0
                        try:
                            self.pmm.add_event(
                                summary="Booster reflection triggered due to emergence plateau",
                                etype="autonomy_booster",
                                effects=[],
                            )
                        except Exception:
                            pass
        except Exception:
            pass

        # Decide whether to emit metrics (adaptive cadence)
        # Environment-configurable thresholds
        try:
            min_interval = int(os.getenv("PMM_METRICS_MIN_INTERVAL", "3600"))  # seconds
        except Exception:
            min_interval = 3600
        try:
            active_interval = int(
                os.getenv("PMM_METRICS_ACTIVE_INTERVAL", "900")
            )  # seconds
        except Exception:
            active_interval = 900
        try:
            min_events = int(os.getenv("PMM_METRICS_MIN_EVENTS", "12"))
        except Exception:
            min_events = 12

        # Compute elapsed time and new-event count
        latest_row = None
        try:
            latest_row = self.db.conn.execute("SELECT MAX(id) FROM events").fetchone()
        except Exception:
            latest_row = None
        latest_id = int(latest_row[0]) if latest_row and latest_row[0] else 0
        new_events = (
            (latest_id - int(self._last_metrics_event_id or 0)) if latest_id else 0
        )
        elapsed = (
            (now - self._last_metrics_at).total_seconds()
            if self._last_metrics_at
            else 9e9
        )

        # Choose target interval based on activity/decline
        declining = (di is not None and di < 0) or (dg is not None and dg < 0)
        target_interval = (
            active_interval
            if (declining or reflected or booster_triggered)
            else min_interval
        )
        should_emit = (elapsed >= target_interval) or (new_events >= min_events)

        meta = {
            "reason": reason,
            "reflected": reflected or booster_triggered,
            "reflection_id": reflection_id,
            "ias": ias,
            "gas": gas,
            "ias_delta": di,
            "gas_delta": dg,
            "events_analyzed": scores.get("events_analyzed", 0),
            "stage": scores.get("stage", "Unknown"),
            "plateau_counter": self._plateau_counter,
            "booster": booster_triggered,
        }
        # Skip emission when we lack sufficient recent signal
        min_state_events = 5
        try:
            min_state_events = int(os.getenv("PMM_METRICS_STATE_MIN_EVENTS", "5"))
        except Exception:
            pass
        has_state = int(scores.get("events_analyzed", 0) or 0) >= min_state_events

        if should_emit and has_state:
            try:
                self.pmm.add_event(
                    summary=f"Autonomy tick: {reason} | IAS={ias} GAS={gas}",
                    etype="autonomy_tick",
                    effects=[],
                    evidence=meta,
                )
                self._last_metrics_at = now
                self._last_metrics_event_id = latest_id
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

        # ---- Integrations: drift watch, experiments, policy evolution ----
        track_snapshot = {
            "IAS": float(ias or 0.0),
            "GAS": float(gas or 0.0),
            "stage": scores.get("stage", "Unknown"),
        }
        # Drift Watch + Self-Heal
        try:
            from pmm.drift_watch import watch_and_heal

            watch_and_heal(self.pmm, track_snapshot)
        except Exception:
            pass
        # Run due micro-experiments (if any)
        try:
            from pmm.experiments import ExperimentManager

            ExperimentManager(self.db).run_due()
        except Exception:
            pass
        # Policy evolution (stagnation-based tuning)
        try:
            from pmm.policy.evolution import PolicyEvolution

            PolicyEvolution(self.pmm).maybe_adjust()
        except Exception:
            pass

    # -------- minimal helpers: last user prompt + intent detection --------
    def _get_last_user_prompt(self) -> str:
        """Return content of the most recent 'prompt' event (user input)."""
        try:
            row = self.db.conn.execute(
                "SELECT content FROM events WHERE kind='prompt' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return str(row[0]) if row and row[0] else ""
        except Exception:
            return ""

    def _user_blocked_reflection(self, text: str) -> bool:
        """Heuristic for explicit user opt-out of background reflection."""
        try:
            s = (text or "").lower()
            import re

            patterns = [
                r"\bdo not reflect\b",
                r"\bdon't reflect\b",
                r"\bno reflection\b",
                r"\bstop reflecting\b",
                r"\bstop reflection\b",
                r"\bdisable reflection\b",
            ]
            return any(re.search(p, s) for p in patterns)
        except Exception:
            return False
