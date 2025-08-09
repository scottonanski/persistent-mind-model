from __future__ import annotations
import json
import threading
import os
from dataclasses import asdict
from datetime import datetime
from typing import Optional, List

from .model import (
    PersistentMindModel,
    Event, EffectHypothesis, Thought, Insight, IdentityChange,
)
from .validation import SchemaValidator
from .metrics import compute_identity_coherence, compute_self_consistency
from .drift import apply_effects

# Minimal debug logging
DEBUG = os.environ.get("PMM_DEBUG", "0") == "1"

def _log(*a):
    if DEBUG:
        print("[PMM]", *a)

class SelfModelManager:
    """Interface to the persistent self-model: handles loading, saving, and structured updates."""

    def __init__(self, filepath: Optional[str] = None, schema_path: Optional[str] = None):
        self.filepath = filepath or "persistent_self_model.json"
        self.schema_path = schema_path or "schema/pmm.schema.json"
        self.validator = SchemaValidator(schema_path=self.schema_path)
        self.lock = threading.RLock()
        self.model = self.load_model()

    # -------- persistence --------
    def load_model(self) -> PersistentMindModel:
        with self.lock:
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                model = PersistentMindModel()
                self.save_model(model)
                return model

            # --- hydrate dict -> dataclasses (defaults first, then overlay) ---
            model = PersistentMindModel()

            # core_identity
            ci = data.get("core_identity", {}) or {}
            model.core_identity.id = ci.get("id", model.core_identity.id)
            model.core_identity.name = ci.get("name", model.core_identity.name)
            model.core_identity.birth_timestamp = ci.get("birth_timestamp", model.core_identity.birth_timestamp)
            model.core_identity.aliases = ci.get("aliases", model.core_identity.aliases)

            # personality.traits.big5 / hexaco
            for grp in ("big5", "hexaco"):
                src = (((data.get("personality") or {}).get("traits") or {}).get(grp) or {})
                dst = getattr(model.personality.traits, grp)
                for k, v in src.items():
                    ts = getattr(dst, k, None)
                    if ts and isinstance(v, dict):
                        ts.score = v.get("score", ts.score)
                        ts.conf = v.get("conf", ts.conf)
                        ts.last_update = v.get("last_update", ts.last_update)
                        ts.origin = v.get("origin", ts.origin)

            # mbti, values, prefs, emotion
            mb = (data.get("personality") or {}).get("mbti") or {}
            model.personality.mbti.label = mb.get("label", model.personality.mbti.label)
            model.personality.mbti.conf = mb.get("conf", model.personality.mbti.conf)
            model.personality.mbti.last_update = mb.get("last_update", model.personality.mbti.last_update)
            model.personality.mbti.origin = mb.get("origin", model.personality.mbti.origin)
            if isinstance(mb.get("poles"), dict):
                for pole, val in mb["poles"].items():
                    if hasattr(model.personality.mbti.poles, pole):
                        setattr(model.personality.mbti.poles, pole, val)

            vals = (data.get("personality") or {}).get("values_schwartz")
            if isinstance(vals, list):
                model.personality.values_schwartz = vals

            prefs = (data.get("personality") or {}).get("preferences") or {}
            model.personality.preferences.style = prefs.get("style", model.personality.preferences.style)
            model.personality.preferences.risk_tolerance = prefs.get("risk_tolerance", model.personality.preferences.risk_tolerance)
            model.personality.preferences.collaboration_bias = prefs.get("collaboration_bias", model.personality.preferences.collaboration_bias)

            emo = (data.get("personality") or {}).get("emotional_tendencies") or {}
            model.personality.emotional_tendencies.baseline_stability = emo.get("baseline_stability", model.personality.emotional_tendencies.baseline_stability)
            model.personality.emotional_tendencies.assertiveness = emo.get("assertiveness", model.personality.emotional_tendencies.assertiveness)
            model.personality.emotional_tendencies.cooperativeness = emo.get("cooperativeness", model.personality.emotional_tendencies.cooperativeness)

            # Self knowledge: patterns, events, thoughts, insights (convert to dataclasses where needed)
            sk = data.get("self_knowledge", {}) or {}
            if isinstance(sk.get("behavioral_patterns"), dict):
                model.self_knowledge.behavioral_patterns = sk["behavioral_patterns"]

            def _to_effects(lst):
                out: List[EffectHypothesis] = []
                for e in lst or []:
                    if isinstance(e, dict):
                        out.append(EffectHypothesis(
                            target=e.get("target", ""),
                            delta=float(e.get("delta", 0.0) or 0.0),
                            confidence=float(e.get("confidence", 0.0) or 0.0),
                        ))
                return out

            events = []
            for ev in sk.get("autobiographical_events", []) or []:
                if isinstance(ev, dict):
                    events.append(Event(
                        id=ev.get("id", ""),
                        t=ev.get("t", ""),
                        type=ev.get("type", "experience"),
                        summary=ev.get("summary", ""),
                        valence=ev.get("valence", 0.5),
                        arousal=ev.get("arousal", 0.5),
                        salience=ev.get("salience", 0.5),
                        tags=ev.get("tags", []) or [],
                        effects_hypothesis=_to_effects(ev.get("effects_hypothesis")),
                        meta=ev.get("meta", {"processed": False}) or {"processed": False},
                    ))
            if events:
                model.self_knowledge.autobiographical_events = events

            thoughts = []
            for th in sk.get("thoughts", []) or []:
                if isinstance(th, dict):
                    thoughts.append(Thought(
                        id=th.get("id", ""), t=th.get("t", ""), content=th.get("content", ""), trigger=th.get("trigger", "")
                    ))
            if thoughts:
                model.self_knowledge.thoughts = thoughts

            insights = []
            for ins in sk.get("insights", []) or []:
                if isinstance(ins, dict):
                    insights.append(Insight(
                        id=ins.get("id", ""), t=ins.get("t", ""), content=ins.get("content", ""),
                        references=ins.get("references", {}) or {}
                    ))
            if insights:
                model.self_knowledge.insights = insights

            # Metrics overlay
            met = data.get("metrics", {}) or {}
            if "identity_coherence" in met:
                model.metrics.identity_coherence = met["identity_coherence"]
            if "self_consistency" in met:
                model.metrics.self_consistency = met["self_consistency"]
            if isinstance(met.get("drift_velocity"), dict):
                model.metrics.drift_velocity = met["drift_velocity"]
            if "reflection_cadence_days" in met:
                model.metrics.reflection_cadence_days = met["reflection_cadence_days"]
            if "last_reflection_at" in met:
                model.metrics.last_reflection_at = met["last_reflection_at"]

            # Drift config overlay
            dc = data.get("drift_config", {}) or {}
            for k in ("maturity_principle", "inertia", "max_delta_per_reflection", "cooldown_days", "event_sensitivity", "notes"):
                if k in dc:
                    setattr(model.drift_config, k, dc[k])
            if isinstance(dc.get("bounds"), dict):
                if "min" in dc["bounds"]:
                    model.drift_config.bounds.min = dc["bounds"]["min"]
                if "max" in dc["bounds"]:
                    model.drift_config.bounds.max = dc["bounds"]["max"]
            if isinstance(dc.get("locks"), list):
                model.drift_config.locks = dc["locks"]

            # Meta cognition overlay
            mc = data.get("meta_cognition", {}) or {}
            for k in ("times_accessed_self", "self_modification_count"):
                if k in mc:
                    setattr(model.meta_cognition, k, mc[k])
            if isinstance(mc.get("identity_evolution"), list):
                model.meta_cognition.identity_evolution = [
                    IdentityChange(t=item.get("t", "") if isinstance(item, dict) else "",
                                   change=item.get("change", "") if isinstance(item, dict) else "")
                    for item in mc["identity_evolution"] if isinstance(item, dict)
                ]

            # validate hydrated model
            self.validator.validate_model(model)
            _log("loaded", self.filepath)
            return model

    def save_model(self, model: Optional[PersistentMindModel] = None) -> None:
        if model is None:
            model = self.model
        with self.lock:
            # recompute metrics before save
            model.metrics.identity_coherence = compute_identity_coherence(model)
            model.metrics.self_consistency = compute_self_consistency(model)
            self.validator.validate_model(model)
            with open(self.filepath, "w") as f:
                json.dump(asdict(model), f, indent=2, sort_keys=False)
            _log("saved", self.filepath)

    # -------- convenience APIs --------
    def add_event(self, summary: str, effects: Optional[List[dict]] = None, *, etype: str = "experience") -> Event:
        ev_id = f"ev{len(self.model.self_knowledge.autobiographical_events)+1}"
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        eff_objs: List[EffectHypothesis] = []
        for e in effects or []:
            eff_objs.append(EffectHypothesis(
                target=e.get("target", ""),
                delta=float(e.get("delta", 0.0) or 0.0),
                confidence=float(e.get("confidence", 0.0) or 0.0),
            ))
        ev = Event(
            id=ev_id, t=ts, type=etype, summary=summary,
            effects_hypothesis=eff_objs
        )
        self.model.self_knowledge.autobiographical_events.append(ev)
        self.save_model()
        return ev

    def add_thought(self, content: str, trigger: str = "") -> Thought:
        th_id = f"th{len(self.model.self_knowledge.thoughts)+1}"
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        th = Thought(id=th_id, t=ts, content=content, trigger=trigger)
        self.model.self_knowledge.thoughts.append(th)
        self.save_model()
        return th

    def add_insight(self, content: str) -> Insight:
        in_id = f"in{len(self.model.self_knowledge.insights)+1}"
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        ins = Insight(id=in_id, t=ts, content=content)
        self.model.self_knowledge.insights.append(ins)
        self.save_model()
        return ins

    def apply_drift_and_save(self) -> dict:
        with self.lock:
            # Check pattern signals to steer drift
            patterns = self.model.self_knowledge.behavioral_patterns
            recent_insights = self.model.self_knowledge.insights[-10:] if self.model.self_knowledge.insights else []
            
            # Calculate pattern momentum from recent insights
            exp_count = patterns.get("experimentation", 0)
            align_count = patterns.get("user_goal_alignment", 0)
            calib_count = patterns.get("calibration", 0)
            error_count = patterns.get("error_correction", 0)
            
            # Adjust drift aggressiveness based on patterns
            original_delta = self.model.drift_config.max_delta_per_reflection
            
            # If experimentation/alignment up, allow upper half of delta for openness/conscientiousness
            if exp_count > 5 or align_count > 3:
                self.model.drift_config.max_delta_per_reflection = min(0.05, original_delta * 1.5)
                _log("drift", f"Boosted delta to {self.model.drift_config.max_delta_per_reflection} due to experimentation/alignment")
            
            # If calibration/error_correction up, bias neuroticism downward
            if calib_count > 3 and error_count > 2:
                # Apply small downward bias to neuroticism
                neuro_trait = self.model.personality.traits.big5.neuroticism
                if neuro_trait.score > 0.3:  # Only if not already low
                    neuro_trait.score = max(0.05, neuro_trait.score - 0.01)
                    _log("drift", f"Applied neuroticism stability bias: {neuro_trait.score}")
            
            net = apply_effects(self.model)
            
            # Restore original delta
            self.model.drift_config.max_delta_per_reflection = original_delta
            
            self.save_model(self.model)
            _log("drift", net)
            return net

    # -------- extra helpers for duel/mentor loops --------
    def get_big5(self) -> dict:
        """Return a flat dict of Big Five scores from the nested dataclasses."""
        b5 = self.model.personality.traits.big5
        return {
            "openness": b5.openness.score,
            "conscientiousness": b5.conscientiousness.score,
            "extraversion": b5.extraversion.score,
            "agreeableness": b5.agreeableness.score,
            "neuroticism": b5.neuroticism.score,
        }

    def set_big5(self, updates: dict, origin: str = "manual") -> None:
        """Set Big Five scores with clamping to drift bounds; updates last_update and origin."""
        if not updates:
            return
        bmin = self.model.drift_config.bounds.min
        bmax = self.model.drift_config.bounds.max
        today = datetime.utcnow().strftime("%Y-%m-%d")
        b5 = self.model.personality.traits.big5
        with self.lock:
            for k, v in updates.items():
                if not hasattr(b5, k):
                    continue
                v = max(bmin, min(bmax, float(v)))
                ts: TraitScore = getattr(b5, k)
                ts.score = v
                ts.last_update = today
                ts.origin = origin
            self.save_model(self.model)

    def update_patterns(self, text: str) -> None:
        """Very simple keyword-based pattern incrementer to populate behavioral_patterns."""
        if not text:
            return
        low = text.lower()
        patterns = self.model.self_knowledge.behavioral_patterns
        # lightweight, extend as needed
        kw = {
            "stability": ["stable", "stability", "consistent"],
            "identity": ["identity", "who i am", "self"],
            "growth": ["grow", "growth", "improve", "adapt"],
            "reflection": ["reflect", "reflection", "summariz", "journal"],
            # newly tracked meta-behaviors
            "calibration": ["unsure", "uncertain", "confidence", "probability", "estimate"],
            "error_correction": ["mistake", "fix", "correct", "regression", "bug"],
            "source_citation": ["`", ".py", "class ", "def ", "path/", "file:"],
            "experimentation": ["ablation", "test", "benchmark", "experiment", "hypothesis"],
            "user_goal_alignment": ["objective", "goal", "align", "constraint", "tradeoff"],
        }
        changed = False
        for label, terms in kw.items():
            if any(t in low for t in terms):
                patterns[label] = int(patterns.get(label, 0)) + 1
                changed = True
        if changed:
            self.save_model(self.model)

    def set_drift_params(
        self,
        *,
        max_delta_per_reflection: float | None = None,
        locks: list[str] | None = None,
        notes_append: str | None = None,
    ) -> None:
        """Adjust drift configuration on the live model and persist it."""
        dc = self.model.drift_config
        with self.lock:
            if max_delta_per_reflection is not None:
                dc.max_delta_per_reflection = float(max_delta_per_reflection)
            if locks is not None:
                dc.locks = list(locks)
            if notes_append:
                dc.notes = (dc.notes + "\n" if dc.notes else "") + str(notes_append)
            self.save_model(self.model)
