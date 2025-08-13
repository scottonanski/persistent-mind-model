from __future__ import annotations
import json
import threading
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional, List

from .model import (
    PersistentMindModel,
    Event,
    EffectHypothesis,
    Thought,
    Insight,
    IdentityChange,
    TraitScore,
)
from .validation import SchemaValidator
from .metrics import compute_identity_coherence, compute_self_consistency
from .drift import apply_effects
from .storage.sqlite_store import SQLiteStore
from .commitments import CommitmentTracker

# Minimal debug logging
DEBUG = os.environ.get("PMM_DEBUG", "0") == "1"


def _log(*a):
    if DEBUG:
        print("[PMM]", *a)


class SelfModelManager:
    """Interface to the persistent self-model: handles loading, saving, and structured updates."""

    def __init__(self, model_path: str = "persistent_self_model.json", **kwargs):
        # Back-compat: some tests may pass 'filepath=' instead of 'model_path='
        if "filepath" in kwargs and kwargs["filepath"]:
            model_path = kwargs["filepath"]
        self.model_path = model_path
        self.lock = threading.Lock()
        self.validator = SchemaValidator()
        self.commitment_tracker = CommitmentTracker()

        # Initialize SQLiteStore for API compatibility
        self.sqlite_store = SQLiteStore("pmm.db")

        self.model = self.load_model()
        # Sync commitments from model to tracker
        self._sync_commitments_from_model()

    # -------- persistence --------
    def load_model(self) -> PersistentMindModel:
        with self.lock:
            try:
                with open(self.model_path, "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                model = PersistentMindModel()
                # Save without acquiring lock again (we already have it)
                self._save_model_unlocked(model)
                return model

            # --- hydrate dict -> dataclasses (defaults first, then overlay) ---
            model = PersistentMindModel()

            # core_identity
            ci = data.get("core_identity", {}) or {}
            model.core_identity.id = ci.get("id", model.core_identity.id)
            model.core_identity.name = ci.get("name", model.core_identity.name)
            model.core_identity.birth_timestamp = ci.get(
                "birth_timestamp", model.core_identity.birth_timestamp
            )
            model.core_identity.aliases = ci.get("aliases", model.core_identity.aliases)

            # personality.traits.big5 / hexaco
            for grp in ("big5", "hexaco"):
                src = ((data.get("personality") or {}).get("traits") or {}).get(
                    grp
                ) or {}
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
            model.personality.mbti.last_update = mb.get(
                "last_update", model.personality.mbti.last_update
            )
            model.personality.mbti.origin = mb.get(
                "origin", model.personality.mbti.origin
            )
            if isinstance(mb.get("poles"), dict):
                for pole, val in mb["poles"].items():
                    if hasattr(model.personality.mbti.poles, pole):
                        setattr(model.personality.mbti.poles, pole, val)

            vals = (data.get("personality") or {}).get("values_schwartz")
            if isinstance(vals, list):
                model.personality.values_schwartz = vals

            prefs = (data.get("personality") or {}).get("preferences") or {}
            model.personality.preferences.style = prefs.get(
                "style", model.personality.preferences.style
            )
            model.personality.preferences.risk_tolerance = prefs.get(
                "risk_tolerance", model.personality.preferences.risk_tolerance
            )
            model.personality.preferences.collaboration_bias = prefs.get(
                "collaboration_bias", model.personality.preferences.collaboration_bias
            )

            emo = (data.get("personality") or {}).get("emotional_tendencies") or {}
            model.personality.emotional_tendencies.baseline_stability = emo.get(
                "baseline_stability",
                model.personality.emotional_tendencies.baseline_stability,
            )
            model.personality.emotional_tendencies.assertiveness = emo.get(
                "assertiveness", model.personality.emotional_tendencies.assertiveness
            )
            model.personality.emotional_tendencies.cooperativeness = emo.get(
                "cooperativeness",
                model.personality.emotional_tendencies.cooperativeness,
            )

            # Self knowledge: patterns, events, thoughts, insights (convert to dataclasses where needed)
            sk = data.get("self_knowledge", {}) or {}
            if isinstance(sk.get("behavioral_patterns"), dict):
                model.self_knowledge.behavioral_patterns = sk["behavioral_patterns"]

            def _to_effects(lst):
                out: List[EffectHypothesis] = []
                for e in lst or []:
                    if isinstance(e, dict):
                        out.append(
                            EffectHypothesis(
                                target=e.get("target", ""),
                                delta=float(e.get("delta", 0.0) or 0.0),
                                confidence=float(e.get("confidence", 0.0) or 0.0),
                            )
                        )
                return out

            events = []
            for ev in sk.get("autobiographical_events", []) or []:
                if isinstance(ev, dict):
                    events.append(
                        Event(
                            id=ev.get("id", ""),
                            t=ev.get("t", ""),
                            type=ev.get("type", "experience"),
                            summary=ev.get("summary", ""),
                            valence=ev.get("valence", 0.5),
                            arousal=ev.get("arousal", 0.5),
                            salience=ev.get("salience", 0.5),
                            tags=ev.get("tags", []) or [],
                            effects_hypothesis=_to_effects(
                                ev.get("effects_hypothesis")
                            ),
                            meta=ev.get("meta", {"processed": False})
                            or {"processed": False},
                        )
                    )
            if events:
                model.self_knowledge.autobiographical_events = events

            thoughts = []
            for th in sk.get("thoughts", []) or []:
                if isinstance(th, dict):
                    thoughts.append(
                        Thought(
                            id=th.get("id", ""),
                            t=th.get("t", ""),
                            content=th.get("content", ""),
                            trigger=th.get("trigger", ""),
                        )
                    )
            if thoughts:
                model.self_knowledge.thoughts = thoughts

            insights = []
            for ins in sk.get("insights", []) or []:
                if isinstance(ins, dict):
                    insights.append(
                        Insight(
                            id=ins.get("id", ""),
                            t=ins.get("t", ""),
                            content=ins.get("content", ""),
                            references=ins.get("references", {}) or {},
                        )
                    )
            if insights:
                model.self_knowledge.insights = insights

            # Load commitments
            commitments_data = sk.get("commitments", {}) or {}
            if isinstance(commitments_data, dict):
                model.self_knowledge.commitments = commitments_data

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
            for k in (
                "maturity_principle",
                "inertia",
                "max_delta_per_reflection",
                "cooldown_days",
                "event_sensitivity",
                "notes",
            ):
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
                    IdentityChange(
                        t=item.get("t", "") if isinstance(item, dict) else "",
                        change=item.get("change", "") if isinstance(item, dict) else "",
                    )
                    for item in mc["identity_evolution"]
                    if isinstance(item, dict)
                ]

            # validate hydrated model
            self.validator.validate_model(model)
            _log("loaded", self.model_path)
            return model

    def save_model(self, model: Optional[PersistentMindModel] = None) -> None:
        with self.lock:
            self._save_model_unlocked(model)

    def _save_model_unlocked(self, model: Optional[PersistentMindModel] = None) -> None:
        """Save model without acquiring lock (internal use only)."""
        m = model or self.model
        # recompute metrics before save
        m.metrics.identity_coherence = compute_identity_coherence(m)
        m.metrics.self_consistency = compute_self_consistency(m)
        self.validator.validate_model(m)
        with open(self.model_path, "w") as f:
            json.dump(asdict(m), f, indent=2, sort_keys=False)
        _log("saved", self.model_path)

    # -------- convenience APIs --------
    def add_event(
        self,
        summary: str,
        effects: Optional[List[dict]] = None,
        *,
        etype: str = "experience",
    ) -> Event:
        ev_id = f"ev{len(self.model.self_knowledge.autobiographical_events)+1}"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        eff_objs: List[EffectHypothesis] = []
        for e in effects or []:
            eff_objs.append(
                EffectHypothesis(
                    target=e.get("target", ""),
                    delta=float(e.get("delta", 0.0) or 0.0),
                    confidence=float(e.get("confidence", 0.0) or 0.0),
                )
            )
        ev = Event(
            id=ev_id, t=ts, type=etype, summary=summary, effects_hypothesis=eff_objs
        )
        self.model.self_knowledge.autobiographical_events.append(ev)

        # Also write to SQLite for API compatibility
        try:
            import hashlib

            # Get previous hash for chain integrity
            prev_hash = self.sqlite_store.latest_hash()

            # Create hash for this event
            event_data = f"{ts}|event|{summary}|{ev_id}"
            current_hash = hashlib.sha256(event_data.encode()).hexdigest()

            self.sqlite_store.append_event(
                kind="event",
                content=summary,
                meta={"type": etype, "event_id": ev_id},
                hsh=current_hash,
                prev=prev_hash,
            )
        except Exception as e:
            print(f"Warning: Failed to write event to SQLite: {e}")

        self.save_model()
        return ev

    def add_thought(self, content: str, trigger: str = "") -> Thought:
        th_id = f"th{len(self.model.self_knowledge.thoughts)+1}"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        th = Thought(id=th_id, t=ts, content=content, trigger=trigger)
        self.model.self_knowledge.thoughts.append(th)
        self.save_model()
        return th

    def add_insight(self, content: str) -> Insight:
        in_id = f"in{len(self.model.self_knowledge.insights)+1}"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ins = Insight(id=in_id, t=ts, content=content)
        self.model.self_knowledge.insights.append(ins)

        # Extract and track commitments from insight
        cid = self.commitment_tracker.add_commitment(content, in_id)
        if cid:
            _log("commitment", f"Extracted commitment {cid} from insight {in_id}")

        self.save_model()
        return ins

    def apply_drift_and_save(self) -> dict:
        with self.lock:
            # Check pattern signals to steer drift with evidence weighting
            patterns = self.model.self_knowledge.behavioral_patterns
            _recent_insights = (
                self.model.self_knowledge.insights[-10:]
                if self.model.self_knowledge.insights
                else []
            )

            # Calculate pattern deltas (momentum from recent activity)
            exp_count = patterns.get("experimentation", 0)
            align_count = patterns.get("user_goal_alignment", 0)
            calib_count = patterns.get("calibration", 0)
            error_count = patterns.get("error_correction", 0)

            # Get commitment metrics for close rate
            commitment_metrics = self.commitment_tracker.get_commitment_metrics()
            close_rate = commitment_metrics.get("close_rate", 0)

            # Update model metrics with commitment data
            self.model.metrics.commitments_open = commitment_metrics.get(
                "commitments_open", 0
            )
            self.model.metrics.commitments_closed = commitment_metrics.get(
                "commitments_closed", 0
            )

            # Evidence-weighted signals (GPT-5's formula)
            exp_delta = max(0, exp_count - 3)  # Above baseline
            align_delta = max(0, align_count - 2)  # Above baseline
            close_rate_delta = max(0, close_rate - 0.3)  # Above 30% close rate

            signals = exp_delta + align_delta + close_rate_delta
            evidence_weight = min(1, signals / 3)  # Cap at 1.0

            # Adjust drift aggressiveness based on evidence-weighted patterns
            original_delta = self.model.drift_config.max_delta_per_reflection

            # If experimentation/alignment up with evidence, boost delta
            if evidence_weight > 0.3:
                boost_factor = 1 + (0.5 * evidence_weight)  # 1.0 to 1.5x
                self.model.drift_config.max_delta_per_reflection = min(
                    0.05, original_delta * boost_factor
                )
                _log(
                    "drift",
                    f"Evidence-weighted delta boost: {boost_factor:.2f}x (signals: {signals:.2f})",
                )

            # Symmetry guard for neuroticism (GPT-5's recommendation)
            if calib_count > 3 and error_count > 2 and close_rate > 0.4:
                # Apply downward bias only with both calibration AND commitment completion
                neuro_trait = self.model.personality.traits.big5.neuroticism
                if neuro_trait.score > 0.3:
                    bias_strength = min(0.02, 0.01 * evidence_weight)
                    neuro_trait.score = max(0.05, neuro_trait.score - bias_strength)
                    _log(
                        "drift",
                        f"Applied neuroticism stability bias: -{bias_strength:.3f} (close_rate: {close_rate:.2f})",
                    )
            elif calib_count > 3 and close_rate <= 0.4:
                # Reduced bias if calibration up but commitments not closing
                neuro_trait = self.model.personality.traits.big5.neuroticism
                if neuro_trait.score > 0.3:
                    bias_strength = 0.005  # 50% of planned decrease
                    neuro_trait.score = max(0.05, neuro_trait.score - bias_strength)
                    _log(
                        "drift",
                        f"Reduced neuroticism bias: -{bias_strength:.3f} (low close rate: {close_rate:.2f})",
                    )

            net = apply_effects(self.model)

            # Restore original delta
            self.model.drift_config.max_delta_per_reflection = original_delta

            # Save without acquiring lock again (we already have it)
            self._save_model_unlocked(self.model)
            _log("drift", f"Applied drift with evidence weight: {evidence_weight:.2f}")
            return net

    def set_drift_params(
        self,
        max_delta: float = 0.02,
        maturity_factor: float = 0.8,
        max_delta_per_reflection: float = None,
        notes_append: str = None,
    ):
        """Set drift parameters for this agent."""
        # Handle both parameter names for backward compatibility
        if max_delta_per_reflection is not None:
            self.model.drift_config.max_delta_per_reflection = max_delta_per_reflection
        else:
            self.model.drift_config.max_delta_per_reflection = max_delta

        self.model.drift_config.maturity_factor = maturity_factor

        # Handle notes append
        if notes_append:
            dc = self.model.drift_config
            dc.notes = (dc.notes + "\n" if dc.notes else "") + str(notes_append)

        # Save the changes
        self.save_model()

    # -------- commitment tracking --------
    def add_commitment(
        self, text: str, source_insight_id: str, due: Optional[str] = None
    ) -> str:
        """Add a new commitment and return its ID."""
        return self.commitment_tracker.add_commitment(text, source_insight_id, due)

    def mark_commitment(
        self, cid: str, status: str, note: Optional[str] = None
    ) -> bool:
        """Manually mark a commitment as closed/completed."""
        return self.commitment_tracker.mark_commitment(cid, status, note)

    def get_open_commitments(self) -> List[dict]:
        """Get all open commitments."""
        return self.commitment_tracker.get_open_commitments()

    def auto_close_commitments_from_event(self, event_text: str) -> List[str]:
        """Auto-close commitments mentioned in event descriptions."""
        return self.commitment_tracker.auto_close_from_event(event_text)

    def auto_close_commitments_from_reflection(self, reflection_text: str) -> List[str]:
        """Auto-close commitments based on reflection completion signals."""
        closed_cids = self.commitment_tracker.auto_close_from_reflection(
            reflection_text
        )
        if closed_cids:
            self._sync_commitments_to_model()
        return closed_cids

    def _sync_commitments_from_model(self):
        """Load commitments from model into tracker."""
        if hasattr(self.model.self_knowledge, "commitments"):
            for cid, commitment_data in self.model.self_knowledge.commitments.items():
                # Reconstruct Commitment object from dict
                from .commitments import Commitment

                commitment = Commitment(
                    cid=commitment_data["cid"],
                    text=commitment_data["text"],
                    created_at=commitment_data["created_at"],
                    source_insight_id=commitment_data["source_insight_id"],
                    status=commitment_data.get("status", "open"),
                    closed_at=commitment_data.get("closed_at"),
                    due=commitment_data.get("due"),
                    close_note=commitment_data.get("close_note"),
                    ngrams=commitment_data.get("ngrams", []),
                )
                self.commitment_tracker.commitments[cid] = commitment

    def _sync_commitments_to_model(self):
        """Save commitments from tracker to model."""
        commitment_dict = {}
        for cid, commitment in self.commitment_tracker.commitments.items():
            commitment_dict[cid] = {
                "cid": commitment.cid,
                "text": commitment.text,
                "created_at": commitment.created_at,
                "source_insight_id": commitment.source_insight_id,
                "status": commitment.status,
                "closed_at": commitment.closed_at,
                "due": commitment.due,
                "close_note": commitment.close_note,
                "ngrams": commitment.ngrams or [],
            }
        self.model.self_knowledge.commitments = commitment_dict

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
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
            "stability": [
                "stable",
                "stability",
                "consistent",
                "reliable",
                "predictable",
            ],
            "identity": ["identity", "who i am", "self", "recognize", "observe"],
            "growth": [
                "grow",
                "growth",
                "improve",
                "adapt",
                "expand",
                "develop",
                "enhance",
                "evolve",
            ],
            "reflection": [
                "reflect",
                "reflection",
                "summariz",
                "journal",
                "notice",
                "observed",
                "recognize",
            ],
            # newly tracked meta-behaviors
            "calibration": [
                "unsure",
                "uncertain",
                "confidence",
                "probability",
                "estimate",
                "assess",
            ],
            "error_correction": [
                "mistake",
                "fix",
                "correct",
                "regression",
                "bug",
                "adjust",
                "refine",
            ],
            "source_citation": [
                "`",
                ".py",
                "class ",
                "def ",
                "path/",
                "file:",
                "reference",
            ],
            "experimentation": [
                "ablation",
                "test",
                "benchmark",
                "experiment",
                "hypothesis",
                "explore",
                "try",
                "challenge",
                "innovative",
                "new approaches",
                "different",
                "diverse",
                "stimulate",
                "fresh",
            ],
            "user_goal_alignment": [
                "objective",
                "goal",
                "align",
                "constraint",
                "tradeoff",
                "allocate",
                "dedicate",
                "focus",
                "aim",
            ],
        }
        changed = False
        for label, terms in kw.items():
            if any(t in low for t in terms):
                patterns[label] = int(patterns.get(label, 0)) + 1
                changed = True
        if changed:
            self.save_model(self.model)
