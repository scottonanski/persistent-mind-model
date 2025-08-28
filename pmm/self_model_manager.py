from __future__ import annotations
import json
import threading
import os
import hashlib
from dataclasses import asdict
import re
from datetime import datetime, timezone
from typing import Optional, List

from .model import (
    PersistentMindModel,
    Event,
    EffectHypothesis,
    Thought,
    Insight,
    IdentityChange,
)
from .validation import SchemaValidator
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
        self.lock = threading.RLock()  # FIXED: Use RLock for nested calls
        self.validator = SchemaValidator()
        self.commitment_tracker = CommitmentTracker()

        # Initialize SQLiteStore for API compatibility
        # Prefer explicit env override, otherwise co-locate DB with the JSON model path
        db_path = os.environ.get("PMM_DB_PATH")
        if not db_path:
            try:
                model_dir = os.path.dirname(os.path.abspath(self.model_path))
                db_path = os.path.join(model_dir, "pmm.db")
            except Exception:
                db_path = "pmm.db"
        self.sqlite_store = SQLiteStore(db_path)

        self.model = self.load_model()
        # Sync commitments from model to tracker
        self._sync_commitments_from_model()

        # Phase 2: Archive legacy generic commitments on startup
        try:
            archived = self.commitment_tracker.archive_legacy_commitments()
            if archived:
                print(
                    f"🔍 DEBUG: Archived {len(archived)} legacy commitments on startup"
                )
                self.save_model()  # Persist the archival
        except Exception:
            # Silently handle archival errors - not critical for operation
            pass

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
            except json.JSONDecodeError:
                # Corrupt JSON — back up and start fresh to preserve availability
                try:
                    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    backup_path = f"{self.model_path}.corrupt.{ts}.bak"
                    try:
                        os.replace(self.model_path, backup_path)
                    except Exception:
                        pass
                except Exception:
                    pass
                model = PersistentMindModel()
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

            vals = (data.get("personality") or {}).get("values") or {}
            for k, v in vals.items():
                if hasattr(model.personality.values, k):
                    setattr(model.personality.values, k, v)

            prefs = (data.get("personality") or {}).get("preferences") or {}
            for k, v in prefs.items():
                if hasattr(model.personality.preferences, k):
                    setattr(model.personality.preferences, k, v)

            emo = (data.get("personality") or {}).get("emotion") or {}
            for k, v in emo.items():
                if hasattr(model.personality.emotion, k):
                    setattr(model.personality.emotion, k, v)

            # self_knowledge
            sk = data.get("self_knowledge", {}) or {}

            # autobiographical_events
            events_data = sk.get("autobiographical_events", [])
            for ev_dict in events_data:
                # Create Event with proper defaults
                ev = Event(
                    id=ev_dict.get("id", ""),
                    t=ev_dict.get("t", ""),
                    summary=ev_dict.get("summary", ""),
                    type=ev_dict.get("type", "experience"),
                    tags=ev_dict.get("tags", []),
                    effects_hypothesis=self._to_effects(
                        ev_dict.get("effects_hypothesis", [])
                    ),
                    evidence=ev_dict.get("evidence"),
                )
                model.self_knowledge.autobiographical_events.append(ev)

            # thoughts
            thoughts_data = sk.get("thoughts", [])
            for th_dict in thoughts_data:
                th = Thought(
                    id=th_dict.get("id", ""),
                    t=th_dict.get("t", ""),
                    content=th_dict.get("content", ""),
                    trigger=th_dict.get("trigger", ""),
                )
                model.self_knowledge.thoughts.append(th)

            # insights
            insights_data = sk.get("insights", [])
            for ins_dict in insights_data:
                ins = Insight(
                    id=ins_dict.get("id", ""),
                    t=ins_dict.get("t", ""),
                    content=ins_dict.get("content", ""),
                )
                model.self_knowledge.insights.append(ins)

            # identity_evolution (stored in meta_cognition)
            meta = data.get("meta_cognition", {})
            identity_evolution_data = meta.get("identity_evolution", [])
            for ch_dict in identity_evolution_data:
                ch = IdentityChange(
                    t=ch_dict.get("t", ""),
                    change=ch_dict.get("change", ""),
                )
                model.meta_cognition.identity_evolution.append(ch)

            # bandit persistence fields (optional)
            try:
                bc = meta.get("bandit_counts")
                br = meta.get("bandit_rewards")
                if isinstance(bc, list):
                    model.meta_cognition.bandit_counts = bc
                if isinstance(br, list):
                    model.meta_cognition.bandit_rewards = br
            except Exception:
                pass

            # behavioral_patterns
            patterns = sk.get("behavioral_patterns", {})
            for k, v in patterns.items():
                model.self_knowledge.behavioral_patterns[k] = v

            # commitments - ensure it's always a dictionary
            commitments_data = sk.get("commitments", {})
            if isinstance(commitments_data, list):
                # Convert legacy list format to dictionary
                commitments_dict = {}
                for i, commitment in enumerate(commitments_data):
                    if isinstance(commitment, dict) and "id" in commitment:
                        commitments_dict[commitment["id"]] = commitment
                    else:
                        # Generate ID for legacy commitments without IDs
                        commitments_dict[f"legacy_{i}"] = commitment
                model.self_knowledge.commitments = commitments_dict
            else:
                model.self_knowledge.commitments = commitments_data

            # drift_params
            drift = data.get("drift_params", {})
            for k, v in drift.items():
                if hasattr(model.drift_params, k):
                    setattr(model.drift_params, k, v)

            # metadata
            meta = data.get("metadata", {})
            for k, v in meta.items():
                if hasattr(model.metadata, k):
                    setattr(model.metadata, k, v)

            return model

        def _to_effects(self, lst):
            """Convert list of effect dicts to EffectHypothesis objects."""
            out = []
            for item in lst:
                if isinstance(item, dict):
                    out.append(
                        EffectHypothesis(
                            trait=item.get("trait", ""),
                            direction=item.get("direction", ""),
                            magnitude=item.get("magnitude", 0.0),
                            confidence=item.get("confidence", 0.0),
                        )
                    )
            return out

    def save_model(self, model: Optional[PersistentMindModel] = None):
        with self.lock:
            self._save_model_unlocked(model)

    def _save_model_unlocked(self, model: Optional[PersistentMindModel] = None):
        """Save model without acquiring lock (internal use only)."""
        if model is None:
            model = self.model
        with open(self.model_path, "w") as f:
            json.dump(asdict(model), f, indent=2, ensure_ascii=False)

    # -------- structured updates --------

    def add_event(
        self,
        summary: str,
        effects: Optional[List[dict]] = None,
        *,
        etype: str = "experience",
        tags: Optional[List[str]] = None,
        full_text: Optional[str] = None,
        embedding: Optional[bytes] = None,
        evidence: Optional[dict] = None,
    ):
        """Add an autobiographical event with optional trait effects."""

        # Sanitize non-JSON types in evidence/tags to avoid model JSON corruption
        def _sanitize(obj):
            try:
                if obj is None:
                    return None
                if isinstance(obj, (str, int, float, bool)):
                    return obj
                if isinstance(obj, (list, tuple)):
                    return [_sanitize(x) for x in obj]
                if isinstance(obj, dict):
                    return {str(k): _sanitize(v) for k, v in obj.items()}
                # Fallback: stringify
                return str(obj)
            except Exception:
                return "<non-serializable>"

        if evidence is not None:
            evidence = _sanitize(evidence)
        if tags is not None:
            tags = _sanitize(tags)
        with self.lock:
            # FIXED: Generate ID atomically inside lock
            ev_id = f"ev{len(self.model.self_knowledge.autobiographical_events)+1}"

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # FIXED: Ensure tags consistency between JSON and SQLite
            if tags is None:
                tags = []

            # Create event for JSON model
            effects_list = self._to_effects(effects or [])
            event = Event(
                id=ev_id,
                t=ts,
                summary=summary,
                type=etype,
                tags=tags,  # Consistent tags
                effects=effects_list,  # For drift system
                effects_hypothesis=effects_list,  # For backward compatibility
                evidence=evidence,
            )
            self.model.self_knowledge.autobiographical_events.append(event)

            # FIXED: True hash chaining with prev_hash and full payload
            prev_hash = self.sqlite_store.latest_hash()

            # Include all relevant data in hash computation
            event_payload = {
                "ts": ts,
                "kind": "event",
                "summary": summary,
                "id": ev_id,
                "tags": tags,
                "effects": effects or [],
                "evidence": evidence,
                "prev_hash": prev_hash,
            }
            event_data = json.dumps(event_payload, sort_keys=True, ensure_ascii=False)
            current_hash = hashlib.sha256(event_data.encode()).hexdigest()

            # Write to SQLite with consistent fields
            self.sqlite_store.append_event(
                kind="event",
                content=summary,
                meta={
                    "id": ev_id,
                    "type": etype,
                    "tags": tags,  # FIXED: Consistent tags field
                    "effects": effects or [],
                    "evidence": evidence,
                    "full_text": full_text,
                },
                hsh=current_hash,
                prev=prev_hash,
                embedding=embedding,
            )

            self._save_model_unlocked()
            return ev_id

    def add_thought(self, content: str, trigger: str = ""):
        """Add a thought/reflection."""
        with self.lock:
            # FIXED: Generate ID atomically inside lock
            th_id = f"th{len(self.model.self_knowledge.thoughts)+1}"

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            thought = Thought(id=th_id, t=ts, content=content, trigger=trigger)
            self.model.self_knowledge.thoughts.append(thought)
            self._save_model_unlocked()

    def add_insight(self, content: str):
        """Add an insight with optional trait effects."""
        with self.lock:
            # FIXED: Generate ID atomically inside lock
            ins_id = f"ins{len(self.model.self_knowledge.insights)+1}"

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            insight = Insight(id=ins_id, t=ts, content=content)
            self.model.self_knowledge.insights.append(insight)
            self._save_model_unlocked()
            return ins_id

    def apply_drift_and_save(self):
        """Apply trait drift from accumulated effects and save."""
        with self.lock:
            # Collect all effects from events and insights
            all_effects = []
            for event in self.model.self_knowledge.autobiographical_events:
                all_effects.extend(event.effects)
            for insight in self.model.self_knowledge.insights:
                all_effects.extend(insight.effects)

            if not all_effects:
                return []

            # Apply drift
            old_traits = asdict(self.model.personality.traits)
            apply_effects(self.model)
            new_traits = asdict(self.model.personality.traits)

            # Log significant changes
            changes = []
            for group in ("big5", "hexaco"):
                for trait_name in old_traits[group]:
                    old_score = old_traits[group][trait_name]["score"]
                    new_score = new_traits[group][trait_name]["score"]
                    delta = abs(new_score - old_score)
                    if delta > 0.01:  # threshold for logging
                        changes.append(
                            f"{group}.{trait_name}: {old_score:.3f} → {new_score:.3f} (Δ{delta:+.3f})"
                        )

            if changes:
                _log(f"Trait drift applied: {', '.join(changes)}")

            self._save_model_unlocked()
            return changes

    def set_drift_params(
        self,
        max_delta: float = 0.02,
        maturity_factor: float = 0.8,
        max_delta_per_reflection: float = None,
        notes_append: str = None,
    ):
        """Set drift parameters for this agent."""
        with self.lock:
            self.model.drift_params.max_delta = max_delta
            self.model.drift_params.maturity_factor = maturity_factor
            if max_delta_per_reflection is not None:
                self.model.drift_params.max_delta_per_reflection = (
                    max_delta_per_reflection
                )
            if notes_append:
                self.model.drift_params.notes += f" {notes_append}"
            self._save_model_unlocked()

    # -------- commitment management --------

    def add_commitment(
        self, text: str, source_insight_id: str, due: Optional[str] = None
    ):
        """Add a new commitment and return its ID."""
        cid = self.commitment_tracker.add_commitment(text, source_insight_id, due)
        if not cid:
            return ""

        # Sync to model's self_knowledge for persistence
        commitment = self.commitment_tracker.commitments[cid]

        # Reinforcement path: duplicate intent recorded -> log reinforcement instead of new commitment
        if getattr(commitment, "_just_reinforced", False):
            try:
                meta = {
                    "cid": cid,
                    "source_insight_id": source_insight_id,
                    "attempts": getattr(commitment, "attempts", 1),
                    "reinforcements": getattr(commitment, "reinforcements", 0),
                    "status": commitment.status,
                }
                self.sqlite_store.append_event(
                    kind="commitment_reinforcement", content=commitment.text, meta=meta
                )
            except Exception:
                pass
            # Update self-model record if exists
            rec = self.model.self_knowledge.commitments.get(cid, {})
            rec["text"] = commitment.text
            rec["status"] = commitment.status
            rec["attempts"] = getattr(commitment, "attempts", 1)
            rec["reinforcements"] = getattr(commitment, "reinforcements", 0)
            rec["source_insight_id"] = source_insight_id
            rec["due"] = commitment.due
            self.model.self_knowledge.commitments[cid] = rec
            commitment._just_reinforced = False
            self.save_model()
            return cid

        # Emit a canonical 'commitment' event to the SQLite chain for auditability
        # and to enable evidence linkage by event hash.
        try:
            meta = {
                "cid": cid,
                "source_insight_id": source_insight_id,
                "due": due,
                "status": commitment.status,
            }
            res = self.sqlite_store.append_event(
                kind="commitment", content=commitment.text, meta=meta
            )
            # Store the canonical event hash on the commitment for future evidence linking
            try:
                commitment.event_hash = res.get("hash")
            except Exception:
                pass
        except Exception:
            # Best-effort: continue even if DB append fails
            res = None

        # Persist commitment details to the JSON self-model (include event hash when available)
        commit_record = {
            "text": commitment.text,
            "created_at": commitment.created_at,
            "status": commitment.status,
            "source_insight_id": commitment.source_insight_id,
            "due": commitment.due,
            "attempts": getattr(commitment, "attempts", 1),
            "reinforcements": getattr(commitment, "reinforcements", 0),
        }
        if res and res.get("hash"):
            commit_record["hash"] = res.get("hash")
            commit_record["event_id"] = res.get("event_id")

        self.model.self_knowledge.commitments[cid] = commit_record
        self.save_model()
        return cid

    def mark_commitment(self, cid: str, status: str, note: Optional[str] = None):
        """Manually mark a commitment as closed/completed."""
        self.commitment_tracker.mark_commitment(cid, status, note)

    def get_open_commitments(self):
        """Get all open commitments."""
        return self.commitment_tracker.get_open_commitments()

    def purge_legacy_commitments(self):
        """Permanently remove legacy commitments that don't align with autonomous development."""
        legacy_phrases = [
            "daily reflection practice",
            "weekly review",
            "unfamiliar individuals",
            "engage in brief",
            "daily conversations",
            "expand my network",
            "enhance engagement",
            "seek feedback",
            "solicit feedback",
            "initiate conversations",
            "challenge my comfort zone",
            "enrich my social perspective",
            "outside my usual circle",
            "feedback for improvement",
            "actively solicit feedback",
            "guide my understanding",
        ]

        # Get all commitments and mark legacy ones as closed
        all_commitments = self.commitment_tracker.commitments.copy()
        purged_count = 0

        for cid, commitment in all_commitments.items():
            commitment_text = (
                commitment.text if hasattr(commitment, "text") else str(commitment)
            )
            if any(phrase in commitment_text.lower() for phrase in legacy_phrases):
                self.commitment_tracker.mark_commitment(
                    cid, "closed", "Purged legacy commitment"
                )
                purged_count += 1

        # Also remove from model's commitments list
        if hasattr(self.model.self_knowledge, "commitments"):
            original_count = len(self.model.self_knowledge.commitments)
            self.model.self_knowledge.commitments = [
                c
                for c in self.model.self_knowledge.commitments
                if not any(
                    phrase
                    in (c.get("text", "") if isinstance(c, dict) else str(c)).lower()
                    for phrase in legacy_phrases
                )
            ]
            model_purged = original_count - len(self.model.self_knowledge.commitments)
            purged_count += model_purged

        if purged_count > 0:
            self.save_model()
            print(f"🧹 Purged {purged_count} legacy commitments")

        return purged_count

    def auto_close_commitments_from_event(self, event_text: str):
        """Auto-close commitments mentioned in event descriptions."""
        self.commitment_tracker.auto_close_from_event(event_text)

    def auto_close_commitments_from_reflection(self, reflection_text: str):
        """Auto-close commitments based on reflection completion signals."""
        patterns = ["completed", "finished", "done with", "accomplished"]
        for pattern in patterns:
            if pattern in reflection_text.lower():
                # This is a simple heuristic; could be more sophisticated
                break

    def provisional_close_commitments_from_reflection(self, reflection_text: str):
        """Emit non-evidence closure hints for commitments mentioned as completed in reflections.

        This does NOT mark commitments as closed. Instead, it writes lightweight
        'closure_hint' events to SQLite with a commit_ref meta field for the
        referenced commitment hash. Emergence GAS can use these hints to
        provisionally boost growth-seeking without violating evidence-only
        permanent closure rules.

        Returns: list of commitment hashes hinted.
        """
        try:
            text = (reflection_text or "").lower()
            if not text:
                return []

            # Only trigger when completion phrasing appears
            completion_markers = [
                "completed",
                "finished",
                "done with",
                "accomplished",
                "wrapped up",
                "i did",
                "i have done",
            ]
            if not any(m in text for m in completion_markers):
                return []

            # Collect open commitments and look for either short-hash or simple text overlap
            hinted: list[str] = []
            try:
                open_commitments = self.get_open_commitments() or []
            except Exception:
                open_commitments = []

            for c in open_commitments:
                try:
                    chash = (c.get("hash") or "").lower()
                    ctext = (c.get("text") or c.get("title") or "").lower()
                except Exception:
                    chash, ctext = "", ""
                if not chash and not ctext:
                    continue

                short = chash[:8] if chash else ""
                # Heuristics: mention of short-hash OR 2+ shared keywords
                hash_hit = bool(short and short in text)
                keyword_hit = False
                if not hash_hit and ctext:
                    # Tokenize on whitespace, keep words >=4 chars to reduce noise
                    words = [w for w in re.split(r"\W+", ctext) if len(w) >= 4]
                    if words:
                        shared = sum(1 for w in words if w in text)
                        keyword_hit = shared >= 2

                if hash_hit or keyword_hit:
                    # Write a non-evidence hint row for audit without affecting permanent closure
                    meta = {
                        "commit_ref": chash,
                        "source": "reflection",
                        "hint_type": "provisional_completion",
                    }
                    try:
                        prev = self.sqlite_store.latest_hash()
                        payload = {
                            "ts": datetime.now(timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                            "kind": "closure_hint",
                            "commit_ref": chash,
                            "prev_hash": prev,
                        }
                        # Minimal deterministic hash for chaining
                        data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
                        hsh = hashlib.sha256(data.encode()).hexdigest()

                        self.sqlite_store.append_event(
                            kind="closure_hint",
                            content=f"Provisional closure hint for {chash[:8]}",
                            meta=meta,
                            hsh=hsh,
                            prev=prev,
                        )
                    except Exception:
                        pass
                    hinted.append(chash)

            return hinted
        except Exception:
            return []

    def _sync_commitments_from_model(self):
        """Load commitments from model into tracker."""
        with self.lock:
            for cid, commitment_data in self.model.self_knowledge.commitments.items():
                try:
                    # Convert dict to Commitment object if needed
                    if isinstance(commitment_data, dict):
                        from pmm.commitments import Commitment

                        commitment_obj = Commitment(
                            cid=cid,
                            text=commitment_data.get("text", ""),
                            created_at=commitment_data.get("created_at", ""),
                            status=commitment_data.get("status", "open"),
                            source_insight_id=commitment_data.get(
                                "source_insight_id", ""
                            ),
                            due=commitment_data.get("due"),
                        )
                        self.commitment_tracker.commitments[cid] = commitment_obj
                    else:
                        self.commitment_tracker.commitments[cid] = commitment_data
                except (KeyError, TypeError):
                    continue

    def _sync_commitments_to_model(self):
        """Save commitments from tracker to model."""
        with self.lock:
            self.model.self_knowledge.commitments = dict(
                self.commitment_tracker.commitments
            )

    # -------- trait management --------

    def get_big5(self):
        """Return a flat dict of Big Five scores from the nested dataclasses."""
        with self.lock:
            return {
                "openness": self.model.personality.traits.big5.openness.score,
                "conscientiousness": self.model.personality.traits.big5.conscientiousness.score,
                "extraversion": self.model.personality.traits.big5.extraversion.score,
                "agreeableness": self.model.personality.traits.big5.agreeableness.score,
                "neuroticism": self.model.personality.traits.big5.neuroticism.score,
            }

    def set_big5(self, updates: dict, origin: str = "manual"):
        """Set Big Five scores with clamping to drift bounds; updates last_update and origin."""
        with self.lock:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            for trait_name, new_score in updates.items():
                trait_obj = getattr(
                    self.model.personality.traits.big5, trait_name, None
                )
                if trait_obj:
                    trait_obj.score = max(0.0, min(1.0, float(new_score)))
                    trait_obj.last_update = ts
                    trait_obj.origin = origin
            self._save_model_unlocked()

    def set_name(self, new_name: str, origin: str = "manual"):
        """Persistently set the agent's name and log an identity change event.

        FIXED: Relaxed validation and consistent identity logging.
        """
        import re
        from .name_detect import _STOPWORDS

        # Enhanced validation with stopwords check
        if not new_name:
            return
        name = new_name.strip().strip('.,!?;"')
        _NAME_RX = re.compile(r"[A-Za-z][A-Za-z .'-]{1,63}$")
        if not _NAME_RX.fullmatch(name) or name.lower() in _STOPWORDS:
            print(f"🔍 DEBUG: Rejected suspicious name: {name!r}")
            return

        with self.lock:
            old = self.model.core_identity.name
            if old == name:
                return
            self.model.core_identity.name = name

            # FIXED: Consistent identity change logging in both JSON and SQLite
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Add to JSON model (meta_cognition.identity_evolution)
            identity_change = IdentityChange(
                t=ts,
                change=f"Name changed from '{old}' to '{name}' (origin={origin})",
            )
            self.model.meta_cognition.identity_evolution.append(identity_change)

            # Also add as autobiographical event for test compatibility
            ev_id = f"ev{len(self.model.self_knowledge.autobiographical_events)+1}"
            identity_event = Event(
                id=ev_id,
                t=ts,
                type="identity_change",
                summary=f"Name changed from '{old}' to '{name}' (origin={origin})",
                effects=[],
                effects_hypothesis=[],
                meta={
                    "field": "name",
                    "old_value": old,
                    "new_value": name,
                    "origin": origin,
                },
            )
            self.model.self_knowledge.autobiographical_events.append(identity_event)

            # FIXED: Also log to SQLite for consistency
            prev_hash = self.sqlite_store.latest_hash()
            change_payload = {
                "ts": ts,
                "kind": "identity_change",
                "field": "name",
                "old_value": old,
                "new_value": new_name,
                "origin": origin,
                "prev_hash": prev_hash,
            }
            change_data = json.dumps(change_payload, sort_keys=True, ensure_ascii=False)
            current_hash = hashlib.sha256(change_data.encode()).hexdigest()

            self.sqlite_store.append_event(
                kind="identity_change",
                content=f"Name changed from '{old}' to '{new_name}' (origin={origin})",
                meta={
                    "field": "name",
                    "old_value": old,
                    "new_value": new_name,
                    "origin": origin,
                },
                hsh=current_hash,
                prev=prev_hash,
            )

            self._save_model_unlocked()

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

    def _to_effects(self, lst):
        """Convert list of effect dicts to EffectHypothesis objects."""
        out = []
        for item in lst:
            if isinstance(item, dict):
                out.append(
                    EffectHypothesis(
                        target=item.get("target", item.get("trait", "")),
                        delta=item.get("delta", item.get("magnitude", 0.0)),
                        confidence=item.get("confidence", 0.0),
                    )
                )
        return out
