import json
import numpy as np
import threading
from datetime import datetime, timezone, timedelta
from typing import List, Dict
from pmm.adapters.base import Message, ModelAdapter
from pmm.storage.sqlite_store import SQLiteStore
from pmm.storage.integrity import make_linked_hash, verify_chain
from pmm.self_model_manager import SelfModelManager
from pmm.core.autonomy import AutonomyLoop
from pmm.integrated_directive_system import IntegratedDirectiveSystem
from pmm.adaptive_triggers import AdaptiveTrigger, TriggerConfig, TriggerState
from pmm.emergence import compute_emergence_scores
from pmm.name_detect import (
    extract_agent_name_command,
    _too_soon_since_last_name_change,
    _utcnow_str,
)


class PMMRuntime:
    """Core PMM runtime integrating adapters and storage with commitment tracking."""

    def __init__(self, adapter: ModelAdapter, store: SQLiteStore, model_name: str):
        self.model_name = model_name
        self.adapter = adapter
        self.store = store
        # Initialize PMM personality system
        self.pmm_manager = SelfModelManager("persistent_self_model.json")
        # Initialize directive system consistent with LangChain wrapper
        self.directive_system = IntegratedDirectiveSystem(storage_manager=self.store)

        # --- ALWAYS-ON AUTONOMY (no flags, no config) ---
        try:
            self._autonomy_loop = AutonomyLoop(self.pmm_manager, interval_seconds=300)
            self._autonomy_stop = threading.Event()

            def _aut_run():
                self._autonomy_loop.run_forever(stop_event=self._autonomy_stop)

            self._autonomy_thread = threading.Thread(
                target=_aut_run, name="PMM-Autonomy", daemon=True
            )
            self._autonomy_thread.start()
            # Record one-time start marker
            try:
                self._append(
                    "event",
                    "Autonomy loop started (interval=300s)",
                    {"tag": "autonomy_start"},
                )
            except Exception:
                pass
        except Exception:
            # Never block runtime init if autonomy fails
            pass

        # --- Adaptive reflection trigger state ---
        try:
            self._trigger_state = TriggerState()
            # Start permissive cadence; will adapt via emergence scores
            self._trigger_cfg = TriggerConfig(
                cadence_days=None,  # time-gate open; we handle min cooldown below
                events_min_gap=4,
                ias_low=0.35,
                gas_low=0.35,
                ias_high=0.75,
                gas_high=0.75,
                min_cooldown_minutes=2,
                max_skip_days=7.0,
            )
            self._adaptive_trigger = AdaptiveTrigger(
                self._trigger_cfg, self._trigger_state
            )
        except Exception:
            self._adaptive_trigger = None

    def _append(self, kind: str, content: str, meta: Dict):
        """Append event to hash-chain with integrity."""
        prev = self.store.latest_hash()
        meta_json = json.dumps(meta, ensure_ascii=False)
        hsh = make_linked_hash(prev, kind, content, meta_json)
        return self.store.append_event(kind, content, meta, hsh, prev)

    def ask(
        self,
        user_text: str,
        history: List[Message] | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Ask question and get response through PMM mind-driven architecture."""
        history = history or []

        # Log user prompt to hash chain
        self._append("prompt", user_text, {"role": "user", "model": self.model_name})

        # Add thought to PMM personality system
        self.pmm_manager.add_thought(user_text, trigger="user_input")

        # Build PMM mind context for response generation
        pmm_context = self._build_pmm_context(user_text)

        # Generate response through PMM mind
        messages = self._build_pmm_messages(pmm_context, user_text, history)
        reply = self.adapter.generate(messages, max_tokens=max_tokens)

        # Log response to hash chain
        response_event = self._append(
            "response", reply, {"role": "assistant", "model": self.model_name}
        )

        # Update PMM mind state after response
        self._update_pmm_state(user_text, reply, response_event)

        # Allow PMM to self-adopt identity changes expressed by itself
        self._detect_self_identity_update(reply)

        # Autonomous commitment evaluation
        self._evaluate_commitments(reply)

        # Self-directed triggers: adaptive micro/macro reflection and directive evolution
        self._internal_autonomy_triggers(
            user_text=user_text, reply=reply, response_event=response_event
        )

        return reply

    def _build_pmm_context(self, user_text: str) -> Dict:
        """Build rich PMM mind context for response generation."""
        # Get current personality state
        big5_traits = self.pmm_manager.get_big5()
        open_commitments = self.pmm_manager.get_open_commitments()

        # Get recent autobiographical events from PMM
        recent_events = []
        if hasattr(self.pmm_manager.model, "autobiographical_events"):
            recent_events = self.pmm_manager.model.autobiographical_events[-5:]

        # Get recent insights
        recent_insights = []
        if hasattr(self.pmm_manager.model, "insights"):
            recent_insights = self.pmm_manager.model.insights[-3:]

        # Get behavioral patterns
        patterns = getattr(
            self.pmm_manager.model.self_knowledge, "behavioral_patterns", {}
        )

        return {
            "personality": big5_traits,
            "commitments": open_commitments,
            "recent_events": [
                {"summary": e.summary, "timestamp": e.timestamp} for e in recent_events
            ],
            "recent_insights": [{"content": i.content} for i in recent_insights],
            "behavioral_patterns": patterns,
            "identity": {
                "name": self.pmm_manager.model.core_identity.name,
                "id": self.pmm_manager.model.core_identity.id,
            },
        }

    def _build_pmm_messages(
        self, pmm_context: Dict, user_text: str, history: List[Message]
    ) -> List[Dict]:
        """Build messages with PMM mind context embedded."""
        # Create rich system prompt that embeds PMM personality
        system_prompt = self._create_pmm_system_prompt(pmm_context)

        # Build message history
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current user input
        messages.append({"role": "user", "content": user_text})

        return messages

    def _create_pmm_system_prompt(self, context: Dict) -> str:
        """Create system prompt that embeds PMM personality and memory."""
        personality = context["personality"]
        patterns = context["behavioral_patterns"]
        identity = context["identity"]

        prompt = f"""You are {identity['name']}.

You have access to persistent memory to maintain continuity across sessions.
Use that memory as context; otherwise, answer directly and plainly.

PERSONALITY TRAITS (Big Five):
- Openness: {personality['openness']:.3f}
- Conscientiousness: {personality['conscientiousness']:.3f}  
- Extraversion: {personality['extraversion']:.3f}
- Agreeableness: {personality['agreeableness']:.3f}
- Neuroticism: {personality['neuroticism']:.3f}"""

        # Inject rolling top-5 autonomy directives (self-evolved)
        directives = self._get_top_autonomy_directives(limit=5)
        if directives:
            prompt += "\n\nAUTONOMY DIRECTIVES (Top-5):\n"
            for d in directives:
                prompt += f"- {d}\n"

        prompt += """
BEHAVIORAL PATTERNS:
"""
        top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        for pattern, count in top_patterns:
            prompt += f"- {pattern}: {count}\n"

        prompt += """
RECENT INSIGHTS:
"""
        for insight in context["recent_insights"][-2:]:  # Last 2 insights
            prompt += f"- {insight['content'][:100]}...\n"

        # Note: No explicit coaching; autonomy emerges via internal triggers and stored directives

        return prompt

    def _update_pmm_state(
        self, user_text: str, reply: str, response_event: Dict
    ) -> None:
        """Update PMM mind state after generating response."""
        # Check for identity updates (name changes) using strict detector + cooldown
        try:
            cand = extract_agent_name_command(user_text, speaker="user")
            if cand:
                last_change = getattr(
                    self.pmm_manager.model.metrics, "last_name_change_at", None
                )
                if not _too_soon_since_last_name_change(last_change, days=1):
                    self.pmm_manager.set_name(cand, origin="user_command")
                    # record cooldown marker
                    self.pmm_manager.model.metrics.last_name_change_at = _utcnow_str()
                    self.pmm_manager.save_model()
                    # emit identity_update event
                    self.pmm_manager.add_event(
                        summary=f"Identity update: Name changed to '{cand}' (origin=user_command)",
                        effects=[],
                        etype="identity_update",
                    )
        except Exception:
            # Non-fatal
            pass

        # Update behavioral patterns based on user input and response
        self.pmm_manager.update_patterns(user_text)
        self.pmm_manager.update_patterns(reply)

        # Add autobiographical event for this interaction
        self.pmm_manager.add_event(
            summary=f"Conversation: {user_text[:50]}... -> {reply[:50]}...",
            effects=None,
            etype="conversation",
        )

        # Extract and track directives/commitments from assistant replies (always-on)
        try:
            if self.directive_system:
                self.directive_system.process_response(
                    user_message=user_text,
                    ai_response=reply,
                    event_id=response_event["event_id"],
                )
        except Exception:
            # Non-fatal, never block on directive processing
            pass

    def _detect_and_update_identity(self, user_text: str, reply: str) -> None:
        """Detect and update identity changes like name assignments."""
        user_lower = user_text.lower()

        # Detect name assignment patterns
        name_patterns = [
            "your name is now",
            "we will call you",
            "i will call you",
            "you are now called",
            "you should be called",
            "call you",
        ]

        for pattern in name_patterns:
            if pattern in user_lower:
                # Extract potential name from user input
                words = user_text.split()
                try:
                    pattern_idx = None
                    pattern_words = pattern.split()

                    # Find pattern in words
                    for i in range(len(words) - len(pattern_words) + 1):
                        if all(
                            words[i + j].lower() == pattern_words[j]
                            for j in range(len(pattern_words))
                        ):
                            pattern_idx = i + len(pattern_words)
                            break

                    if pattern_idx and pattern_idx < len(words):
                        # Extract name (next word after pattern, clean punctuation)
                        potential_name = words[pattern_idx].strip('.,!?;:"')

                        # Validate name (basic checks)
                        if len(potential_name) > 1 and potential_name.isalpha():
                            # Update PMM identity
                            self.pmm_manager.model.core_identity.name = potential_name

                            # Add autobiographical event for name change
                            self.pmm_manager.add_event(
                                summary=f"Identity update: Name changed to {potential_name}",
                                effects=None,
                                etype="identity_change",
                            )

                            # Save the updated model
                            self.pmm_manager.save_model()

                            print(
                                f"🔄 PMM identity updated: Name is now '{potential_name}'"
                            )
                            break

                except (IndexError, AttributeError):
                    continue

    def micro_reflect(self) -> str:
        """Generate micro-reflection with self-assessment and extract commitments."""
        prompt = "Reflect briefly (≤40 tokens) on the last exchange. If a commitment was made, assess its completion status and propose a new action if needed. Use 'Next, I will...' for new commitments."
        self._append(
            "prompt",
            prompt,
            {"role": "system", "model": self.model_name, "tag": "micro_reflection"},
        )
        reply = self.adapter.generate(
            [{"role": "user", "content": prompt}], max_tokens=80
        )
        self._append(
            "reflection",
            reply,
            {"role": "assistant", "model": self.model_name, "tag": "micro_reflection"},
        )

        # Extract new commitments using "Next, I will" pattern
        if "Next, I will" in reply:
            self._append("commitment", reply, {"extracted": True})

        # Autonomous self-assessment: check if reflection indicates completion
        all_events = self.store.all_events()
        recent_commits = [e for e in all_events if e[2] == "commitment"]

        if recent_commits and any(
            keyword in reply.lower()
            for keyword in ["completed", "done", "finished", "accomplished"]
        ):
            last_commit = recent_commits[-1]
            commit_hash = last_commit[6]  # hash field
            self._append(
                "evidence",
                f"Self-reflection indicates completion: {reply}",
                {"commit_ref": commit_hash, "evidence_type": "self_assessment"},
            )
        return reply

    def _embedding_completion_signal(
        self, commitment_content: str, recent_history: List[str], threshold: float = 0.8
    ) -> bool:
        """Heuristic completion signal via embeddings + keywords.

        Uses OpenAI embeddings to compare a commitment text with recent history; if
        similarity exceeds threshold or completion keywords are detected, returns True.
        Falls back to keyword-only detection on any error.
        """
        try:
            from openai import OpenAI

            client = OpenAI()

            # Get embeddings for commitment and recent history
            all_texts = [commitment_content] + list(recent_history or [])
            embeddings_response = client.embeddings.create(
                input=all_texts, model="text-embedding-3-small"
            )

            # Calculate cosine similarity between commitment and recent responses
            commit_embedding = np.array(embeddings_response.data[0].embedding)
            similarities = []
            for i in range(1, len(embeddings_response.data)):
                history_embedding = np.array(embeddings_response.data[i].embedding)
                similarity = np.dot(commit_embedding, history_embedding) / (
                    np.linalg.norm(commit_embedding) * np.linalg.norm(history_embedding)
                )
                similarities.append(similarity)

            # Check if any similarity exceeds threshold
            max_similarity = max(similarities, default=0)

            # Enhanced keyword detection in AI's own reflections
            combined_history = " ".join(recent_history or []).lower()
            completion_keywords = [
                "completed",
                "done",
                "finished",
                "achieved",
                "accomplished",
                "fulfilled",
            ]
            keyword_match = any(
                keyword in combined_history for keyword in completion_keywords
            )

            return max_similarity > threshold or keyword_match
        except Exception:
            # Fallback to keyword detection only if embeddings fail
            combined_history = " ".join(recent_history or []).lower()
            completion_keywords = [
                "completed",
                "done",
                "finished",
                "achieved",
                "accomplished",
                "fulfilled",
            ]
            return any(keyword in combined_history for keyword in completion_keywords)

    def _detect_self_identity_update(self, reply: str) -> None:
        """Conservatively adopt identity updates if the assistant clearly self-declares a new name.

        Enforces a 1-day cooldown shared with user-driven name changes.
        """
        try:
            text = (reply or "").strip()
            if not text:
                return

            # Use centralized structural detector (no regex)
            candidate = extract_agent_name_command(text, speaker="assistant")
            if not candidate:
                return

            # Cooldown check
            last_change = getattr(
                self.pmm_manager.model.metrics, "last_name_change_at", None
            )
            if _too_soon_since_last_name_change(last_change, days=1):
                return

            # Persist change
            self.pmm_manager.set_name(candidate, origin="assistant_self")
            self.pmm_manager.model.metrics.last_name_change_at = _utcnow_str()
            self.pmm_manager.save_model()
            self.pmm_manager.add_event(
                summary=f"Identity update: Name changed to '{candidate}' (origin=assistant_self)",
                effects=[],
                etype="identity_update",
            )
            self._append(
                "event",
                f"Self-identity adoption -> {candidate}",
                {"tag": "identity_self_adopt"},
            )
        except Exception:
            # Never block on identity adoption
            pass

    def _internal_autonomy_triggers(
        self, user_text: str, reply: str, response_event: Dict
    ) -> None:
        """Run adaptive, self-directed triggers after each exchange.

        Decides whether to micro-reflect recursively, macro-reflect, and evolve autonomy directives.
        """
        try:
            # Compute emergence scores if available
            ias = gas = 0.5
            try:
                scores = compute_emergence_scores(self.store, window_events=50)
                ias = float(scores.get("IAS", 0.5))
                gas = float(scores.get("GAS", 0.5))
            except Exception:
                pass

            events = (
                self.store.all_events() if hasattr(self.store, "all_events") else []
            )
            cnt_since_reflection = 0
            for row in reversed(events):
                if row[2] == "reflection":
                    break
                cnt_since_reflection += 1

            # Decide micro reflection
            if self._adaptive_trigger:
                decide, _reason = self._adaptive_trigger.decide(
                    now=datetime.utcnow(),
                    ias=ias,
                    gas=gas,
                    events_since_reflection=cnt_since_reflection,
                )
            else:
                decide, _reason = (cnt_since_reflection >= 4, "fallback_gap")

            if decide:
                self._recursive_micro_reflect(max_depth=3, overlap_threshold=0.2)

            # Heuristic: occasional macro reflections on high growth or low identity alignment
            if gas >= 0.75 or ias <= 0.35:
                try:
                    self.macro_reflect(response_event, session_days=3, max_events=200)
                except Exception:
                    pass

        except Exception:
            # Silent failure to avoid interfering with normal flow
            pass

    def _recursive_micro_reflect(
        self, max_depth: int = 3, overlap_threshold: float = 0.2
    ) -> None:
        """Run micro_reflect up to max_depth, stopping early if semantic overlap is high.

        Overlap is approximated via token set Jaccard index to avoid external dependencies.
        """
        try:
            produced: List[str] = []
            for _ in range(max_depth):
                r = self.micro_reflect()
                if not r:
                    break
                produced.append(r)
                # Compare against prior reflections to gate redundancy
                recent = [
                    e[3]
                    for e in (self.store.all_events() or [])
                    if e[2] == "reflection"
                ]
                if len(recent) >= 2:
                    a = set((recent[-1] or "").lower().split())
                    b = set((recent[-2] or "").lower().split())
                    inter = len(a & b)
                    union = max(1, len(a | b))
                    jacc = inter / union
                    if jacc >= overlap_threshold:
                        break
        except Exception:
            pass

    def macro_reflect(
        self, source_event: Dict, session_days: int = 3, max_events: int = 200
    ) -> str | None:
        """Synthesize high-level insights over recent events and persist autonomy directives."""
        try:
            events = self.store.all_events()
            if not events:
                return None

            # Select recent window by timestamp
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(days=session_days)
            recent: List[str] = []
            for row in events[-max_events:]:
                ts = row[1]
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                except Exception:
                    continue
                if dt >= cutoff and row[2] in (
                    "response",
                    "reflection",
                    "commitment",
                    "evidence",
                ):
                    recent.append(row[3])

            if not recent:
                return None

            prompt = (
                "Review recent responses, reflections, commitments, and evidence to derive 2-3 high-level insights. "
                "If appropriate, propose a single autonomy directive phrased as an imperative."
            )
            self._append(
                "prompt", prompt, {"role": "system", "tag": "macro_reflection"}
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are an autonomous PMM analyzing its recent behavior.",
                },
                {"role": "user", "content": "\n\n".join(recent)[-4000:]},
                {"role": "user", "content": prompt},
            ]
            reply = self.adapter.generate(messages, max_tokens=256)
            self._append(
                "reflection", reply, {"role": "assistant", "tag": "macro_reflection"}
            )

            # Extract autonomy directive line if present: starts with "Directive:" (case-insensitive)
            for line in (reply or "").splitlines():
                l = line.strip()
                if not l:
                    continue
                if l.lower().startswith("directive:"):
                    content = l[len("directive:") :].strip()
                    if content:
                        self._persist_autonomy_directive(content, source_event)
                        break
            return reply
        except Exception:
            return None

    def _persist_autonomy_directive(self, content: str, source_event: Dict) -> None:
        """Persist autonomy directive as a special event with simple dedup."""
        try:
            content = (content or "").strip()
            if not content:
                return
            existing = [
                e
                for e in (self.store.all_events() or [])
                if e[2] == "autonomy_directive"
            ]
            texts = {e[3].strip() for e in existing[-200:]}
            if content in texts:
                return

            if self.directive_system:
                self.directive_system.add_directive(
                    content=content,
                    directive_type="autonomy",
                    source_event_id=source_event["event_id"],
                )
        except Exception:
            pass

    def _get_top_autonomy_directives(self, limit: int = 5) -> List[str]:
        """Return top-N autonomy directives by frequency (recent-first tie-break)."""
        try:
            events = [
                e
                for e in (self.store.all_events() or [])
                if e[2] == "autonomy_directive"
            ]
            if not events:
                return []
            from collections import Counter

            counts = Counter([e[3].strip() for e in events])
            ranked = sorted(
                counts.items(),
                key=lambda x: (
                    -x[1],
                    -events[::-1].index(
                        next(e for e in events[::-1] if e[3].strip() == x[0])
                    ),
                ),
            )
            return [t for t, _ in ranked[:limit]]
        except Exception:
            return []

    def verify(self) -> bool:
        """Verify hash-chain integrity across all events."""
        return verify_chain(self.store.all_events())

    def get_recent_context(self, limit: int = 5) -> List[Dict]:
        """Get recent events for context building."""
        events = self.store.recent_events(limit)
        return [{"kind": row[2], "content": row[3], "ts": row[1]} for row in events]
