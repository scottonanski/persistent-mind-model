"""
LangChain Memory Wrapper for Persistent Mind Model

This module provides a LangChain-compatible memory interface that integrates
PMM's persistent personality system with any LangChain application.

Key Features:
- Drop-in replacement for LangChain memory systems
- Persistent personality traits (Big Five, HEXACO)
- Automatic commitment extraction and tracking
- Model-agnostic consciousness transfer
- Behavioral pattern evolution over time

Usage:
    from pmm.langchain_memory import PersistentMindMemory

    memory = PersistentMindMemory(
        agent_path="my_agent.json",
        personality_config={
            "openness": 0.7,
            "conscientiousness": 0.8
        }
    )

    # Use with any LangChain chain
    from langchain.chains import ConversationChain
    chain = ConversationChain(memory=memory, llm=your_llm)
"""

from typing import Any, Dict, List, Optional
import os
import threading
import atexit
from collections import deque

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone

    UTC = timezone.utc
from langchain.memory.chat_memory import BaseChatMemory
from pydantic import Field

from .self_model_manager import SelfModelManager
from .reflection import reflect_once
from .commitments import CommitmentTracker
from .commitments import get_identity_turn_commitments
from .integrated_directive_system import IntegratedDirectiveSystem
from .semantic_analysis import get_semantic_analyzer
from .introspection import IntrospectionEngine, IntrospectionConfig
from .phrase_deduper import PhraseDeduper
from .stance_filter import StanceFilter
from .model_baselines import ModelBaselineManager
from .atomic_reflection import AtomicReflectionManager
from pmm.policy import bandit as pmm_bandit
from .reflection_cooldown import ReflectionCooldownManager
from .commitment_ttl import CommitmentTTLManager
from .ngram_ban import NGramBanSystem
from .emergence_stages import EmergenceStageManager
from .core.autonomy import AutonomyLoop
from .logging_config import pmm_tlog, pmm_dlog


class PersistentMindMemory(BaseChatMemory):
    """
    LangChain memory wrapper that provides persistent AI personality.

    This memory system goes beyond simple conversation history to maintain
    a persistent personality with evolving traits, commitments, and behavioral
    patterns that influence all interactions.
    """

    pmm: SelfModelManager = Field(default=None, exclude=True)
    directive_system: IntegratedDirectiveSystem = Field(default=None, exclude=True)
    introspection: IntrospectionEngine = Field(default=None, exclude=True)
    personality_context: str = Field(default="")
    commitment_context: str = Field(default="")
    memory_key: str = Field(default="history")
    input_key: str = Field(default="input")
    output_key: str = Field(default="response")
    conversation_count: int = Field(default=0)
    enable_summary: bool = Field(default=False)
    enable_embeddings: bool = Field(default=False)
    turns_since_last_reflection: int = Field(default=0)
    active_model_config: Optional[Dict] = Field(default=None, exclude=True)
    trigger_config: Optional[Any] = Field(default=None, exclude=True)
    trigger_state: Optional[Any] = Field(default=None, exclude=True)
    adaptive_trigger: Optional[Any] = Field(default=None, exclude=True)
    phrase_deduper: Optional[Any] = Field(default=None, exclude=True)
    stance_filter: Optional[Any] = Field(default=None, exclude=True)
    model_baselines: Optional[Any] = Field(default=None, exclude=True)
    atomic_reflection: Optional[Any] = Field(default=None, exclude=True)
    reflection_cooldown: Optional[Any] = Field(default=None, exclude=True)
    commitment_ttl: Optional[Any] = Field(default=None, exclude=True)
    ngram_ban: Optional[Any] = Field(default=None, exclude=True)
    emergence_stages: Optional[Any] = Field(default=None, exclude=True)

    def __init__(
        self,
        agent_path: str,
        personality_config: Optional[Dict[str, float]] = None,
        *,
        enable_summary: bool = False,
        enable_embeddings: bool = False,
    ):
        """
        Initialize the LangChain-compatible memory wrapper.

        Args:
            agent_path: Path to save/load the agent's persistent state
            personality_config: Optional initial personality configuration
        """
        super().__init__()
        model_file_path = os.path.join(agent_path, "persistent_self_model.json")
        self.pmm = SelfModelManager(model_file_path)
        self.directive_system = IntegratedDirectiveSystem(
            storage_manager=self.pmm.sqlite_store
        )
        self.enable_summary = bool(enable_summary)
        self.enable_embeddings = bool(enable_embeddings)

        # Initialize introspection engine
        self.introspection = IntrospectionEngine(
            storage_manager=self.pmm.sqlite_store, config=IntrospectionConfig()
        )

        # Initialize personality if provided
        if personality_config:
            for trait, value in personality_config.items():
                if hasattr(self.pmm.model.personality.traits.big5, trait):
                    trait_obj = getattr(self.pmm.model.personality.traits.big5, trait)
                    trait_obj.score = max(0.0, min(1.0, float(value)))
            self.pmm.save_model()

        # Initialize reflection cooldown tracking
        self.turns_since_last_reflection = 0

        # Initialize adaptive trigger system
        from pmm.adaptive_triggers import AdaptiveTrigger, TriggerConfig, TriggerState

        self.trigger_config = TriggerConfig(
            cadence_days=None,  # Disable time-based for now
            events_min_gap=2,  # Relax gap to increase cadence
            ias_low=0.35,
            gas_low=0.35,
            ias_high=0.65,
            gas_high=0.65,
            min_cooldown_minutes=6,  # Shorter cooldown for tighter cadence
            max_skip_days=7.0,
        )
        self.trigger_state = TriggerState()
        self.adaptive_trigger = AdaptiveTrigger(self.trigger_config, self.trigger_state)

        # Initialize components
        self.phrase_deduper = PhraseDeduper()
        self.stance_filter = StanceFilter()
        self.model_baselines = ModelBaselineManager()
        self.atomic_reflection = AtomicReflectionManager(self.pmm)
        # Configure reflection cooldown gates
        # Loosened by default to encourage chaining during active sessions; override via env
        try:
            cooldown_seconds = int(os.getenv("PMM_REFLECTION_COOLDOWN_SECONDS", "45"))
        except Exception:
            cooldown_seconds = 45
        self.reflection_cooldown = ReflectionCooldownManager(
            min_turns=2,
            min_wall_time_seconds=cooldown_seconds,
            novelty_threshold=0.85,
        )
        self.commitment_ttl = CommitmentTTLManager()
        self.ngram_ban = NGramBanSystem()
        self.emergence_stages = EmergenceStageManager(self.model_baselines)

        # ---- Bandit horizon tracker (Step 4) ----
        try:
            horizon = int(os.getenv("PMM_BANDIT_HORIZON_TURNS", "5"))
        except Exception:
            horizon = 5
        self._bandit_horizon_turns: int = max(1, min(50, horizon))
        self._bandit_events: deque = deque(
            maxlen=20
        )  # records of recent bandit actions
        self._bandit_turn_id: int = 0
        self._bandit_last_close_event_id: int = 0

        # ---- Bandit DB ensure (Step 8) ----
        try:
            flag = os.getenv("PMM_BANDIT_ENABLED")
            enabled = (
                True if flag is None else flag.lower() in ("1", "true", "yes", "on")
            )
            if enabled and hasattr(self.pmm, "sqlite_store"):
                # Ensure bandit tables exist at startup
                pmm_bandit.set_store(self.pmm.sqlite_store)
        except Exception:
            pass

        # LangChain memory interface requirements - ConversationChain uses "response" as output key
        self.memory_key = "history"
        self.input_key = "input"
        self.output_key = "response"

        # Track conversation context for commitments
        self.conversation_count = 0
        self.commitment_context = ""

        # Update context strings
        self._update_personality_context()
        self._update_commitment_context()

        # === Autonomy autostart (opt-in via env PMM_AUTONOMY_AUTOSTART) ===
        self._autonomy_thread = None
        self._autonomy_stop = None
        self._autonomy_loop = None
        try:
            # Default ON for autonomy; allow explicit opt-out via env
            enable = str(os.environ.get("PMM_AUTONOMY_AUTOSTART", "1")).lower() in (
                "1",
                "true",
                "yes",
            )
            if enable:
                interval = 300  # fixed cadence (seconds)
                self._autonomy_loop = AutonomyLoop(
                    self.pmm,
                    interval_seconds=interval,
                    directive_system=self.directive_system,
                )
                self._autonomy_stop = threading.Event()

                def _run():
                    # Each loop uses the PMM’s own SQLite store; work is self-contained
                    self._autonomy_loop.run_forever(stop_event=self._autonomy_stop)

                self._autonomy_thread = threading.Thread(
                    target=_run, name="PMM-Autonomy", daemon=True
                )
                self._autonomy_thread.start()
                # optional: log an internal event for provenance
                try:
                    self.pmm.add_event(
                        summary=f"Autonomy loop started (interval={interval}s)",
                        etype="autonomy_start",
                        effects=[],
                    )
                except Exception:
                    pass
        except Exception as _e:
            # never crash init
            pmm_dlog(f"[PMM] Autonomy autostart failed: {_e}")

        # Ensure clean shutdown on process exit
        atexit.register(self._shutdown_autonomy)

        # === Background Embedding Backlog (autonomous) ===
        self._embedder_thread = None
        self._embedder_stop = None
        try:
            if self.enable_embeddings:
                self._embedder_stop = threading.Event()

                def _embed_backlog():
                    import time
                    import numpy as np
                    from pmm.semantic_analysis import get_semantic_analyzer

                    analyzer = get_semantic_analyzer()
                    store = getattr(self.pmm, "sqlite_store", None)
                    if not store:
                        return
                    idle_sleep = 20.0
                    active_sleep = 0.2
                    batch = 20
                    while not self._embedder_stop.is_set():
                        try:
                            # Pull a small batch of unembedded events
                            items = store.get_unembedded_events(limit=batch)
                            if not items:
                                time.sleep(idle_sleep)
                                continue
                            for ev in items:
                                if self._embedder_stop.is_set():
                                    break
                                content = (
                                    ev.get("summary") or ev.get("content") or ""
                                ).strip()
                                if not content:
                                    # Mark empty content with zero vector to avoid re-processing
                                    store.set_event_embedding(ev["id"], b"")
                                    time.sleep(active_sleep)
                                    continue
                                try:
                                    vec = analyzer._get_embedding(content)
                                    emb = np.array(vec, dtype=np.float32).tobytes()
                                except Exception:
                                    emb = None
                                try:
                                    store.set_event_embedding(ev["id"], emb)
                                except Exception:
                                    pass
                                time.sleep(active_sleep)
                        except Exception:
                            time.sleep(idle_sleep)

                self._embedder_thread = threading.Thread(
                    target=_embed_backlog, name="PMM-Embedder", daemon=True
                )
                self._embedder_thread.start()
        except Exception:
            pass

        # Ensure background embedder stops on exit
        atexit.register(self._shutdown_embedder)

        # === Lightweight Scene Compactor (autonomous) ===
        self._compactor_thread = None
        self._compactor_stop = None
        try:
            self._compactor_stop = threading.Event()

            def _compact_scenes():
                import time
                from datetime import datetime, timezone
                from pmm.model import Scene

                # Parameters
                interval = 120.0  # seconds
                window_events = 20
                min_gap = 30  # require at least this many new events since last compact

                while not self._compactor_stop.is_set():
                    try:
                        events = self.pmm.model.self_knowledge.autobiographical_events
                        scenes = self.pmm.model.narrative_identity.scenes

                        # Bootstrap: if we have few/no scenes, synthesize up to 3 from history
                        try:
                            if len(scenes) < 3 and events and len(events) >= 12:
                                total = len(events)
                                # Divide recent history into up to 3 chunks (newest-first)
                                chunk = max(4, total // 3)
                                created = 0
                                for idx in range(3):
                                    end = total - (idx * chunk)
                                    start = max(0, end - chunk)
                                    seg = events[start:end]
                                    if not seg:
                                        break
                                    parts = []
                                    tags = set()
                                    for ev in seg:
                                        s = getattr(ev, "summary", "") or ""
                                        if s:
                                            parts.append(s)
                                        for tg in getattr(ev, "tags", []) or []:
                                            tags.add(str(tg))
                                    summary = "; ".join(parts)[:480]
                                    now_iso = datetime.now(timezone.utc).strftime(
                                        "%Y-%m-%dT%H:%M:%SZ"
                                    )
                                    scene = Scene(
                                        id=f"boot{len(scenes)+1}",
                                        t=now_iso,
                                        type="scene",
                                        summary=summary,
                                        tags=list(tags)[:10],
                                    )
                                    self.pmm.model.narrative_identity.scenes.append(
                                        scene
                                    )
                                    created += 1
                                    if (
                                        len(self.pmm.model.narrative_identity.scenes)
                                        >= 3
                                    ):
                                        break
                                if created:
                                    self.pmm.save_model()
                        except Exception:
                            pass

                        if not events or len(events) < window_events:
                            time.sleep(interval)
                            continue

                        # Determine if enough new events since last scene
                        last_scene_time = None
                        if scenes:
                            try:
                                last_scene_time = scenes[-1].t
                            except Exception:
                                last_scene_time = None

                        # Count new events since last scene timestamp
                        if last_scene_time:
                            new_count = 0
                            for ev in reversed(events):
                                try:
                                    if ev.t <= last_scene_time:
                                        break
                                except Exception:
                                    break
                                new_count += 1
                            if new_count < min_gap:
                                time.sleep(interval)
                                continue

                        # Build window summary from last N events
                        slice_events = events[-window_events:]
                        parts = []
                        tags = set()
                        for ev in slice_events:
                            s = getattr(ev, "summary", "") or ""
                            if s:
                                parts.append(s)
                            for tg in getattr(ev, "tags", []) or []:
                                tags.add(str(tg))
                        summary = "; ".join(parts)[:480]
                        now_iso = datetime.now(timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        )

                        # Create and append scene
                        scene = Scene(
                            id=f"scn{len(scenes)+1}",
                            t=now_iso,
                            type="scene",
                            summary=summary,
                            tags=list(tags)[:10],
                        )
                        self.pmm.model.narrative_identity.scenes.append(scene)
                        self.pmm.save_model()
                    except Exception:
                        pass
                    finally:
                        time.sleep(interval)

            self._compactor_thread = threading.Thread(
                target=_compact_scenes, name="PMM-SceneCompactor", daemon=True
            )
            self._compactor_thread.start()
        except Exception:
            pass
        atexit.register(self._shutdown_compactor)

    def _shutdown_autonomy(self) -> None:
        """Stop autonomy thread if it was started (best-effort)."""
        try:
            if self._autonomy_stop is not None:
                self._autonomy_stop.set()
            if self._autonomy_thread is not None and self._autonomy_thread.is_alive():
                # Don't block indefinitely; thread is daemon=True
                self._autonomy_thread.join(timeout=0.5)
        except Exception:
            # best-effort cleanup
            pass

    def _shutdown_embedder(self) -> None:
        """Stop background embedder thread (best-effort)."""
        try:
            if self._embedder_stop is not None:
                self._embedder_stop.set()
            if self._embedder_thread is not None and self._embedder_thread.is_alive():
                self._embedder_thread.join(timeout=0.5)
        except Exception:
            pass

    def _shutdown_compactor(self) -> None:
        """Stop scene compactor thread (best-effort)."""
        try:
            if self._compactor_stop is not None:
                self._compactor_stop.set()
            if self._compactor_thread is not None and self._compactor_thread.is_alive():
                self._compactor_thread.join(timeout=0.5)
        except Exception:
            pass

    def _apply_personality_config(self, config: Dict[str, float]) -> None:
        """Apply initial personality configuration to PMM agent."""
        big5_traits = [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]

        for trait, score in config.items():
            if trait in big5_traits:
                trait_obj = getattr(self.pmm.model.personality.traits.big5, trait)
                trait_obj.score = max(0.0, min(1.0, score))

        self.pmm.save_model()

    def _update_personality_context(self) -> None:
        """Generate personality context for LLM prompts."""
        traits = self.pmm.model.personality.traits.big5
        patterns = self.pmm.model.self_knowledge.behavioral_patterns

        context_parts = [
            "Personality Profile (Big Five):",
            f"• Openness: {traits.openness.score:.2f} - {'Creative, curious' if traits.openness.score > 0.6 else 'Practical, conventional'}",
            f"• Conscientiousness: {traits.conscientiousness.score:.2f} - {'Organized, disciplined' if traits.conscientiousness.score > 0.6 else 'Flexible, spontaneous'}",
            f"• Extraversion: {traits.extraversion.score:.2f} - {'Outgoing, energetic' if traits.extraversion.score > 0.6 else 'Reserved, quiet'}",
            f"• Agreeableness: {traits.agreeableness.score:.2f} - {'Cooperative, trusting' if traits.agreeableness.score > 0.6 else 'Competitive, skeptical'}",
            f"• Neuroticism: {traits.neuroticism.score:.2f} - {'Anxious, sensitive' if traits.neuroticism.score > 0.6 else 'Calm, resilient'}",
        ]

        if patterns:
            context_parts.append(
                f"Behavioral Patterns: {', '.join(f'{k}({v})' for k, v in patterns.items())}"
            )

        # Add recent memories and insights with model-namespaced filtering
        from .memory_keys import agent_namespace
        from .llm_factory import get_llm_factory

        # Get current model config for namespacing
        llm_factory = get_llm_factory()
        active_config = llm_factory.get_active_config()
        install_id = getattr(self.pmm, "install_id", "default")

        if active_config:
            agent_namespace(active_config, install_id)
            current_model = active_config.get("name", "unknown")
        else:
            current_model = "unknown"

        # Filter events by model source to prevent cross-model memory bleeding
        recent_events = []
        for event in self.pmm.model.self_knowledge.autobiographical_events[-10:]:
            # Only include events from current model or model-agnostic events
            event_source = getattr(event, "model_source", None)
            if event_source is None or event_source == current_model:
                recent_events.append(event)
                if len(recent_events) >= 3:
                    break

        if recent_events:
            context_parts.append(f"\nRecent Memories ({current_model}):")
            for event in recent_events:
                context_parts.append(f"• {event.summary}")

        # Filter insights by model source
        recent_insights = []
        for insight in self.pmm.model.self_knowledge.insights[-10:]:
            insight_source = getattr(insight, "model_source", None)
            if insight_source is None or insight_source == current_model:
                recent_insights.append(insight)
                if len(recent_insights) >= 2:
                    break

        if recent_insights:
            context_parts.append(f"\nRecent Insights ({current_model}):")
            for insight in recent_insights:
                context_parts.append(
                    f"• {insight.content[:100]}{'...' if len(insight.content) > 100 else ''}"
                )

        self.personality_context = "\n".join(context_parts)

    def _mentions_identity_signal(self, text: str) -> bool:
        """Heuristic, regex‑free identity signal detection.

        Triggers on clear self‑ascriptions like:
        - "my name is ..."
        - "identity confirm" / "identity:"
        - "name is ..."
        - "I am <CapitalizedWord>" (single token name)
        """
        try:
            if not text:
                return False
            raw = text.strip()
            low = raw.lower()
            # Direct keyword cues
            if any(
                k in low
                for k in ("my name is", "identity confirm", "identity:", "name is")
            ):
                return True

            # Simple pattern: "I am <CapitalizedWord>"
            tokens = [t for t in raw.split() if t]
            for i in range(len(tokens) - 2):
                if tokens[i].lower() == "i" and tokens[i + 1].lower() == "am":
                    cand = tokens[i + 2].strip('.,!?;:"')
                    if cand and cand.isalpha() and cand[0].isupper():
                        return True
            return False
        except Exception:
            return False

    def _auto_extract_key_info(self, user_input: str) -> None:
        """
        Automatically extract and remember key information from user input.

        This method detects:
        - Names ("My name is X", "I am X", "Call me X")
        - Preferences ("I like X", "I prefer X")
        - Important facts about the user
        """
        try:
            raw = (user_input or "").strip()
            user_lower = raw.lower()

            # Extract names (more conservative to avoid false positives like "I'm just...")
            stopwords = {
                "just",
                "good",
                "fine",
                "okay",
                "ok",
                "testing",
                "running",
                "logging",
                "ready",
                "back",
                "here",
                "there",
                "busy",
                "tired",
                "great",
                "awesome",
            }

            def _remember_user_name(name: str) -> None:
                if not name:
                    return
                # Title-case single/multi-token name
                clean = name.strip().strip('.,!?;:"')
                if not clean:
                    return
                # Limit to 1-3 tokens, alphabetic, capitalized tokens
                parts = [p for p in clean.split() if p]
                if not (1 <= len(parts) <= 3):
                    return
                for p in parts:
                    if not p.isalpha() or not p[0].isupper():
                        return
                remembered = " ".join(parts)
                self.pmm.add_event(
                    summary=f"IMPORTANT: User's name is {remembered}",
                    effects=[],
                    etype="identity_info",
                )
                pmm_dlog(f" Automatically remembered: User's name is {remembered}")

            # Pattern order: strongest first, using original casing for capitalization heuristics
            low = raw.lower()

            # 1) "My name is X" (case-insensitive phrase, preserve original for name extraction)
            phrase = "my name is "
            idx = low.find(phrase)
            if idx != -1:
                tail = raw[idx + len(phrase) :]
                # Stop at delimiter
                for delim in [".", "!", "?", ",", ";", ":", "\n"]:
                    cut = tail.find(delim)
                    if cut != -1:
                        tail = tail[:cut]
                        break
                _remember_user_name(tail.strip())
            else:
                # 2) "Call me X"
                phrase2 = "call me "
                idx2 = low.find(phrase2)
                if idx2 != -1:
                    tail = raw[idx2 + len(phrase2) :]
                    for delim in [".", "!", "?", ",", ";", ":", "\n"]:
                        cut = tail.find(delim)
                        if cut != -1:
                            tail = tail[:cut]
                            break
                    _remember_user_name(tail.strip())
                else:
                    # 3) "I am X" or "I'm X" only if next token is Capitalized and not a stopword/common verb
                    tokens = [t for t in raw.strip().split() if t.strip()]
                    for i, tok in enumerate(tokens[:-1]):
                        tl = tok.lower()
                        if tl in ("i", "i'm", "i’m") or (
                            tl == "i"
                            and i + 1 < len(tokens)
                            and tokens[i + 1].lower() == "am"
                        ):
                            # Determine candidate next token
                            j = i + 1
                            if tl == "i":
                                # if pattern was "I am"
                                if j < len(tokens) and tokens[j].lower() == "am":
                                    j += 1
                            if j < len(tokens):
                                candidate = tokens[j].strip('.,!?;:"')
                                if (
                                    candidate
                                    and candidate[0].isupper()
                                    and candidate.isalpha()
                                ):
                                    if (
                                        candidate.lower() not in stopwords
                                        and candidate.lower()
                                        not in {"doing", "going", "working"}
                                    ):
                                        _remember_user_name(candidate)
                                        break

            # Detect agent name assignments and persist them
            # Only accept explicit *agent* rename directives.
            # DO NOT infer from casual phrasing like "you're ...".
            # IMPORTANT POLICY: Do not accept user-driven agent renames here.
            # User inputs should never rename the agent. Only assistant self‑declarations may.
            # (Name changes, if any, are handled later from assistant output with cooldown.)

            # Extract preferences and other key info
            # Preference extraction without regex; capture short clauses after cues
            def capture_after(
                prefix: str, text_raw: str, text_low: str
            ) -> Optional[str]:
                pos = text_low.find(prefix)
                if pos == -1:
                    return None
                tail_raw = text_raw[pos + len(prefix) :]
                # Stop at delimiters or conjunctions
                stops = [" and ", " but ", ",", ".", "!", "?", ";", ":", "\n"]
                cut_idx = len(tail_raw)
                low_tail = tail_raw.lower()
                for s in stops:
                    idxs = low_tail.find(s)
                    if idxs != -1:
                        cut_idx = min(cut_idx, idxs)
                chunk = tail_raw[:cut_idx].strip(' \t\r\n.,;:!?"')
                return chunk if chunk else None

            prefs = [
                capture_after("i like ", raw, user_lower),
                capture_after("i prefer ", raw, user_lower),
                capture_after("i work on ", raw, user_lower)
                or capture_after("i work at ", raw, user_lower)
                or capture_after("i work with ", raw, user_lower),
                capture_after("i am ", raw, user_lower),
            ]
            for pref in prefs:
                if pref and 2 < len(pref) < 50:
                    self.pmm.add_event(
                        summary=f"PREFERENCE: User {pref}",
                        effects=[],
                        etype="preference_info",
                    )
                    break

        except Exception:
            # Silently handle errors in auto-extraction
            pass

    def _update_commitment_context(self) -> None:
        """Generate commitment context for LLM prompts."""
        try:
            open_commitments = self.pmm.get_open_commitments()
            if open_commitments:
                commitment_list = [
                    f"• {c['text']}" for c in open_commitments[:3]
                ]  # Show top 3
                self.commitment_context = "Active Commitments:\n" + "\n".join(
                    commitment_list
                )
            else:
                self.commitment_context = ""
        except Exception:
            self.commitment_context = ""

    def _semantic_intent_is_non_behavioral(self, text: str) -> Optional[bool]:
        """Lightweight semantic intent classifier.

        Returns True if input is semantically non-behavioral (logs/debug/pastes),
        False if behavioral (questions/requests/dialogue), or None if uncertain.
        """
        try:
            raw = (text or "").strip()
            if not raw:
                return None

            # Quick structure signals
            # High symbol density and long length suggests paste/log
            symbol_ratio = sum(ch in "{}[]<>:=#;/\\|`~$%^&*()" for ch in raw) / max(
                1, len(raw)
            )
            if len(raw) > 1200 and symbol_ratio > 0.03:
                return True

            analyzer = get_semantic_analyzer()

            non_behavioral_exemplars = [
                "DEBUG: initialized service; pid=4321; ts=2025-08-29T12:00:00Z",
                "Traceback (most recent call last):\n  File 'x.py', line 10, in <module>\n  raise ValueError('oops')",
                '{"event":"metrics","values":[1,2,3],"ok":true}',
                "[LOG] GET /health 200 in 12ms",
                "ERROR: connection refused at 10.0.0.2:5432",
                "INFO 2025-08-30T09:00:00Z Job completed successfully",
                "npm ERR! code ERESOLVE",
                "[API] POST /v1/items payload size=3",
                "<html><body>500 Internal Server Error</body></html>",
                "2025-08-30 10:12:01,233 | module | level=INFO | message=started",
            ]

            behavioral_exemplars = [
                "Can you help me summarize this?",
                "What should we try next to improve performance?",
                "Let's plan the next step and assign an action.",
                "I have a question about the results.",
                "Please analyze this output and suggest changes.",
                "How do I fix this bug?",
            ]

            def max_sim(text: str, refs: List[str]) -> float:
                best = 0.0
                for r in refs:
                    best = max(best, analyzer.cosine_similarity(text, r))
                return best

            nb_score = max_sim(raw, non_behavioral_exemplars)
            b_score = max_sim(raw, behavioral_exemplars)

            # Margin decision: require clear separation
            margin = 0.12
            if nb_score - b_score > margin:
                return True
            if b_score - nb_score > margin:
                return False

            # Additional signal: many newlines typical of pastes/logs
            if raw.count("\n") > 12 and symbol_ratio > 0.01:
                return True

            return None
        except Exception:
            # Fail open to heuristic path
            return None

    def _is_non_behavioral_input(self, text: str) -> bool:
        """
        Determine if input should be treated as non-behavioral (debug/log/paste).

        Non-behavioral inputs are stored for provenance but don't trigger
        reflections, commitment extraction, or behavioral patterns.
        """
        if not text or not text.strip():
            return False

        # 1) Semantic classifier first
        decision = self._semantic_intent_is_non_behavioral(text)
        if decision is True:
            return True
        if decision is False:
            return False

        # 2) Fallback heuristics
        lines = text.strip().split("\n")

        # Single line checks
        if len(lines) == 1:
            line = lines[0].strip()
            # Debug/log prefixes
            if line.startswith(
                ("DEBUG:", "🔍 DEBUG:", "[API]", "[LOG]", "ERROR:", "WARNING:")
            ):
                return True
            # JSON-like structures
            if (line.startswith(("{", "[")) and line.endswith(("}", "]"))) and len(
                line
            ) > 20:
                return True
            return False

        # Multi-line paste detection
        if len(lines) > 10:  # Threshold for "paste cascade"
            return True

        # Check if majority of lines are debug/log
        debug_lines = 0
        for line in lines:
            line = line.strip()
            if line.startswith(
                (
                    "DEBUG:",
                    "🔍 DEBUG:",
                    "[API]",
                    "[LOG]",
                    "ERROR:",
                    "WARNING:",
                    "  at ",
                    "Traceback",
                )
            ):
                debug_lines += 1

        # If >50% are debug lines, treat as non-behavioral
        return debug_lines > len(lines) * 0.5

    def is_non_behavioral_input(self, text: str) -> bool:
        """
        Public wrapper for `_is_non_behavioral_input`.

        Exposes input hygiene classification as a stable API for tests and clients
        without relying on private methods.
        """
        return self._is_non_behavioral_input(text)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save conversation context to PMM system.

        This method:
        1) Stores the conversation as PMM events + internal thoughts
        2) Extracts and tracks commitments from responses
        3) Updates behavioral patterns
        4) Auto‑closes commitments from new evidence
        5) Triggers reflection on cadence and on new commitments
        6) Applies personality drift immediately after reflection
        """
        # ---- 0) Normalize IO ----
        # Advance bandit turn counter for this user interaction
        try:
            self._bandit_turn_id += 1
        except Exception:
            self._bandit_turn_id = 1
        human_input = (
            inputs.get(self.input_key, "")
            or inputs.get("input", "")
            or inputs.get("question", "")
        )

        ai_output = ""
        if outputs:
            ai_output = (
                outputs.get(self.output_key, "")
                or outputs.get("response", "")
                or outputs.get("text", "")
                or outputs.get("answer", "")
            )
            if not ai_output and outputs:
                ai_output = list(outputs.values())[0] if outputs.values() else ""

        # ---- 0.25) Input Hygiene: Check if input is non-behavioral ----
        is_non_behavioral = self._is_non_behavioral_input(human_input)
        if is_non_behavioral:
            pmm_dlog(
                "🔍 DEBUG: Non-behavioral input detected, skipping behavioral triggers"
            )

        # ---- 0.3) Apply stance filter and phrase deduplication ----
        if ai_output and not is_non_behavioral:
            # Get current model name for phrase deduplication
            current_model = getattr(self, "_active_model_config", {}).get(
                "name", "unknown"
            )

            # Apply stance filter to remove anthropomorphic language
            # Determine emergence stage label for stage-aware filtering
            stage_label = None
            try:
                from pmm.emergence import compute_emergence_scores

                scores = compute_emergence_scores(
                    window=5, storage_manager=getattr(self.pmm, "sqlite_store", None)
                )
                ias = float(scores.get("IAS", 0.0) or 0.0)
                gas = float(scores.get("GAS", 0.0) or 0.0)
                model_name = current_model
                profile = self.emergence_stages.calculate_emergence_profile(
                    model_name, ias, gas
                )
                stage_label = getattr(profile.stage, "value", None) or str(
                    profile.stage
                )
            except Exception:
                # Fall back to internal resolution in filter if stage cannot be computed
                stage_label = None

            filtered_output, stance_filters = self.stance_filter.filter_response(
                ai_output, stage=stage_label
            )
            if stance_filters:
                pmm_dlog(
                    f"🔍 DEBUG: Applied stance filters: {len(stance_filters)} changes (stage={stage_label or 'auto'})"
                )

            # Check for phrase repetition
            is_repetitive, repeated_phrases, repetition_score = (
                self.phrase_deduper.check_response(current_model, filtered_output)
            )

            if is_repetitive:
                pmm_dlog(
                    f"🔍 DEBUG: Repetitive phrases detected: {repeated_phrases[:3]}... (score: {repetition_score:.3f})"
                )
                # Could implement re-generation here, for now just log

            # Add response to phrase cache for future deduplication
            self.phrase_deduper.add_response(current_model, filtered_output)

            # Use filtered output
            ai_output = filtered_output

        # ---- 0.5) Input Hygiene: Check if input is non-behavioral ----
        is_non_behavioral = self._is_non_behavioral_input(human_input)
        if is_non_behavioral:
            pmm_dlog(
                "🔍 DEBUG: Non-behavioral input detected, skipping behavioral triggers"
            )

        # Store non-behavioral inputs for provenance but with special marking
        behavioral_input = human_input if not is_non_behavioral else ""

        # ---- 0.75) Handle Introspection Commands ----
        introspection_result = None
        if human_input:
            command_type = self.introspection.parse_user_command(human_input)
            if command_type:
                pmm_dlog(
                    f"🔍 DEBUG: Processing introspection command: {command_type.value}"
                )

                if human_input.lower().strip() == "@introspect help":
                    # Special case: show available commands
                    commands = self.introspection.get_available_commands()
                    help_text = "🔍 **Available Introspection Commands:**\n\n"
                    for cmd, desc in commands.items():
                        help_text += f"• `{cmd}` - {desc}\n"
                    help_text += "\n💡 **Automatic Introspection:**\n"
                    help_text += (
                        "• PMM also performs automatic introspection when it detects:\n"
                    )
                    help_text += "  - Failed commitments\n"
                    help_text += "  - Significant trait drift\n"
                    help_text += "  - Reflection quality issues\n"
                    help_text += "  - Emergence score plateaus\n"
                    help_text += (
                        "\n🔔 You'll be notified when automatic analysis occurs.\n"
                    )

                    # Return help immediately without further processing
                    return help_text
                else:
                    # Process the introspection command
                    introspection_result = self.introspection.user_introspect(
                        command_type
                    )
                    formatted_result = self.introspection.format_result_for_user(
                        introspection_result
                    )

                    # Log the introspection as a special event
                    self.pmm.add_event(
                        summary=f"User requested {command_type.value} introspection",
                        etype="introspection_command",
                        tags=["introspection", "user_command", command_type.value],
                        effects=[],
                        evidence="",
                        full_text=human_input,
                    )

                    # Return the introspection result immediately
                    return formatted_result

        # ---- 1) Log human event + auto‑extract key info ----
        if human_input:
            try:
                # Always log for provenance, but mark non-behavioral inputs
                event_type = "non_behavioral" if is_non_behavioral else "conversation"

                # Generate embedding for semantic search if enabled
                embedding = None
                if self.enable_embeddings and human_input.strip():
                    try:
                        semantic_analyzer = get_semantic_analyzer()
                        import numpy as np

                        embedding_list = semantic_analyzer._get_embedding(human_input)
                        embedding = np.array(embedding_list, dtype=np.float32).tobytes()
                    except Exception as e:
                        pmm_dlog(
                            f"[warn] Failed to generate embedding for user input: {e}"
                        )

                self.pmm.add_event(
                    summary=f"User said: {human_input[:200]}{'...' if len(human_input) > 200 else ''}",
                    effects=[],
                    etype=event_type,
                    embedding=embedding,
                )

                # Only extract key info and auto-close from behavioral inputs
                if behavioral_input:
                    self._auto_extract_key_info(behavioral_input)

                    # Name handling: ignore any user-driven rename attempts (policy).
                    if self.pmm:
                        pmm_dlog(
                            "🔍 DEBUG: Ignoring user-driven agent rename attempts by policy"
                        )

                    # Evidence detection for user text is disabled to avoid
                    # duplicate or premature closures. Evidence is inferred
                    # from assistant behavior/output instead.
            except Exception:
                pass  # never crash chat on memory write

        # ---- 2) Log assistant thought + event ----
        if ai_output:
            try:
                self.pmm.add_thought(ai_output, trigger="langchain_conversation")
                # Tag identity replies when identity mode is active or explicit self-ascription is present
                tags: List[str] = []
                try:
                    active_idents = get_identity_turn_commitments(self.pmm) or []
                    if any(
                        (it or {}).get("remaining_turns", 0) > 0 for it in active_idents
                    ):
                        tags.append("identity")
                except Exception:
                    pass
                try:
                    # Heuristic identity-signal detector (regex-free)
                    if (
                        self._mentions_identity_signal(ai_output or "")
                        and "identity" not in tags
                    ):
                        tags.append("identity")
                except Exception:
                    pass
                response_event_id = self.pmm.add_event(
                    summary=f"I responded: {ai_output[:200]}{'...' if len(ai_output) > 200 else ''}",
                    effects=[],
                    etype="response",
                    tags=tags,
                )

                # Check for assistant self-declarations
                if self.pmm:
                    from .name_detect import (
                        extract_agent_name_command,
                        _too_soon_since_last_name_change,
                        _utcnow_str,
                    )

                    cand = extract_agent_name_command(ai_output, "assistant")
                    if cand:
                        # Apply cooldown and persist assistant self‑declaration
                        last_change = getattr(
                            self.pmm.model.metrics, "last_name_change_at", None
                        )
                        if not _too_soon_since_last_name_change(last_change, days=1):
                            try:
                                self.pmm.set_name(cand, origin="assistant_self")
                                # Record cooldown marker
                                self.pmm.model.metrics.last_name_change_at = (
                                    _utcnow_str()
                                )
                                self.pmm.save_model()
                                self.pmm.add_event(
                                    summary=f"Identity update: Name changed to '{cand}' (origin=assistant_self)",
                                    effects=[],
                                    etype="identity_update",
                                )
                                pmm_dlog(f"🔍 DEBUG: Assistant self‑named to: {cand}")
                            except Exception as e:
                                pmm_dlog(f"🔍 DEBUG: Assistant self‑naming failed: {e}")
                        else:
                            pmm_dlog("🔍 DEBUG: Name change blocked by cooldown")
                    else:
                        pmm_dlog(
                            "🔍 DEBUG: No assistant self‑declaration found; skipping"
                        )
                # Process evidence events from assistant output as well
                # Emit canonical evidence based on assistant behavior/output only
                try:
                    from .evidence.behavior_engine import process_reply_for_evidence

                    _ = process_reply_for_evidence(self.pmm, ai_output)
                except Exception:
                    pass
            except Exception:
                pass

            # ---- 3) Commitments: extract + add from AI RESPONSE (always) and USER INPUT (gated) ----
            new_commitment_text = None
            # Always analyze assistant output for directives/commitments regardless of user input being behavioral
            try:
                # Use integrated directive system instead of old CommitmentTracker
                detected_directives = self.directive_system.process_response(
                    user_message=human_input,
                    ai_response=ai_output,
                    event_id=response_event_id,
                )

                if detected_directives:
                    pmm_dlog(
                        f"🔍 DEBUG: Detected {len(detected_directives)} directives:"
                    )
                    for directive in detected_directives:
                        pmm_dlog(
                            f"  - {directive.__class__.__name__}: {directive.content[:80]}..."
                        )

                        # Add to PMM for backward compatibility
                        if hasattr(directive, "content"):
                            self.pmm.add_commitment(
                                text=directive.content,
                                source_insight_id="langchain_interaction",
                            )
                            new_commitment_text = directive.content
                else:
                    pmm_dlog("🔍 DEBUG: No directives found in conversation")

                # Check for evolution triggers
                evolution_triggered = (
                    self.directive_system.trigger_evolution_if_needed()
                )
                if evolution_triggered:
                    pmm_dlog("🔍 DEBUG: Meta-principle triggered natural evolution")

            except Exception as e:
                pmm_dlog(f"🔍 DEBUG: Directive processing error: {e}")
                # Fallback to old system
                tracker = CommitmentTracker()
                new_commitment_text, _ = tracker.extract_commitment(ai_output)
                if new_commitment_text:
                    self.pmm.add_commitment(
                        text=new_commitment_text,
                        source_insight_id="langchain_interaction",
                    )

                try:
                    evidence_events = self._process_evidence_events(human_input)
                    if evidence_events:
                        pmm_dlog(
                            f"🔍 DEBUG: Found {len(evidence_events)} evidence events"
                        )
                    else:
                        pmm_dlog(
                            f"🔍 DEBUG: No evidence found in: {human_input[:100]}..."
                        )
                except Exception as e:
                    pmm_dlog(f"🔍 DEBUG: Evidence processing failed: {e}")
                    pass

                # Also process evidence in assistant output
                try:
                    evidence_events_ai = self._process_evidence_events(ai_output)
                    if evidence_events_ai:
                        pmm_dlog(
                            f"🔍 DEBUG: Found {len(evidence_events_ai)} evidence events in assistant output"
                        )
                except Exception as e:
                    pmm_dlog(f"🔍 DEBUG: Assistant evidence processing failed: {e}")
                    pass

                try:
                    self.pmm.auto_close_commitments_from_event(ai_output)
                except Exception:
                    pass

                # ---- 4) Patterns update ----
                try:
                    self.pmm.update_patterns(ai_output)
                except Exception:
                    pass

            # ---- 5) Adaptive Reflection Triggers (Phase 3C) ----
            # Skip reflection triggers for non-behavioral inputs
            should_reflect = False
            if not is_non_behavioral:
                try:
                    from datetime import datetime
                    from pmm.adaptive_triggers import (
                        AdaptiveTrigger,
                        TriggerConfig,
                        TriggerState,
                    )

                    # Build state from PMM
                    events_total = len(
                        self.pmm.model.self_knowledge.autobiographical_events
                    )
                    last_ref_ts = getattr(
                        self.pmm.model.self_knowledge, "last_reflection_ts", None
                    )
                    last_dt = None
                    if isinstance(last_ref_ts, str):
                        try:
                            last_dt = datetime.fromisoformat(last_ref_ts)
                        except (ValueError, TypeError):
                            last_dt = None
                    elif hasattr(last_ref_ts, "isoformat"):
                        last_dt = last_ref_ts

                    # Calculate emergence scores in real-time for adaptive triggers
                    try:
                        from pmm.emergence import EmergenceAnalyzer, EmergenceEvent

                        # Reuse the existing PMM SQLite store to avoid DB drift
                        store = getattr(self.pmm, "sqlite_store", None)
                        if not store or not getattr(store, "conn", None):
                            raise RuntimeError("PMM sqlite_store is not available")

                        # Get recent events with comprehensive filtering like probe API
                        conn = store.conn
                        cursor = conn.execute(
                            """
                            SELECT id, ts, kind, content, meta 
                            FROM events 
                            WHERE kind IN ('response', 'event', 'reflection', 'evidence', 'commitment')
                            ORDER BY ts DESC 
                            LIMIT 15
                        """
                        )
                        rows = cursor.fetchall()

                        # Convert to EmergenceEvent objects
                        events = []
                        for row in rows:
                            import json

                            meta = {}
                            try:
                                meta = json.loads(row[4]) if row[4] else {}
                            except (json.JSONDecodeError, TypeError):
                                pass
                            events.append(
                                EmergenceEvent(
                                    id=row[0],
                                    timestamp=row[1],
                                    kind=row[2],
                                    content=row[3],
                                    meta=meta,
                                )
                            )

                        # Create analyzer and override get_recent_events like probe API
                        analyzer = EmergenceAnalyzer(storage_manager=store)
                        analyzer.get_recent_events = (
                            lambda kind="response", limit=15: events
                        )

                        scores = analyzer.compute_scores(
                            window=15
                        )  # Use Phase 3C window size
                        ias = scores.get("IAS", 0.0)
                        gas = scores.get("GAS", 0.0)
                        pmm_dlog(
                            f"🔍 DEBUG: Real-time emergence scores - IAS: {ias}, GAS: {gas}"
                        )
                        pmm_dlog(
                            f"🔍 DEBUG: Events analyzed: {scores.get('events_analyzed', 0)}, Stage: {scores.get('stage', 'Unknown')}"
                        )
                    except Exception as e:
                        pmm_dlog(f"🔍 DEBUG: Failed to calculate emergence scores: {e}")
                        import traceback

                        pmm_dlog(f"🔍 DEBUG: Traceback: {traceback.format_exc()}")
                        ias = None
                        gas = None

                    # Determine events since last reflection
                    events_since_reflection = getattr(
                        self.pmm.model.self_knowledge, "events_since_reflection", None
                    )
                    if events_since_reflection is None:
                        # fallback heuristic: use total events modulo
                        events_since_reflection = events_total % 6

                    # Respect config reflection_cadence_days if present
                    cadence_days = getattr(
                        self.pmm.model.metrics, "reflection_cadence_days", 7.0
                    )

                    trigger = AdaptiveTrigger(
                        TriggerConfig(
                            cadence_days=cadence_days,
                            events_min_gap=4,  # tune in Phase 3C tests
                        ),
                        TriggerState(
                            last_reflection_at=last_dt,
                            events_since_reflection=events_since_reflection,
                        ),
                    )

                    # Force reflection on new commitment regardless of other conditions
                    if new_commitment_text:
                        should_reflect = True
                        reason = "new-commitment"
                        pmm_dlog(
                            f"🔍 DEBUG: Commitment trigger - new commitment: {new_commitment_text}"
                        )
                    else:
                        should_reflect, reason = trigger.decide(
                            datetime.now(UTC), ias, gas, events_since_reflection
                        )

                    pmm_dlog(
                        f"🔍 DEBUG: Adaptive trigger decision: {should_reflect} ({reason})"
                    )
                    pmm_dlog(
                        f"🔍 DEBUG: Events since reflection: {events_since_reflection}, IAS: {ias}, GAS: {gas}"
                    )

                except Exception as e:
                    pmm_dlog(f"🔍 DEBUG: Adaptive reflection trigger check failed: {e}")
                    # Fallback to simple event count trigger
                    behavioral_events = [
                        e
                        for e in self.pmm.model.self_knowledge.autobiographical_events
                        if e.type != "non_behavioral"
                    ]
                    event_count = len(behavioral_events)
                    should_reflect = (event_count > 0 and event_count % 4 == 0) or bool(
                        new_commitment_text
                    )
            else:
                pmm_dlog(
                    "🔍 DEBUG: Skipping reflection triggers for non-behavioral input"
                )

            if should_reflect:
                pmm_dlog("🔍 DEBUG: Reflection triggered, checking for topic loops...")
                # Skip reflection if the last few events look like a topical loop
                try:
                    recent = self.pmm.sqlite_store.recent_events(limit=10)
                    window = []
                    for event_dict in recent:
                        # Handle dictionary format from recent_events()
                        if (
                            isinstance(event_dict, dict)
                            and "kind" in event_dict
                            and "content" in event_dict
                        ):
                            kind = event_dict["kind"]  # event kind
                            content = event_dict["content"]  # event content
                            if kind in ("conversation", "self_expression"):
                                window.append(content)
                    joined = " ".join(window).lower()
                    if joined.count("slop code") >= 3:
                        pmm_dlog(
                            "🔍 DEBUG: Suppressing reflection due to topic loop (slop code)"
                        )
                        insight = None  # suppress looped reflections
                    else:
                        pmm_dlog(
                            "🔍 DEBUG: No topic loop detected, proceeding with reflection..."
                        )
                        insight = self._auto_reflect()
                        if insight:
                            if self._is_similar_to_recent_insights(insight):
                                pmm_dlog(
                                    "🔍 DEBUG: Suppressing reflection due to similarity to recent insights"
                                )
                                insight = None
                        pmm_dlog(
                            f"🔍 DEBUG: Reflection completed, insight: {bool(insight)}"
                        )
                except Exception as e:
                    pmm_dlog(f"🔍 DEBUG: Reflection failed: {e}")
                    insight = None

                if insight:
                    pmm_tlog(
                        f"\n🧠 Insight: {insight[:160]}{'...' if len(insight) > 160 else ''}"
                    )

                    # ---- Automatic Introspection Triggers ----
                    try:
                        automatic_results = (
                            self.introspection.check_automatic_triggers()
                        )
                        for auto_result in automatic_results:
                            if (
                                auto_result.user_visible
                                and auto_result.confidence
                                >= self.introspection.config.notify_threshold
                            ):
                                formatted_auto = (
                                    self.introspection.format_result_for_user(
                                        auto_result
                                    )
                                )
                                pmm_tlog(
                                    f"\n🤖 Automatic Introspection Triggered:\n{formatted_auto}"
                                )

                                # Log automatic introspection as event
                                self.pmm.add_event(
                                    summary=f"Automatic {auto_result.type.value} introspection triggered",
                                    etype="introspection_automatic",
                                    tags=[
                                        "introspection",
                                        "automatic",
                                        auto_result.type.value,
                                        auto_result.trigger_reason.value,
                                    ],
                                    effects=[],
                                    evidence="",
                                    full_text=formatted_auto,
                                )
                    except Exception as e:
                        pmm_dlog(f"🔍 DEBUG: Automatic introspection check failed: {e}")

                    # (c) auto‑close from reflection + apply drift immediately
                    try:
                        pmm_dlog(
                            "🔍 DEBUG: Auto-closing commitments from reflection..."
                        )
                        self.pmm.auto_close_commitments_from_reflection(insight)
                        pmm_dlog("🔍 DEBUG: Auto-close completed")
                    except Exception as e:
                        pmm_dlog(f"🔍 DEBUG: Auto-close failed: {e}")
                    try:
                        pmm_dlog("🔍 DEBUG: Applying trait drift...")
                        self.pmm.apply_drift_and_save()
                        pmm_dlog("🔍 DEBUG: Trait drift completed")
                    except Exception as e:
                        pmm_dlog(f"🔍 DEBUG: Trait drift failed: {e}")

                    # Phase 3C: Persist reflection bookkeeping for adaptive triggers
                    try:
                        from datetime import datetime

                        self.pmm.model.self_knowledge.last_reflection_ts = datetime.now(
                            UTC
                        ).isoformat()
                        self.pmm.model.self_knowledge.events_since_reflection = 0
                        # ---- 6) Update commitment context for next turn ----
                        self._update_commitment_context()

                        # ---- 7) Update reflection bookkeeping for adaptive triggers ----
                        try:
                            self.adaptive_trigger.update_reflection_bookkeeping()
                        except Exception:
                            pass  # Never crash on bookkeeping
                    except Exception as e:
                        pmm_dlog(
                            f"🔍 DEBUG: Failed to update reflection bookkeeping: {e}"
                        )
                else:
                    # Increment events_since_reflection counter even if reflection failed
                    try:
                        cur = (
                            getattr(
                                self.pmm.model.self_knowledge,
                                "events_since_reflection",
                                0,
                            )
                            or 0
                        )
                        self.pmm.model.self_knowledge.events_since_reflection = (
                            int(cur) + 1
                        )
                    except Exception:
                        pass

        # ---- 6) Refresh prompt contexts ----
        self._update_personality_context()
        self._update_commitment_context()

        # ---- 7) Increment reflection cooldown turn counter ----
        if human_input:  # Only increment on actual user turns
            self.reflection_cooldown.increment_turn()

            # Also feed the novelty gate with a compact current context snapshot
            try:
                recent_events = self.pmm.model.self_knowledge.autobiographical_events[
                    -3:
                ]
                current_context = " ".join(
                    [getattr(event, "summary", str(event)) for event in recent_events]
                )
            except Exception:
                current_context = ""
            if current_context:
                self.reflection_cooldown.add_context(current_context)

        # ---- 8) Persist PMM ----
        self.pmm.save_model()

        # ---- 9) Keep LangChain compatibility ----
        # FIXED: Commented out to prevent hanging - PMM handles all persistence internally
        # super().save_context(inputs, outputs)

        # ---- 10) Behavior-based evidence (heuristic) ----
        # Duplicate evidence processing removed; assistant evidence is already handled earlier

        # ---- 10.1) Tick identity TTL commitments (turn-scoped) ----
        try:
            if ai_output:
                from pmm.commitments import tick_turn_scoped_identity_commitments

                tick_turn_scoped_identity_commitments(self.pmm, ai_output)
        except Exception:
            pass

        # ---- 10.2) Bandit: scan for commitment closes and evaluate rewards ----
        try:
            self._bandit_scan_commit_closes()
            self._bandit_evaluate_rewards()
        except Exception:
            pass

    # ---- Bandit helper hooks (Step 4) ----
    def _bandit_note_event(
        self, kind: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an outcome-related event for recent bandit actions within horizon.

        This does not compute rewards yet (handled in Step 5); it simply tags
        buffered actions with observations to be evaluated.
        """
        # Respect bandit enable flag
        try:
            flag = os.getenv("PMM_BANDIT_ENABLED")
            if flag is not None and flag.lower() not in ("1", "true", "yes", "on"):
                return
        except Exception:
            pass
        try:
            turn_now = int(getattr(self, "_bandit_turn_id", 0) or 0)
            horizon = int(getattr(self, "_bandit_horizon_turns", 5) or 5)
            for rec in list(getattr(self, "_bandit_events", []) or []):
                if rec.get("finalized"):
                    continue
                age = max(0, turn_now - int(rec.get("turn", 0) or 0))
                if age > horizon:
                    rec["finalized"] = True
                    continue
                ev = {"turn": turn_now, "kind": kind, "details": details or {}}
                try:
                    rec.setdefault("events", []).append(ev)
                except Exception:
                    pass
            pmm_dlog(f"🔍 DEBUG: bandit: observed event kind={kind} turn={turn_now}")
        except Exception:
            pass

    def _bandit_scan_commit_closes(self) -> None:
        """Scan recent SQLite events and tag any new commitment.close events for bandit."""
        # Respect bandit enable flag
        try:
            flag = os.getenv("PMM_BANDIT_ENABLED")
            if flag is not None and flag.lower() not in ("1", "true", "yes", "on"):
                return
        except Exception:
            pass
        try:
            store = getattr(self.pmm, "sqlite_store", None)
            if not store:
                return
            with store._lock:
                rows = list(
                    store.conn.execute(
                        "SELECT id, kind FROM events WHERE kind='commitment.close' ORDER BY id DESC LIMIT 20"
                    )
                )
            if not rows:
                return
            # Process oldest-first to preserve order
            for ev_id, kind in reversed(rows):
                try:
                    eid = int(ev_id)
                except Exception:
                    continue
                if eid <= int(getattr(self, "_bandit_last_close_event_id", 0) or 0):
                    continue
                # New close observed this turn
                self._bandit_note_event("commit_close", {"event_id": eid})
                self._bandit_last_close_event_id = eid
        except Exception:
            pass

    def _bandit_evaluate_rewards(self) -> None:
        """Evaluate rewards for actions whose horizon expired and persist outcomes."""
        # Respect bandit enable flag
        try:
            flag = os.getenv("PMM_BANDIT_ENABLED")
            if flag is not None and flag.lower() not in ("1", "true", "yes", "on"):
                return
        except Exception:
            pass
        try:
            turn_now = int(getattr(self, "_bandit_turn_id", 0) or 0)
            horizon = int(getattr(self, "_bandit_horizon_turns", 5) or 5)

            # Ensure bandit is wired to our store
            try:
                if hasattr(self.pmm, "sqlite_store"):
                    pmm_bandit.set_store(self.pmm.sqlite_store)
            except Exception:
                pass

            # Reward constants (env-tunable)
            def _f(name: str, default: float) -> float:
                try:
                    return float(os.getenv(name, str(default)))
                except Exception:
                    return default

            POS_ACCEPTED = _f("PMM_BANDIT_POS_ACCEPTED", 0.7)
            POS_CLOSE = _f("PMM_BANDIT_POS_CLOSE", 0.3)
            NEG_INERT2 = _f("PMM_BANDIT_NEG_INERT2", 0.15)
            NEG_REJECT = _f("PMM_BANDIT_NEG_REJECT", 0.05)
            POS_CONT_CLOSE = _f("PMM_BANDIT_POS_CONTINUE_CLOSE", 0.1)

            for rec in list(self._bandit_events):
                if rec.get("finalized"):
                    continue
                age = max(0, turn_now - int(rec.get("turn", 0) or 0))
                if age <= horizon:
                    continue  # Not yet at horizon

                events = sorted(
                    rec.get("events", []), key=lambda e: int(e.get("turn", 0))
                )
                action = rec.get("action")
                ctx = rec.get("ctx", {})
                reward = 0.0
                components = {
                    "accepted": False,
                    "close": False,
                    "inert2": False,
                    "reject": False,
                    "continue_close": False,
                }

                if action == "reflect_now":
                    # Positive: accepted reflection
                    acc_turns = [
                        int(e.get("turn", 0))
                        for e in events
                        if e.get("kind") == "reflection_accepted"
                    ]
                    if acc_turns:
                        components["accepted"] = True
                        reward += POS_ACCEPTED
                        first_acc_turn = min(acc_turns)
                        # Additional positive: commitment close within horizon after acceptance
                        has_close = any(
                            (e.get("kind") == "commit_close")
                            and int(e.get("turn", 0)) >= first_acc_turn
                            for e in events
                        )
                        if has_close:
                            components["close"] = True
                            reward += POS_CLOSE

                    # Negative: inert twice in a row within horizon
                    consec = 0
                    max_consec = 0
                    for e in events:
                        if e.get("kind") == "reflection_inert":
                            consec += 1
                            max_consec = max(max_consec, consec)
                        elif e.get("kind") == "reflection_accepted":
                            # reset streak on acceptance
                            consec = 0
                        else:
                            # other events do not affect inert streak
                            pass
                    if max_consec >= 2:
                        components["inert2"] = True
                        reward -= NEG_INERT2

                    # Negative: rejected by dedup/novelty without acceptance (no INERT)
                    has_inert = any(e.get("kind") == "reflection_inert" for e in events)
                    if (
                        (not components["accepted"])
                        and (not has_inert)
                        and any(e.get("kind") == "reflection_rejected" for e in events)
                    ):
                        components["reject"] = True
                        reward -= NEG_REJECT

                else:  # action == "continue"
                    has_close = any(e.get("kind") == "commit_close" for e in events)
                    has_accept = any(
                        e.get("kind") == "reflection_accepted" for e in events
                    )
                    if has_close and not has_accept:
                        components["continue_close"] = True
                        reward += POS_CONT_CLOSE

                # Clamp reward
                if reward > 1.0:
                    reward = 1.0
                if reward < -1.0:
                    reward = -1.0

                # Persist outcome
                try:
                    note = (
                        f"accepted={components['accepted']} close={components['close']} "
                        f"inert2={components['inert2']} reject={components['reject']} "
                        f"continue_close={components['continue_close']}"
                    )
                    pmm_bandit.record_outcome(
                        ctx, action, float(reward), horizon=horizon, notes=note
                    )
                except Exception:
                    pass

                # Telemetry: bandit reward components
                try:
                    telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                except Exception:
                    telemetry = False
                if telemetry:
                    try:
                        comp_str = (
                            f"accepted={int(components['accepted'])}, close={int(components['close'])}, "
                            f"inert2={int(components['inert2'])}, reject={int(components['reject'])}, "
                            f"continue_close={int(components['continue_close'])}"
                        )
                        pmm_tlog(
                            f"[PMM_TELEMETRY] bandit_reward: action={action}, reward={reward:.3f}, components={{{ {comp_str} }}}, horizon={horizon}"
                        )
                    except Exception:
                        pass

            rec["finalized"] = True
        except Exception:
            pass

    def _process_evidence_events(self, text: str) -> List[tuple]:
        """Process evidence events from text and emit them to PMM system.

        Returns a list of detected evidence tuples for observability.
        """
        detected: List[tuple] = []
        try:
            detected = self.pmm.commitment_tracker.detect_evidence_events(text) or []
            for evidence_type, commit_ref, description, artifact in detected:
                # If it's a 'done' evidence, close the commitment (pass named args)
                if evidence_type == "done" and commit_ref:
                    try:
                        self.pmm.commitment_tracker.close_commitment_with_evidence(
                            commit_hash=commit_ref,
                            evidence_type="done",
                            description=description,
                            artifact=artifact,
                        )
                    except Exception:
                        # Best-effort; never crash memory flow on evidence persistence
                        pass
        except Exception:
            # Best-effort; never crash memory flow on evidence detection
            pass
        return detected

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer with prepended PMM context."""
        # Get the chat history messages from the buffer
        from langchain.schema.messages import HumanMessage, SystemMessage

        messages = self.chat_memory.messages

        # Assemble PMM context parts
        pmm_context_parts = []
        try:
            # Core personality context
            self._update_personality_context()
            if self.personality_context:
                pmm_context_parts.append(self.personality_context)

            # Directives: Meta-principles, Guiding Principles, and Commitments
            if hasattr(self, "directive_system") and self.directive_system:
                # Helper to safely extract text from dicts or objects
                def _content_of(x):
                    try:
                        if isinstance(x, dict):
                            return x.get("content") or x.get("text") or str(x)
                        return (
                            getattr(x, "content", None)
                            or getattr(x, "text", None)
                            or str(x)
                        )
                    except Exception:
                        return str(x)

                meta_principles = self.directive_system.get_meta_principles()
                if meta_principles:
                    pmm_context_parts.append(
                        "\n".join(
                            [
                                f"[Meta-Principle] {_content_of(p)}"
                                for p in meta_principles
                            ]
                        )
                    )

                active_principles = self.directive_system.get_active_principles()
                if active_principles:
                    pmm_context_parts.append(
                        "\n".join(
                            [
                                f"[Guiding Principle] {_content_of(p)}"
                                for p in active_principles
                            ]
                        )
                    )

                active_commitments = self.directive_system.get_active_commitments()
                if active_commitments:
                    pmm_context_parts.append(
                        "\n".join(
                            [
                                f"[Commitment] {_content_of(c)}"
                                for c in active_commitments
                            ]
                        )
                    )
                else:
                    # Fallback: include commitments directly from PMM model if available
                    try:
                        commitments = (
                            getattr(self.pmm.model.self_knowledge, "commitments", None)
                            if hasattr(self.pmm, "model") and hasattr(self.pmm, "model")
                            else None
                        )
                        if commitments:

                            def _get_content(c):
                                return (
                                    getattr(c, "content", c.get("content", str(c)))
                                    if isinstance(c, dict) or hasattr(c, "content")
                                    else str(c)
                                )

                            formatted = [
                                f"[Commitment] {_get_content(c)}" for c in commitments
                            ]
                            if formatted:
                                pmm_context_parts.append("\n".join(formatted))
                    except Exception:
                        pass
            else:
                # No directive system: best-effort to surface commitments from PMM model
                try:
                    if hasattr(self.pmm, "model") and hasattr(
                        self.pmm.model, "self_knowledge"
                    ):
                        commitments = getattr(
                            self.pmm.model.self_knowledge, "commitments", None
                        )
                        if commitments:

                            def _get_content(c):
                                return (
                                    getattr(c, "content", c.get("content", str(c)))
                                    if isinstance(c, dict) or hasattr(c, "content")
                                    else str(c)
                                )

                            formatted = [
                                f"[Commitment] {_get_content(c)}" for c in commitments
                            ]
                            if formatted:
                                pmm_context_parts.append("\n".join(formatted))
                except Exception:
                    pass

        except Exception as e:
            pmm_dlog(f"[error] Failed to load PMM context: {e}")

        # Prepend the PMM context to the first message in the buffer
        if pmm_context_parts:
            pmm_context = "\n\n".join(pmm_context_parts)
            # Ensure there is at least one message to prepend to
            if messages:
                first_message = messages[0]
                # It's safest to prepend to a human message
                if first_message.type == "human":
                    # Create a new message with the combined content
                    new_content = f"{pmm_context}\n\n{first_message.content}"
                    # Create a new list of messages with the updated first message
                    messages = [HumanMessage(content=new_content)] + messages[1:]
                else:
                    # If the first message isn't human, insert a new system message
                    # This is a safe fallback to avoid losing context
                    messages.insert(0, SystemMessage(content=pmm_context))
            else:
                # If there are no messages, create a new system message with the context
                messages = [SystemMessage(content=pmm_context)]

        # Return the messages in the format expected by LangChain
        return {self.memory_key: messages}

    def get_personality_summary(self) -> Dict[str, Any]:
        """Get current personality state for debugging/monitoring."""
        return {
            "agent_id": self.pmm.model.core_identity.id,
            "name": self.pmm.model.core_identity.name,
            "personality_traits": {
                trait: getattr(self.pmm.model.personality.traits.big5, trait).score
                for trait in [
                    "openness",
                    "conscientiousness",
                    "extraversion",
                    "agreeableness",
                    "neuroticism",
                ]
            },
            "behavioral_patterns": self.pmm.model.self_knowledge.behavioral_patterns,
            "total_events": len(self.pmm.model.self_knowledge.autobiographical_events),
            "total_insights": len(self.pmm.model.self_knowledge.insights),
            "open_commitments": len(self.pmm.model.self_knowledge.commitments),
        }

    def handle_introspection_command(self, user_input: str) -> Optional[str]:
        """
        Handle user introspection commands and return formatted results.

        Args:
            user_input: User input that may contain introspection commands

        Returns:
            Formatted introspection result if command was processed, None otherwise
        """
        if not hasattr(self, "introspection") or not self.introspection:
            return None

        command_type = self.introspection.parse_user_command(user_input)
        if not command_type:
            return None

        pmm_dlog(f"🔍 DEBUG: Processing introspection command: {command_type.value}")

        if user_input.lower().strip() == "@introspect help":
            # Special case: show available commands
            commands = self.introspection.get_available_commands()
            help_text = "🔍 **Available Introspection Commands:**\n\n"
            for cmd, desc in commands.items():
                help_text += f"• `{cmd}` - {desc}\n"
            help_text += "\n💡 **Automatic Introspection:**\n"
            help_text += (
                "• PMM also performs automatic introspection when it detects:\n"
            )
            help_text += "  - Failed commitments\n"
            help_text += "  - Significant trait drift\n"
            help_text += "  - Reflection quality issues\n"
            help_text += "  - Emergence score plateaus\n"
            help_text += "\n🔔 You'll be notified when automatic analysis occurs.\n"

            return help_text
        else:
            # Process the introspection command
            introspection_result = self.introspection.user_introspect(command_type)
            formatted_result = self.introspection.format_result_for_user(
                introspection_result
            )

            # Log the introspection as a special event
            self.pmm.add_event(
                summary=f"User requested {command_type.value} introspection",
                etype="introspection_command",
                tags=["introspection", "user_command", command_type.value],
                effects=[],
                evidence="",
                full_text=user_input,
            )

            return formatted_result

    def _get_semantic_context(
        self, current_input: str, max_results: int = 8
    ) -> List[str]:
        """Retrieve semantically relevant memories based on current input."""
        try:
            if not current_input.strip() or not hasattr(self.pmm, "sqlite_store"):
                return []

            # Get semantic analyzer
            semantic_analyzer = get_semantic_analyzer()

            # Generate embedding for current input
            import numpy as np

            embedding_list = semantic_analyzer._get_embedding(current_input)
            query_embedding = np.array(embedding_list, dtype=np.float32).tobytes()

            # Search for semantically similar events
            similar_events = self.pmm.sqlite_store.semantic_search(
                query_embedding,
                limit=max_results,
                kind_filter=None,  # Search all event types
            )

            # Format relevant memories for context
            relevant_memories = []
            for event in similar_events:
                content = event.get("summary") or event.get("content", "")
                kind = event.get("kind", "")
                event.get("ts", "")

                # Format based on event type
                if kind in ["prompt", "response"]:
                    if "User said:" in content:
                        user_msg = content.replace("User said: ", "")
                        relevant_memories.append(f"[Relevant] Human: {user_msg}")

                    elif "I responded:" in content:
                        ai_msg = content.replace("I responded: ", "")
                        relevant_memories.append(f"[Relevant] Assistant: {ai_msg}")
                elif kind == "event":
                    relevant_memories.append(f"[Context] {content}")
                elif kind == "commitment":
                    relevant_memories.append(f"[Commitment] {content}")
                elif kind == "reflection":
                    relevant_memories.append(f"[Insight] {content}")

            return relevant_memories

        except Exception as e:
            pmm_dlog(f"🔍 DEBUG: Semantic context retrieval failed: {e}")
            return []

    def _is_similar_to_recent_insights(self, content: str) -> bool:
        """Delegate to AtomicReflectionManager embedding dedup (env-driven threshold).

        Falls back to a simple token-overlap check if the atomic call fails.
        """
        if not content or not content.strip():
            return True

        # Prefer centralized atomic dedup to ensure parity across the system
        try:
            arm = AtomicReflectionManager(self.pmm)
            is_dup = arm._is_duplicate_embedding(content)
            if is_dup:
                # Surface effective threshold for observability
                pmm_dlog(
                    f"🔍 DEBUG: High similarity via AtomicReflectionManager (threshold: {arm.embedding_threshold})"
                )
            return is_dup
        except Exception as e:
            pmm_dlog(f"🔍 DEBUG: Atomic dedup check failed, using fallback: {e}")

        # Fallback to simple text comparison against recent insights
        recent_insights = self.pmm.model.self_knowledge.insights[-8:]
        if not recent_insights:
            return False

        for insight in recent_insights:
            if (
                getattr(insight, "content", None)
                and len(
                    set(content.lower().split()) & set(insight.content.lower().split())
                )
                > len(content.split()) * 0.7
            ):
                return True

        return False

    def _auto_reflect(self) -> Optional[str]:
        """
        Internal automatic reflection with atomic validation, cooldown, and TTL management.
        This method is only called by the adaptive trigger system - no manual invocation.
        """
        if not hasattr(self.pmm, "model") or not self.pmm.model:
            return None

        # Get active model config - fail fast if missing
        from pmm.llm_factory import get_llm_factory

        llm_factory = get_llm_factory()
        active_model_config = llm_factory.get_active_config()

        if (
            not active_model_config
            or not active_model_config.get("name")
            or not active_model_config.get("provider")
        ):
            pmm_dlog(
                f"🔍 DEBUG: No valid active model config for reflection: {active_model_config}"
            )
            return None

        # Build current context for cooldown check - use PMM's own event history
        try:
            recent_events = self.pmm.model.self_knowledge.autobiographical_events[-3:]
            current_context = " ".join(
                [getattr(event, "summary", str(event)) for event in recent_events]
            )
        except (AttributeError, IndexError):
            current_context = ""

        # ---- Bandit policy gate (Step 3) ----
        # Decide reflect vs continue using ε-greedy bandit over context features.
        # Default enabled unless explicitly disabled later (Step 7 will formalize envs).
        try:
            bandit_enabled = os.getenv("PMM_BANDIT_ENABLED")
            bandit_enabled = (
                True
                if bandit_enabled is None
                else bandit_enabled.lower() in ("1", "true", "yes", "on")
            )
        except Exception:
            bandit_enabled = True

        bandit_action = "reflect_now"
        if bandit_enabled:
            try:
                # Ensure bandit uses our SQLite store
                if hasattr(self.pmm, "sqlite_store"):
                    pmm_bandit.set_store(self.pmm.sqlite_store)

                # Gather context features
                try:
                    from pmm.emergence import compute_emergence_scores

                    em = (
                        compute_emergence_scores(
                            window=15,
                            storage_manager=getattr(self.pmm, "sqlite_store", None),
                        )
                        or {}
                    )
                except Exception:
                    em = {}

                gas_now = float(em.get("GAS", 0.0) or 0.0)
                ias_now = float(em.get("IAS", 0.0) or 0.0)
                close_now = float(em.get("commit_close_rate", 0.0) or 0.0)
                ident_signals = float(em.get("identity_signal_count", 0.0) or 0.0)

                # Hot path heuristic consistent with cooldown adaptation
                hot_now = bool(gas_now >= 0.85 and close_now >= 0.60)

                # Time since last reflection (seconds)
                try:
                    from datetime import datetime, timezone

                    last_ts = self.reflection_cooldown.state.last_reflection_time
                    t_since = (
                        (datetime.now(timezone.utc) - last_ts).total_seconds()
                        if last_ts
                        else 0.0
                    )
                except Exception:
                    t_since = 0.0

                # Dedup threshold (effective)
                try:
                    ar_stats = self.atomic_reflection.get_stats()
                    dedup_thr = float(
                        ar_stats.get("embedding_threshold_effective", 0.94) or 0.94
                    )
                except Exception:
                    dedup_thr = 0.94

                # Inert streak (consecutive non-accepted insights)
                inert_streak = 0
                try:
                    insights = (
                        getattr(self.pmm.model.self_knowledge, "insights", []) or []
                    )
                    for ins in reversed(insights):
                        meta = (
                            getattr(ins, "meta", {})
                            or getattr(ins, "__dict__", {}).get("meta", {})
                            or {}
                        )
                        if bool(meta.get("accepted", False)):
                            break
                        inert_streak += 1
                except Exception:
                    inert_streak = 0

                ctx = pmm_bandit.build_context(
                    gas=gas_now,
                    ias=ias_now,
                    close=close_now,
                    hot=hot_now,
                    identity_signal_count=ident_signals,
                    time_since_last_reflection_sec=t_since,
                    dedup_threshold=dedup_thr,
                    inert_streak=float(inert_streak),
                )

                # Select action (with telemetry details)
                hot_bias = False
                try:
                    action_info = pmm_bandit.select_action_info(ctx)
                    # Support both old and new tuple sizes
                    if len(action_info) == 5:
                        bandit_action, sel_eps, q_reflect, q_continue, hot_bias = (
                            action_info
                        )
                    else:
                        bandit_action, sel_eps, q_reflect, q_continue = action_info  # type: ignore[misc]
                except Exception:
                    bandit_action = pmm_bandit.select_action(ctx)
                    st = pmm_bandit.load_policy()
                    float(os.getenv("PMM_BANDIT_EPSILON", "0.10") or 0.10)
                    float((st.get("reflect_now", {}) or {}).get("value", 0.0))
                    float((st.get("continue", {}) or {}).get("value", 0.0))
                    hot_bias = False

                # Safety guardrails (Step 2: Re-balanced for hot ticks)
                safety_override = False
                try:
                    turn_now = int(getattr(self, "_bandit_turn_id", 0) or 0)
                    hot_strength = float(ctx.get("hot_strength", 0.0))

                    # Never block the first reflection inside a hot window
                    first_hot_always = os.getenv(
                        "PMM_SAFETY_FIRST_HOT_ALWAYS", "1"
                    ).lower() in ("1", "true", "yes", "on")
                    if first_hot_always and hot_strength >= 0.5:
                        # Check if no reflect attempt in last 3 turns
                        recent_reflects_3 = 0
                        for r in list(getattr(self, "_bandit_events", []) or []):
                            rt = int(r.get("turn", 0) or 0)
                            if (turn_now - 3) < rt < turn_now and str(
                                r.get("action")
                            ) == "reflect_now":
                                recent_reflects_3 += 1
                        if recent_reflects_3 == 0:
                            # Skip safety override for first hot reflection
                            pass
                        else:
                            # Apply normal safety rules
                            max_reflect_per_5 = int(
                                os.getenv("PMM_SAFETY_MAX_REFLECT_PER_5", "2")
                            )
                            recent_reflects = 0
                            for r in list(getattr(self, "_bandit_events", []) or []):
                                rt = int(r.get("turn", 0) or 0)
                                if (turn_now - 5) < rt < turn_now and str(
                                    r.get("action")
                                ) == "reflect_now":
                                    # Only count actual attempts, not blocked by time/turns gate
                                    if r.get("finalized", False):
                                        recent_reflects += 1
                            if (
                                bandit_action == "reflect_now"
                                and recent_reflects >= max_reflect_per_5
                            ):
                                safety_override = True
                    else:
                        # Normal safety: max 2 reflections per 5 turns
                        max_reflect_per_5 = int(
                            os.getenv("PMM_SAFETY_MAX_REFLECT_PER_5", "2")
                        )
                        recent_reflects = 0
                        for r in list(getattr(self, "_bandit_events", []) or []):
                            rt = int(r.get("turn", 0) or 0)
                            if (turn_now - 5) < rt < turn_now and str(
                                r.get("action")
                            ) == "reflect_now":
                                # Only count actual attempts, not blocked by time/turns gate
                                if r.get("finalized", False):
                                    recent_reflects += 1
                        if (
                            bandit_action == "reflect_now"
                            and recent_reflects >= max_reflect_per_5
                        ):
                            safety_override = True

                    # Inert loop breaker: do not trigger on hot ticks
                    if hot_strength < 0.5:
                        no_close_recent = True
                        for r in list(getattr(self, "_bandit_events", []) or []):
                            rt = int(r.get("turn", 0) or 0)
                            if rt >= (turn_now - 5):
                                for ev in r.get("events") or []:
                                    if str(ev.get("kind")) == "commit_close":
                                        no_close_recent = False
                                        break
                            if not no_close_recent:
                                break
                        if (
                            bandit_action == "reflect_now"
                            and (float(inert_streak) >= 2.0)
                            and no_close_recent
                        ):
                            safety_override = True
                except Exception:
                    safety_override = False
                if safety_override:
                    bandit_action = "continue"

                # Telemetry for selection
                try:
                    telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                except Exception:
                    telemetry = False
                if telemetry:
                    try:
                        cd_status = self.reflection_cooldown.get_status()
                    except Exception:
                        cd_status = {
                            "turns_gate_passed": None,
                            "time_gate_passed": None,
                            "novelty_threshold": None,
                        }
                    try:
                        ar_stats = self.atomic_reflection.get_stats()
                        eff = float(
                            ar_stats.get("embedding_threshold_effective", 0.0) or 0.0
                        )
                        cfg = float(
                            ar_stats.get("embedding_threshold_configured", 0.0) or 0.0
                        )
                    except Exception:
                        eff, cfg = 0.0, 0.0
                    extra = ", bandit_safety_override=True" if safety_override else ""
                    hb = ", bandit_hot_bias=True" if hot_bias else ""
                    pmm_tlog(
                        f"[PMM_TELEMETRY] reflection_attempt: decision=blocked, reason=bandit_continue, "
                        f"turns_gate={cd_status.get('turns_gate_passed')}, time_gate={cd_status.get('time_gate_passed')}, "
                        f"novelty_threshold={cd_status.get('novelty_threshold')}, "
                        f"embedding_threshold_effective={eff:.3f}, embedding_threshold_configured={cfg:.3f}, "
                        f"adaptive_enabled={getattr(self.atomic_reflection, '_adaptive_enabled', True)}, turns_override=False, bandit_action=continue{hb}{extra}"
                    )
                    return None
            except Exception:
                # Graceful degradation: fall through to legacy behavior
                bandit_action = "reflect_now"

        # Stage-adapt the novelty threshold before checking cooldown gates
        try:
            from pmm.emergence import compute_emergence_scores

            scores = compute_emergence_scores(
                window=5, storage_manager=getattr(self.pmm, "sqlite_store", None)
            )
            ias = float(scores.get("IAS", 0.0) or 0.0)
            gas = float(scores.get("GAS", 0.0) or 0.0)

            model_name = active_model_config.get("name", "unknown")
            profile = self.emergence_stages.calculate_emergence_profile(
                model_name, ias, gas
            )

            base_thresh = float(self.reflection_cooldown.novelty_threshold)
            adapted_thresh = float(
                self.emergence_stages.adapt_novelty_threshold(
                    base_thresh, profile.stage
                )
            )
            # Day-1 tuning: reduce novelty threshold slightly in S0/S1 to encourage reflection
            try:
                if str(profile.stage.value).startswith("S0") or str(
                    profile.stage.value
                ).startswith("S1"):
                    adapted_thresh = max(0.0, min(1.0, adapted_thresh - 0.05))
            except Exception:
                # Fallback to string profile
                stage_label = str(getattr(profile, "stage", ""))
                if stage_label.startswith("S0") or stage_label.startswith("S1"):
                    adapted_thresh = max(0.0, min(1.0, adapted_thresh - 0.05))
            # Set adapted novelty threshold for this decision
            self.reflection_cooldown.novelty_threshold = adapted_thresh

            # Telemetry: record adaptation
            try:
                telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
            except Exception:
                telemetry = False
            if telemetry:
                pmm_tlog(
                    f"[PMM_TELEMETRY] novelty_adapt: base={base_thresh:.2f}, adapted={adapted_thresh:.2f}, "
                    f"stage={profile.stage.value}, IAS={ias:.3f}, GAS={gas:.3f}"
                )
        except Exception as _e:
            # Non-fatal: keep existing threshold on any failure
            pass

        # Dynamic cooldown adaptation (hot path + inert quick-retry) before cooldown gates
        try:
            from pmm.emergence import compute_emergence_scores

            # Base from env or current manager
            try:
                base_cd = int(
                    os.getenv(
                        "PMM_REFLECTION_COOLDOWN_SECONDS",
                        str(int(self.reflection_cooldown.min_wall_time_seconds)),
                    )
                )
            except Exception:
                base_cd = int(self.reflection_cooldown.min_wall_time_seconds)

            # Read broader window for stability
            em = (
                compute_emergence_scores(
                    window=15, storage_manager=getattr(self.pmm, "sqlite_store", None)
                )
                or {}
            )
            gas_now = float(em.get("GAS", 0.0) or 0.0)
            close_now = float(em.get("commit_close_rate", 0.0) or 0.0)

            # Detect last reflection inert status (accepted=False)
            def _last_inert() -> bool:
                try:
                    insights = (
                        getattr(self.pmm.model.self_knowledge, "insights", []) or []
                    )
                    if not insights:
                        return False
                    last = insights[-1]
                    meta = (
                        getattr(last, "meta", {})
                        or getattr(last, "__dict__", {}).get("meta", {})
                        or {}
                    )
                    return not bool(meta.get("accepted", False))
                except Exception:
                    return False

            try:
                hot_factor = float(os.getenv("PMM_REFLECTION_HOT_FACTOR", "0.35"))
            except Exception:
                hot_factor = 0.35

            hot_now = bool(gas_now >= 0.85 and close_now >= 0.60)
            if hot_now:
                dyn_cd = max(int(base_cd * hot_factor), 4)
            elif _last_inert():
                dyn_cd = max(int(base_cd * 0.5), 6)
            else:
                dyn_cd = base_cd

            # Apply dynamic cooldown to the manager so novelty/turns gates still apply
            self.reflection_cooldown.min_wall_time_seconds = int(dyn_cd)

            # Telemetry: record cooldown adaptation
            try:
                telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
            except Exception:
                telemetry = False
            if telemetry:
                pmm_tlog(
                    f"[PMM_TELEMETRY] cooldown_adapt: base={base_cd}s, dynamic={dyn_cd}s, GAS={gas_now:.3f}, close={close_now:.3f}, inert={_last_inert()}, hot={hot_now}"
                )
        except Exception:
            # Non-fatal: keep current cooldown
            pass

        # Check reflection cooldown gates with bandit hot override
        orig_min_turns = int(getattr(self.reflection_cooldown, "min_turns", 2) or 2)
        turns_override_applied = False
        bandit_override_turns = False

        try:
            # Reuse previously computed gas_now/close_now if available; otherwise recompute
            try:
                hot_now  # type: ignore # noqa
            except Exception:
                try:
                    em = (
                        compute_emergence_scores(
                            window=15,
                            storage_manager=getattr(self.pmm, "sqlite_store", None),
                        )
                        or {}
                    )
                    gas_now = float(em.get("GAS", 0.0) or 0.0)
                    close_now = float(em.get("commit_close_rate", 0.0) or 0.0)
                    hot_now = bool(gas_now >= 0.85 and close_now >= 0.60)
                except Exception:
                    hot_now = False

            # BANDIT HOT OVERRIDE: Skip turns gate when bandit says reflect_now and hot_strength >= 0.5
            if bandit_enabled and bandit_action == "reflect_now":
                try:
                    hot_strength = float(ctx.get("hot_strength", 0.0))
                    if hot_strength >= 0.5:
                        bandit_override_turns = True
                        # Set turns gate to 0 for this decision only
                        self.reflection_cooldown.min_turns = 0
                        turns_override_applied = True
                except Exception:
                    pass
            elif hot_now:
                # Legacy hot path: allow turns gate to be one lower (but at least 1)
                new_turns = max(1, orig_min_turns - 1)
                if new_turns != orig_min_turns:
                    self.reflection_cooldown.min_turns = int(new_turns)
                    turns_override_applied = True

            should_reflect, cooldown_reason = self.reflection_cooldown.should_reflect(
                current_context
            )
        finally:
            # Restore configured turns gate
            try:
                self.reflection_cooldown.min_turns = int(orig_min_turns)
            except Exception:
                pass
        if not should_reflect:
            pmm_dlog(f"🔍 DEBUG: Reflection blocked by cooldown - {cooldown_reason}")
            # Telemetry: consolidate attempt even when blocked
            try:
                telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
            except Exception:
                telemetry = False
            if telemetry:
                try:
                    cd_status = self.reflection_cooldown.get_status()
                    ar_stats = self.atomic_reflection.get_stats()
                    override_reason = (
                        "bandit_override_turns=True"
                        if bandit_override_turns
                        else f"turns_override={turns_override_applied}"
                    )
                    pmm_tlog(
                        f"[PMM_TELEMETRY] reflection_attempt: decision=blocked, reason={cooldown_reason}, "
                        f"turns_gate={cd_status.get('turns_gate_passed')}, time_gate={cd_status.get('time_gate_passed')}, "
                        f"novelty_threshold={cd_status.get('novelty_threshold')}, "
                        f"embedding_threshold_effective={ar_stats.get('embedding_threshold_effective'):.3f}, "
                        f"embedding_threshold_configured={ar_stats.get('embedding_threshold_configured'):.3f}, "
                        f"adaptive_enabled={ar_stats.get('adaptive_enabled')}, {override_reason}, bandit_action={bandit_action}"
                    )
                except Exception:
                    pass
            return None

        # Passed cooldown gates
        pmm_dlog(f"🔍 DEBUG: Reflection cooldown passed - {cooldown_reason}")

        success = False
        try:
            # Generate a candidate insight
            insight_obj = reflect_once(self.pmm, None, active_model_config)

            if not insight_obj or not getattr(insight_obj, "content", None):
                pmm_dlog("🔍 DEBUG: No insight generated")
                try:
                    self._bandit_note_event("reflection_inert")
                except Exception:
                    pass
                return None

            content = insight_obj.content.strip()
            if len(content) < 10:
                pmm_dlog("🔍 DEBUG: Insight too short, skipping")
                try:
                    self._bandit_note_event("reflection_inert")
                except Exception:
                    pass
                return None

            # Apply n-gram ban filtering (stage-aware)
            try:
                stage_label = profile.stage.value  # from earlier computation
            except Exception:
                stage_label = None

            filtered_content, ban_replacements = self.ngram_ban.postprocess_style(
                content, model_name, stage=stage_label
            )
            if ban_replacements:
                pmm_dlog(
                    f"🔍 DEBUG: N-gram ban applied: {ban_replacements} (stage={stage_label or 'auto'})"
                )
            content = filtered_content

            # Step 4: Generate reflection ID for credit assignment tracking
            import uuid

            str(uuid.uuid4())[:8]

            # Atomic reflection validation and persistence
            success = self.atomic_reflection.add_insight(
                content, active_model_config, active_model_config.get("epoch", 0)
            )
            if success:
                # Update baselines with current IAS/GAS (safe computation)
                try:
                    from pmm.emergence import compute_emergence_scores

                    scores = compute_emergence_scores(
                        window=5,
                        storage_manager=getattr(self.pmm, "sqlite_store", None),
                    )
                    ias = float(scores.get("IAS", 0.0) or 0.0)
                    gas = float(scores.get("GAS", 0.0) or 0.0)
                    self.model_baselines.add_scores(model_name, ias, gas)

                    # Calculate emergence profile
                    profile = self.emergence_stages.calculate_emergence_profile(
                        model_name, ias, gas
                    )
                    pmm_dlog(
                        f"🔍 DEBUG: Emergence stage: {profile.stage.value} (confidence: {profile.confidence:.2f})"
                    )
                except Exception:
                    # Non-fatal; continue without emergence update
                    pass

                pmm_dlog(f"🔍 DEBUG: Insight atomically persisted: {content[:100]}...")
                # Telemetry: consolidated acceptance line
                try:
                    telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                except Exception:
                    telemetry = False
                if telemetry:
                    try:
                        cd_status = self.reflection_cooldown.get_status()
                        ar_stats = self.atomic_reflection.get_stats()
                        pmm_tlog(
                            f"[PMM_TELEMETRY] reflection_attempt: decision=accepted, reason={cooldown_reason}, "
                            f"turns_gate={cd_status.get('turns_gate_passed')}, time_gate={cd_status.get('time_gate_passed')}, "
                            f"novelty_threshold={cd_status.get('novelty_threshold')}, "
                            f"embedding_threshold_effective={ar_stats.get('embedding_threshold_effective'):.3f}, "
                            f"embedding_threshold_configured={ar_stats.get('embedding_threshold_configured'):.3f}, "
                            f"adaptive_enabled={ar_stats.get('adaptive_enabled')}, turns_override={turns_override_applied}, bandit_action={bandit_action}"
                        )
                    except Exception:
                        pass
                try:
                    self._bandit_note_event("reflection_accepted")
                except Exception:
                    pass
                return content
            else:
                pmm_dlog("🔍 DEBUG: Insight rejected by atomic validation")
                # Telemetry: consolidated rejection line
                try:
                    telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                except Exception:
                    telemetry = False
                if telemetry:
                    try:
                        cd_status = self.reflection_cooldown.get_status()
                        ar_stats = self.atomic_reflection.get_stats()
                        pmm_tlog(
                            f"[PMM_TELEMETRY] reflection_attempt: decision=rejected, reason={cooldown_reason}, "
                            f"turns_gate={cd_status.get('turns_gate_passed')}, time_gate={cd_status.get('time_gate_passed')}, "
                            f"novelty_threshold={cd_status.get('novelty_threshold')}, "
                            f"embedding_threshold_effective={ar_stats.get('embedding_threshold_effective'):.3f}, "
                            f"embedding_threshold_configured={ar_stats.get('embedding_threshold_configured'):.3f}, "
                            f"adaptive_enabled={ar_stats.get('adaptive_enabled')}, turns_override={turns_override_applied}, bandit_action={bandit_action}"
                        )
                    except Exception:
                        pass
                try:
                    self._bandit_note_event("reflection_rejected")
                except Exception:
                    pass
                return None
        except Exception as e:
            pmm_dlog(f"🔍 DEBUG: Reflection error: {e}")
            return None
        finally:
            pmm_dlog(f"🔍 DEBUG: Reflection completed, insight: {success}")

    @property
    def memory_variables(self) -> List[str]:
        """Return list of memory variables."""
        return [self.memory_key]

    def clear(self) -> None:
        """Clear conversation history but preserve personality."""
        super().clear()
        # Note: We don't clear PMM state - personality persists across conversations
