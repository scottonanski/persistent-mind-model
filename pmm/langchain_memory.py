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
from .integrated_directive_system import IntegratedDirectiveSystem
from .semantic_analysis import get_semantic_analyzer
from .introspection import IntrospectionEngine, IntrospectionConfig
from .phrase_deduper import PhraseDeduper
from .stance_filter import StanceFilter
from .model_baselines import ModelBaselineManager
from .atomic_reflection import AtomicReflectionManager
from .reflection_cooldown import ReflectionCooldownManager
from .commitment_ttl import CommitmentTTLManager
from .ngram_ban import NGramBanSystem
from .emergence_stages import EmergenceStageManager


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
        self.pmm = SelfModelManager(agent_path)
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
            events_min_gap=3,  # Minimum 3 turns between reflections
            ias_low=0.35,
            gas_low=0.35,
            ias_high=0.65,
            gas_high=0.65,
            min_cooldown_minutes=10,
            max_skip_days=7.0,
        )
        self.trigger_state = TriggerState()
        self.adaptive_trigger = AdaptiveTrigger(self.trigger_config, self.trigger_state)

        # Initialize components
        self.phrase_deduper = PhraseDeduper()
        self.stance_filter = StanceFilter()
        self.model_baselines = ModelBaselineManager()
        self.atomic_reflection = AtomicReflectionManager(self.pmm)
        self.reflection_cooldown = ReflectionCooldownManager()
        self.commitment_ttl = CommitmentTTLManager()
        self.ngram_ban = NGramBanSystem()
        self.emergence_stages = EmergenceStageManager(self.model_baselines)

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

    def _auto_extract_key_info(self, user_input: str) -> None:
        """
        Automatically extract and remember key information from user input.

        This method detects:
        - Names ("My name is X", "I am X", "Call me X")
        - Preferences ("I like X", "I prefer X")
        - Important facts about the user
        """
        try:
            raw = user_input.strip()
            user_lower = raw.lower()

            # Extract names (more conservative to avoid false positives like "I'm just...")
            import re

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
                print(f" Automatically remembered: User's name is {remembered}")

            # Pattern order: strongest first, using original casing for capitalization heuristics
            # 1) "My name is X"
            m = re.search(
                r"\bMy name is ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b", raw
            )
            if m:
                _remember_user_name(m.group(1))
            else:
                # 2) "Call me X"
                m = re.search(
                    r"\bCall me ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b", raw
                )
                if m:
                    _remember_user_name(m.group(1))
                else:
                    # 3) "I am X" or "I'm X" only if the next token isn't a stopword and is Capitalized
                    m = re.search(
                        r"\bI\s*(?:am|'m)\s+([A-Z][a-zA-Z]+)(?:\b|\s*$|[\.,!?:;])", raw
                    )
                    if m:
                        candidate = m.group(1)
                        if (
                            candidate.lower() not in stopwords
                            and candidate.lower() not in {"doing", "going", "working"}
                        ):
                            _remember_user_name(candidate)

            # Detect agent name assignments and persist them
            # Only accept explicit *agent* rename directives.
            # DO NOT infer from casual phrasing like "you're ...".
            agent_name_patterns = [
                r"\byour name is (\w+)\b",
                r"\bwe will call you (\w+)\b",
                r"\blet['']s call you (\w+)\b",
                r"\bi['']ll call you (\w+)\b",
            ]

            # Prevent silly names like "doing", "working", "fine", etc.
            forbidden_names = {
                "doing",
                "working",
                "fine",
                "ok",
                "okay",
                "good",
                "thanks",
                "cool",
            }
            for pattern in agent_name_patterns:
                match = re.search(pattern, user_lower)
                if match:
                    candidate = match.group(1).strip().lower()
                    if candidate in forbidden_names:
                        continue
                    agent_name = candidate.title()
                    self.pmm.set_name(agent_name, origin="chat_detect")

                    # PHASE 3B+: Emit explicit identity_update event for traceability
                    self.pmm.add_event(
                        summary=f"IDENTITY UPDATE: Agent name officially adopted as '{agent_name}' via user affirmation",
                        effects=[],
                        etype="identity_update",
                    )

                    print(f" Persisted agent name change to: {agent_name}")
                    print(" Emitted identity_update event for traceability")
                    break

            # Extract preferences and other key info
            preference_patterns = [
                r"i like (.+)",
                r"i prefer (.+)",
                r"i work (?:on|at|with) (.+)",
                r"i am (.+?) (?:and|but|,|\.|$)",
            ]

            for pattern in preference_patterns:
                match = re.search(pattern, user_lower)
                if match and len(match.group(1)) < 50:  # Avoid capturing too much
                    info = match.group(1).strip()
                    if info and len(info) > 2:  # Valid info
                        self.pmm.add_event(
                            summary=f"PREFERENCE: User {match.group(0)}",
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

    def _is_non_behavioral_input(self, text: str) -> bool:
        """
        Determine if input should be treated as non-behavioral (debug/log/paste).

        Non-behavioral inputs are stored for provenance but don't trigger
        reflections, commitment extraction, or behavioral patterns.
        """
        if not text or not text.strip():
            return False

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
            print(
                "🔍 DEBUG: Non-behavioral input detected, skipping behavioral triggers"
            )

        # ---- 0.3) Apply stance filter and phrase deduplication ----
        if ai_output and not is_non_behavioral:
            # Get current model name for phrase deduplication
            current_model = getattr(self, "_active_model_config", {}).get(
                "name", "unknown"
            )

            # Apply stance filter to remove anthropomorphic language
            filtered_output, stance_filters = self.stance_filter.filter_response(
                ai_output
            )
            if stance_filters:
                print(
                    f"🔍 DEBUG: Applied stance filters: {len(stance_filters)} changes"
                )

            # Check for phrase repetition
            is_repetitive, repeated_phrases, repetition_score = (
                self.phrase_deduper.check_response(current_model, filtered_output)
            )

            if is_repetitive:
                print(
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
            print(
                "🔍 DEBUG: Non-behavioral input detected, skipping behavioral triggers"
            )

        # Store non-behavioral inputs for provenance but with special marking
        behavioral_input = human_input if not is_non_behavioral else ""

        # ---- 0.75) Handle Introspection Commands ----
        introspection_result = None
        if human_input:
            command_type = self.introspection.parse_user_command(human_input)
            if command_type:
                print(
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
                        print(
                            f"Warning: Failed to generate embedding for user input: {e}"
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

                    # Check for agent name adoption patterns using strict detector
                    if self.pmm:
                        from .name_detect import (
                            extract_agent_name_command,
                            _too_soon_since_last_name_change,
                            _utcnow_str,
                        )

                        cand = extract_agent_name_command(behavioral_input, "user")
                        if cand:
                            # Check cooldown
                            last_change = getattr(
                                self.pmm.model.metrics, "last_name_change_at", None
                            )
                            if not _too_soon_since_last_name_change(
                                last_change, days=1
                            ):
                                print(
                                    f"🔍 DEBUG: Explicit agent naming detected → '{cand}'"
                                )
                                try:
                                    self.pmm.set_name(cand, origin="user_command")
                                    # Record cooldown marker
                                    self.pmm.model.metrics.last_name_change_at = (
                                        _utcnow_str()
                                    )
                                    self.pmm.save_model()
                                    # Emit identity_update event for traceability
                                    self.pmm.add_event(
                                        summary=f"Identity update: Name changed to '{cand}' (origin=user_command)",
                                        effects=[],
                                        etype="identity_update",
                                    )
                                    print(f"🔍 DEBUG: Successfully set name to: {cand}")
                                except Exception as e:
                                    print(f"🔍 DEBUG: Failed to set name: {e}")
                            else:
                                print("🔍 DEBUG: Name change blocked by cooldown")
                        else:
                            print("🔍 DEBUG: No explicit agent naming found; skipping")

                    # NEW: Phase 3 - Detect and process evidence events from human input
                    try:
                        self._process_evidence_events(behavioral_input)
                    except Exception as e:
                        print(f"🔍 DEBUG: Evidence processing failed: {e}")
                        pass
            except Exception:
                pass  # never crash chat on memory write

        # ---- 2) Log assistant thought + event ----
        if ai_output:
            try:
                self.pmm.add_thought(ai_output, trigger="langchain_conversation")
                self.pmm.add_event(
                    summary=f"I responded: {ai_output[:200]}{'...' if len(ai_output) > 200 else ''}",
                    effects=[],
                    etype="self_expression",
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
                        # Assistant self-naming disabled to prevent pollution
                        print("🔍 DEBUG: Name change blocked by cooldown")
                    else:
                        print("🔍 DEBUG: No assistant self-declaration found; skipping")
            except Exception:
                pass

            # ---- 3) Commitments: extract + add from USER INPUT and AI RESPONSE ----
            # Skip commitment extraction and behavioral processing for non-behavioral inputs
            new_commitment_text = None
            if not is_non_behavioral:
                try:
                    # Use integrated directive system instead of old CommitmentTracker
                    detected_directives = self.directive_system.process_response(
                        user_message=human_input,
                        ai_response=ai_output,
                        event_id=f"langchain_{self.conversation_count}",
                    )

                    if detected_directives:
                        print(
                            f"🔍 DEBUG: Detected {len(detected_directives)} directives:"
                        )
                        for directive in detected_directives:
                            print(
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
                        print("🔍 DEBUG: No directives found in conversation")

                    # Check for evolution triggers
                    evolution_triggered = (
                        self.directive_system.trigger_evolution_if_needed()
                    )
                    if evolution_triggered:
                        print("🔍 DEBUG: Meta-principle triggered natural evolution")

                except Exception as e:
                    print(f"🔍 DEBUG: Directive processing error: {e}")
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
                        print(f"🔍 DEBUG: Found {len(evidence_events)} evidence events")
                    else:
                        print(f"🔍 DEBUG: No evidence found in: {human_input[:100]}...")
                except Exception as e:
                    print(f"🔍 DEBUG: Evidence processing failed: {e}")
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
                        from pmm.storage.sqlite_store import SQLiteStore

                        # Use same database access pattern as probe API
                        db_path = "pmm.db"  # Standard PMM database path
                        store = SQLiteStore(db_path)

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
                        print(
                            f"🔍 DEBUG: Real-time emergence scores - IAS: {ias}, GAS: {gas}"
                        )
                        print(
                            f"🔍 DEBUG: Events analyzed: {scores.get('events_analyzed', 0)}, Stage: {scores.get('stage', 'Unknown')}"
                        )
                    except Exception as e:
                        print(f"🔍 DEBUG: Failed to calculate emergence scores: {e}")
                        import traceback

                        print(f"🔍 DEBUG: Traceback: {traceback.format_exc()}")
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
                        print(
                            f"🔍 DEBUG: Commitment trigger - new commitment: {new_commitment_text}"
                        )
                    else:
                        should_reflect, reason = trigger.decide(
                            datetime.now(UTC), ias, gas, events_since_reflection
                        )

                    print(
                        f"🔍 DEBUG: Adaptive trigger decision: {should_reflect} ({reason})"
                    )
                    print(
                        f"🔍 DEBUG: Events since reflection: {events_since_reflection}, IAS: {ias}, GAS: {gas}"
                    )

                except Exception as e:
                    print(f"🔍 DEBUG: Adaptive reflection trigger check failed: {e}")
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
                print("🔍 DEBUG: Skipping reflection triggers for non-behavioral input")

            if should_reflect:
                print("🔍 DEBUG: Reflection triggered, checking for topic loops...")
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
                        print(
                            "🔍 DEBUG: Suppressing reflection due to topic loop (slop code)"
                        )
                        insight = None  # suppress looped reflections
                    else:
                        print(
                            "🔍 DEBUG: No topic loop detected, proceeding with reflection..."
                        )
                        insight = self._auto_reflect()
                        if insight:
                            if self._is_similar_to_recent_insights(insight):
                                print(
                                    "🔍 DEBUG: Suppressing reflection due to similarity to recent insights"
                                )
                                insight = None
                        print(
                            f"🔍 DEBUG: Reflection completed, insight: {bool(insight)}"
                        )
                except Exception as e:
                    print(f"🔍 DEBUG: Reflection failed: {e}")
                    insight = None

                if insight:
                    print(
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
                                print(
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
                        print(f"🔍 DEBUG: Automatic introspection check failed: {e}")

                    # (c) auto‑close from reflection + apply drift immediately
                    try:
                        print("🔍 DEBUG: Auto-closing commitments from reflection...")
                        self.pmm.auto_close_commitments_from_reflection(insight)
                        print("🔍 DEBUG: Auto-close completed")
                    except Exception as e:
                        print(f"🔍 DEBUG: Auto-close failed: {e}")
                    try:
                        print("🔍 DEBUG: Applying trait drift...")
                        self.pmm.apply_drift_and_save()
                        print("🔍 DEBUG: Trait drift completed")
                    except Exception as e:
                        print(f"🔍 DEBUG: Trait drift failed: {e}")

                    # Phase 3C: Persist reflection bookkeeping for adaptive triggers
                    try:
                        from datetime import datetime

                        self.pmm.model.self_knowledge.last_reflection_ts = datetime.now(
                            UTC
                        ).isoformat()
                        self.pmm.model.self_knowledge.events_since_reflection = 0
                        # ---- 6) Update commitment context for next turn ----
                        self._update_commitment_context()

                        # ---- 7) Increment reflection cooldown turn counter ----
                        if human_input:  # Only increment on actual user turns
                            self.reflection_cooldown.increment_turn()

                        # ---- 8) Update reflection bookkeeping for adaptive triggers ----
                        try:
                            self.adaptive_trigger.update_reflection_bookkeeping()
                        except Exception:
                            pass  # Never crash on bookkeeping
                    except Exception as e:
                        print(f"🔍 DEBUG: Failed to update reflection bookkeeping: {e}")
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

        # ---- 8) Persist PMM ----
        self.pmm.save_model()

        # ---- 9) Keep LangChain compatibility ----
        # FIXED: Commented out to prevent hanging - PMM handles all persistence internally
        # super().save_context(inputs, outputs)

    def _process_evidence_events(self, text: str) -> None:
        """Process evidence events from text and emit them to PMM system."""
        try:
            evidence_events = self.pmm.commitment_tracker.detect_evidence_events(text)
            for evidence_type, commit_ref, description, artifact in evidence_events:
                # Create evidence event data
                evidence_data = {
                    "evidence_type": evidence_type,
                    "commit_ref": commit_ref,
                    "description": description,
                    "artifact": artifact,
                }

                # Add evidence event to PMM
                self.pmm.add_event(
                    summary=f"Evidence: {description}",
                    etype=f"evidence:{evidence_type}",
                    evidence=evidence_data,
                )

                # If it's a 'done' evidence, close the commitment
                if evidence_type == "done":
                    self.pmm.commitment_tracker.close_commitment_with_evidence(
                        commit_ref, description, artifact
                    )

        except Exception as e:
            print(f"🔍 DEBUG: Evidence event processing error: {e}")

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
                Load memory variables for LLM prompts.

        {{ ... }}
                Returns personality context, commitments, and conversation history
                formatted for optimal LLM performance.
        """
        # Get base memory variables from LangChain
        base_variables = super().load_memory_variables(inputs)
        if base_variables is None:
            base_variables = {}

        # Add PMM personality context
        pmm_context_parts = []

        if self.personality_context:
            pmm_context_parts.append(self.personality_context)

        if self.commitment_context:
            pmm_context_parts.append(self.commitment_context)

        # Add directive hierarchy context
        try:
            self.directive_system.get_directive_summary()

            # Add meta-principles
            meta_principles = self.directive_system.get_meta_principles()
            if meta_principles:
                mp_text = "Core Meta-Principles (evolutionary self-rules):\n"
                for mp in meta_principles[:3]:  # Top 3 most important
                    mp_text += f"• {mp['content']}\n"
                pmm_context_parts.append(mp_text)

            # Add active principles
            principles = self.directive_system.get_active_principles()
            if principles:
                p_text = "Active Guiding Principles:\n"
                for p in principles[:5]:  # Top 5 most important
                    p_text += f"• {p['content']}\n"
                pmm_context_parts.append(p_text)

            # Add active commitments with new format
            commitments = self.directive_system.get_active_commitments()
            if commitments:
                c_text = "Current Behavioral Commitments:\n"
                for c in commitments[:8]:  # Top 8 most recent
                    c_text += f"• {c['text']}\n"
                pmm_context_parts.append(c_text)

        except Exception as e:
            print(f"🔍 DEBUG: Failed to load directive context: {e}")
            # Fallback to legacy commitment loading
            pass

        # Load conversation history using hybrid approach: semantic + chronological
        try:
            if hasattr(self.pmm, "sqlite_store"):
                conversation_history = []
                key_facts = []  # Extract key facts like names

                # Get current input for semantic search
                current_input = inputs.get(self.input_key, "")

                # 1. Semantic search for relevant context (if we have input)
                if current_input and self.enable_embeddings:
                    semantic_memories = self._get_semantic_context(
                        current_input, max_results=6
                    )
                    if semantic_memories:
                        conversation_history.extend(semantic_memories)
                        conversation_history.append("---")  # Separator

                # 2. Recent chronological events for immediate context
                recent_events = self.pmm.sqlite_store.recent_events(limit=30)
                if recent_events:
                    for event in reversed(recent_events):  # chronological order
                        # Extract fields from dictionary (recent_events returns dicts via _row_to_dict)
                        kind = event.get("kind", "")
                        content = event.get("content", "")
                        summary = event.get("summary", "")
                        keywords = event.get("keywords", [])

                        # Prefer summary when available
                        display_text = summary or content or ""

                        if kind in ["event", "response", "prompt"]:
                            # Format for LLM context
                            if "User said:" in display_text:
                                user_msg = display_text.replace("User said: ", "")
                                conversation_history.append(f"Human: {user_msg}")

                                # Extract key information automatically
                                if (
                                    "my name is" in user_msg.lower()
                                    or "i am" in user_msg.lower()
                                ):
                                    key_facts.append(f"IMPORTANT: {user_msg}")

                            elif "I responded:" in display_text:
                                ai_msg = display_text.replace("I responded: ", "")
                                if not self._is_non_behavioral_input(ai_msg):
                                    conversation_history.append(f"Assistant: {ai_msg}")

                                    # Extract commitments and identity info
                                    if (
                                        "next, i will" in ai_msg.lower()
                                        or "scott" in ai_msg.lower()
                                    ):
                                        key_facts.append(
                                            f"COMMITMENT/IDENTITY: {ai_msg}"
                                        )

                            elif kind == "event":
                                conversation_history.append(f"Context: {display_text}")

                            # Include keywords when available (already parsed as list)
                            try:
                                if keywords and isinstance(keywords, list) and keywords:
                                    # Add a compact keywords line
                                    key_facts.append(
                                        "KEYWORDS: " + ", ".join(map(str, keywords[:6]))
                                    )
                            except Exception:
                                pass

                if conversation_history:
                    # Add key facts at the top for emphasis
                    if key_facts:
                        pmm_context_parts.append(
                            "Key Information to Remember:\n" + "\n".join(key_facts[-5:])
                        )

                    # Add recent conversation history
                    pmm_context_parts.append(
                        "Recent conversation history:\n"
                        + "\n".join(conversation_history[-15:])
                    )  # Last 15 exchanges
        except Exception as e:
            print(f"Warning: Failed to load conversation history from SQLite: {e}")

        # Combine PMM context with conversation history
        if pmm_context_parts:
            pmm_context = "\n\n".join(pmm_context_parts)

            # Inject personality context into the conversation
            if self.memory_key in base_variables:
                base_variables[self.memory_key] = (
                    f"{pmm_context}\n\n{base_variables[self.memory_key]}"
                )
            else:
                base_variables[self.memory_key] = pmm_context

        return base_variables

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

        print(f"🔍 DEBUG: Processing introspection command: {command_type.value}")

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
            print(f"🔍 DEBUG: Semantic context retrieval failed: {e}")
            return []

    def _is_similar_to_recent_insights(self, content: str) -> bool:
        """Check if content is too similar to recent insights using 0.88 threshold."""
        if not content or not content.strip():
            return True

        # Get recent insights for comparison
        recent_insights = self.pmm.model.self_knowledge.insights[
            -8:
        ]  # Last 8 insights for better dedup
        if not recent_insights:
            return False

        try:
            from openai import OpenAI

            client = OpenAI()

            # Get embedding for new content
            new_embedding = (
                client.embeddings.create(
                    input=content.strip(), model="text-embedding-ada-002"
                )
                .data[0]
                .embedding
            )

            # Compare with recent insights
            for insight in recent_insights:
                if not insight.content:
                    continue

                existing_embedding = (
                    client.embeddings.create(
                        input=insight.content.strip(), model="text-embedding-ada-002"
                    )
                    .data[0]
                    .embedding
                )

                # Calculate cosine similarity
                import numpy as np

                similarity = np.dot(new_embedding, existing_embedding) / (
                    np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
                )

                # Use 0.88 threshold as specified in requirements
                if similarity > 0.88:
                    print(
                        f"🔍 DEBUG: High similarity detected: {similarity:.3f} (threshold: 0.88)"
                    )
                    return True

        except Exception as e:
            print(f"🔍 DEBUG: Similarity check failed: {e}")
            # Fallback to simple text comparison
            for insight in recent_insights:
                if (
                    insight.content
                    and len(
                        set(content.lower().split())
                        & set(insight.content.lower().split())
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
            print(
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

        # Check reflection cooldown gates
        should_reflect, cooldown_reason = self.reflection_cooldown.should_reflect(
            current_context
        )
        if not should_reflect:
            print(f"🔍 DEBUG: Reflection blocked by cooldown - {cooldown_reason}")
            return None

        print(f"🔍 DEBUG: Reflection cooldown passed - {cooldown_reason}")

        # Run synchronously to avoid threading issues
        success = False
        try:
            # Generate insight with current model config
            insight_obj = reflect_once(self.pmm, None, active_model_config)

            if not insight_obj or not insight_obj.content:
                print("🔍 DEBUG: No insight generated")
                return None

            content = insight_obj.content.strip()
            if len(content) < 10:
                print("🔍 DEBUG: Insight too short, skipping")
                return None

            # Apply n-gram ban filtering
            model_name = active_model_config.get("name", "unknown")
            filtered_content, ban_replacements = self.ngram_ban.postprocess_style(
                content, model_name
            )
            if ban_replacements:
                print(f"🔍 DEBUG: N-gram ban applied: {ban_replacements}")
                content = filtered_content

            # Atomic reflection validation and persistence
            success = self.atomic_reflection.add_insight(
                content, active_model_config, active_model_config.get("epoch", 0)
            )
            if success:
                # Update baselines with current IAS/GAS
                emergence_context = self.pmm.get_emergence_context()
                if emergence_context:
                    ias = emergence_context.get("ias", 0.0)
                    gas = emergence_context.get("gas", 0.0)
                    self.model_baselines.add_scores(model_name, ias, gas)

                    # Calculate emergence profile
                    profile = self.emergence_stages.calculate_emergence_profile(
                        model_name, ias, gas
                    )
                    print(
                        f"🔍 DEBUG: Emergence stage: {profile.stage.value} (confidence: {profile.confidence:.2f})"
                    )

                print(f"🔍 DEBUG: Insight atomically persisted: {content[:100]}...")
                return content
            else:
                print("🔍 DEBUG: Insight rejected by atomic validation")
                return None

        except Exception as e:
            print(f"🔍 DEBUG: Reflection error: {e}")
            return None
        finally:
            print(f"🔍 DEBUG: Reflection completed, insight: {success}")

    @property
    def memory_variables(self) -> List[str]:
        """Return list of memory variables."""
        return [self.memory_key]

    def clear(self) -> None:
        """Clear conversation history but preserve personality."""
        super().clear()
        # Note: We don't clear PMM state - personality persists across conversations
