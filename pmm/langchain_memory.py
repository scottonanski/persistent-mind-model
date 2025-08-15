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
from threading import Thread
from langchain.memory.chat_memory import BaseChatMemory
from pydantic import Field

from .self_model_manager import SelfModelManager
from .reflection import reflect_once
from .adapters.openai_adapter import OpenAIAdapter
from .commitments import CommitmentTracker


class PersistentMindMemory(BaseChatMemory):
    """
    LangChain memory wrapper that provides persistent AI personality.

    This memory system goes beyond simple conversation history to maintain
    a persistent personality with evolving traits, commitments, and behavioral
    patterns that influence all interactions.
    """

    pmm: SelfModelManager = Field(default=None, exclude=True)
    personality_context: str = Field(default="")
    commitment_context: str = Field(default="")
    memory_key: str = Field(default="history")
    input_key: str = Field(default="input")
    output_key: str = Field(default="response")
    conversation_count: int = Field(default=0)
    enable_summary: bool = Field(default=False)
    enable_embeddings: bool = Field(default=False)

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
        self.enable_summary = bool(enable_summary)
        self.enable_embeddings = bool(enable_embeddings)

        # Initialize personality if provided
        if personality_config:
            for trait, value in personality_config.items():
                if hasattr(self.pmm.model.personality.traits.big5, trait):
                    trait_obj = getattr(self.pmm.model.personality.traits.big5, trait)
                    trait_obj.score = max(0.0, min(1.0, float(value)))
            self.pmm.save_model()

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

        # Add recent memories and insights
        recent_events = self.pmm.model.self_knowledge.autobiographical_events[
            -3:
        ]  # Last 3 events
        if recent_events:
            context_parts.append("\nRecent Memories:")
            for event in recent_events:
                context_parts.append(f"• {event.summary}")

        recent_insights = self.pmm.model.self_knowledge.insights[-2:]  # Last 2 insights
        if recent_insights:
            context_parts.append("\nRecent Insights:")
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
                        if candidate.lower() not in stopwords:
                            _remember_user_name(candidate)

            # Detect agent name assignments and persist them
            agent_name_patterns = [
                r"your name is (\w+)",
                r"we will call you (\w+)",
                r"let's call you (\w+)",
                r"i'll call you (\w+)",
                r"you're (\w+)",
            ]

            for pattern in agent_name_patterns:
                match = re.search(pattern, user_lower)
                if match:
                    agent_name = match.group(1).title()
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

        # ---- 0.5) Input Hygiene: Check if input is non-behavioral ----
        is_non_behavioral = self._is_non_behavioral_input(human_input)
        if is_non_behavioral:
            print(
                "🔍 DEBUG: Non-behavioral input detected, skipping behavioral triggers"
            )

        # Store non-behavioral inputs for provenance but with special marking
        behavioral_input = human_input if not is_non_behavioral else ""

        # ---- 1) Log human event + auto‑extract key info ----
        if human_input:
            try:
                # Always log for provenance, but mark non-behavioral inputs
                event_type = "non_behavioral" if is_non_behavioral else "conversation"
                self.pmm.add_event(
                    summary=f"User said: {human_input[:200]}{'...' if len(human_input) > 200 else ''}",
                    effects=[],
                    etype=event_type,
                )

                # Only extract key info and auto-close from behavioral inputs
                if behavioral_input:
                    self._auto_extract_key_info(behavioral_input)

                    # Check for agent name adoption patterns
                    if self.pmm and any(
                        pattern in behavioral_input.lower()
                        for pattern in [
                            "call me",
                            "my name is",
                            "i am",
                            "i'm",
                            "adopt the name",
                            "officially adopt",
                        ]
                    ):
                        print(
                            f"🔍 DEBUG: Identity pattern detected in: {behavioral_input[:100]}..."
                        )
                        # Extract potential name from the conversation
                        import re

                        name_patterns = [
                            r"call me (\w+)",
                            r"my name is (\w+)",
                            r"i am (\w+)",
                            r"i'm (\w+)",
                            r"adopt(?:ing)? the name (\w+)",
                            r"officially adopt (?:the name )?(\w+)",
                        ]

                        for pattern in name_patterns:
                            match = re.search(pattern, behavioral_input.lower())
                            if match:
                                potential_name = match.group(1).capitalize()
                                if potential_name and len(potential_name) > 1:
                                    print(
                                        f"🔍 DEBUG: Attempting to set name to: {potential_name}"
                                    )
                                    try:
                                        self.pmm.set_name(
                                            potential_name, origin="conversation"
                                        )
                                        # Emit identity_update event for traceability
                                        self.pmm.add_event(
                                            summary=f"Identity update: Name changed to '{potential_name}' (origin=conversation)",
                                            effects=[],
                                            etype="identity_update",
                                        )
                                        print(
                                            f"🔍 DEBUG: Successfully set name to: {potential_name}"
                                        )
                                    except Exception as e:
                                        print(f"🔍 DEBUG: Failed to set name: {e}")
                                    break
                        else:
                            print("🔍 DEBUG: No name extracted from identity patterns")

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
            except Exception:
                pass

            # ---- 3) Commitments: extract + add from USER INPUT and AI RESPONSE ----
            # Skip commitment extraction and behavioral processing for non-behavioral inputs
            new_commitment_text = None
            if not is_non_behavioral:
                try:
                    tracker = CommitmentTracker()
                    # Check user input first
                    new_commitment_text, _ = tracker.extract_commitment(human_input)
                    if new_commitment_text:
                        print(
                            f"🔍 DEBUG: Found commitment in user input: {new_commitment_text}"
                        )
                        self.pmm.add_commitment(
                            text=new_commitment_text,
                            source_insight_id="langchain_interaction",
                        )
                    else:
                        # Check AI response for commitments
                        ai_commitment_text, _ = tracker.extract_commitment(ai_output)
                        if ai_commitment_text:
                            print(
                                f"🔍 DEBUG: Found commitment in AI response: {ai_commitment_text}"
                            )
                            self.pmm.add_commitment(
                                text=ai_commitment_text,
                                source_insight_id="langchain_interaction",
                            )
                            new_commitment_text = (
                                ai_commitment_text  # Set for reflection trigger
                            )
                        else:
                            print(
                                f"🔍 DEBUG: No commitment found in: {human_input[:100]}..."
                            )
                except Exception as e:
                    print(f"🔍 DEBUG: Commitment extraction error: {e}")
                    pass

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
                    from pmm.adaptive_triggers import AdaptiveTrigger, TriggerConfig, TriggerState
                    
                    # Build state from PMM
                    events_total = len(self.pmm.model.self_knowledge.autobiographical_events)
                    last_ref_ts = getattr(self.pmm.model.self_knowledge, "last_reflection_ts", None)
                    last_dt = None
                    if isinstance(last_ref_ts, str):
                        try:
                            last_dt = datetime.fromisoformat(last_ref_ts)
                        except:
                            last_dt = None
                    elif hasattr(last_ref_ts, 'isoformat'):
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
                        cursor = conn.execute("""
                            SELECT id, ts, kind, content, meta 
                            FROM events 
                            WHERE kind IN ('response', 'event', 'reflection', 'evidence', 'commitment')
                            ORDER BY ts DESC 
                            LIMIT 15
                        """)
                        rows = cursor.fetchall()
                        
                        # Convert to EmergenceEvent objects
                        events = []
                        for row in rows:
                            import json
                            meta = {}
                            try:
                                meta = json.loads(row[4]) if row[4] else {}
                            except:
                                pass
                            events.append(EmergenceEvent(
                                id=row[0], timestamp=row[1], kind=row[2], 
                                content=row[3], meta=meta
                            ))
                        
                        # Create analyzer and override get_recent_events like probe API
                        analyzer = EmergenceAnalyzer(storage_manager=store)
                        analyzer.get_recent_events = lambda kind="response", limit=15: events
                        
                        scores = analyzer.compute_scores(window=15)  # Use Phase 3C window size
                        ias = scores.get("IAS", 0.0)
                        gas = scores.get("GAS", 0.0)
                        print(f"🔍 DEBUG: Real-time emergence scores - IAS: {ias}, GAS: {gas}")
                        print(f"🔍 DEBUG: Events analyzed: {scores.get('events_analyzed', 0)}, Stage: {scores.get('stage', 'Unknown')}")
                    except Exception as e:
                        print(f"🔍 DEBUG: Failed to calculate emergence scores: {e}")
                        import traceback
                        print(f"🔍 DEBUG: Traceback: {traceback.format_exc()}")
                        ias = None
                        gas = None

                    # Determine events since last reflection
                    events_since_reflection = getattr(self.pmm.model.self_knowledge, "events_since_reflection", None)
                    if events_since_reflection is None:
                        # fallback heuristic: use total events modulo
                        events_since_reflection = events_total % 6

                    # Respect config reflection_cadence_days if present
                    cadence_days = getattr(self.pmm.model.metrics, "reflection_cadence_days", 7.0)

                    trigger = AdaptiveTrigger(
                        TriggerConfig(
                            cadence_days=cadence_days,
                            events_min_gap=4,     # tune in Phase 3C tests
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
                        print(f"🔍 DEBUG: Commitment trigger - new commitment: {new_commitment_text}")
                    else:
                        should_reflect, reason = trigger.decide(datetime.utcnow(), ias, gas, events_since_reflection)

                    print(f"🔍 DEBUG: Adaptive trigger decision: {should_reflect} ({reason})")
                    print(f"🔍 DEBUG: Events since reflection: {events_since_reflection}, IAS: {ias}, GAS: {gas}")
                    
                except Exception as e:
                    print(f"🔍 DEBUG: Adaptive reflection trigger check failed: {e}")
                    # Fallback to simple event count trigger
                    behavioral_events = [
                        e for e in self.pmm.model.self_knowledge.autobiographical_events
                        if e.type != "non_behavioral"
                    ]
                    event_count = len(behavioral_events)
                    should_reflect = (event_count > 0 and event_count % 4 == 0) or bool(new_commitment_text)
            else:
                print("🔍 DEBUG: Skipping reflection triggers for non-behavioral input")

            if should_reflect:
                print("🔍 DEBUG: Reflection triggered, starting...")
                try:
                    insight = self.trigger_reflection()
                    print(f"🔍 DEBUG: Reflection completed, insight: {bool(insight)}")
                except Exception as e:
                    print(f"🔍 DEBUG: Reflection failed: {e}")
                    insight = None

                if insight:
                    print(
                        f"\n🧠 Insight: {insight[:160]}{'...' if len(insight) > 160 else ''}"
                    )
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
                        self.pmm.model.self_knowledge.last_reflection_ts = datetime.utcnow().isoformat()
                        self.pmm.model.self_knowledge.events_since_reflection = 0
                        print("🔍 DEBUG: Updated reflection bookkeeping for adaptive triggers")
                    except Exception as e:
                        print(f"🔍 DEBUG: Failed to update reflection bookkeeping: {e}")
                else:
                    # Increment events_since_reflection counter even if reflection failed
                    try:
                        cur = getattr(self.pmm.model.self_knowledge, "events_since_reflection", 0) or 0
                        self.pmm.model.self_knowledge.events_since_reflection = int(cur) + 1
                    except Exception:
                        pass

        # ---- 6) Refresh prompt contexts ----
        self._update_personality_context()
        self._update_commitment_context()

        # ---- 7) Persist PMM ----
        self.pmm.save_model()

        # ---- 8) Keep LangChain compatibility ----
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

        # Load recent conversation history from SQLite database
        try:
            if hasattr(self.pmm, "sqlite_store"):
                # Load more events to capture key information like names
                recent_events = self.pmm.sqlite_store.recent_events(limit=50)
                if recent_events:
                    conversation_history = []
                    key_facts = []  # Extract key facts like names

                    for event in reversed(recent_events):  # chronological order
                        # Support both legacy 7-column and new 10-column schemas
                        event_id, ts, kind, content, meta, prev_hash, hash_val = event[
                            :7
                        ]
                        summary = None
                        keywords = None
                        try:
                            summary = event[7]
                        except Exception:
                            summary = None
                        try:
                            keywords = event[8]
                        except Exception:
                            keywords = None
                        # embedding at event[9] if present (unused here)

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

                            # Include keywords when available (JSON string)
                            try:
                                if keywords:
                                    import json as _json

                                    kw_list = _json.loads(keywords)
                                    if isinstance(kw_list, list) and kw_list:
                                        # Add a compact keywords line
                                        key_facts.append(
                                            "KEYWORDS: "
                                            + ", ".join(map(str, kw_list[:6]))
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
            "behavioral_patterns": dict(
                self.pmm.model.self_knowledge.behavioral_patterns
            ),
            "total_events": len(self.pmm.model.self_knowledge.autobiographical_events),
            "total_insights": len(self.pmm.model.self_knowledge.insights),
            "open_commitments": len(self.pmm.model.self_knowledge.commitments),
        }

    def trigger_reflection(self) -> Optional[str]:
        """
        Manually trigger PMM reflection process with a short timeout to avoid UI stalls.

        Returns the generated insight content or None if reflection times out or fails.
        """
        result: Dict[str, Optional[Any]] = {"insight": None, "error": None}

        def _worker():
            try:
                print("🔍 DEBUG: Worker thread starting reflect_once...")
                insight_obj = reflect_once(self.pmm, OpenAIAdapter())
                print(
                    f"🔍 DEBUG: reflect_once returned: {type(insight_obj)} - {bool(insight_obj)}"
                )
                result["insight"] = insight_obj
            except Exception as e:
                print(f"🔍 DEBUG: Worker thread exception: {e}")
                result["error"] = str(e)
                result["insight"] = None

        try:
            print("🔍 DEBUG: Starting reflection worker thread...")
            t = Thread(target=_worker, daemon=True)
            t.start()
            t.join(timeout=15.0)  # Increased timeout for gpt-4o-mini

            if t.is_alive():
                print("🔍 DEBUG: Reflection timed out after 15 seconds")
                return None

            print(f"🔍 DEBUG: Worker completed. Error: {result.get('error')}")
            insight = result.get("insight")
            print(f"🔍 DEBUG: Insight object: {type(insight)} - {bool(insight)}")

            if insight:
                print(f"🔍 DEBUG: Insight content length: {len(insight.content)}")
                self._update_personality_context()
                self._update_commitment_context()
                return insight.content
            else:
                print("🔍 DEBUG: No insight generated")
        except Exception as e:
            print(f"🔍 DEBUG: trigger_reflection exception: {e}")
        return None

    @property
    def memory_variables(self) -> List[str]:
        """Return list of memory variables."""
        return [self.memory_key]

    def clear(self) -> None:
        """Clear conversation history but preserve personality."""
        super().clear()
        # Note: We don't clear PMM state - personality persists across conversations
