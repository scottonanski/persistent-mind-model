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
                    print(f" Persisted agent name change to: {agent_name}")
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

        # ---- 1) Log human event + auto‑extract key info ----
        if human_input:
            try:
                self.pmm.add_event(
                    summary=f"User said: {human_input[:200]}{'...' if len(human_input) > 200 else ''}",
                    effects=[],
                    etype="conversation",
                )
                self._auto_extract_key_info(human_input)
                # NEW: auto‑close commitments from human message
                try:
                    self.pmm.auto_close_commitments_from_event(human_input)
                except Exception:
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

            # ---- 3) Commitments: extract + add; auto‑close from assistant output ----
            new_commitment_text = None
            try:
                tracker = CommitmentTracker()
                new_commitment_text, _ = tracker.extract_commitment(ai_output)
                if new_commitment_text:
                    self.pmm.add_commitment(
                        text=new_commitment_text,
                        source_insight_id="langchain_interaction",
                    )
            except Exception:
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

            # ---- 5) Reflection triggers ----
            # (a) cadence: every 4 events (~2 back‑and‑forths)
            # (b) immediate after creating a new commitment (planning reflection)
            should_reflect = False
            try:
                event_count = len(self.pmm.model.self_knowledge.autobiographical_events)
                print(
                    f"🔍 DEBUG: Event count: {event_count}, new_commitment: {bool(new_commitment_text)}"
                )
                if event_count > 0 and event_count % 4 == 0:
                    print(
                        f"🔍 DEBUG: Cadence trigger - event count {event_count} divisible by 4"
                    )
                    should_reflect = True
                if new_commitment_text:
                    print(
                        f"🔍 DEBUG: Commitment trigger - new commitment: {new_commitment_text}"
                    )
                    should_reflect = True
                print(f"🔍 DEBUG: Should reflect: {should_reflect}")
            except Exception as e:
                print(f"🔍 DEBUG: Reflection trigger check failed: {e}")

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

        # ---- 6) Refresh prompt contexts ----
        self._update_personality_context()
        self._update_commitment_context()

        # ---- 7) Persist PMM ----
        self.pmm.save_model()

        # ---- 8) Keep LangChain compatibility ----
        super().save_context(inputs, outputs)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        Load memory variables for LLM prompts.

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
                                conversation_history.append(f"Assistant: {ai_msg}")

                                # Extract commitments and identity info
                                if (
                                    "next, i will" in ai_msg.lower()
                                    or "scott" in ai_msg.lower()
                                ):
                                    key_facts.append(f"COMMITMENT/IDENTITY: {ai_msg}")

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
                                "Key Information to Remember:\n"
                                + "\n".join(key_facts[-5:])
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
