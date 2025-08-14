"""
PersistentMindMemory (LangChain adapter)

Purpose:
- Persist explicit, provenance-backed identity/event summaries to PMM.
- Return a concise, provenance-first context block (facts + recent turns)
  for use by an LLM.

Notes:
- No heuristic inference. Only store explicit identity statements.
- No external APIs required. Reflection gated/disabled by default.
"""

from typing import Any, Dict, List, Optional
import sys
from threading import Thread
from langchain.memory.chat_memory import BaseChatMemory
from pydantic import Field

from .self_model_manager import SelfModelManager
from .reflection import reflect_once
from .adapters.openai_adapter import OpenAIAdapter
from .commitments import CommitmentTracker

# Reflection is present but disabled by default to keep this module local-only
REFLECTION_ENABLED = False

def _log(level: str, msg: str) -> None:
    print(f"[pmm][{level}] {msg}", file=sys.stderr)


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

    def __init__(
        self, agent_path: str, personality_config: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the LangChain-compatible memory wrapper.

        Args:
            agent_path: Path to save/load the agent's persistent state
            personality_config: Optional initial personality configuration
        """
        super().__init__()
        self.pmm = SelfModelManager(agent_path)

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
        Store ONLY explicit identity statements with provenance.
        Patterns:
          - "My name is X"
          - "Call me X"
        """
        try:
            raw = user_input.strip()

            import re
            name_pats = [
                r"\bMy name is ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b",
                r"\bCall me ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b",
            ]

            def _store_user_name(name: str) -> None:
                if not name:
                    return
                tokens = [t for t in name.strip().split() if t]
                if not (1 <= len(tokens) <= 3):
                    return
                if any((not t.isalpha()) or (not t[0].isupper()) for t in tokens):
                    return
                clean = " ".join(tokens)
                self.pmm.add_event(
                    summary=f"User’s name is {clean}.",
                    effects=[],
                    etype="identity_info",
                    full_text=raw,
                    tags=["identity","name"],
                )
                _log("info", f"remembered explicit identity: name={clean}")

            for pat in name_pats:
                m = re.search(pat, raw)
                if m:
                    _store_user_name(m.group(1))
                    break

            # No role/job/relationship heuristics. No "I'm/I am" capture.
        except Exception:
            # Do not throw from auto-extraction
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
        1. Stores the conversation as PMM events
        2. Extracts and tracks commitments from responses
        3. Updates behavioral patterns
        4. Triggers personality evolution if needed
        """
        # Get human input and AI output - handle various LangChain key formats
        human_input = (
            inputs.get(self.input_key, "")
            or inputs.get("input", "")
            or inputs.get("question", "")
        )

        # LangChain ConversationChain typically uses the first available output value
        ai_output = ""
        if outputs:
            # Try common output keys
            ai_output = (
                outputs.get(self.output_key, "")
                or outputs.get("response", "")
                or outputs.get("text", "")
                or outputs.get("answer", "")
            )

            # If no standard keys, get the first value
            if not ai_output and outputs:
                ai_output = list(outputs.values())[0] if outputs.values() else ""

        # Store conversation as PMM event
        if human_input:
            try:
                # Add user input as an autobiographical event
                self.pmm.add_event(
                    summary=f"User said: {human_input[:100]}{'...' if len(human_input) > 100 else ''}",
                    effects=[],
                    etype="conversation",
                )

                # Automatically extract and remember key information
                self._auto_extract_key_info(human_input)

                pass  # Event added successfully
            except Exception:
                pass  # Silently handle errors in production

        if ai_output:
            try:
                # Add AI response as thought
                self.pmm.add_thought(ai_output, trigger="langchain_conversation")

                # Add AI response as an event too
                self.pmm.add_event(
                    summary=f"I responded: {ai_output[:100]}{'...' if len(ai_output) > 100 else ''}",
                    effects=[],
                    etype="self_expression",
                )
            except Exception:
                pass  # Silently handle errors in production

            # Extract and track commitments
            try:
                tracker = CommitmentTracker()
                commitment_text, _ = tracker.extract_commitment(ai_output)
                if commitment_text:
                    self.pmm.add_commitment(
                        text=commitment_text, source_insight_id="langchain_interaction"
                    )
            except Exception:
                pass

            # Update behavioral patterns based on conversation
            try:
                self.pmm.update_patterns(ai_output)
            except Exception:
                pass

            # Trigger reflection if we have enough events (every 3 interactions)
            try:
                event_count = len(self.pmm.model.self_knowledge.autobiographical_events)
                if (
                    event_count > 0 and event_count % 6 == 0
                ):  # Every 3 back-and-forth exchanges
                    insight = self.trigger_reflection()
                    if insight:
                        print(f"\n🧠 Generated insight: {insight[:100]}...")
            except Exception:
                pass

        # Update context for next interaction
        self._update_personality_context()
        self._update_commitment_context()

        # Save PMM state
        self.pmm.save_model()

        # Call parent save_context for LangChain compatibility
        super().save_context(inputs, outputs)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return a concise, provenance-first PMM context under `history`."""
        base = super().load_memory_variables(inputs) or {}
        pmm_context_parts: List[str] = []

        if self.personality_context:
            pmm_context_parts.append(self.personality_context)
        if self.commitment_context:
            pmm_context_parts.append(self.commitment_context)

        facts: List[str] = []
        recent_lines: List[str] = []
        try:
            if hasattr(self.pmm, "sqlite_store"):
                rows = self.pmm.sqlite_store.recent_events(limit=120)
                rows = list(reversed(rows))  # chronological
                import json as _json
                for (eid, ts, kind, content, meta, prev_hash, hash_val) in rows:
                    m = {}
                    if meta:
                        try:
                            m = _json.loads(meta) if isinstance(meta, str) else (meta or {})
                        except Exception:
                            m = {}
                    etype = m.get("type") or m.get("etype")
                    summary = (content or "")[:200]
                    if etype == "identity_info":
                        facts.append(f"- E{eid} ({ts}): {summary}")
                    if kind in ("event", "response", "prompt"):
                        if summary.startswith("User said:"):
                            recent_lines.append("Human: " + summary.replace("User said: ", ""))
                        elif summary.startswith("I responded:"):
                            recent_lines.append("Assistant: " + summary.replace("I responded: ", ""))
                        else:
                            recent_lines.append("Context: " + summary)

                if facts:
                    pmm_context_parts.append("PMM FACTS (provenance)\n" + "\n".join(facts[-8:]))
                if recent_lines:
                    pmm_context_parts.append("Recent conversation\n" + "\n".join(recent_lines[-12:]))
        except Exception as e:
            _log("warn", f"load_memory_variables: {e}")

        if pmm_context_parts:
            block = "\n\n".join(pmm_context_parts)
            if self.memory_key in base and base[self.memory_key]:
                base[self.memory_key] = block + "\n\n" + base[self.memory_key]
            else:
                base[self.memory_key] = block
        return base

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
        Manually trigger PMM reflection process. Disabled by default.

        Returns the generated insight content or None.
        """
        if not REFLECTION_ENABLED:
            return None
        result: Dict[str, Optional[Any]] = {"insight": None}

        def _worker():
            try:
                result["insight"] = reflect_once(self.pmm, OpenAIAdapter())
            except Exception:
                result["insight"] = None

        try:
            t = Thread(target=_worker, daemon=True)
            t.start()
            t.join(timeout=8.0)  # prevent blocking the chat loop
            if t.is_alive():
                return None

            insight = result.get("insight")
            if insight:
                self._update_personality_context()
                self._update_commitment_context()
                return insight.content
        except Exception:
            pass
        return None

    @property
    def memory_variables(self) -> List[str]:
        """Return list of memory variables."""
        return [self.memory_key]

    def clear(self) -> None:
        """Clear conversation history but preserve personality."""
        super().clear()
        # Note: We don't clear PMM state - personality persists across conversations
