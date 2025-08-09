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
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from pydantic import Field

from .self_model_manager import SelfModelManager
from .reflection import reflect_once
from .llm import OpenAIClient
from .commitments import extract_commitments


class PersistentMindMemory(BaseChatMemory):
    """
    LangChain memory wrapper that provides persistent AI personality.
    
    This memory system goes beyond simple conversation history to maintain
    a persistent personality with evolving traits, commitments, and behavioral
    patterns that influence all interactions.
    """
    
    pmm: SelfModelManager = Field(exclude=True)
    personality_context: str = Field(default="")
    commitment_context: str = Field(default="")
    
    def __init__(
        self,
        agent_path: str,
        personality_config: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize PMM-powered memory system.
        
        Args:
            agent_path: Path to PMM agent file (created if doesn't exist)
            personality_config: Initial personality traits (Big Five scores 0-1)
            **kwargs: Additional LangChain memory arguments
        """
        super().__init__(**kwargs)
        
        # Initialize PMM system
        self.pmm = SelfModelManager(agent_path)
        
        # Set initial personality if provided
        if personality_config:
            self._apply_personality_config(personality_config)
        
        # Update context strings
        self._update_personality_context()
        self._update_commitment_context()
    
    def _apply_personality_config(self, config: Dict[str, float]) -> None:
        """Apply initial personality configuration to PMM agent."""
        big5_traits = ["openness", "conscientiousness", "extraversion", 
                      "agreeableness", "neuroticism"]
        
        for trait, score in config.items():
            if trait in big5_traits:
                setattr(
                    self.pmm.model.personality.traits.big5[trait], 
                    "score", 
                    max(0.0, min(1.0, score))
                )
        
        self.pmm.save_model()
    
    def _update_personality_context(self) -> None:
        """Generate personality context for LLM prompts."""
        traits = self.pmm.model.personality.traits.big5
        patterns = self.pmm.model.self_knowledge.behavioral_patterns
        
        context_parts = [
            f"Personality Profile (Big Five):",
            f"• Openness: {traits.openness.score:.2f} - {'Creative, curious' if traits.openness.score > 0.6 else 'Practical, conventional'}",
            f"• Conscientiousness: {traits.conscientiousness.score:.2f} - {'Organized, disciplined' if traits.conscientiousness.score > 0.6 else 'Flexible, spontaneous'}",
            f"• Extraversion: {traits.extraversion.score:.2f} - {'Outgoing, energetic' if traits.extraversion.score > 0.6 else 'Reserved, quiet'}",
            f"• Agreeableness: {traits.agreeableness.score:.2f} - {'Cooperative, trusting' if traits.agreeableness.score > 0.6 else 'Competitive, skeptical'}",
            f"• Neuroticism: {traits.neuroticism.score:.2f} - {'Anxious, sensitive' if traits.neuroticism.score > 0.6 else 'Calm, resilient'}"
        ]
        
        if patterns:
            context_parts.append(f"Behavioral Patterns: {', '.join(f'{k}({v})' for k, v in patterns.items())}")
        
        self.personality_context = "\n".join(context_parts)
    
    def _update_commitment_context(self) -> None:
        """Generate commitment context for LLM prompts."""
        try:
            open_commitments = self.pmm.get_open_commitments()
            if open_commitments:
                commitment_list = [f"• {c['text']}" for c in open_commitments[:3]]  # Show top 3
                self.commitment_context = f"Active Commitments:\n" + "\n".join(commitment_list)
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
        # Get human input and AI output
        human_input = inputs.get(self.input_key, "")
        ai_output = outputs.get(self.output_key, "")
        
        # Store conversation as PMM event
        if human_input:
            self.pmm.add_event(
                summary=f"User interaction: {human_input[:100]}...",
                effects=[]
            )
        
        if ai_output:
            # Add AI response as thought
            self.pmm.add_thought(ai_output, trigger="langchain_conversation")
            
            # Extract and track commitments
            try:
                commitments = extract_commitments(ai_output)
                for commitment_text in commitments:
                    self.pmm.add_commitment(
                        text=commitment_text,
                        source_insight_id="langchain_interaction"
                    )
            except Exception:
                pass
            
            # Update behavioral patterns based on conversation
            try:
                self.pmm.update_patterns(ai_output)
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
        """
        Load memory variables for LLM prompts.
        
        Returns personality context, commitments, and conversation history
        formatted for optimal LLM performance.
        """
        # Get base memory variables from LangChain
        base_variables = super().load_memory_variables(inputs)
        
        # Add PMM personality context
        pmm_context_parts = []
        
        if self.personality_context:
            pmm_context_parts.append(self.personality_context)
        
        if self.commitment_context:
            pmm_context_parts.append(self.commitment_context)
        
        # Combine PMM context with conversation history
        if pmm_context_parts:
            pmm_context = "\n\n".join(pmm_context_parts)
            
            # Inject personality context into the conversation
            if self.memory_key in base_variables:
                base_variables[self.memory_key] = f"{pmm_context}\n\n{base_variables[self.memory_key]}"
            else:
                base_variables[self.memory_key] = pmm_context
        
        return base_variables
    
    def get_personality_summary(self) -> Dict[str, Any]:
        """Get current personality state for debugging/monitoring."""
        return {
            "agent_id": self.pmm.model.core_identity.agent_id,
            "name": self.pmm.model.core_identity.name,
            "personality_traits": {
                trait: getattr(self.pmm.model.personality.traits.big5, trait).score
                for trait in ["openness", "conscientiousness", "extraversion", 
                             "agreeableness", "neuroticism"]
            },
            "behavioral_patterns": dict(self.pmm.model.self_knowledge.behavioral_patterns),
            "total_events": len(self.pmm.model.autobiographical_memory.events),
            "total_insights": len(self.pmm.model.self_knowledge.insights),
            "open_commitments": len(self.pmm.get_open_commitments()) if hasattr(self.pmm, 'get_open_commitments') else 0
        }
    
    def trigger_reflection(self) -> Optional[str]:
        """
        Manually trigger PMM reflection process.
        
        Returns the generated insight or None if reflection fails.
        """
        try:
            insight = reflect_once(self.pmm, OpenAIClient())
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
