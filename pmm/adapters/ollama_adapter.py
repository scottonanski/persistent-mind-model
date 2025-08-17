"""
Ollama Adapter for PMM Reflection System

Provides a consistent interface for Ollama models to be used in PMM reflection
and other LLM-dependent operations.
"""

from typing import Optional
from langchain_ollama import OllamaLLM


class OllamaAdapter:
    """Adapter to provide consistent interface for Ollama models in PMM."""

    def __init__(self, model: str = "gemma3:4b", temperature: float = 0.7):
        """Initialize Ollama adapter with specified model."""
        self.model = model
        self.temperature = temperature
        self.llm = OllamaLLM(model=model, temperature=temperature)

    def chat(self, system: str, user: str) -> Optional[str]:
        """
        Generate response using Ollama model.

        Args:
            system: System prompt/instructions
            user: User message/query

        Returns:
            Generated response text or None if failed
        """
        try:
            # Format prompt for Ollama (single string format)
            formatted_prompt = f"System: {system}\n\nUser: {user}\n\nAssistant: "
            response = self.llm.invoke(formatted_prompt)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            print(f"Ollama adapter error: {e}")
            return None
