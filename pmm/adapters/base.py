from abc import ABC, abstractmethod
from typing import List, Dict

Message = Dict[str, str]  # {"role": "user"|"assistant"|"system", "content": str}


class ModelAdapter(ABC):
    """Abstract base class for LLM model adapters."""

    @abstractmethod
    def generate(self, messages: List[Message], max_tokens: int = 512) -> str:
        """Generate response from messages with token limit."""
        pass
