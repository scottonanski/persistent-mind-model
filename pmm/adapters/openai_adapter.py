import os
import time
from typing import List
from .base import ModelAdapter, Message

# openai v1 client
from openai import OpenAI

_OPENAI_TIMEOUT = 30

class OpenAIAdapter(ModelAdapter):
    """OpenAI adapter with retry logic and timeout handling."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = 3
        self.timeout = _OPENAI_TIMEOUT

    def generate(self, messages: List[Message], max_tokens: int = 512) -> str:
        """Generate response using OpenAI API with retry logic."""
        print(f"[API] Calling OpenAI with prompt: {messages[-1]['content'][:60]}...")
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=self.timeout
                )
                result = response.choices[0].message.content.strip()
                print(f"[API] Response received: {len(result)} chars")
                return result
            except Exception as e:
                print(f"[API] Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        raise RuntimeError("All retry attempts failed")
