import requests
import json
from typing import Optional


class OllamaClient:
    """Ollama client that implements the same interface as OpenAIClient for model-agnostic PMM."""
    
    def __init__(
        self,
        model: str = "llama3.1:8b",
        base: str = "http://localhost:11434",
        temp: float = 0.4,
        timeout: int = 60
    ):
        self.model = model
        self.base = base
        self.temp = temp
        self.timeout = timeout
    
    def chat(self, system: str, user: str) -> str:
        """Chat with Ollama model using the same interface as OpenAIClient."""
        url = f"{self.base}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "options": {
                "temperature": self.temp
            },
            "stream": False
        }
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("message", {}).get("content", "").strip()
            
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base}. "
                "Make sure Ollama is running with: ollama serve"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")


class HuggingFaceClient:
    """HuggingFace client for local transformers models."""
    
    def __init__(
        self,
        model: str = "microsoft/DialoGPT-medium",
        temp: float = 0.4,
        max_length: int = 512
    ):
        self.model = model
        self.temp = temp
        self.max_length = max_length
        self._pipeline = None
    
    def _get_pipeline(self):
        """Lazy load the transformers pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    temperature=self.temp,
                    max_length=self.max_length
                )
            except ImportError:
                raise RuntimeError(
                    "transformers library not installed. "
                    "Install with: pip install transformers torch"
                )
        return self._pipeline
    
    def chat(self, system: str, user: str) -> str:
        """Chat with HuggingFace model using the same interface."""
        pipeline = self._get_pipeline()
        
        # Combine system and user messages for text generation models
        prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:"
        
        try:
            result = pipeline(prompt, max_length=self.max_length, do_sample=True)
            response = result[0]["generated_text"]
            
            # Extract just the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            raise RuntimeError(f"HuggingFace model error: {e}")


# Factory function for easy model switching
def create_llm_client(provider: str = "openai", **kwargs):
    """Factory function to create LLM clients."""
    if provider.lower() == "openai":
        from .llm import OpenAIClient
        return OpenAIClient(**kwargs)
    elif provider.lower() == "ollama":
        return OllamaClient(**kwargs)
    elif provider.lower() == "huggingface":
        return HuggingFaceClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
