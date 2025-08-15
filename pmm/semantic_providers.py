# pmm/semantic_providers.py
from typing import List, Protocol
import os

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers to maintain model-agnostic architecture."""
    def embed_text(self, text: str) -> List[float]: ...

class OpenAIEmbeddingProvider:
    """OpenAI embedding provider using text-embedding-3-small by default."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = None  # Lazy initialization
        
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self.client is None:
            try:
                from openai import OpenAI
                import os
                # Load environment variables if not already loaded
                from dotenv import load_dotenv
                load_dotenv()
                
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                    
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self.client
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API."""
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * 1536  # text-embedding-3-small dimension
        
        try:
            client = self._get_client()
            response = client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: OpenAI embedding failed: {e}")
            # Return zero vector on failure
            return [0.0] * 1536

class LocalEmbeddingProvider:
    """Local embedding provider using sentence-transformers (placeholder for future)."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        # TODO: Implement sentence-transformers integration
        # from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(model_name)
        
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using local model (placeholder)."""
        # TODO: Implement actual local embedding
        # return self.model.encode(text).tolist()
        
        # For now, return zero vector as placeholder
        return [0.0] * 384  # all-MiniLM-L6-v2 dimension

def get_embedding_provider(provider_type: str = "openai") -> EmbeddingProvider:
    """Factory function to get embedding provider based on configuration."""
    provider_type = provider_type.lower()
    
    if provider_type == "openai":
        return OpenAIEmbeddingProvider()
    elif provider_type == "local":
        return LocalEmbeddingProvider()
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")

# Default provider based on environment
def get_default_provider() -> EmbeddingProvider:
    """Get default embedding provider based on environment configuration."""
    provider_type = os.getenv("PMM_EMBEDDING_PROVIDER", "openai")
    return get_embedding_provider(provider_type)
