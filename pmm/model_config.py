# pmm/model_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelConfig:
    """Immutable model configuration for consistent LLM behavior."""

    provider: str  # "openai" | "ollama"
    name: str
    family: str  # "gpt" | "gemma" | "llama" | "qwen" | "claude"
    version: str = "unknown"  # "3.3" | "4o-mini" | etc
    epoch: int = 0  # monotonically increasing on change
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: Optional[int] = None  # ollama only
    seed: Optional[int] = 42
    max_tokens: Optional[int] = 1024
    stop: Tuple[str, ...] = ()
    system_prompt: Optional[str] = None

    def __post_init__(self):
        """Validate configuration on creation."""
        if self.provider not in ("openai", "ollama"):
            raise ValueError(f"Unsupported provider: {self.provider}")
        if not self.name:
            raise ValueError("Model name cannot be empty")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"Temperature must be 0.0-2.0, got {self.temperature}")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be 0.0-1.0, got {self.top_p}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider,
            "name": self.name,
            "family": self.family,
            "version": self.version,
            "epoch": self.epoch,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "stop": list(self.stop),
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Create from dictionary."""
        return cls(
            provider=data["provider"],
            name=data["name"],
            family=data.get("family", "unknown"),
            version=data.get("version", "unknown"),
            epoch=data.get("epoch", 0),
            temperature=data.get("temperature", 0.3),
            top_p=data.get("top_p", 0.9),
            top_k=data.get("top_k"),
            seed=data.get("seed", 42),
            max_tokens=data.get("max_tokens", 1024),
            stop=tuple(data.get("stop", [])),
            system_prompt=data.get("system_prompt"),
        )

    def get_model_key(self) -> str:
        """Get unique key for this model configuration."""
        return f"{self.provider}:{self.name}"

    def is_compatible(self, other: "ModelConfig") -> bool:
        """Check if two configs are compatible (same model, different params OK)."""
        return self.provider == other.provider and self.name == other.name


# Common model configurations
COMMON_CONFIGS = {
    "gpt-4": ModelConfig(
        provider="openai",
        name="gpt-4",
        family="gpt",
        version="4",
        temperature=0.3,
        max_tokens=1024,
        seed=42,
    ),
    "gpt-4o-mini": ModelConfig(
        provider="openai",
        name="gpt-4o-mini",
        family="gpt",
        version="4o-mini",
        temperature=0.3,
        max_tokens=1024,
        seed=42,
    ),
    "gemma2:9b": ModelConfig(
        provider="ollama",
        name="gemma2:9b",
        family="gemma",
        version="2:9b",
        temperature=0.3,
        top_k=40,
        max_tokens=1024,
        seed=42,
    ),
    "gemma3:4b": ModelConfig(
        provider="ollama",
        name="gemma3:4b",
        family="gemma",
        version="3:4b",
        temperature=0.3,
        top_k=40,
        max_tokens=1024,
        seed=42,
    ),
    "llama3:8b": ModelConfig(
        provider="ollama",
        name="llama3:8b",
        family="llama",
        version="3:8b",
        temperature=0.3,
        top_k=40,
        max_tokens=1024,
        seed=42,
    ),
}
