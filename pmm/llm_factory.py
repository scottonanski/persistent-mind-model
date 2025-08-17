# pmm/llm_factory.py
from __future__ import annotations
from typing import Optional, Any
import threading
from .model_config import ModelConfig


class LLMFactory:
    """Unified factory for creating LLM instances with epoch-based invalidation."""

    def __init__(self):
        self._epoch = 0
        self._lock = threading.Lock()
        self._cache = {}
        self._active_config = None

    def increment_epoch(self) -> None:
        """Increment epoch to invalidate in-flight operations on model switch."""
        with self._lock:
            self._epoch += 1
            self._cache.clear()
            print(
                f"🔄 LLM Factory: Epoch incremented to {self._epoch} (invalidating stale operations)"
            )

    def get_current_epoch(self) -> int:
        """Get current epoch for validation."""
        with self._lock:
            return self._epoch

    def set_active_config(self, config) -> None:
        """Set the active model configuration."""
        with self._lock:
            if hasattr(config, "provider"):
                # ModelConfig object
                self._active_config = {
                    "provider": config.provider,
                    "name": config.name,
                    "family": getattr(config, "family", "unknown"),
                    "version": getattr(config, "version", "unknown"),
                    "epoch": self._epoch,
                }
            else:
                # Dictionary
                self._active_config = {
                    "provider": config.get("provider"),
                    "name": config.get("name"),
                    "family": config.get("family", "unknown"),
                    "version": config.get("version", "unknown"),
                    "epoch": config.get("epoch", self._epoch),
                }

    def get_active_config(self) -> Optional[dict]:
        """Get the active model configuration."""
        with self._lock:
            return self._active_config.copy() if self._active_config else None

    def get(self, config: ModelConfig) -> Any:
        """Get LLM instance with validation."""
        if not config.name or not config.provider:
            raise ValueError(
                f"Invalid ModelConfig: provider={config.provider}, name={config.name}"
            )

        cache_key = f"{config.provider}:{config.name}:epoch={self._epoch}"

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        llm = self.resolve_llm(config)

        with self._lock:
            self._cache[cache_key] = llm

        return llm

    def resolve_llm(self, config: ModelConfig) -> Any:
        """Create LLM instance from configuration."""
        cache_key = (config.provider, config.name, config.temperature, config.seed)

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        if config.provider == "ollama":
            llm = self._create_ollama_llm(config)
        elif config.provider == "openai":
            llm = self._create_openai_llm(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

        with self._lock:
            self._cache[cache_key] = llm

        return llm

    def _create_ollama_llm(self, config: ModelConfig) -> Any:
        """Create Ollama LLM instance."""
        try:
            from langchain_ollama import OllamaLLM
        except ImportError:
            # Fallback to older import
            from langchain.llms import Ollama as OllamaLLM

        # Map common params to Ollama "options"
        options = {
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.top_k is not None:
            options["top_k"] = config.top_k
        if config.seed is not None:
            options["seed"] = config.seed
        if config.max_tokens is not None:
            options["num_predict"] = config.max_tokens
        if config.stop:
            options["stop"] = list(config.stop)

        return OllamaLLM(model=config.name, **options)

    def _create_openai_llm(self, config: ModelConfig) -> Any:
        """Create OpenAI LLM instance."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from langchain.chat_models import ChatOpenAI

        kwargs = {
            "model": config.name,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
        }

        if config.stop:
            kwargs["stop"] = list(config.stop)

        return ChatOpenAI(**kwargs)


class EpochGuardedOperation:
    """Context manager for epoch-guarded operations."""

    def __init__(self, factory: LLMFactory, operation_name: str = "operation"):
        self.factory = factory
        self.operation_name = operation_name
        self.start_epoch = None
        self.result = None

    def __enter__(self):
        self.start_epoch = self.factory.get_current_epoch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_epoch = self.factory.get_current_epoch()
        if self.start_epoch != current_epoch:
            print(
                f"⚠️  {self.operation_name} invalidated: epoch {self.start_epoch} → {current_epoch}"
            )
            self.result = None

    def is_valid(self) -> bool:
        """Check if operation is still valid."""
        return self.start_epoch == self.factory.get_current_epoch()

    def set_result(self, result: Any) -> None:
        """Set result if operation is still valid."""
        if self.is_valid():
            self.result = result
        else:
            print(f"⚠️  {self.operation_name} result discarded due to epoch mismatch")
            self.result = None


# Global factory instance
_factory = LLMFactory()


def get_llm_factory() -> LLMFactory:
    """Get global LLM factory instance."""
    return _factory


def resolve_llm(config: ModelConfig) -> Any:
    """Convenience function to resolve LLM from config."""
    return _factory.resolve_llm(config)


def on_model_switch() -> None:
    """Call when model configuration changes to invalidate stale operations."""
    _factory.increment_epoch()


def create_epoch_guard(operation_name: str = "operation") -> EpochGuardedOperation:
    """Create epoch guard for long-running operations."""
    return EpochGuardedOperation(_factory, operation_name)
