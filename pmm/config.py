#!/usr/bin/env python3
"""
Centralized configuration management for Persistent Mind Model.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    api_key: str = ""
    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    base_url: Optional[str] = None


@dataclass
class DriftConfig:
    """Configuration for personality drift."""

    maturity_principle: bool = True
    inertia: float = 0.9
    max_delta_per_reflection: float = 0.02
    cooldown_days: int = 7
    event_sensitivity: float = 0.4
    bounds_min: float = 0.05
    bounds_max: float = 0.95
    locks: List[str] = field(default_factory=list)


@dataclass
class ReflectionConfig:
    """Configuration for reflection system."""

    cadence_days: int = 7
    max_context_events: int = 10
    max_context_thoughts: int = 5
    enable_provenance: bool = True


@dataclass
class PersistenceConfig:
    """Configuration for data persistence."""

    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 10
    atomic_writes: bool = True
    validation_enabled: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    log_file: Optional[str] = None
    enable_file_logging: bool = False
    max_log_size_mb: int = 10
    backup_count: int = 5


@dataclass
class PMMConfig:
    """Master configuration for Persistent Mind Model."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_env(cls) -> "PMMConfig":
        """Load configuration from environment variables."""
        config = cls()

        # LLM configuration
        config.llm.api_key = os.getenv("OPENAI_API_KEY", "")
        config.llm.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        config.llm.max_tokens = int(os.getenv("PMM_MAX_TOKENS", "1000"))
        config.llm.temperature = float(os.getenv("PMM_TEMPERATURE", "0.7"))
        config.llm.timeout = int(os.getenv("PMM_TIMEOUT", "30"))
        config.llm.max_retries = int(os.getenv("PMM_MAX_RETRIES", "3"))

        # Drift configuration
        config.drift.max_delta_per_reflection = float(
            os.getenv("PMM_MAX_DELTA", "0.02")
        )
        config.drift.inertia = float(os.getenv("PMM_INERTIA", "0.9"))
        config.drift.event_sensitivity = float(
            os.getenv("PMM_EVENT_SENSITIVITY", "0.4")
        )

        # Reflection configuration
        config.reflection.cadence_days = int(os.getenv("PMM_REFLECTION_CADENCE", "7"))
        config.reflection.max_context_events = int(
            os.getenv("PMM_MAX_CONTEXT_EVENTS", "10")
        )

        # Logging configuration
        config.logging.level = os.getenv("PMM_LOG_LEVEL", "INFO")
        config.logging.log_file = os.getenv("PMM_LOG_FILE")
        config.logging.enable_file_logging = (
            os.getenv("PMM_FILE_LOGGING", "false").lower() == "true"
        )

        return config

    def validate(self) -> None:
        """Validate configuration values."""
        errors = []

        if not self.llm.api_key:
            errors.append("LLM API key is required")

        if not 0 <= self.llm.temperature <= 2:
            errors.append("LLM temperature must be between 0 and 2")

        if not 0 < self.drift.max_delta_per_reflection <= 1:
            errors.append("Drift max_delta must be between 0 and 1")

        if not 0 <= self.drift.inertia <= 1:
            errors.append("Drift inertia must be between 0 and 1")

        if self.drift.bounds_min >= self.drift.bounds_max:
            errors.append("Drift bounds_min must be less than bounds_max")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")


# Global configuration instance
_config: Optional[PMMConfig] = None


def get_config() -> PMMConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = PMMConfig.from_env()
        _config.validate()
    return _config


def set_config(config: PMMConfig) -> None:
    """Set the global configuration instance."""
    global _config
    config.validate()
    _config = config
