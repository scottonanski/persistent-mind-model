#!/usr/bin/env python3
"""
Centralized logging configuration for Persistent Mind Model.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Configure structured logging for PMM components."""

    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Get root logger for PMM
    logger = logging.getLogger("pmm")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for specific PMM components."""
    return logging.getLogger(f"pmm.{name}")


# Component-specific loggers
persistence_logger = get_logger("persistence")
reflection_logger = get_logger("reflection")
drift_logger = get_logger("drift")
validation_logger = get_logger("validation")
llm_logger = get_logger("llm")


# Minimal telemetry print helper (stderr) guarded by PMM_TELEMETRY
def pmm_tlog(*args, **kwargs):
    """Minimal telemetry print when PMM_TELEMETRY is truthy."""
    try:
        flag = os.getenv("PMM_TELEMETRY", "")
        if flag:
            print(*args, **kwargs, file=sys.stderr, flush=True)
    except Exception:
        # Never let telemetry printing crash runtime
        pass
