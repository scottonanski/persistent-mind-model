import sys
from typing import Literal

Level = Literal["info", "warn", "error", "debug"]


def log(level: Level, msg: str) -> None:
    """Write a tagged system notice to stderr.

    Example output: [pmm][info] message
    """
    try:
        sys.stderr.write(f"[pmm][{level}] {msg}\n")
        sys.stderr.flush()
    except Exception:
        # Never let logging crash the app
        pass
