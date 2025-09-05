import os


def semantic_mode_enabled() -> bool:
    """Return True when semantic relaxations are enabled via env flag.

    Enable with: PMM_SEMANTIC_MODE=on|true|1|yes (case-insensitive)
    Default is off for test-first behavior.
    """
    return str(os.getenv("PMM_SEMANTIC_MODE", "off")).strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )
