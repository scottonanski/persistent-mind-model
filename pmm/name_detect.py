# name_detect.py - Strict agent name detection to prevent false positives

import re
from datetime import datetime, timezone, timedelta

# very common non-names you saw in logs + generic adverbs/verbs
_STOPWORDS = {
    "just",
    "continuing",
    "trying",
    "well",
    "okay",
    "ok",
    "hey",
    "hello",
    "thanks",
    "today",
    "tomorrow",
    "yesterday",
    "now",
    "sure",
}

# Core patterns:
#  (1) User explicitly naming the AGENT: "your name is X", "I'll call you X", "let's call you X"
#  (2) Agent self-declaration: "I am X" / "I'm X" — but only when speaker == assistant
_PATTERNS_USER_COMMAND = re.compile(
    r"""(?ix)
        (?:your\ name\ (?:is|shall\ be)\s+|
           let'?s\ call\ you\s+|
           i'?ll\ call\ you\s+|
           from\ now\ on[, ]*\s*(?:you|your\ name)\s*(?:are|is)\s+)
        (?P<name>[A-Z][A-Za-z .'\-]{1,63}?)(?=\s|$|[.!?,:;])
    """
)

_PATTERNS_ASSISTANT_SELF = re.compile(
    r"""(?ix)
        (?:^|\s)(?:i\ am|i'm)\s+(?P<name>[A-Z][A-Za-z .'\-]{1,63}?)(?=\s|$|[.!?,:;])
    """
)


def extract_agent_name_command(text: str, speaker: str) -> str | None:
    """
    Returns a validated agent-name to set, or None.
    `speaker` is "user" or "assistant".
    """
    if not text:
        return None

    # 1) user explicitly names the agent
    if speaker == "user":
        m = _PATTERNS_USER_COMMAND.search(text)
        if m:
            cand = m.group("name").strip().strip(".")
            if cand.lower() in _STOPWORDS:
                return None
            return cand
        return None

    # 2) assistant self-declaration
    if speaker == "assistant":
        m = _PATTERNS_ASSISTANT_SELF.search(text)
        if m:
            cand = m.group("name").strip().strip(".")
            if cand.lower() in _STOPWORDS:
                return None
            return cand

    return None


def _utcnow_str():
    """Helper to get current UTC time as ISO string"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _too_soon_since_last_name_change(last_change_at: str | None, days: int = 1) -> bool:
    """Check if it's too soon since the last name change (cooldown period)"""
    if not last_change_at:
        return False
    try:
        dt_last = datetime.strptime(last_change_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except Exception:
        return False
    return datetime.now(timezone.utc) < dt_last + timedelta(days=days)
