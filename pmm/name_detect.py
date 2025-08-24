# name_detect.py - Strict agent name detection to prevent false positives

import re
from datetime import datetime, timezone, timedelta
from typing import Optional

# Enhanced patterns with separation of user self-intro vs agent-directed rename
# 1a) Strict user self-introduction: only "my name is ..." (extraction only, no rename)
USER_MY_NAME_IS_PATTERN = re.compile(
    r"""(?ix)
    (?:^|[^\w])
    my\s+name\s+is\s+
    (?!not\b)
    ["'""]?
    ([A-ZÀ-ÖØ-Ý][\w''\-]{0,62}(?:\s+[A-ZÀ-ÖØ-Ý][\w''\-]{1,30})?)
    ["'""]?
    (?=\s|[^\w]|$)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# 1b) User-own-name phrases (for user self-identification only, not agent rename)
USER_SELF_OWN_NAME_PATTERN = re.compile(
    r"""(?ix)
    (?:^|[^\w])
    (?:
        my\s+name\s+is|
        call\s+me|
        you\s+can\s+call\s+me
    )\s+
    (?!not\b)
    ["'""]?
    ([A-ZÀ-ÖØ-Ý][\w''\-]{0,62}(?:\s+[A-ZÀ-ÖØ-Ý][\w''\-]{1,30})?)
    ["'""]?
    (?=\s|[^\w]|$)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# 1c) General self-introduction phrases (used for assistant self-declaration only)
USER_SELF_NAME_PATTERN = re.compile(
    r"""(?ix)
    (?:^|[^\w])
    (?:
        my\s+name\s+is|
        call\s+me|
        you\s+can\s+call\s+me|
        i'm|
        i\s+am
    )\s+
    (?!not\b)
    ["'""]?
    ([A-ZÀ-ÖØ-Ý][\w''\-]{0,62}(?:\s+[A-ZÀ-ÖØ-Ý][\w''\-]{1,30})?)
    ["'""]?
    (?=\s|[^\w]|$)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# 2) Agent-directed rename commands (ONLY these should rename the agent)
AGENT_RENAME_PATTERN = re.compile(
    r"""(?ix)
    (?:^|[^\w])
    (?:
        your\s+name\s+is|
        let's\s+call\s+you|
        i['’]ll\s+call\s+you|
        from\s+now\s+on,?\s+you\s+are|
        your\s+name\s+shall\s+be
    )\s+
    (?!not\b)
    ["'""]?
    ([A-ZÀ-ÖØ-Ý][\w''\-]{0,62}(?:\s+[A-ZÀ-ÖØ-Ý][\w''\-]{1,30})?)
    ["'""]?
    (?=\s|[^\w]|$)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Pattern to detect and remove code blocks
CODE_FENCE_PATTERN = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")

# Stopwords to filter out common false positives (reduced set, more focused)
_STOPWORDS = {
    "asking",
    "calling",
    "going",
    "trying",
    "working",
    "looking",
    "thinking",
    "feeling",
    "being",
    "doing",
    "making",
    "getting",
    "having",
    "saying",
    "coming",
    "running",
    "walking",
    "talking",
    "reading",
    "writing",
    "playing",
    "learning",
    "teaching",
    "helping",
    "building",
    "creating",
    "developing",
    "testing",
    "debugging",
    "coding",
    "programming",
    "analyzing",
    "studying",
    "not",
    "none",
    "nothing",
    "nobody",
    "nowhere",
    "never",
    "no",
    "false",
    "true",
    "yes",
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "certainly",
    "definitely",
    "absolutely",
    "exactly",
    "precisely",
    "specifically",
    "generally",
    "usually",
    "normally",
    "typically",
    "commonly",
    "frequently",
    "often",
    "sometimes",
    "occasionally",
    "rarely",
    "seldom",
    "hardly",
    "barely",
    "almost",
    "nearly",
    "quite",
    "very",
    "extremely",
    "incredibly",
    "amazingly",
    "surprisingly",
    "interestingly",
    "unfortunately",
    "fortunately",
    "just",
    "continuing",
    "also",
    "still",
    "even",
    "only",
    "really",
    "actually",
}

# Export for backward compatibility
STOPWORDS = _STOPWORDS


def extract_agent_name_command(text: str, speaker: str) -> Optional[str]:
    """
    Enhanced name extraction with multilingual support and code block filtering.
    Returns a validated agent-name to set, or None.
    `speaker` is "user" or "assistant".
    """
    if not text:
        return None

    # Strip code/log blocks to avoid capturing from traces
    text = CODE_FENCE_PATTERN.sub(" ", text)
    text = INLINE_CODE_PATTERN.sub(" ", text)

    name: Optional[str] = None

    # If this is a user message, allow extraction for user's own name via
    # 'my name is', 'call me', 'you can call me'; ignore "I'm/I am"
    if speaker == "user" and USER_SELF_OWN_NAME_PATTERN.search(text):
        match_own = USER_SELF_OWN_NAME_PATTERN.search(text)
        if match_own:
            name = match_own.group(1).strip().rstrip(".,!?;:")

    # Only accept agent-directed rename patterns from the user (unless we already
    # extracted user's own name via strict self-intro above)
    if speaker == "user":
        if name is None:
            match = AGENT_RENAME_PATTERN.search(text)
            if not match:
                return None
            name = match.group(1).strip().rstrip(".,!?;:")
    # Allow assistant self-declaration (e.g., "I'm Echo", "I am Echo")
    elif speaker == "assistant":
        match = USER_SELF_NAME_PATTERN.search(text)
        if not match:
            return None
        name = match.group(1).strip().rstrip(".,!?;:")
    else:
        return None

    # Remove common trailing words that shouldn't be part of names
    common_words = [
        "and",
        "that",
        "the",
        "but",
        "or",
        "so",
        "yet",
        "for",
        "nor",
        "with",
        "by",
        "from",
        "to",
        "in",
        "on",
        "at",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "must",
        "shall",
    ]
    name_parts = name.split()
    while name_parts and name_parts[-1].lower() in common_words:
        name_parts.pop()
    name = " ".join(name_parts)

    # Token-level validation and normalization
    tokens = [t for t in name.split() if t]
    if not (1 <= len(tokens) <= 3):
        return None

    # Reject leading negations like "not Scott" or "no Echo"
    if tokens and tokens[0].lower() in {"not", "no"}:
        return None

    # All tokens must be alphabetic and not stopwords by themselves
    for t in tokens:
        if not t.isalpha():
            return None
        if t.lower() in _STOPWORDS:
            return None

    # Preserve original casing while normalizing whitespace
    normalized = " ".join(tokens)

    # Length and starting letter checks on normalized form
    if len(normalized) < 2 or len(normalized) > 63:
        return None
    if not normalized[0].isalpha():
        return None

    return normalized


def _utcnow_str():
    """Helper to get current UTC time as ISO string"""
    from datetime import datetime, timezone

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
