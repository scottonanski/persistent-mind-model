# name_detect.py - Strict agent name detection to prevent false positives

import re
from datetime import datetime, timezone, timedelta
from typing import Optional

# Enhanced multilingual name pattern with code block filtering
NAME_PATTERN = re.compile(
    r"""(?ix)
    (?:^|[^\w])                              # boundary
    (?:
        my\s+name\s+is|
        call\s+me|
        you\s+can\s+call\s+me|
        your\s+name\s+is|
        let's\s+call\s+you|
        i'll\s+call\s+you|
        from\s+now\s+on,?\s+you\s+are|
        your\s+name\s+shall\s+be|
        i'm|
        i\s+am
    )\s+
    (?!not\b)                                # "my name is not"
    ["'""]?                                  # optional quote
    ([A-ZÀ-ÖØ-Ý][\w''\-]{0,62}(?:\s+[A-ZÀ-ÖØ-Ý][\w''\-]{1,30}(?!\s+(?:and|that|the|but|or|so|yet|for|nor|with|by|from|to|in|on|at|as|is|was|are|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might|can|must|shall)))?)  # Single or multi-word name, but not followed by common words
    ["'""]?
    (?=\s|[^\w]|$)                          # word boundary
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

    # Look for name declaration patterns
    match = NAME_PATTERN.search(text)
    if not match:
        return None

    # Extract the matched pattern to determine if it's assistant-only
    pattern_text = match.group(0).lower()
    assistant_only_patterns = ["i'm", "i am"]

    # Check speaker restrictions
    if any(pattern in pattern_text for pattern in assistant_only_patterns):
        # Assistant self-declaration patterns
        if speaker != "assistant":
            return None
    else:
        # User naming patterns - only allow from user
        if speaker != "user":
            return None

    # Extract and clean the name
    name = match.group(1).strip().rstrip(".,!?;:")

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

    # Filter out stopwords and invalid names
    if name.lower() in _STOPWORDS:
        return None

    # Additional validation
    if len(name) < 2 or len(name) > 63:
        return None

    # Must start with letter
    if not name[0].isalpha():
        return None

    return name


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
