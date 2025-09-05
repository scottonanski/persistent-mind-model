# name_detect.py - Strict agent name detection to prevent false positives

from datetime import datetime, timezone, timedelta
from typing import Optional

# Phrase lists for structural detection (no regex)
USER_OWN_NAME_CUES = [
    "my name is ",
    "call me ",
    "you can call me ",
]

ASSISTANT_SELF_NAME_CUES = USER_OWN_NAME_CUES + [
    "i am ",
    "i'm ",
    "i’m ",
]

AGENT_RENAME_CUES = [
    "your name is ",
    "let's call you ",
    "lets call you ",
    "i'll call you ",
    "i’ll call you ",
    "from now on, you are ",
    "from now on you are ",
    "your name shall be ",
]


def _strip_code_blocks(text: str) -> str:
    """Remove fenced and inline code blocks without regex."""
    if not text:
        return ""
    s = str(text)
    out_chars = []
    i = 0
    n = len(s)
    in_fence = False
    in_inline = False
    while i < n:
        # Detect triple backtick fence
        if not in_inline and s.startswith("```", i):
            in_fence = not in_fence
            i += 3
            continue
        # Detect single backtick inline code
        if not in_fence and s[i] == "`":
            in_inline = not in_inline
            i += 1
            continue
        if not in_fence and not in_inline:
            out_chars.append(s[i])
        i += 1
    return "".join(out_chars)


def _extract_name_after_phrase(tail: str) -> Optional[str]:
    """Extract a candidate name from the tail following a cue phrase.

    Stops at common delimiters and validates capitalization and token rules.
    """
    if not tail:
        return None
    # Disallow immediate negation (e.g., "not Scott")
    if tail.lstrip().lower().startswith("not "):
        return None
    # Cut at first delimiter
    cut_tail = tail
    for delim in [".", "!", "?", ",", ";", ":", "\n", "\r"]:
        pos = cut_tail.find(delim)
        if pos != -1:
            cut_tail = cut_tail[:pos]
            break
    # Also cut at common conjunction markers introducing follow-up clauses
    low_ct = " " + cut_tail.lower() + " "
    for marker in [" and ", " but ", " so ", " then ", " because ", " that "]:
        mpos = low_ct.find(marker)
        if mpos != -1:
            # Map back to original string index; mpos is in the padded low string
            real_pos = max(0, mpos - 1)
            cut_tail = cut_tail[:real_pos]
            break
    candidate = cut_tail.strip().strip("'\" ")
    if not candidate:
        return None
    # Limit tokens and enforce capitalization/alphabetic
    parts = [p for p in candidate.split() if p]
    if not (1 <= len(parts) <= 3):
        return None
    for p in parts:
        if not p.isalpha() or not p[0].isupper():
            return None
    return " ".join(parts)


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
    text = _strip_code_blocks(text)

    name: Optional[str] = None

    # Lower/normalize for cue scanning while preserving original for capitalization
    raw = text
    low = text.lower()

    # If this is a user message, allow extraction for user's own name via
    # 'my name is', 'call me', 'you can call me'; ignore "I'm/I am"
    if speaker == "user":
        for cue in USER_OWN_NAME_CUES:
            pos = low.find(cue)
            if pos != -1:
                tail = raw[pos + len(cue) :]
                name = _extract_name_after_phrase(tail)
                if name:
                    break

    # Only accept agent-directed rename patterns from the user (unless we already
    # extracted user's own name via strict self-intro above)
    if speaker == "user":
        if name is None:
            for cue in AGENT_RENAME_CUES:
                pos = low.find(cue)
                if pos != -1:
                    tail = raw[pos + len(cue) :]
                    name = _extract_name_after_phrase(tail)
                    if name:
                        break
            if name is None:
                return None
    # Allow assistant self-declaration (e.g., "I'm Echo", "I am Echo")
    elif speaker == "assistant":
        for cue in ASSISTANT_SELF_NAME_CUES:
            pos = low.find(cue)
            if pos != -1:
                tail = raw[pos + len(cue) :]
                name = _extract_name_after_phrase(tail)
                if name:
                    break
        if name is None:
            return None
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
    # Also cut at common conjunctions introducing follow-ups (e.g., "and that's final")
    cut_markers = {" and ", " but ", " so ", " then ", " because ", " that "}
    low_name_scan = " " + name.lower() + " "
    cut_pos = min(
        [low_name_scan.find(m) for m in cut_markers if m in low_name_scan]
        or [len(low_name_scan)]
    )
    if cut_pos < len(low_name_scan):
        name = name[: max(0, cut_pos - 1)].strip()
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
