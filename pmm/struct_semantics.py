#!/usr/bin/env python3
"""
Structural and semantic utilities with zero keyword/regex heuristics.

- Sentence splitter: period/newline segmentation without regex
- POS tagging: best-effort using available lightweight libraries; falls back to unknown tags
- CentroidModel: vector-only positive/negative centroids and scoring
"""
from __future__ import annotations
from typing import List, Tuple, Sequence, Optional

# Avoid regex and keyword lists entirely


def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    # Simple segmentation: split on newlines first, then on periods; keep minimal trimming
    out: List[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        # naive period segmentation; do not use regex
        cur = []
        for ch in block:
            cur.append(ch)
            if ch == ".":
                s = "".join(cur).strip()
                if s:
                    out.append(s)
                cur = []
        if cur:
            s = "".join(cur).strip()
            if s:
                out.append(s)
    return out


def pos_tag(text: str) -> List[Tuple[str, str]]:
    """
    Best-effort POS tagging without relying on lexeme lists in code.
    Tries spaCy or NLTK if available; otherwise returns tokens with 'X'.
    """
    toks: List[str] = [t for t in (text or "").strip().split() if t]
    # Try spaCy
    try:
        import spacy  # type: ignore

        try:
            nlp = spacy.load("en_core_web_sm")  # type: ignore
        except Exception:
            # lightweight blank model with tagger if available
            nlp = spacy.blank("en")  # type: ignore
        doc = nlp(" ".join(toks))
        out = []
        for t in doc:
            tg = t.tag_
            if not tg:
                tg = "X"
            out.append((t.text, tg))
        return out
    except Exception:
        pass
    # Try NLTK
    try:
        import nltk  # type: ignore

        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except Exception:
            pass
        return list(nltk.pos_tag(toks))  # type: ignore
    except Exception:
        pass
    # Fallback: unknown tags
    return [(t, "X") for t in toks]


class CentroidModel:
    def __init__(self):
        self._pos: Optional[List[float]] = None
        self._neg: Optional[List[float]] = None

    @staticmethod
    def _mean(vectors: Sequence[Sequence[float]]) -> Optional[List[float]]:
        vecs = [list(v) for v in vectors if isinstance(v, (list, tuple)) and len(v) > 0]
        if not vecs:
            return None
        n = len(vecs)
        dim = len(vecs[0])
        acc = [0.0] * dim
        for v in vecs:
            if len(v) != dim:
                continue
            for i in range(dim):
                acc[i] += float(v[i])
        return [x / max(1, n) for x in acc]

    @staticmethod
    def _cos(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        la = len(a)
        lb = len(b)
        if la == 0 or lb == 0:
            return 0.0
        dim = min(la, lb)
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(dim):
            ai = float(a[i])
            bi = float(b[i])
            dot += ai * bi
            na += ai * ai
            nb += bi * bi
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / ((na**0.5) * (nb**0.5))

    def fit(
        self,
        pos_vectors: Sequence[Sequence[float]],
        neg_vectors: Sequence[Sequence[float]],
    ) -> None:
        self._pos = self._mean(pos_vectors)
        self._neg = self._mean(neg_vectors)

    def score(self, vec: Sequence[float]) -> float:
        # score = cos(vec, posC) - cos(vec, negC)
        if not isinstance(vec, (list, tuple)):
            return 0.0
        pos = self._pos or []
        neg = self._neg or []
        return self._cos(vec, pos) - self._cos(vec, neg)


# --- Structural, regex-free helpers -------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and trim."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def _strip_punct(tok: str) -> str:
    """Strip leading/trailing punctuation around a token."""
    if not tok:
        return tok
    return tok.strip("\"'\t\r\n.,;:!?()[]{}")


def detect_ev_ids_and_hashes(text: str) -> set[str]:
    """Extract event IDs like 'ev123' and 16-char lowercase hex hashes from text without regex."""
    if not text:
        return set()
    low = text.lower()
    refs: set[str] = set()

    # Scan tokens for 'ev' + digits and 16-char hex
    for raw in low.split():
        tok = _strip_punct(raw)
        if not tok:
            continue
        # ev + digits
        if tok.startswith("ev") and tok[2:].isdigit():
            refs.add(tok)
            continue
        # 16-char lowercase hex
        if len(tok) == 16 and all(c in "0123456789abcdef" for c in tok):
            refs.add(tok)

    # Fallback: simple linear scan to catch ev123 inside punctuation without spaces
    n = len(low)
    i = 0
    while i < n - 2:
        if low[i] == "e" and low[i + 1] == "v":
            j = i + 2
            while j < n and low[j].isdigit():
                j += 1
            if j > i + 2:
                refs.add(low[i:j])
                i = j
                continue
        i += 1
    return refs


def has_banned_coachlike(text: str) -> bool:
    """Deprecated: keyword/phrase-based blocking removed.

    No keyword lists or regex are permitted. This function now returns False
    unconditionally to comply with the global no-keyword policy.
    """
    return False


def detect_event_numbers(text: str) -> list[str]:
    """Detect references like 'event 123' or 'event_123' (case-insensitive).

    Returns a list of canonical strings like 'event_123'.
    """
    if not text:
        return []
    low = text.lower()
    out: list[str] = []
    tokens = low.replace("\n", " ").split()
    for i, raw in enumerate(tokens):
        tok = _strip_punct(raw)
        if tok == "event" and i + 1 < len(tokens):
            nxt = _strip_punct(tokens[i + 1])
            if nxt.isdigit():
                out.append(f"event_{nxt}")
        elif tok.startswith("event_"):
            suf = tok.split("_", 1)[-1]
            if suf.isdigit():
                out.append(f"event_{suf}")
        elif tok.startswith("event") and len(tok) > 5:
            # handle 'event123' without separator
            suf = tok[5:]
            if suf.isdigit():
                out.append(f"event_{suf}")
    return out


def parse_identity_name_change(content: str) -> tuple[str | None, str | None]:
    """Deprecated: identity phrase parsing removed.

    No keyword lists or exemplar phrases are allowed. This now returns (None, None)
    and callers should instead derive identity from structural metadata or vectors.
    """
    return (None, None)


def user_opted_out_of_reflection(text: str) -> bool:
    """Deprecated: explicit opt-out phrase detection removed.

    To satisfy the no-keyword requirement, this always returns False. If
    opt-out is required, implement a structural control path outside of text.
    """
    return False
