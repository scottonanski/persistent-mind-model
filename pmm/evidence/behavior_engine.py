from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from urllib.parse import urlparse
from pathlib import PurePosixPath
from pmm.semantic_analysis import get_semantic_analyzer

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _best_open_commitment_token(smm, reply_text: str) -> Tuple[Optional[str], Optional[dict]]:
    """Return (commit_ref, commit_dict) via simple token overlap without regex."""
    try:
        opens = smm.get_open_commitments() or []
        if not opens:
            return None, None
        rt = set((reply_text or "").lower().split())
        best = (0.0, None, None)
        for c in opens:
            ct = set((c.get("text", "") or "").lower().split())
            if not ct:
                continue
            inter = len(rt & ct)
            score = inter / max(1, len(ct))
            if score > best[0]:
                best = (score, c.get("hash"), c)
        return best[1], best[2]
    except Exception:
        return None, None


def _best_open_commitment_semantic(
    smm, reply_text: str
) -> Tuple[Optional[str], Optional[dict], float]:
    """Return (commit_ref, commit_dict, similarity) using semantic similarity."""
    try:
        opens = smm.get_open_commitments() or []
        if not opens:
            return None, None, 0.0
        analyzer = get_semantic_analyzer()
        best = (0.0, None, None)
        for c in opens:
            ctext = (c.get("text", "") or "").strip()
            if not ctext:
                continue
            sim = analyzer.cosine_similarity(reply_text, ctext)
            if sim > best[0]:
                best = (sim, c.get("hash"), c)
        return best[1], best[2], best[0]
    except Exception:
        return None, None, 0.0


def _parse_artifact(text: str) -> Optional[str]:
    """Parse a likely artifact using standard parsers only (no regex).

    - Prefer valid HTTP/HTTPS URIs via urllib.parse
    - Else detect file-like tokens via pathlib suffix
    - Else detect bare commit/event/issue tokens if they look like ids (simple tokens with digits)
    """
    if not isinstance(text, str) or not text:
        return None
    # Check URI
    try:
        parsed = urlparse(text.strip())
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return text.strip()
    except Exception:
        pass
    # Look through whitespace tokens for a plausible path
    for tok in text.split():
        try:
            p = PurePosixPath(tok)
            if p.suffix:
                return tok
        except Exception:
            continue
        # Detect simple ids like '#123' or 'EV123' without regex: check digits presence
        if any(ch.isdigit() for ch in tok):
            return tok
    return None


def _noop_learn(*args, **kwargs) -> None:
    # Removed exemplar phrase learning to comply with no literal phrases policy
    return None


def process_reply_for_evidence(smm, reply_text: str) -> Optional[str]:
    """Semantic-only evidence mapping.

    - Map reply to nearest open commitment via cosine similarity; require threshold.
    - Parse artifact via urllib.parse/pathlib only.
    - Emit a single 'evidence' event with meta.commit_ref if mapped.
    - Do NOT auto-close commitments here; closure is inferred elsewhere from evidence linkages.
    """
    try:
        text = (reply_text or "").strip()
        if not text:
            return None
        # Map to nearest open commitment by semantic similarity, else token overlap
        s_cref, s_cdict, s_sim = _best_open_commitment_semantic(smm, text)
        t_cref, t_cdict = _best_open_commitment_token(smm, text)
        sim_thresh = _env_float("PMM_EVIDENCE_SEM_THRESHOLD", 0.58)
        commit_ref = None
        if s_cref and (s_sim >= sim_thresh):
            commit_ref = s_cref
        elif t_cref:
            commit_ref = t_cref
        if not commit_ref:
            return None
        # Artifact parsing
        art = _parse_artifact(text)
        # Confidence: monotone mapping from semantic similarity with small artifact boost
        base = max(0.0, min(1.0, (s_sim if s_cref == commit_ref else 0.0)))
        # Map [sim_thresh..1] -> [0.6..0.9]
        if base > 0.0:
            span = max(1e-6, 1.0 - sim_thresh)
            base = 0.6 + (max(0.0, base - sim_thresh) / span) * 0.3
        conf = min(0.95, base + (0.05 if art else 0.0))
        summary = (text.splitlines()[0] if text else "").strip()[:240]
        meta = {"type": "done", "commit_ref": commit_ref}
        content = {
            "type": "done",
            "summary": summary,
            "artifact": art,
            "confidence": round(conf, 2),
        }
        res = smm.sqlite_store.append_event(
            kind="evidence", content=json.dumps(content, ensure_ascii=False), meta=meta
        )
        # No auto-close here; authoritative closure is evidence-linked in analyzer
        _noop_learn()
        return res.get("hash")
    except Exception:
        return None
