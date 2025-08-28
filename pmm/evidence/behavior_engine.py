from __future__ import annotations
import re
import json
from typing import Optional


DONE_RX = re.compile(r"\bDone:\s*(.+)", re.IGNORECASE)
SYN_RX = re.compile(
    r"\b(completed|implemented|fixed|merged|published|shipped|closed|deployed|uploaded|documented|refactored)\b",
    re.IGNORECASE,
)
URL_RX = re.compile(r"https?://\S+")
EV_RX = re.compile(r"\bev\d+\b", re.IGNORECASE)
PR_RX = re.compile(r"(#\d+|[A-Z]{2,}-\d+)")
FILE_RX = re.compile(r"\b[\w/\.-]+\.(py|md|txt|json|yaml|yml|ipynb)\b")


def process_reply_for_evidence(smm, reply_text: str) -> Optional[str]:
    """Best-effort evidence detector.

    Emits a minimal 'evidence' event when it spots a 'Done:' marker, with an
    artifact if a URL or evID is present. If an open commitment exists with a
    canonical hash, attach it via meta.commit_ref. Returns the event hash if
    written.
    """
    try:
        text = (reply_text or "").strip()
        if not text:
            return None
        # Primary: explicit Done: summary
        m = DONE_RX.search(text)
        summary = m.group(1).strip()[:240] if m else None

        # Secondary: completion synonyms — use first sentence as summary
        if not summary and SYN_RX.search(text):
            # Take up to the first 240 chars of the line containing the synonym
            for line in text.splitlines():
                if SYN_RX.search(line):
                    summary = line.strip()[:240]
                    break
        if not summary:
            return None

        # artifact: URL, PR/issue id, evID, or file path
        art = None
        for rx in (URL_RX, PR_RX, EV_RX, FILE_RX):
            mrx = rx.search(text)
            if mrx:
                art = mrx.group(0)
                break
        # try to grab an open commitment hash for commit_ref
        commit_ref = None
        try:
            opens = smm.get_open_commitments() or []
            if opens:
                commit_ref = opens[0].get("hash") or None
        except Exception:
            commit_ref = None
        meta = {"type": "done"}
        if commit_ref:
            meta["commit_ref"] = commit_ref
        # Confidence: base 0.65 + 0.1 if artifact present, capped at 0.9
        conf = 0.65 + (0.1 if art else 0.0)
        conf = min(0.9, conf)
        content = {
            "type": "done",
            "summary": summary,
            "artifact": art,
            "confidence": round(conf, 2),
        }
        res = smm.sqlite_store.append_event(
            kind="evidence", content=json.dumps(content, ensure_ascii=False), meta=meta
        )
        return res.get("hash")
    except Exception:
        return None
