from __future__ import annotations
import re
from typing import Optional


DONE_RX = re.compile(r"\bDone: (.+)", re.IGNORECASE)
URL_RX = re.compile(r"https?://\S+")
EV_RX = re.compile(r"\bev\d+\b", re.IGNORECASE)


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
        m = DONE_RX.search(text)
        if not m:
            return None
        summary = m.group(1).strip()[:240]
        # artifact: prefer URL then evID
        art = None
        murl = URL_RX.search(text)
        if murl:
            art = murl.group(0)
        else:
            mev = EV_RX.search(text)
            if mev:
                art = mev.group(0)
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
        content = {
            "type": "done",
            "summary": summary,
            "artifact": art,
            "confidence": 0.65,
        }
        res = smm.sqlite_store.append_event(kind="evidence", content=content, meta=meta)
        return res.get("hash")
    except Exception:
        return None

