from __future__ import annotations
import re
import json
from typing import Optional, List, Tuple


DONE_RX = re.compile(r"\bDone:\s*(.+)", re.IGNORECASE)
# Broadened completion cues so user never needs to type "Done"
SYN_RX = re.compile(
    r"\b(completed|implemented|fixed|merged|published|shipped|closed|deployed|uploaded|documented|refactored|adopted|renamed|set up|configured)\b",
    re.IGNORECASE,
)
URL_RX = re.compile(r"https?://\S+")
EV_RX = re.compile(r"\bev\d+\b", re.IGNORECASE)
PR_RX = re.compile(r"(#\d+|[A-Z]{2,}-\d+)")
FILE_RX = re.compile(r"\b[\w/\.-]+\.(py|md|txt|json|yaml|yml|ipynb)\b")

# Identity adoption patterns (e.g., "I am now officially Quest", "my name is now Quest")
IDENTITY_NAME_RXES: List[re.Pattern] = [
    re.compile(r"\bmy\s+name\s+is\s+(?:now\s+)?['\"]?(?P<name>[A-Z][\w\- ]{1,40})['\"]?", re.IGNORECASE),
    re.compile(r"\bi\s*(?:am|'m)\s+now\s+(?:officially\s+)?(?P<name>[A-Z][\w\- ]{1,40})\b", re.IGNORECASE),
    re.compile(r"\bofficially\s+(?:named|name)\s+['\"]?(?P<name>[A-Z][\w\- ]{1,40})['\"]?", re.IGNORECASE),
]


def _best_open_commitment(smm, reply_text: str) -> Tuple[Optional[str], Optional[dict]]:
    """Return (commit_ref, commit_dict) with highest token overlap vs reply_text."""
    try:
        opens = smm.get_open_commitments() or []
        if not opens:
            return None, None
        # simple token overlap score
        import re as _re

        def toks(s: str) -> set:
            return set(t for t in _re.split(r"\W+", (s or "").lower()) if t)

        rt = toks(reply_text)
        best = (0.0, None, None)
        for c in opens:
            ct = toks(c.get("text", ""))
            if not ct:
                continue
            inter = len(rt & ct)
            score = inter / max(1, len(ct))
            if score > best[0]:
                best = (score, c.get("hash"), c)
        return best[1], best[2]
    except Exception:
        return None, None


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
        # Specialized: identity adoption as evidence
        identity_name = None
        for rx in IDENTITY_NAME_RXES:
            m = rx.search(text)
            if m and m.group("name"):
                identity_name = m.group("name").strip()
                break

        # If identity adoption is detected, try to map to an open commitment explicitly about name adoption
        mapped_commit = None
        commit_ref = None
        if identity_name:
            try:
                opens = smm.get_open_commitments() or []
            except Exception:
                opens = []
            for c in opens:
                t = (c.get("text", "") or "").lower()
                if ("adopt" in t or "name" in t) and identity_name.lower() in t:
                    commit_ref = c.get("hash") or None
                    mapped_commit = c
                    break
            # As an extra safety, verify the self-model name matches to boost confidence
            try:
                current_name = str(getattr(smm.model.core_identity, "name", ""))
            except Exception:
                current_name = ""

            if mapped_commit and identity_name and identity_name.lower() in current_name.lower():
                # Strong evidence — construct evidence content now
                meta = {"type": "done", "commit_ref": commit_ref}
                content = {
                    "type": "done",
                    "summary": f"Identity adopted: {identity_name}",
                    "artifact": None,
                    "confidence": 0.9,
                }
                res = smm.sqlite_store.append_event(
                    kind="evidence", content=json.dumps(content, ensure_ascii=False), meta=meta
                )
                try:
                    smm.commitment_tracker.close_commitment_with_evidence(
                        commit_hash=commit_ref,
                        evidence_type="done",
                        description=f"Adopted name {identity_name}",
                        artifact=None,
                        confidence=0.9,
                    )
                except Exception:
                    pass
                return res.get("hash")

        # General: explicit Done: summary
        m = DONE_RX.search(text)
        summary = m.group(1).strip()[:240] if m else None

        # Secondary: completion synonyms — use first line containing cue as summary
        if not summary and SYN_RX.search(text):
            for line in text.splitlines():
                if SYN_RX.search(line):
                    summary = line.strip()[:240]
                    break
        if not summary:
            # No positive completion cues — attempt high-overlap mapping as final fallback
            cref, cdict = _best_open_commitment(smm, text)
            if not cref:
                return None
            # Require some overlap to avoid spurious evidence
            summary = (text.splitlines()[0] if text else "").strip()[:240]
            commit_ref = cref
        else:
            commit_ref, _ = _best_open_commitment(smm, text)

        # artifact: URL, PR/issue id, evID, or file path
        art = None
        for rx in (URL_RX, PR_RX, EV_RX, FILE_RX):
            mrx = rx.search(text)
            if mrx:
                art = mrx.group(0)
                break
        # try to grab an open commitment hash for commit_ref
        # commit_ref selected above; if missing, leave evidence unattached
        meta = {"type": "done"}
        if commit_ref:
            meta["commit_ref"] = commit_ref
        # Confidence: base 0.65 + 0.1 if artifact present, capped at 0.9
        conf = 0.7 + (0.1 if art else 0.0)
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
        # Mirror JSON evidence event into self-model for narrative auditability
        try:
            # Structure matching tests: type prefix "evidence:done" and evidence payload
            ev = {
                "evidence_type": "done",
                "commit_ref": meta.get("commit_ref"),
                "description": summary or "Done",
                "artifact": art,
            }
            smm.add_event(
                summary=f"Evidence: {summary}",
                etype="evidence:done",
                evidence=ev,
            )
        except Exception:
            pass

        # Optional: auto-close mapped commitment when above threshold
        try:
            from pmm.config.models import get_evidence_confidence_threshold

            thresh = float(get_evidence_confidence_threshold())
            cref = meta.get("commit_ref")
            if cref and conf >= thresh:
                try:
                    smm.commitment_tracker.close_commitment_with_evidence(
                        commit_hash=cref,
                        evidence_type="done",
                        description=summary or "Done",
                        artifact=art,
                        confidence=conf,
                    )
                except Exception:
                    pass
        except Exception:
            pass
        return res.get("hash")
    except Exception:
        return None
