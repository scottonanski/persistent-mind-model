#!/usr/bin/env python3
"""
Commitment lifecycle management for Persistent Mind Model.
Tracks agent commitments from creation to completion.

This module also contains lightweight helpers for "turn‑scoped" identity
commitments that live for N assistant turns and are enforced via the
append‑only SQLite event log. These helpers are designed to be immutable:
we append `commitment.update` rows to decrement remaining turns and then
emit `evidence` and `commitment.close` when TTL reaches zero.
"""

import hashlib
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from urllib.parse import urlparse
from pathlib import PurePosixPath
from pmm.classifiers import CommitmentExtractor

# Note: Avoid importing SelfModelManager here to prevent circular import.
# Functions below accept an object with a `sqlite_store` attribute.
from pmm.config.models import IDENTITY_COMMIT


@dataclass
class Commitment:
    """A commitment made by an agent during reflection."""

    cid: str
    text: str
    created_at: str
    source_insight_id: str
    status: str = "open"  # open, closed, expired, tentative, ongoing
    # Tier indicates permanence; tentative items can be promoted on reinforcement/evidence
    tier: str = "permanent"  # permanent, tentative, ongoing
    due: Optional[str] = None
    closed_at: Optional[str] = None
    close_note: Optional[str] = None
    ngrams: List[str] = None  # 3-grams for matching
    # Hash of the associated 'commitment' event in the SQLite chain
    # If present, this becomes the canonical reference for evidence
    event_hash: Optional[str] = None
    # Reinforcement tracking
    attempts: int = 1
    reinforcements: int = 0
    last_reinforcement_ts: Optional[str] = None
    _just_reinforced: bool = False


class CommitmentTracker:
    """Manages commitment lifecycle and completion detection."""

    def __init__(self):
        self.commitments: Dict[str, Commitment] = {}

    def _is_valid_commitment(self, text: str) -> bool:
        """Validate using only structural features and centroid score.

        - No keyword or phrase lists
        - No normalization to canonical phrases
        - Structure: first token POS verb OR presence of first-person POS; token_count >= 3
        - Semantics: CommitmentExtractor score >= commit_thresh
        """
        if not isinstance(text, str):
            return False
        s = text.strip()
        if len(s) < 3:
            return False
        # Structural gating
        try:
            from pmm.struct_semantics import pos_tag

            tags = pos_tag(s)
        except Exception:
            tags = []
        first_is_verb = bool(
            tags
            and isinstance(tags[0], (list, tuple))
            and str(tags[0][1]).startswith("VB")
        )
        pos_unknown = (not tags) or all(
            (isinstance(t, (list, tuple)) and str(t[1]) == "X") for t in (tags or [])
        )
        toks = [t for t in s.split() if t]
        first_token_lower = toks[0].lower() if toks else ""
        token_count = len([t for t in s.split() if t])
        # Pattern: PRP MD VB* (e.g., I will do, We should consider)
        prp_md_vb = False
        if isinstance(tags, list) and len(tags) >= 3:
            try:
                t0 = str(tags[0][1])
                t1 = str(tags[1][1])
                t2 = str(tags[2][1])
                prp_md_vb = (
                    t0.startswith("PRP") and t1.startswith("MD") and t2.startswith("VB")
                )
            except Exception:
                prp_md_vb = False
        # Semantic-only policy: avoid rejecting imperative forms due to missing POS tagger.
        # Require only minimal length; semantics gate below does the heavy lifting.
        structure_ok = token_count >= 2
        # Reject short pronoun+modal constructions as too vague unless verb-first
        if prp_md_vb and token_count < 6 and not first_is_verb:
            if os.getenv("PMM_DEBUG") == "1":
                print(f"[PMM_DEBUG] Reject: short PRP-MD-VB | text={s}")
            return False
        # Reject short pronoun-first non-imperative statements in general
        if first_token_lower in {"i", "we"} and token_count < 6 and not first_is_verb:
            if os.getenv("PMM_DEBUG") == "1":
                print(
                    f"[PMM_DEBUG] Reject: short pronoun-first non-imperative | text={s}"
                )
            return False
        # Reject short comma-prefixed sequences (e.g., "Next, ...") unless long enough
        if isinstance(tags, list) and len(tags) >= 4:
            try:
                first_tok_text = str(tags[0][0])
                comma_prefixed = first_tok_text.endswith(",")
                if comma_prefixed and token_count < 7 and not first_is_verb:
                    if os.getenv("PMM_DEBUG") == "1":
                        print(f"[PMM_DEBUG] Reject: short comma-prefixed | text={s}")
                    return False
            except Exception:
                pass
        if not structure_ok:
            return False
        # Semantic gating
        extractor = CommitmentExtractor()
        # If we cannot compute a vector (e.g., analyzer unavailable), be conservative:
        # require imperative (verb-first) to pass; otherwise reject.
        try:
            vec = extractor._vector(s)  # type: ignore[attr-defined]
        except Exception:
            vec = []
        vec_missing = (not isinstance(vec, (list, tuple))) or (len(vec) == 0)
        if vec_missing:
            # If POS is known and indicates non-imperative, reject
            imperative_like = first_is_verb or (prp_md_vb and token_count >= 6)
            if not pos_unknown and not imperative_like:
                if os.getenv("PMM_DEBUG") == "1":
                    print(
                        f"[PMM_DEBUG] Reject: vec_missing & non-imperative (POS known) | text={s}"
                    )
                return False
            # If POS unknown, do not over-reject; rely on structural score + thresholds
        score = extractor.score(s)
        return score >= extractor.commit_thresh

    def extract_commitment(self, text: str) -> Tuple[Optional[str], List[str]]:
        """Extract a normalized commitment string and its 3-grams from free text.

        Rules:
        - Choose the best sentence by structural+semantic score
        - Apply structural validation via _is_valid_commitment
        Returns: (commitment_text_or_None, ngrams)
        """
        if not isinstance(text, str) or not text.strip():
            return None, []
        extractor = CommitmentExtractor()
        cand = extractor.extract_best_sentence(text.strip())
        if not cand or not self._is_valid_commitment(cand):
            return None, []
        ngrams = self._ngram3(cand)
        return cand, ngrams

    def add_commitment(
        self, text: str, source_insight_id: str, due: Optional[str] = None
    ) -> Optional[str]:
        """Validate and add a commitment; returns cid or None if rejected.

        - Enforces ownership and structural checks
        - Rejects duplicates by semantic similarity (threshold via env)
        - Stores 3-grams for later matching
        """
        # Removed all literal special-casing; only structural+semantic extraction is allowed

        commit_text, ngrams = self.extract_commitment(text)
        if not commit_text:
            return None
        dup_cid = self._is_duplicate_commitment(commit_text)
        if dup_cid:
            # Record reinforcement on the existing commitment
            try:
                c = self.commitments[dup_cid]
                c.reinforcements += 1
                c.last_reinforcement_ts = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                c._just_reinforced = True
            except Exception:
                pass
            return None
        # Create CID and store
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cid = hashlib.sha256(
            f"{ts}:{commit_text}:{source_insight_id}".encode("utf-8")
        ).hexdigest()[:12]
        self.commitments[cid] = Commitment(
            cid=cid,
            text=commit_text,
            created_at=ts,
            source_insight_id=str(source_insight_id),
            status="open",
            tier="permanent",
            due=due,
            ngrams=ngrams,
        )
        return cid

    def _ngram3(self, s: str) -> List[str]:
        toks = [t for t in (s or "").lower().split() if t]
        return [" ".join(toks[i : i + 3]) for i in range(max(0, len(toks) - 2))]

    def _is_duplicate_commitment(self, new_text: str) -> Optional[str]:
        """Return cid of a semantically duplicate open commitment, else None.

        Uses cosine similarity when available, otherwise token Jaccard.
        Threshold is configured by PMM_DUPLICATE_SIM_THRESHOLD (default 0.85).
        Only compares against non-archived, non-closed commitments.
        """
        try:
            thresh = float(os.getenv("PMM_DUPLICATE_SIM_THRESHOLD", "0.60"))
        except Exception:
            thresh = 0.60
        candidates = [
            c
            for c in self.commitments.values()
            if c.status in ("open", "tentative", "ongoing")
        ]
        if not candidates:
            return None
        # Try semantic analyzer
        analyzer = None
        try:
            from pmm.semantic_analysis import get_semantic_analyzer

            analyzer = get_semantic_analyzer()
        except Exception:
            analyzer = None
        best_cid = None
        best_sim = 0.0
        for c in candidates:
            sim_sem = 0.0
            sim_tok = 0.0
            if analyzer is not None:
                try:
                    sim_sem = float(analyzer.cosine_similarity(new_text, c.text))
                except Exception:
                    sim_sem = 0.0
            else:
                # Jaccard on lowercased tokens without regex
                a = set((new_text or "").lower().split()) - {""}
                b = set((c.text or "").lower().split()) - {""}
                inter = len(a & b)
                union = len(a | b) or 1
                sim_tok = inter / union
            # Always compute token Jaccard as a fallback signal
            if sim_tok == 0.0:
                a = set((new_text or "").lower().split()) - {""}
                b = set((c.text or "").lower().split()) - {""}
                inter = len(a & b)
                union = len(a | b) or 1
                sim_tok = inter / union

            sim = max(sim_sem, sim_tok)
            if sim > best_sim:
                best_sim = sim
                best_cid = c.cid
        if best_sim >= thresh:
            return best_cid
        return None

    def get_commitment_hash(self, commitment: Commitment) -> str:
        """Return a stable 16-hex hash for a commitment for evidence linking."""
        base = f"{commitment.cid}:{commitment.text}:{commitment.created_at}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

    def _extract_artifact(self, description: str) -> Optional[str]:
        """Extract a likely artifact reference from evidence text.

        Handles:
        - URLs via urllib.parse
        - File-like tokens via pathlib suffix
        - Simple IDs if token contains digits
        - ISO-like dates are not regex-detected; rely on tokens containing '-' and digits
        """
        if not isinstance(description, str) or not description:
            return None
        d = description.strip()

        # Tokens (strip basic punctuation)
        def _sp(tok: str) -> str:
            return tok.strip("\"'`.,;:()[]{}")

        for tok in d.split():
            st = _sp(tok)
            if not st:
                continue
            # URL per-token
            try:
                parsed = urlparse(st)
                if parsed.scheme in ("http", "https") and parsed.netloc:
                    return st
            except Exception:
                pass
            try:
                p = PurePosixPath(st)
                if p.suffix:
                    return st
            except Exception:
                pass
            if any(ch.isdigit() for ch in st):
                return st
        return None

    def detect_evidence_events(
        self, text: str
    ) -> List[Tuple[str, str, str, Optional[str]]]:
        """Deprecated: keyword-based evidence detection removed.

        Evidence mapping is handled semantically in pmm/evidence/behavior_engine.py.
        This function now returns an empty list.
        """
        return []

    def _estimate_evidence_confidence(
        self,
        commitment_text: str,
        description: str,
        artifact: Optional[str] = None,
    ) -> float:
        """Heuristically estimate confidence that evidence supports the commitment."""
        try:
            ct = (commitment_text or "").lower()
            desc = (description or "").lower()
            base = 0.35

            def bigrams(s: str) -> set:
                toks = [t for t in (s or "").split() if t]
                return set(
                    " ".join(toks[i : i + 2]) for i in range(max(0, len(toks) - 1))
                )

            def trigrams(s: str) -> set:
                toks = [t for t in (s or "").split() if t]
                return set(
                    " ".join(toks[i : i + 3]) for i in range(max(0, len(toks) - 2))
                )

            b_ct, b_desc = bigrams(ct), bigrams(desc)
            t_ct, t_desc = trigrams(ct), trigrams(desc)
            b_overlap = (len(b_ct & b_desc) / len(b_ct)) if b_ct else 0.0
            t_overlap = (len(t_ct & t_desc) / len(t_ct)) if t_ct else 0.0

            score = base + 0.35 * b_overlap + 0.25 * t_overlap

            # Semantic boost
            try:
                from pmm.semantic_analysis import get_semantic_analyzer

                analyzer = get_semantic_analyzer()
                cos = analyzer.cosine_similarity(commitment_text, description)
                score += 0.35 * max(0.0, min(1.0, float(cos)))
            except Exception:
                pass

            if artifact:
                score += 0.10
                # URL boost
                try:
                    parsed = urlparse(artifact)
                    if parsed.scheme in ("http", "https") and parsed.netloc:
                        score += 0.05
                except Exception:
                    pass
                # File suffix boost
                try:
                    p = PurePosixPath(artifact)
                    if p.suffix:
                        score += 0.05
                except Exception:
                    pass
                # Bare id token boost if contains digits
                try:
                    if any(ch.isdigit() for ch in str(artifact)):
                        score += 0.03
                except Exception:
                    pass

            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5

    def get_open_commitments(self) -> List[Dict[str, Any]]:
        """Return a simple list of open (non-archived, non-closed) commitments."""
        out: List[Dict[str, Any]] = []
        for c in self.commitments.values():
            if c.status in ("open", "tentative", "ongoing"):
                out.append(
                    {
                        "cid": c.cid,
                        "text": c.text,
                        "status": c.status,
                        "created_at": c.created_at,
                    }
                )
        return out

    def archive_legacy_commitments(self) -> List[str]:
        """Deprecated: removal of keyword-based hygiene archiving.

        Returns an empty list to comply with no-keyword policy.
        """
        return []

    def mark_commitment(
        self, cid: str, status: str, note: Optional[str] = None
    ) -> bool:
        """Update commitment status and optional note.

        Returns True if found and updated; False otherwise.
        """
        c = self.commitments.get(cid)
        if not c:
            return False
        # Allow closing or expiring ongoing via explicit mark; protect in evidence path only
        valid_status = {
            "open",
            "closed",
            "expired",
            "tentative",
            "ongoing",
            "archived_legacy",
        }
        if status not in valid_status:
            return False
        c.status = status
        if status in ("closed", "expired", "archived_legacy"):
            c.closed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if note:
            c.close_note = note
        return True

    def auto_close_from_reflection(self, text: str) -> List[str]:
        """Heuristic auto-closure from reflection content.

        Current implementation is conservative: does not auto-close any items.
        Returns the list of cids that were closed (empty).
        """
        return []

    def close_commitment_with_evidence(
        self,
        commit_hash: str,
        evidence_type: str,
        description: str,
        artifact: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """
        Close a commitment with evidence.

        Args:
            commit_hash: Hash of the commitment to close
            evidence_type: Type of evidence (done, blocked, delegated)
            description: Description of the evidence
            artifact: Optional artifact (file, URL, etc.)
            confidence: Optional confidence score (0-1)

        Returns:
            bool: True if commitment was closed, False otherwise
        """
        # Find commitment by hash
        target_cid = None
        target_commitment = None

        for cid, commitment in self.commitments.items():
            if self.get_commitment_hash(commitment) == commit_hash:
                target_cid = cid
                target_commitment = commitment
                break

        if not target_commitment:
            print(f"[PMM_EVIDENCE] commitment not found: {commit_hash}")
            return False

        # Protect ongoing items; record evidence upstream without closing
        if (
            getattr(target_commitment, "status", "open") == "ongoing"
            or getattr(target_commitment, "tier", "permanent") == "ongoing"
        ):
            print(
                f"[PMM_EVIDENCE] ongoing commitment; recording evidence only: {target_cid}"
            )
            return False
        # Permit closing 'tentative' commitments too, since the UI lists them among open items
        if target_commitment.status not in ("open", "tentative"):
            try:
                st = str(getattr(target_commitment, "status", "unknown"))
            except Exception:
                st = "unknown"
            print(f"[PMM_EVIDENCE] commitment not closable (status={st}): {target_cid}")
            return False

        # Only 'done' evidence can close commitments
        if evidence_type != "done":
            print(
                f"[PMM_EVIDENCE] non_done_evidence: {evidence_type} recorded but not closing"
            )
            return False

        # Validate evidence input (requires artifact by default; allows specific test-only override)
        if not self._is_valid_evidence(evidence_type, description, artifact):
            return False

        # Close on valid 'done' evidence with acceptable artifact
        target_commitment.status = "closed"
        target_commitment.closed_at = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        target_commitment.close_note = f"Evidence: {description} | artifact={artifact}"
        return True

    def _is_valid_evidence(
        self,
        evidence_type: str,
        description: str,
        artifact: Optional[str] = None,
    ) -> bool:
        """Validate evidence inputs for closing a commitment.

        Policy:
        - Only 'done' evidence is eligible for closure (checked by caller).
        - By default, require a non-empty artifact string; text-only evidence is insufficient.
        - Tests can opt-in to allow text-only evidence via PMM_TEST_ALLOW_TEXT_ONLY_EVIDENCE=1.
        """
        if not isinstance(description, str) or not description.strip():
            return False
        allow_text_only = os.getenv("PMM_TEST_ALLOW_TEXT_ONLY_EVIDENCE", "0") == "1"
        if isinstance(artifact, str) and artifact.strip():
            return True
        # No artifact provided
        if allow_text_only:
            return True
        return False

    def expire_old_commitments(self, days_old: int = 30) -> List[str]:
        """Mark old commitments as expired."""
        expired_cids = []

        # Calculate cutoff timestamp
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_old)
        except Exception:
            cutoff = datetime.now(timezone.utc)

        for cid, commitment in self.commitments.items():
            if commitment.status != "open":
                continue

            try:
                created = datetime.fromisoformat(commitment.created_at.replace("Z", ""))
                # Normalize to aware datetime in UTC for comparison
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                if created < cutoff:
                    self.mark_commitment(
                        cid, "expired", f"Auto-expired after {days_old} days"
                    )
                    expired_cids.append(cid)
            except Exception:
                continue

        return expired_cids


# =============================
# Identity turn-scoped helpers
# =============================


def canonical_identity_dedupe_key(policy: str, scope: str, ttl_turns: int) -> str:
    """Return a canonical dedupe key for identity commitments.

    Example: identity::express_core_principles::turns::3
    """
    return f"identity::{policy}::{scope}::{int(ttl_turns)}"


def _safe_json_loads(s: Optional[str]) -> Dict[str, Any]:
    try:
        import json

        if not s:
            return {}
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _list_open_identity_turn_commitments(smm: Any) -> List[Dict[str, Any]]:
    """List open identity commitments scoped to turns with latest remaining_turns.

    Returns a list of dicts like:
    {"event_hash": str, "open_event_id": int, "content": { ... }, "meta": { ... }}
    """
    try:
        store = getattr(smm, "sqlite_store", None)
        if store is None:
            return []
        # Query only the relevant kinds for efficiency
        with store._lock:  # type: ignore[attr-defined]
            rows = list(
                store.conn.execute(
                    "SELECT id, ts, kind, content, meta, hash FROM events "
                    "WHERE kind IN ('commitment.open','commitment.update','commitment.close') "
                    "ORDER BY id"
                )
            )

        opens: Dict[str, Dict[str, Any]] = {}
        updates: Dict[str, int] = {}
        closed: set[str] = set()

        for r in rows:
            kind = r[2]
            content_raw = r[3]
            meta_raw = r[4]
            ev_hash = r[5]

            meta = _safe_json_loads(meta_raw)

            if kind == "commitment.open":
                cont = _safe_json_loads(content_raw)
                if (
                    str(cont.get("category", "")) == "identity"
                    and str(cont.get("scope", "")) == "turns"
                ):
                    opens[ev_hash] = {
                        "event_hash": ev_hash,
                        "open_event_id": int(r[0]),
                        "content": cont,
                        "meta": meta or {},
                    }
                    # Prime updates with the initial remaining_turns
                    try:
                        updates[ev_hash] = int(cont.get("remaining_turns", 0))
                    except Exception:
                        updates[ev_hash] = 0
            elif kind == "commitment.update":
                cref = str((meta or {}).get("commit_ref", ""))
                if cref in opens:
                    cont = _safe_json_loads(content_raw)
                    try:
                        updates[cref] = int(
                            cont.get("remaining_turns", updates.get(cref, 0))
                        )
                    except Exception:
                        pass
            elif kind == "commitment.close":
                cref = str((meta or {}).get("commit_ref", ""))
                if cref:
                    closed.add(cref)

        # Build final list with latest remaining_turns and exclude closed
        out: List[Dict[str, Any]] = []
        for k, rec in opens.items():
            if k in closed:
                continue
            # Overlay latest remaining_turns if present
            if k in updates:
                try:
                    rec["content"] = dict(rec["content"])
                    rec["content"]["remaining_turns"] = int(updates[k])
                except Exception:
                    pass
            out.append(rec)
        return out
    except Exception:
        return []


def _append_evidence_event(
    smm: Any, commit_hash: str, reply_text: str, confidence: float
) -> None:
    try:
        store = getattr(smm, "sqlite_store", None)
        if store is None:
            return
        import json

        evidence_content = {
            "type": "done",
            "summary": "Turn elapsed; identity expression enforced this turn.",
            "artifact": {"reply_excerpt": (reply_text or "")[:400]},
            "confidence": float(confidence),
        }
        store.append_event(
            kind="evidence",
            content=json.dumps(evidence_content, ensure_ascii=False),
            meta={"commit_ref": commit_hash, "subsystem": "identity"},
        )
    except Exception:
        return


def _close_commitment(smm: Any, commit_hash: str) -> None:
    try:
        store = getattr(smm, "sqlite_store", None)
        if store is None:
            return
        import json

        store.append_event(
            kind="commitment.close",
            content=json.dumps({"reason": "ttl_exhausted"}, ensure_ascii=False),
            meta={"commit_ref": commit_hash, "subsystem": "identity"},
        )
    except Exception:
        return


def tick_turn_scoped_identity_commitments(smm: Any, reply_text: str) -> None:
    """Decrement remaining_turns for open identity commitments; auto-close at zero.

    Appends immutable `commitment.update` events for each tick. When TTL hits
    zero, emits a minimal `evidence` row then a `commitment.close` row.
    """
    items = _list_open_identity_turn_commitments(smm)
    if not items:
        return
    for item in items:
        content = dict(item.get("content", {}) or {})
        commit_hash = item.get("event_hash")
        try:
            remaining = int(content.get("remaining_turns", 0))
        except Exception:
            remaining = 0

        if remaining <= 0:
            _close_commitment(smm, commit_hash)
            continue

        # Decrement and append update
        remaining -= 1
        upd_content = {"remaining_turns": remaining, "note": "decrement after turn"}
        try:
            smm.sqlite_store.append_event(
                kind="commitment.update",
                content=__import__("json").dumps(upd_content, ensure_ascii=False),
                meta={"commit_ref": commit_hash, "subsystem": "identity"},
            )
        except Exception:
            pass

        if remaining == 0:
            _append_evidence_event(
                smm,
                commit_hash=commit_hash,
                reply_text=reply_text or "",
                confidence=float(IDENTITY_COMMIT.min_confidence),
            )
            _close_commitment(smm, commit_hash)


def open_identity_commitment(
    smm: Any,
    policy: str = "express_core_principles",
    ttl_turns: Optional[int] = None,
    note: Optional[str] = None,
) -> str:
    """Open a turn-scoped identity commitment and return its event hash.

    Dedupe is enforced by a canonical key (policy+scope+ttl); if an open
    commitment with the same key exists, returns its hash instead of
    creating a new one.
    """
    try:
        ttl = (
            int(ttl_turns) if ttl_turns is not None else int(IDENTITY_COMMIT.ttl_turns)
        )
    except Exception:
        ttl = int(IDENTITY_COMMIT.ttl_turns)

    scope = "turns"
    key = canonical_identity_dedupe_key(policy, scope, ttl)

    # If one is open with same key, return it (no-op)
    try:
        open_items = _list_open_identity_turn_commitments(smm)
        for it in open_items:
            cont = it.get("content", {}) or {}
            if (
                str(cont.get("policy", "")) == policy
                and str(cont.get("scope", "")) == scope
                and int(cont.get("ttl_turns", ttl)) == ttl
            ):
                return str(it.get("event_hash", ""))
    except Exception:
        pass

    # Append the 'commitment.open' event
    import json

    content: Dict[str, Any] = {
        "category": "identity",
        "scope": scope,
        "policy": policy,
        "ttl_turns": ttl,
        "remaining_turns": ttl,
        "note": note or "",
    }
    meta = {"subsystem": "identity", "dedupe_key": key}

    try:
        res = smm.sqlite_store.append_event(
            kind="commitment.open",
            content=json.dumps(content, ensure_ascii=False),
            meta=meta,
        )
        return str(res.get("hash", ""))
    except Exception:
        return ""


def get_identity_turn_commitments(smm: Any) -> List[Dict[str, Any]]:
    """Return a simplified list of open identity turn-scoped commitments.

    Each item contains: policy, ttl_turns, remaining_turns, event_hash.
    """
    items = _list_open_identity_turn_commitments(smm)
    out: List[Dict[str, Any]] = []
    for it in items:
        cont = dict(it.get("content", {}) or {})
        out.append(
            {
                "policy": str(cont.get("policy", "")),
                "ttl_turns": int(cont.get("ttl_turns", 0) or 0),
                "remaining_turns": int(cont.get("remaining_turns", 0) or 0),
                "event_hash": str(it.get("event_hash", "")),
            }
        )
    return out


def close_identity_turn_commitments(
    smm: Any, commit_hashes: Optional[List[str]] = None
) -> int:
    """Force-close identity turn-scoped commitments.

    If commit_hashes is None, closes all currently open identity turn commitments.
    Returns the number of commitments closed.
    """
    try:
        items = _list_open_identity_turn_commitments(smm)
        targets = []
        if commit_hashes:
            want = set(str(h) for h in commit_hashes)
            for it in items:
                if str(it.get("event_hash", "")) in want:
                    targets.append(str(it.get("event_hash", "")))
        else:
            targets = [str(it.get("event_hash", "")) for it in items]
        for h in targets:
            if h:
                _close_commitment(smm, h)
        return len(targets)
    except Exception:
        return 0
