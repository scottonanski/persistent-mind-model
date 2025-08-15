from __future__ import annotations
import json
from typing import List, Optional
from fastapi import FastAPI, Query
from pmm.storage.sqlite_store import SQLiteStore
from pmm.storage.integrity import verify_chain
from pmm.emergence import EmergenceAnalyzer, EmergenceEvent

app = FastAPI(title="PMM Probe API", version="0.1.0")

# ---- Helpers --------------------------------------------------------------


def _row_to_dict(row):
    id_, ts, kind, content, meta, prev_hash, hsh = row
    try:
        meta_obj = json.loads(meta) if isinstance(meta, str) else meta
    except Exception:
        meta_obj = {"_raw": meta}
    return {
        "id": id_,
        "ts": ts,
        "kind": kind,
        "content": content,
        "meta": meta_obj,
        "prev_hash": prev_hash,
        "hash": hsh,
    }


def _load_events(db_path: str, limit: int, kind: Optional[str]) -> List[dict]:
    store = SQLiteStore(db_path)
    q = "SELECT id,ts,kind,content,meta,prev_hash,hash FROM events"
    args = []
    if kind:
        q += " WHERE kind=?"
        args.append(kind)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    rows = list(store.conn.execute(q, args))
    return [_row_to_dict(r) for r in rows]


def _all_rows(db_path: str):
    store = SQLiteStore(db_path)
    return list(
        store.conn.execute(
            "SELECT id,ts,kind,content,meta,prev_hash,hash FROM events ORDER BY id"
        )
    )


def _commitments_with_status(db_path: str, limit: int):
    """
    Convention:
      - commitment rows: kind='commitment' (content includes 'Next, I will …')
      - evidence rows:   kind='evidence' with meta.commit_ref = <hash of commitment event>
    """
    rows = _all_rows(db_path)
    commits = [r for r in rows if r[2] == "commitment"]
    evidence = [r for r in rows if r[2] == "evidence"]

    # build a set of commitment hashes that have evidence
    ev_refs = set()
    for _, _, _, _, meta_json, _, _ in evidence:
        try:
            m = json.loads(meta_json) if isinstance(meta_json, str) else meta_json
            ref = m.get("commit_ref")
            if ref:
                ev_refs.add(ref)
        except Exception:
            continue

    out = []
    for r in commits[::-1][-limit:]:  # oldest to newest within limit
        d = _row_to_dict(r)
        d["status"] = "closed" if d["hash"] in ev_refs else "open"
        out.append(d)
    return out[::-1]  # return newest first


def _get_emergence_events(
    db_path: str, kinds: List[str] = None, limit: int = 15
) -> List[EmergenceEvent]:
    """Convert database rows to EmergenceEvent objects for analysis."""
    if kinds is None:
        kinds = ["response", "reflection", "evidence", "commitment"]

    # Get events of all specified kinds
    all_events = []
    for kind in kinds:
        rows = _load_events(db_path, limit, kind)
        all_events.extend(rows)

    # Sort by timestamp (most recent first) and limit
    all_events.sort(key=lambda x: x["ts"], reverse=True)
    all_events = all_events[:limit]

    events = []
    for row in all_events:
        event = EmergenceEvent(
            id=row["id"],
            timestamp=row["ts"],
            kind=row["kind"],
            content=row["content"],
            meta=row["meta"],
        )
        events.append(event)
    return events


# ---- Routes ---------------------------------------------------------------


@app.get("/health")
def health(db: str = Query("pmm.db", description="Path to PMM SQLite DB")):
    rows = _all_rows(db)
    return {
        "ok": True,
        "db": db,
        "events": len(rows),
        "last_kind": rows[-1][2] if rows else None,
    }


@app.get("/integrity")
def integrity(db: str = Query("pmm.db", description="Path to PMM SQLite DB")):
    rows = _all_rows(db)
    ok = verify_chain(rows)
    return {"ok": ok, "events": len(rows)}


@app.get("/events/recent")
def recent_events(
    db: str = Query("pmm.db"),
    limit: int = Query(50, ge=1, le=500),
    kind: Optional[str] = Query(
        None,
        description="Filter by kind (prompt|response|reflection|commitment|evidence)",
    ),
):
    return {"items": _load_events(db, limit, kind)}


@app.get("/commitments")
def commitments(db: str = Query("pmm.db"), limit: int = Query(100, ge=1, le=500)):
    return {"items": _commitments_with_status(db, limit)}


# PHASE 3B+: Identity endpoint to verify current agent name
@app.get("/identity")
def get_identity(db: str = Query("pmm.db", description="Path to PMM SQLite DB")):
    """Get current agent identity from latest identity_update event."""
    try:
        # Find the most recent identity_update event from database
        events = _load_events(db, limit=1000, kind="identity_update")

        if events:
            latest_event = events[0]  # most recent first
            # Extract name from event content/summary
            import re

            content = latest_event.get("content", "")
            name_match = re.search(r"Name changed to '([^']+)'", content)
            if name_match:
                return {
                    "name": name_match.group(1),
                    "event_id": latest_event["id"],
                    "timestamp": latest_event["ts"],
                    "source": "identity_update_event",
                }

        # Fallback to default
        return {
            "name": "Agent",
            "source": "default",
            "note": "No identity_update events found",
        }

    except Exception as e:
        return {"error": str(e), "name": "Agent", "source": "error"}


# PHASE 3B: Reflection hygiene and evidence analysis endpoint
@app.get("/reflections")
def reflections(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    limit: int = Query(
        20, ge=1, le=100, description="Number of recent reflections to analyze"
    ),
):
    """Analyze reflection hygiene: referential vs non-referential insights."""
    try:
        reflection_events = _load_events(db, limit, "reflection")
        # Load evidence events for reference checking
        _load_events(db, 100, "evidence")

        analysis = {
            "total_reflections": len(reflection_events),
            "referential_count": 0,
            "non_referential_count": 0,
            "evidence_referenced": 0,
            "reflections": [],
        }

        # Analyze each reflection for referential hygiene
        for reflection in reflection_events:
            content = reflection.get("content", "")
            # meta = reflection.get("meta", {})  # Reserved for future use

            # Check if reflection references specific events or evidence
            is_referential = False
            references = []

            # Look for event ID references
            import re

            event_refs = re.findall(r"event[_\s]*(\d+)", content, re.IGNORECASE)
            if event_refs:
                is_referential = True
                references.extend([f"event_{ref}" for ref in event_refs])

            # Look for commitment hash references
            hash_refs = re.findall(r"[a-f0-9]{16}", content)
            if hash_refs:
                is_referential = True
                references.extend([f"hash_{ref}" for ref in hash_refs])

            # Check if it references evidence
            evidence_ref = any(
                word in content.lower()
                for word in ["done:", "completed:", "evidence", "artifact"]
            )
            if evidence_ref:
                analysis["evidence_referenced"] += 1

            if is_referential:
                analysis["referential_count"] += 1
            else:
                analysis["non_referential_count"] += 1

            analysis["reflections"].append(
                {
                    "id": reflection["id"],
                    "timestamp": reflection["ts"],
                    "is_referential": is_referential,
                    "references": references,
                    "evidence_mentioned": evidence_ref,
                    "content_preview": (
                        content[:120] + "..." if len(content) > 120 else content
                    ),
                }
            )

        # Calculate hygiene metrics
        if analysis["total_reflections"] > 0:
            analysis["referential_rate"] = (
                analysis["referential_count"] / analysis["total_reflections"]
            )
            analysis["evidence_rate"] = (
                analysis["evidence_referenced"] / analysis["total_reflections"]
            )
        else:
            analysis["referential_rate"] = 0.0
            analysis["evidence_rate"] = 0.0

        analysis["hygiene_score"] = (analysis["referential_rate"] * 0.7) + (
            analysis["evidence_rate"] * 0.3
        )

        return analysis

    except Exception as e:
        return {"error": str(e), "total_reflections": 0}


# Optional: placeholder traits endpoint. If/when you persist traits, replace with a real query.
@app.get("/traits")
def traits():
    # surface a stable shape even if traits aren't persisted yet
    return {
        "big_five": {
            "openness": None,
            "conscientiousness": None,
            "extraversion": None,
            "agreeableness": None,
            "neuroticism": None,
        },
        "note": "Populate once traits are stored in SQLite.",
    }


@app.get("/emergence")
def emergence(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    window: int = Query(
        15,
        ge=1,
        le=50,
        description="Number of recent events to analyze (expanded for Phase 3B)",
    ),
):
    """
    PMM Emergence Loop: Analyze AI personality convergence through IAS/GAS scoring.

    Phase 3B Enhancement: Now analyzes reflection, evidence, and commitment events alongside responses.

    Returns:
    - IAS (Identity Adoption Score): 0.6 * pmmspec_match + 0.4 * self_ref_rate
    - GAS (Growth Acceleration Score): weighted combination of experience seeking, novelty, commitment closure
    - Stage: S0 (Substrate) → S1 (Resistance) → S2 (Adoption) → S3 (Self-Model) → S4 (Growth-Seeking)
    - Event breakdown by type for transparency
    """
    try:
        # Get recent events of multiple kinds for comprehensive analysis
        events = _get_emergence_events(
            db, kinds=["response", "reflection", "evidence", "commitment"], limit=window
        )

        # Create analyzer with custom event data
        analyzer = EmergenceAnalyzer()

        # Override the get_recent_events method with our data
        analyzer.get_recent_events = lambda kind="response", limit=window: events

        # Compute emergence scores
        scores = analyzer.compute_scores(window)

        # Add Phase 3B metadata about the analysis
        event_breakdown = {}
        for event in events:
            kind = event.kind
            event_breakdown[kind] = event_breakdown.get(kind, 0) + 1

        scores["db_path"] = db
        scores["window_size"] = window
        scores["events_analyzed"] = len(events)
        scores["event_breakdown"] = event_breakdown
        scores["phase"] = "3B"

        return scores

    except Exception as e:
        return {
            "error": str(e),
            "IAS": 0.0,
            "GAS": 0.0,
            "stage": "S0: Substrate",
            "timestamp": "error",
            "events_analyzed": 0,
            "phase": "3B",
        }
