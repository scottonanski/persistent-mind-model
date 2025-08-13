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
    db_path: str, kind: str = "response", limit: int = 5
) -> List[EmergenceEvent]:
    """Convert database rows to EmergenceEvent objects for analysis."""
    rows = _load_events(db_path, limit, kind)
    events = []
    for row in rows:
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
        5, ge=1, le=20, description="Number of recent responses to analyze"
    ),
):
    """
    PMM Emergence Loop: Analyze AI personality convergence through IAS/GAS scoring.

    Returns:
    - IAS (Identity Adoption Score): 0.6 * pmmspec_match + 0.4 * self_ref_rate
    - GAS (Growth Acceleration Score): weighted combination of experience seeking, novelty, commitment closure
    - Stage: S0 (Substrate) → S1 (Resistance) → S2 (Adoption) → S3 (Self-Model) → S4 (Growth-Seeking)
    """
    try:
        # Get recent response events for analysis
        events = _get_emergence_events(db, kind="response", limit=window)

        # Create analyzer with custom event data
        analyzer = EmergenceAnalyzer()

        # Override the get_recent_events method with our data
        analyzer.get_recent_events = lambda kind="response", limit=window: events

        # Compute emergence scores
        scores = analyzer.compute_scores(window)

        # Add metadata about the analysis
        scores["db_path"] = db
        scores["window_size"] = window

        return scores

    except Exception as e:
        return {
            "error": str(e),
            "IAS": 0.0,
            "GAS": 0.0,
            "stage": "S0: Substrate",
            "timestamp": "error",
            "events_analyzed": 0,
        }
