from __future__ import annotations
import json
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pmm.storage.sqlite_store import SQLiteStore
from pmm.emergence import EmergenceAnalyzer, EmergenceEvent
from pmm.semantic_analysis import get_semantic_analyzer
from pmm.meta_reflection import get_meta_reflection_analyzer
from pmm.commitments import get_identity_turn_commitments

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="PMM Probe API", version="0.1.0")

# ---- Pydantic Models ------------------------------------------------------


class EvidenceEvent(BaseModel):
    type: str
    commitment_id: int
    text: str
    source: str
    meta: dict = {}


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


@app.get("/identity")
def identity(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    model_path: str = Query(
        "persistent_self_model.json", description="Path to persistent self model JSON"
    ),
):
    """Return current identity and active identity turn-scoped commitments.

    Identity resolution:
    1) JSON model core_identity (preferred)
    2) Fallback: latest identity change/update event content
    """
    import re

    # 1) Load name/id from JSON model
    name = None
    ident = None
    try:
        import json

        with open(model_path, "r") as f:
            data = json.load(f)
        core = (data or {}).get("core_identity", {}) or {}
        name = core.get("name")
        ident = core.get("id")
    except Exception:
        pass

    # 2) Fallback: try to parse from latest identity_change / identity_update
    if not name:
        try:
            # Prefer current kind, then legacy
            events = _load_events(db, limit=200, kind="identity_change") or []
            if not events:
                events = _load_events(db, limit=200, kind="identity_update") or []
            if events:
                content = events[0].get("content", "") or ""
                m = re.search(
                    r"Name changed from '([^']+)' to '([^']+)'\s*\(origin=.*\)|Name changed to '([^']+)'",
                    content,
                )
                if m:
                    name = m.group(2) or m.group(3)
        except Exception:
            pass

    # List identity turn commitments via a transient SMM-like shim
    class _Shim:
        def __init__(self, path: str):
            self.sqlite_store = SQLiteStore(path)

    shim = _Shim(db)
    items = get_identity_turn_commitments(shim)
    # Project to clean shape (hide event_hash details except short id)
    formatted = [
        {
            "policy": i.get("policy"),
            "ttl_turns": i.get("ttl_turns"),
            "remaining_turns": i.get("remaining_turns"),
            "id": (i.get("event_hash", "") or "")[:8],
        }
        for i in items
    ]

    return {"name": name or "Agent", "id": ident, "identity_commitments": formatted}


# ---- Autonomy Probes --------------------------------------------------------


@app.get("/autonomy/tasks")
def autonomy_tasks(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    limit: int = Query(200, ge=1, le=1000),
):
    """Reconstruct dev tasks from task_* events; return current open + recent closed.

    Event kinds: task_created, task_progress, task_closed with meta.task_id.
    """
    store = SQLiteStore(db)
    rows = list(
        store.conn.execute(
            "SELECT id,ts,kind,content,meta FROM events WHERE kind IN ('task_created','task_progress','task_closed') ORDER BY id DESC LIMIT ?",
            (limit,),
        )
    )
    tasks: dict[str, dict] = {}
    import json

    for rid, ts, kind, content, meta in rows[::-1]:  # oldest->newest
        try:
            m = json.loads(meta) if isinstance(meta, str) else (meta or {})
        except Exception:
            m = {}
        tid = str(m.get("task_id", ""))
        if not tid:
            continue
        rec = tasks.setdefault(
            tid,
            {
                "task_id": tid,
                "status": "open",
                "created_at": ts,
                "kind": None,
                "title": None,
                "policy": None,
                "ttl": None,
                "progress": [],
                "closed_at": None,
            },
        )
        if kind == "task_created":
            try:
                c = json.loads(content) if isinstance(content, str) else (content or {})
            except Exception:
                c = {}
            rec.update({
                "kind": c.get("kind"),
                "title": c.get("title"),
                "policy": c.get("policy"),
                "ttl": c.get("ttl"),
            })
        elif kind == "task_progress":
            rec["progress"].append({"ts": ts, "content": content})
        elif kind == "task_closed":
            rec["status"] = "closed"
            rec["closed_at"] = ts

    open_tasks = [t for t in tasks.values() if t.get("status") == "open"]
    closed_tasks = [t for t in tasks.values() if t.get("status") == "closed"]
    return {"open": open_tasks, "closed": closed_tasks}


@app.get("/autonomy/status")
def autonomy_status(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
):
    """Return a compact autonomy snapshot using recent metrics and task counts."""
    try:
        from pmm.emergence import compute_emergence_scores

        store = SQLiteStore(db)
        scores = compute_emergence_scores(window=15, storage_manager=store)
        # Count open tasks from events
        tasks = autonomy_tasks(db)
        open_count = len(tasks.get("open", []))
        return {
            "stage": scores.get("stage"),
            "IAS": scores.get("IAS"),
            "GAS": scores.get("GAS"),
            "commit_close_rate": scores.get("commit_close_rate"),
            "open_tasks": open_count,
        }
    except Exception as e:
        return {"error": str(e)}


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
    ev_count = len(rows)
    cold_start = ev_count == 0
    return {
        "ok": True,
        "db": db,
        "events": ev_count,
        "last_kind": rows[-1][2] if rows else None,
        "cold_start": cold_start,
        "event_count": ev_count,
    }


@app.get("/endpoints")
def endpoints() -> dict:
    """Return a curated list of Probe API endpoints with short descriptions and examples.

    This is a user-friendly directory intended for CLI discovery (e.g., --@probe list).
    """
    items = [
        {
            "path": "/identity",
            "desc": "Agent name + active identity commitments",
            "example": "/identity",
        },
        {
            "path": "/commitments",
            "desc": "Commitment rows with open/closed status",
            "example": "/commitments?limit=20",
        },
        {
            "path": "/events/recent",
            "desc": "Recent events (id, ts, kind, content)",
            "example": "/events/recent?limit=10&kind=response",
        },
        {
            "path": "/health",
            "desc": "Probe health and last event kind",
            "example": "/health",
        },
        {
            "path": "/emergence",
            "desc": "Emergence snapshot (IAS/GAS/stage)",
            "example": "/emergence",
        },
        {
            "path": "/reflection/quality",
            "desc": "Reflection hygiene analysis",
            "example": "/reflection/quality?limit=20",
        },
        {
            "path": "/introspection",
            "desc": "Introspection results and triggers",
            "example": "/introspection?limit=10",
        },
        {"path": "/traits", "desc": "Big Five scores (live)", "example": "/traits"},
        {
            "path": "/traits/drift",
            "desc": "Trait drift history",
            "example": "/traits/drift?limit=24",
        },
        {
            "path": "/meta-cognition",
            "desc": "Meta-cognitive insights",
            "example": "/meta-cognition?limit=10",
        },
        {
            "path": "/scenes",
            "desc": "Narrative scenes from JSON self-model",
            "example": "/scenes?limit=3",
        },
        {
            "path": "/feedback/summary",
            "desc": "Feedback counts and averages",
            "example": "/feedback/summary?limit=100",
        },
        {
            "path": "/embeddings/backlog",
            "desc": "Unembedded events backlog",
            "example": "/embeddings/backlog",
        },
        {
            "path": "/metrics/hourly",
            "desc": "Autonomy tick metrics (recent)",
            "example": "/metrics/hourly?limit=24",
        },
        {
            "path": "/bandit/reflection",
            "desc": "Reflection bandit stats",
            "example": "/bandit/reflection",
        },
        {
            "path": "/reflection/contract",
            "desc": "Reflection pass/fail metrics",
            "example": "/reflection/contract?limit=50",
        },
    ]
    return {"items": items}


@app.get("/integrity")
def integrity(db: str = Query("pmm.db", description="Path to PMM SQLite DB")):
    """Verify hash-chain integrity using the store's canonical hashing.

    This recomputes each row's hash using the same canonical JSON used by
    SQLiteStore.append_event to avoid parity mismatches.
    """
    import hashlib

    store = SQLiteStore(db)

    def _rehash(row: dict) -> str:
        try:
            meta = row.get("meta") or {}
            if isinstance(meta, str):
                # Best-effort parse if string made it through
                meta = json.loads(meta)
        except Exception:
            meta = {}
        payload = {
            "ts": row.get("ts"),
            "kind": row.get("kind"),
            "content": row.get("content"),
            "meta": meta,
            "prev_hash": row.get("prev_hash"),
        }
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()

    try:
        ok = store.verify_chain(_rehash)
        events = len(store.all_events())
        return {"ok": ok, "events": events}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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


@app.post("/events")
def create_evidence_event(
    evidence: EvidenceEvent,
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
):
    """Create an evidence event to close a commitment with proper hash chaining."""
    try:
        store = SQLiteStore(db)

        # Find the commitment to reference
        commitment_query = """
        SELECT id, ts, kind, content, meta, prev_hash, hash 
        FROM events 
        WHERE kind='commitment' AND id=? 
        LIMIT 1
        """
        commitment_row = list(
            store.conn.execute(commitment_query, (evidence.commitment_id,))
        )

        if not commitment_row:
            raise HTTPException(
                status_code=404, detail=f"Commitment {evidence.commitment_id} not found"
            )

        commitment_dict = _row_to_dict(commitment_row[0])
        commit_ref_hash = commitment_dict["hash"]

        # Build evidence metadata with commitment reference
        evidence_meta = evidence.meta.copy()
        evidence_meta["commit_ref"] = commit_ref_hash
        evidence_meta["commitment_id"] = evidence.commitment_id

        # Use enhanced SQLite store with automatic hash generation and chain integrity
        result = store.append_event(
            kind="evidence", content=evidence.text, meta=evidence_meta
        )

        return {
            "success": True,
            "event_id": result["event_id"],
            "hash": result["hash"],
            "prev_hash": result["prev_hash"],
            "commit_ref": commit_ref_hash,
            "timestamp": result["timestamp"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/commitments")
def commitments(
    db: str = Query("pmm.db"),
    limit: int = Query(100, ge=1, le=500),
    status: Optional[str] = Query(None, description="Filter by status: open|closed"),
    fields: Optional[str] = Query(None, description="Comma-separated fields to return"),
):
    items = _commitments_with_status(db, limit)

    # Filter out test fixtures and invalid entries
    filtered_items = []
    for item in items:
        # Skip test fixtures
        if item.get("meta", {}).get("commitment_id") == "test_commit":
            continue
        # Skip invalid hashes (not 64-char hex)
        if (
            not item.get("hash", "").replace("-", "").isalnum()
            or len(item.get("hash", "")) != 64
        ):
            continue
        # Skip entries with null prev_hash (except intentional genesis)
        if item.get("prev_hash") is None and item.get("id", 0) > 1:
            continue

        filtered_items.append(item)

    # Apply status filter if specified
    if status:
        filtered_items = [
            item for item in filtered_items if item.get("status") == status
        ]

    # Apply field projection if specified
    if fields:
        field_list = [f.strip() for f in fields.split(",")]
        projected_items = []
        for item in filtered_items:
            projected = {
                field: item.get(field) for field in field_list if field in item
            }
            projected_items.append(projected)
        filtered_items = projected_items

    return {"items": filtered_items}


@app.get("/commitments/summary")
def commitments_summary(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    limit: int = Query(500, ge=1, le=5000),
):
    """Summarize commitments and reinforcements from the event log.

    - reinforcement_rate: reinforcements / (commitments + reinforcements)
    - unique_intents: unique commitment contents (hash of normalized text)
    """
    try:
        store = SQLiteStore(db)
        rows = list(
            store.conn.execute(
                "SELECT kind, content, meta FROM events WHERE kind IN ('commitment','commitment_reinforcement') ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        )
        commits = 0
        reinf = 0
        unique = set()
        import hashlib

        for k, content, meta in rows:
            norm = (content or "").strip().lower()
            h = hashlib.sha256(norm.encode()).hexdigest()[:16]
            if k == "commitment":
                commits += 1
                unique.add(h)
            elif k == "commitment_reinforcement":
                reinf += 1
        total = commits + reinf
        rate = (reinf / total) if total else 0.0
        return {
            "commitments": commits,
            "reinforcements": reinf,
            "reinforcement_rate": round(rate, 3),
            "unique_intents": len(unique),
        }
    except Exception as e:
        return {"error": str(e)}


# (removed duplicate /identity endpoint; unified above)


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
def traits(db: str = Query("pmm.db", description="Path to PMM SQLite DB")):
    """
    Live personality traits endpoint showing real-time Big Five scores with drift tracking.

    Returns:
    - Current Big Five trait scores
    - HEXACO trait scores
    - Trait drift history and velocity
    - Last update timestamps and origins
    - Personality evolution metrics
    """
    try:
        from pmm.self_model_manager import SelfModelManager
        from datetime import datetime, timezone
        import os

        # Check if persistent model file exists
        model_file = "persistent_self_model.json"
        if not os.path.exists(model_file):
            return {
                "error": f"PMM model file '{model_file}' not found. Initialize PMM first.",
                "big_five": {
                    "openness": None,
                    "conscientiousness": None,
                    "extraversion": None,
                    "agreeableness": None,
                    "neuroticism": None,
                },
                "note": "PMM model not initialized. Run a conversation first to create the model.",
            }

        # Load PMM model to get live personality data
        manager = SelfModelManager(model_file)

        # Extract Big Five traits
        big5_traits = {}
        for trait_name in [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]:
            trait_obj = getattr(manager.model.personality.traits.big5, trait_name)
            big5_traits[trait_name] = {
                "score": round(trait_obj.score, 3),
                "confidence": round(trait_obj.conf, 3),
                "last_update": trait_obj.last_update,
                "origin": trait_obj.origin,
            }

        # Extract HEXACO traits
        hexaco_traits = {}
        for trait_name in [
            "honesty_humility",
            "emotionality",
            "extraversion",
            "agreeableness",
            "conscientiousness",
            "openness",
        ]:
            trait_obj = getattr(manager.model.personality.traits.hexaco, trait_name)
            hexaco_traits[trait_name] = {
                "score": round(trait_obj.score, 3),
                "confidence": round(trait_obj.conf, 3),
                "last_update": trait_obj.last_update,
                "origin": trait_obj.origin,
            }

        # Get trait drift history from events
        store = SQLiteStore(db)
        recent_events = store.recent_events(limit=100)
        event_count = len(store.all_events())
        cold_start = event_count == 0

        # Calculate trait drift velocity (changes over time)
        trait_changes = []
        for event in recent_events:
            if event.get("kind") == "reflection" and event.get("meta", {}).get(
                "trait_effects"
            ):
                trait_changes.append(
                    {
                        "timestamp": event.get("ts"),
                        "effects": event.get("meta", {}).get("trait_effects", []),
                    }
                )

        # Calculate personality stability metrics
        stability_score = 1.0  # Default high stability
        if len(trait_changes) > 1:
            # Simple stability metric based on frequency of changes
            days_span = 30  # Look at last 30 days
            change_frequency = len(trait_changes) / max(1, days_span)
            stability_score = max(0.0, 1.0 - (change_frequency * 0.1))

        # Get behavioral patterns that influence traits
        behavioral_patterns = dict(manager.model.self_knowledge.behavioral_patterns)

        # Calculate trait-pattern correlations
        trait_influences = {
            "openness": ["experimentation", "growth", "reflection"],
            "conscientiousness": ["stability", "user_goal_alignment"],
            "extraversion": ["identity", "source_citation"],
            "agreeableness": ["user_goal_alignment", "stability"],
            "neuroticism": ["calibration", "error_correction"],
        }

        pattern_influences = {}
        for trait, patterns in trait_influences.items():
            influence_score = sum(behavioral_patterns.get(p, 0) for p in patterns)
            pattern_influences[trait] = influence_score

        return {
            "big_five": big5_traits,
            "hexaco": hexaco_traits,
            "defaults": cold_start,  # treat as defaults when no events yet
            "cold_start": cold_start,
            "event_count": event_count,
            "personality_metrics": {
                "stability_score": round(stability_score, 3),
                "total_trait_changes": len(trait_changes),
                "last_drift_event": (
                    trait_changes[0]["timestamp"] if trait_changes else None
                ),
                "behavioral_pattern_count": len(behavioral_patterns),
                "total_pattern_signals": sum(behavioral_patterns.values()),
            },
            "behavioral_patterns": behavioral_patterns,
            "trait_pattern_influences": pattern_influences,
            "mbti": {
                "label": manager.model.personality.mbti.label,
                "confidence": manager.model.personality.mbti.conf,
                "last_update": manager.model.personality.mbti.last_update,
                "origin": manager.model.personality.mbti.origin,
            },
            "drift_config": {
                "max_delta": manager.model.drift_config.max_delta_per_reflection,
                "maturity_factor": manager.model.drift_config.maturity_principle,
                "cooldown_days": manager.model.drift_config.cooldown_days,
                "locked_traits": manager.model.drift_config.locks,
            },
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "db_path": db,
        }

    except Exception as e:
        return {
            "error": str(e),
            "big_five": {
                "openness": None,
                "conscientiousness": None,
                "extraversion": None,
                "agreeableness": None,
                "neuroticism": None,
            },
            "note": "Failed to load live personality data. Using fallback structure.",
        }


@app.get("/traits/drift")
def trait_drift_history(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    days: int = Query(30, ge=1, le=90, description="Number of days to analyze"),
    trait: Optional[str] = Query(
        None,
        description="Specific trait to analyze (openness, conscientiousness, etc.)",
    ),
):
    """
    Trait drift history endpoint showing personality evolution over time.

    Returns:
    - Historical trait score changes
    - Drift velocity and acceleration
    - Trigger events that caused changes
    - Stability analysis and trend detection
    """
    try:
        from pmm.self_model_manager import SelfModelManager
        from datetime import datetime, timezone, timedelta

        # Load PMM model
        manager = SelfModelManager("persistent_self_model.json")

        # Get current trait scores as baseline
        current_traits = {}
        for trait_name in [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]:
            trait_obj = getattr(manager.model.personality.traits.big5, trait_name)
            current_traits[trait_name] = {
                "score": trait_obj.score,
                "last_update": trait_obj.last_update,
                "origin": trait_obj.origin,
            }

        # Get historical events that might have caused trait changes
        store = SQLiteStore(db)
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        # Get all events since cutoff
        all_events = store.all_events()
        recent_events = [e for e in all_events if e.get("ts", "") >= cutoff_date]

        # Extract trait-affecting events
        trait_events = []
        for event in recent_events:
            if event.get("kind") in ["reflection", "event"] and event.get(
                "meta", {}
            ).get("trait_effects"):
                trait_events.append(
                    {
                        "timestamp": event.get("ts"),
                        "kind": event.get("kind"),
                        "content": event.get("content", "")[:100],
                        "trait_effects": event.get("meta", {}).get("trait_effects", []),
                        "event_id": event.get("id"),
                    }
                )

        # Calculate trait drift metrics
        drift_analysis = {}

        if trait:
            # Analyze specific trait
            trait_history = []
            cumulative_change = 0.0

            for event in trait_events:
                for effect in event.get("trait_effects", []):
                    if effect.get("trait") == trait:
                        delta = effect.get("magnitude", 0.0)
                        direction = effect.get("direction", "neutral")
                        if direction == "decrease":
                            delta = -delta

                        cumulative_change += delta
                        trait_history.append(
                            {
                                "timestamp": event["timestamp"],
                                "delta": delta,
                                "cumulative_change": round(cumulative_change, 4),
                                "trigger_event": event["content"],
                                "confidence": effect.get("confidence", 0.0),
                            }
                        )

            drift_analysis[trait] = {
                "total_change": round(cumulative_change, 4),
                "change_events": len(trait_history),
                "avg_change_per_event": round(
                    cumulative_change / max(1, len(trait_history)), 4
                ),
                "history": trait_history[-20:],  # Last 20 changes
                "current_score": current_traits.get(trait, {}).get("score", 0.5),
            }
        else:
            # Analyze all traits
            for trait_name in current_traits.keys():
                trait_history = []
                cumulative_change = 0.0

                for event in trait_events:
                    for effect in event.get("trait_effects", []):
                        if effect.get("trait") == trait_name:
                            delta = effect.get("magnitude", 0.0)
                            direction = effect.get("direction", "neutral")
                            if direction == "decrease":
                                delta = -delta

                            cumulative_change += delta
                            trait_history.append(
                                {
                                    "timestamp": event["timestamp"],
                                    "delta": delta,
                                    "cumulative_change": round(cumulative_change, 4),
                                }
                            )

                drift_analysis[trait_name] = {
                    "total_change": round(cumulative_change, 4),
                    "change_events": len(trait_history),
                    "avg_change_per_event": round(
                        cumulative_change / max(1, len(trait_history)), 4
                    ),
                    "recent_changes": trait_history[-5:],  # Last 5 changes
                    "current_score": current_traits.get(trait_name, {}).get(
                        "score", 0.5
                    ),
                }

        # Calculate overall personality stability
        total_changes = sum(
            analysis["change_events"] for analysis in drift_analysis.values()
        )
        avg_change_magnitude = sum(
            abs(analysis["total_change"]) for analysis in drift_analysis.values()
        ) / max(1, len(drift_analysis))

        stability_metrics = {
            "total_trait_changes": total_changes,
            "avg_change_magnitude": round(avg_change_magnitude, 4),
            "stability_score": round(max(0.0, 1.0 - (avg_change_magnitude * 2)), 3),
            "most_volatile_trait": (
                max(
                    drift_analysis.keys(),
                    key=lambda t: abs(drift_analysis[t]["total_change"]),
                )
                if drift_analysis
                else None
            ),
            "most_stable_trait": (
                min(
                    drift_analysis.keys(),
                    key=lambda t: abs(drift_analysis[t]["total_change"]),
                )
                if drift_analysis
                else None
            ),
        }

        return {
            "trait_drift_analysis": drift_analysis,
            "stability_metrics": stability_metrics,
            "analysis_period": {
                "days": days,
                "start_date": cutoff_date,
                "end_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "events_analyzed": len(trait_events),
            },
            "current_traits": current_traits,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "db_path": db,
        }

    except Exception as e:
        return {
            "error": str(e),
            "trait_drift_analysis": {},
            "note": "Failed to analyze trait drift history.",
        }


@app.get("/introspection")
def introspection_status(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
):
    """
    Introspection system status and recent analysis results.

    Returns:
    - Recent introspection events (user-prompted and automatic)
    - Available introspection commands
    - Automatic trigger configuration
    - Analysis history and patterns
    """
    try:
        from pmm.introspection import IntrospectionEngine, IntrospectionConfig
        from pmm.storage.sqlite_store import SQLiteStore
        from datetime import datetime, timezone, timedelta

        # Initialize introspection engine
        store = SQLiteStore(db)
        config = IntrospectionConfig()
        engine = IntrospectionEngine(store, config)

        # Get recent introspection events
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        all_events = store.all_events()
        introspection_events = [
            e
            for e in all_events
            if e.get("ts", "") >= cutoff_date
            and e.get("kind") in ["introspection_command", "introspection_automatic"]
        ]

        # Analyze introspection patterns
        user_commands = [
            e for e in introspection_events if e.get("kind") == "introspection_command"
        ]
        automatic_triggers = [
            e
            for e in introspection_events
            if e.get("kind") == "introspection_automatic"
        ]

        # Extract command types from tags
        command_types = {}
        trigger_reasons = {}

        for event in introspection_events:
            tags = event.get("meta", {}).get("tags", [])
            for tag in tags:
                if tag in [
                    "patterns",
                    "decisions",
                    "growth",
                    "commitments",
                    "conflicts",
                    "goals",
                    "emergence",
                    "memory",
                    "reflection",
                ]:
                    command_types[tag] = command_types.get(tag, 0) + 1
                if tag in [
                    "failed_commitment",
                    "trait_drift",
                    "reflection_loop",
                    "emergence_plateau",
                    "pattern_conflict",
                ]:
                    trigger_reasons[tag] = trigger_reasons.get(tag, 0) + 1

        # Get available commands
        available_commands = engine.get_available_commands()

        return {
            "introspection_summary": {
                "total_events": len(introspection_events),
                "user_commands": len(user_commands),
                "automatic_triggers": len(automatic_triggers),
                "analysis_period_days": days,
                "most_used_command": (
                    max(command_types.keys(), key=command_types.get)
                    if command_types
                    else None
                ),
                "most_common_trigger": (
                    max(trigger_reasons.keys(), key=trigger_reasons.get)
                    if trigger_reasons
                    else None
                ),
            },
            "recent_events": [
                {
                    "timestamp": e.get("ts"),
                    "type": e.get("kind"),
                    "summary": e.get("content", "")[:100],
                    "tags": e.get("meta", {}).get("tags", []),
                }
                for e in introspection_events[-10:]  # Last 10 events
            ],
            "command_usage": command_types,
            "trigger_patterns": trigger_reasons,
            "available_commands": available_commands,
            "configuration": {
                "automatic_enabled": config.enable_automatic,
                "notification_enabled": config.notify_automatic,
                "confidence_threshold": config.notify_threshold,
                "lookback_days": config.lookback_days,
                "trait_drift_threshold": config.trait_drift_threshold,
                "emergence_plateau_days": config.emergence_plateau_days,
            },
            "system_status": {
                "engine_initialized": True,
                "last_check": engine.last_automatic_check.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "cache_size": len(engine.analysis_cache),
            },
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "db_path": db,
        }

    except Exception as e:
        return {
            "error": str(e),
            "introspection_summary": {
                "total_events": 0,
                "user_commands": 0,
                "automatic_triggers": 0,
            },
            "note": "Failed to load introspection system status.",
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

        # Create analyzer with SQLite storage for real commitment close rate
        store = SQLiteStore(db)
        analyzer = EmergenceAnalyzer(storage_manager=store)

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


# ---- Phase 3C Enhanced Probe Endpoints ----


@app.get("/reflection/quality")
def reflection_quality(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    limit: int = Query(
        20, ge=1, le=100, description="Number of recent reflections to analyze"
    ),
    window_days: int = Query(7, ge=1, le=30, description="Analysis window in days"),
):
    """
    Phase 3C: Analyze reflection quality with semantic novelty and salience scoring.

    Returns:
    - Semantic novelty scores and duplicate detection
    - Reflection quality metrics (specificity, self-reference, planning)
    - Meta-cognitive pattern analysis
    - Recommendations for improvement
    """
    try:
        # Load recent reflection events
        reflection_events = _load_events(db, limit, "reflection")

        if not reflection_events:
            return {
                "total_reflections": 0,
                "avg_quality": 0.0,
                "avg_novelty": 0.0,
                "duplicate_rate": 0.0,
                "patterns": [],
                "recommendations": [],
            }

        # Convert to format expected by meta-reflection analyzer
        reflections = []
        for event in reflection_events:
            reflections.append(
                {
                    "content": event.get("content", ""),
                    "timestamp": event.get("ts", ""),
                    "meta": event.get("meta", {}),
                }
            )

        # Analyze patterns using meta-reflection analyzer
        meta_analyzer = get_meta_reflection_analyzer()
        analysis = meta_analyzer.analyze_reflection_patterns(reflections, window_days)

        # Add semantic analysis details
        semantic_analyzer = get_semantic_analyzer()
        reflection_texts = [r["content"] for r in reflections]

        # Calculate novelty scores for each reflection
        novelty_details = []
        for i, text in enumerate(reflection_texts):
            previous_texts = reflection_texts[:i] if i > 0 else []
            novelty = semantic_analyzer.semantic_novelty_score(text, previous_texts)
            is_duplicate = semantic_analyzer.is_semantic_duplicate(
                text, previous_texts, threshold=0.8
            )
            novelty_details.append(
                {
                    "reflection_index": i,
                    "novelty_score": novelty,
                    "is_duplicate": is_duplicate,
                    "content_preview": text[:100] + "..." if len(text) > 100 else text,
                }
            )

        return {
            **analysis,
            "novelty_details": novelty_details,
            "semantic_clusters": len(
                semantic_analyzer.cluster_similar_texts(reflection_texts, 0.7)
            ),
            "db_path": db,
            "analysis_window_days": window_days,
        }

    except Exception as e:
        return {
            "error": str(e),
            "total_reflections": 0,
            "avg_quality": 0.0,
            "avg_novelty": 0.0,
        }


@app.get("/emergence/trends")
def emergence_trends(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    days: int = Query(30, ge=1, le=90, description="Number of days to analyze"),
    window_size: int = Query(
        15, ge=5, le=50, description="Window size for each measurement"
    ),
):
    """
    Phase 3C: Historical emergence pattern analysis showing IAS/GAS trends over time.

    Returns:
    - Time-series data of emergence scores
    - Stage progression analysis
    - Commitment close rate trends
    - Personality evolution patterns
    """
    try:
        store = SQLiteStore(db)

        # Get events from the specified time period
        from datetime import datetime, timedelta

        try:
            from datetime import UTC
        except ImportError:
            from datetime import timezone

            UTC = timezone.utc

        cutoff = datetime.now(UTC) - timedelta(days=days)

        # Load all events in time period
        all_events = store.recent_events(limit=days * 50)  # Heuristic limit

        # Filter events by time period and group by day
        daily_groups = {}
        for event in all_events:
            try:
                event_time = datetime.fromisoformat(event[1])  # ts column
                if event_time >= cutoff:
                    day_key = event_time.date().isoformat()
                    if day_key not in daily_groups:
                        daily_groups[day_key] = []
                    daily_groups[day_key].append(event)
            except Exception:
                continue

        # Calculate emergence scores for each day
        analyzer = EmergenceAnalyzer(storage_manager=store)
        trend_data = []

        for day_key in sorted(daily_groups.keys()):
            day_events = daily_groups[day_key]

            # Convert to EmergenceEvent format
            emergence_events = []
            for event in day_events[-window_size:]:  # Use recent events for that day
                emergence_events.append(
                    EmergenceEvent(
                        id=event[0],
                        timestamp=event[1],
                        kind=event[2],
                        content=event[3],
                        meta=(
                            json.loads(event[4])
                            if isinstance(event[4], str)
                            else (event[4] or {})
                        ),
                    )
                )

            if emergence_events:
                # Override get_recent_events for this calculation
                analyzer.get_recent_events = (
                    lambda kind="response", limit=window_size: emergence_events
                )
                scores = analyzer.compute_scores(window_size)

                trend_data.append(
                    {
                        "date": day_key,
                        "IAS": scores.get("IAS", 0.0),
                        "GAS": scores.get("GAS", 0.0),
                        "stage": scores.get("stage", "S0: Substrate"),
                        "commit_close_rate": scores.get("commit_close_rate", 0.0),
                        "events_count": len(emergence_events),
                    }
                )

        # Calculate trends
        if len(trend_data) >= 2:
            ias_trend = trend_data[-1]["IAS"] - trend_data[0]["IAS"]
            gas_trend = trend_data[-1]["GAS"] - trend_data[0]["GAS"]
        else:
            ias_trend = 0.0
            gas_trend = 0.0

        return {
            "trend_data": trend_data,
            "summary": {
                "days_analyzed": len(trend_data),
                "ias_trend": ias_trend,
                "gas_trend": gas_trend,
                "current_stage": (
                    trend_data[-1]["stage"] if trend_data else "S0: Substrate"
                ),
                "avg_close_rate": (
                    sum(d["commit_close_rate"] for d in trend_data) / len(trend_data)
                    if trend_data
                    else 0.0
                ),
            },
            "db_path": db,
            "window_size": window_size,
        }

    except Exception as e:
        return {
            "error": str(e),
            "trend_data": [],
            "summary": {"days_analyzed": 0, "ias_trend": 0.0, "gas_trend": 0.0},
        }


@app.get("/personality/adaptation")
def personality_adaptation(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    limit: int = Query(
        50, ge=10, le=200, description="Number of recent events to analyze"
    ),
):
    """
    Phase 3C: Real-time personality evolution tracking showing trait drift patterns.

    Returns:
    - Recent trait changes and their triggers
    - Adaptation patterns based on emergence signals
    - Evidence-based personality evolution
    """
    try:
        # Load recent events that could trigger personality changes
        events = _load_events(db, limit, None)  # All event types

        # Filter for events that typically trigger trait drift
        adaptation_events = []
        for event in events:
            kind = event.get("kind", "")
            if kind in ["reflection", "evidence", "commitment", "insight"]:
                adaptation_events.append(event)

        # Analyze adaptation patterns
        patterns = {
            "recent_adaptations": len(adaptation_events),
            "adaptation_triggers": {},
            "evidence_based_changes": 0,
            "reflection_driven_changes": 0,
        }

        # Count triggers by type
        for event in adaptation_events:
            kind = event.get("kind", "unknown")
            patterns["adaptation_triggers"][kind] = (
                patterns["adaptation_triggers"].get(kind, 0) + 1
            )

            if kind == "evidence":
                patterns["evidence_based_changes"] += 1
            elif kind == "reflection":
                patterns["reflection_driven_changes"] += 1

        # Calculate adaptation rate (adaptations per day, roughly)
        adaptation_rate = len(adaptation_events) / max(1, limit / 10)  # Rough estimate

        return {
            "adaptation_patterns": patterns,
            "adaptation_rate": adaptation_rate,
            "recent_events": adaptation_events[-10:],  # Last 10 adaptation events
            "db_path": db,
            "events_analyzed": len(adaptation_events),
        }

    except Exception as e:
        return {"error": str(e), "adaptation_patterns": {}, "adaptation_rate": 0.0}


@app.get("/meta-cognition")
def meta_cognition(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    limit: int = Query(
        30, ge=10, le=100, description="Number of recent reflections to analyze"
    ),
):
    """
    Phase 3C: Meta-cognitive analysis showing AI awareness of its own reflection patterns.

    Returns:
    - Self-awareness metrics about reflection quality
    - Pattern recognition in own behavior
    - Meta-insights about cognitive processes
    """
    try:
        # Load recent reflection events
        reflection_events = _load_events(db, limit, "reflection")

        if not reflection_events:
            return {
                "meta_insight": None,
                "self_awareness_score": 0.0,
                "cognitive_patterns": [],
            }

        # Convert to format for meta-reflection analysis
        reflections = []
        for event in reflection_events:
            reflections.append(
                {
                    "content": event.get("content", ""),
                    "timestamp": event.get("ts", ""),
                    "meta": event.get("meta", {}),
                }
            )

        # Generate meta-cognitive analysis
        meta_analyzer = get_meta_reflection_analyzer()
        pattern_analysis = meta_analyzer.analyze_reflection_patterns(
            reflections, window_days=14
        )

        # Generate meta-insight
        meta_insight = meta_analyzer.generate_meta_insight(pattern_analysis)

        # Calculate self-awareness score based on meta-cognitive indicators
        self_awareness_score = 0.0
        if pattern_analysis["avg_quality"] > 0:
            self_awareness_score += 0.3 * pattern_analysis["avg_quality"]
        if pattern_analysis["novelty_trend"] > 0:
            self_awareness_score += 0.3 * pattern_analysis["novelty_trend"]
        if len(pattern_analysis["recommendations"]) > 0:
            self_awareness_score += 0.2  # Ability to generate self-recommendations
        if meta_insight:
            self_awareness_score += 0.2  # Ability to generate meta-insights

        return {
            "meta_insight": meta_insight,
            "self_awareness_score": min(1.0, self_awareness_score),
            "cognitive_patterns": pattern_analysis["patterns"],
            "self_recommendations": pattern_analysis["recommendations"],
            "reflection_analysis": {
                "total_reflections": pattern_analysis["total_reflections"],
                "avg_quality": pattern_analysis["avg_quality"],
                "novelty_trend": pattern_analysis["novelty_trend"],
                "duplicate_rate": pattern_analysis.get("duplicate_rate", 0.0),
            },
            "db_path": db,
        }

    except Exception as e:
        return {"error": str(e), "meta_insight": None, "self_awareness_score": 0.0}


def create_probe_app(db_path: str = "pmm.db") -> FastAPI:
    """Create and configure the probe FastAPI app."""
    return app


# --- Minimal bandit + backlog probes ---


@app.get("/bandit/reflection")
def bandit_reflection():
    """Expose reflection template bandit's simple stats (if available)."""
    try:
        import pmm.reflection as ref

        bandit = getattr(ref, "_bandit", None)
        counts = getattr(bandit, "counts", []) if bandit else []
        rewards = getattr(bandit, "rewards", []) if bandit else []
        epsilon = getattr(bandit, "epsilon", None) if bandit else None
        n = getattr(bandit, "n", None) if bandit else None

        # Fallback to persisted bandit state when runtime is empty
        from pmm.self_model_manager import SelfModelManager

        persisted_counts = []
        persisted_rewards = []
        try:
            mgr = SelfModelManager("persistent_self_model.json")
            mc = mgr.model.meta_cognition
            persisted_counts = list(getattr(mc, "bandit_counts", []) or [])
            persisted_rewards = list(getattr(mc, "bandit_rewards", []) or [])
        except Exception:
            pass

        def _is_nonzero(arr):
            try:
                return any((x or 0) != 0 for x in arr)
            except Exception:
                return False

        source = "runtime" if _is_nonzero(counts) else "persisted"
        if source == "persisted" and _is_nonzero(persisted_counts):
            counts = persisted_counts
            rewards = persisted_rewards
            if not n:
                n = len(counts)
        # compute avg rewards where possible
        avgs = []
        try:
            for i in range(len(counts or [])):
                avgs.append(
                    (float(rewards[i]) / float(counts[i])) if counts[i] else 0.0
                )
        except Exception:
            avgs = []
        return {
            "available": True,
            "source": source,
            "n": n,
            "epsilon": epsilon,
            "counts": counts,
            "rewards": rewards,
            "avg_rewards": avgs,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.get("/embeddings/backlog")
def embeddings_backlog(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    kinds: Optional[str] = Query(None, description="CSV of kinds to filter (optional)"),
):
    """Report approximate backlog of unembedded events."""
    try:
        store = SQLiteStore(db)
        kind_list = [k.strip() for k in kinds.split(",")] if kinds else None
        cnt = store.count_unembedded_events(kind_list)
        return {"db_path": db, "unembedded": cnt}
    except Exception as e:
        return {"error": str(e), "db_path": db}


@app.get("/metrics/hourly")
def metrics_hourly(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    limit: int = Query(
        24, ge=1, le=240, description="Number of recent metrics to return"
    ),
):
    """Return recent autonomy_tick metrics emitted by the autonomy loop.

    These are stored as events with kind='event' and meta.type='autonomy_tick'.
    """
    try:
        store = SQLiteStore(db)
        # Fetch recent 'event' rows with meta.type='autonomy_tick'
        rows = list(
            store.conn.execute(
                "SELECT id,ts,kind,content,meta,prev_hash,hash FROM events WHERE kind='event' ORDER BY id DESC LIMIT ?",
                (limit * 3,),  # oversample, filter below
            )
        )
        out = []
        import json

        for r in rows:
            try:
                meta = json.loads(r[4]) if isinstance(r[4], str) else (r[4] or {})
            except Exception:
                meta = {}
            if str(meta.get("type", "")) != "autonomy_tick":
                continue
            out.append(
                {
                    "id": r[0],
                    "ts": r[1],
                    "content": r[3],
                    "metrics": meta.get("evidence", meta),
                }
            )
            if len(out) >= limit:
                break
        return {"items": out}
    except Exception as e:
        return {"error": str(e), "items": []}


@app.get("/reflection/contract")
def reflection_contract(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    limit: int = Query(
        30, ge=1, le=200, description="Number of recent reflections to inspect"
    ),
):
    """Show pass/fail stats for the 'Next:' contract across recent reflections."""
    try:
        items = _load_events(db, limit, "reflection")
        passed = 0
        failed = 0
        reasons = {}
        rows = []
        first_ok = 0
        final_ok = 0
        rerolled_cnt = 0
        for ev in items:
            meta = ev.get("meta", {}) or {}
            status = meta.get("status")
            reason = meta.get("reason")
            contract = meta.get("contract") or {}
            fp = contract.get("first_pass")
            fr = contract.get("first_reason")
            final = "ok" if status == "ok" else "inert"
            if fp == "ok":
                first_ok += 1
            if final == "ok":
                final_ok += 1
            if contract.get("rerolled"):
                rerolled_cnt += 1
            if status == "inert":
                failed += 1
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
            else:
                passed += 1
            rows.append(
                {
                    "id": ev.get("id"),
                    "ts": ev.get("ts"),
                    "status": (status or "ok"),
                    "reason": reason,
                    "first_pass": fp,
                    "first_reason": fr,
                    "rerolled": bool(contract.get("rerolled")),
                    "preview": (ev.get("content", "")[:100]),
                }
            )
        total = passed + failed
        pass_rate = (passed / total) if total else 0.0
        first_pass_rate = (first_ok / total) if total else 0.0
        final_pass_rate = (final_ok / total) if total else 0.0
        rerolled_share = (rerolled_cnt / total) if total else 0.0
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 3),
            "first_pass_rate": round(first_pass_rate, 3),
            "final_pass_rate": round(final_pass_rate, 3),
            "rerolled_share": round(rerolled_share, 3),
            "reasons": reasons,
            "items": rows,
        }
    except Exception as e:
        return {"error": str(e), "total": 0, "passed": 0, "failed": 0, "items": []}


@app.get("/scenes")
def scenes(
    limit: int = Query(2, ge=1, le=10, description="Number of recent scenes to return"),
):
    """Return last N narrative scenes from the JSON self-model."""
    try:
        from pmm.self_model_manager import SelfModelManager

        mgr = SelfModelManager("persistent_self_model.json")
        sc_list = getattr(mgr.model.narrative_identity, "scenes", []) or []
        out = []
        for sc in sc_list[-limit:][::-1]:
            out.append(
                {
                    "id": getattr(sc, "id", None),
                    "t": getattr(sc, "t", None),
                    "type": getattr(sc, "type", None),
                    "summary": (getattr(sc, "summary", "") or "")[:300],
                    "tags": getattr(sc, "tags", []),
                }
            )
        return {"count": len(sc_list), "items": out}
    except Exception as e:
        return {"error": str(e), "count": 0, "items": []}


# --- Feedback summary probe ---


@app.get("/feedback/summary")
def feedback_summary(
    db: str = Query("pmm.db", description="Path to PMM SQLite DB"),
    limit: int = Query(
        200, ge=1, le=2000, description="Number of recent feedback events to include"
    ),
):
    """Summarize feedback events logged from chat.

    Computes count and average rating (1-5) and returns recent items.
    """
    try:
        store = SQLiteStore(db)
        rows = list(
            store.conn.execute(
                "SELECT id, ts, content, meta FROM events WHERE kind='feedback' ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        )
        items = []
        ratings = []
        conservative = []
        for r in rows:
            try:
                meta = json.loads(r[3]) if isinstance(r[3], str) else (r[3] or {})
            except Exception:
                meta = {}
            rating = meta.get("rating")
            if isinstance(rating, (int, float)):
                try:
                    ratings.append(float(rating))
                except Exception:
                    pass
            # Conservative score heuristic: derive booleans from the linked response, if available
            ok = False
            has_evid = False
            not_dup = False
            novel = False
            try:
                resp_id = int(meta.get("response_ref") or 0)
                if resp_id:
                    rr = store.conn.execute(
                        "SELECT content, meta FROM events WHERE id=? AND kind='response' LIMIT 1",
                        (resp_id,),
                    ).fetchone()
                    if rr:
                        rcontent = rr[0] or ""
                        ok = "Next:" in (rcontent or "")
                        has_evid = "ev" in (rcontent or "").lower()
                        not_dup = True  # best‑effort: no direct signal, assume true
                        novel = True  # best‑effort: no direct signal, assume true
            except Exception:
                pass
            cons = min(5, 1 + int(ok) + int(has_evid) + int(not_dup) + int(novel))
            conservative.append(cons)
            items.append(
                {
                    "id": r[0],
                    "ts": r[1],
                    "content": (r[2] or "")[:200],
                    "rating": rating,
                    "response_ref": meta.get("response_ref"),
                }
            )
        avg = (sum(ratings) / len(ratings)) if ratings else None
        avgc = (sum(conservative) / len(conservative)) if conservative else None
        return {
            "count": len(rows),
            "avg_rating": (round(avg, 3) if avg is not None else None),
            "avg_conservative": (round(avgc, 3) if avgc is not None else None),
            "items": items[:30],
        }
    except Exception as e:
        return {"error": str(e), "count": 0, "avg_rating": None, "items": []}


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="PMM Probe API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"🚀 Starting PMM Probe API on http://{args.host}:{args.port}")
    print("📊 Available endpoints:")
    print("  - /health - Health check")
    print("  - /emergence - Emergence analysis")
    print("  - /reflection/quality - Phase 3C reflection quality analysis")
    print("  - /meta-cognition - Phase 3C meta-cognitive insights")
    print("  - /emergence/trends - Phase 3C emergence trends")
    print("  - /personality/adaptation - Phase 3C personality adaptation")

    uvicorn.run("pmm.api.probe:app", host=args.host, port=args.port, reload=args.reload)
