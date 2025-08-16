from __future__ import annotations
import json
from typing import List, Optional
from fastapi import FastAPI, Query
from dotenv import load_dotenv
from pmm.storage.sqlite_store import SQLiteStore
from pmm.storage.integrity import verify_chain
from pmm.emergence import EmergenceAnalyzer, EmergenceEvent
from pmm.semantic_analysis import get_semantic_analyzer
from pmm.meta_reflection import get_meta_reflection_analyzer

# Load environment variables from .env file
load_dotenv()

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
