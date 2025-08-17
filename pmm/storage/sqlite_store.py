import sqlite3
import json
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

DDL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,         -- 'prompt' | 'response' | 'reflection' | 'commitment' | 'evidence'
  content TEXT NOT NULL,      -- raw text
  meta TEXT NOT NULL,         -- JSON: {"model":"gpt-4o-mini","role":"user",...}
  prev_hash TEXT,             -- hex
  hash TEXT NOT NULL          -- hex
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_hash ON events(hash);
CREATE INDEX IF NOT EXISTS idx_events_prev_hash ON events(prev_hash);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
"""


class SQLiteStore:
    """Append-only SQLite WAL store with hash-chain integrity."""

    def __init__(self, path: str = "pmm.db"):
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(DDL)
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        # Try to add optional efficient-thought columns if they do not exist yet
        try:
            self.conn.execute("ALTER TABLE events ADD COLUMN summary TEXT")
        except Exception:
            pass
        try:
            self.conn.execute(
                "ALTER TABLE events ADD COLUMN keywords TEXT"
            )  # JSON-encoded list[str]
        except Exception:
            pass
        try:
            self.conn.execute("ALTER TABLE events ADD COLUMN embedding BLOB")
        except Exception:
            pass
        self.conn.commit()

    def latest_hash(self) -> Optional[str]:
        """Get the hash of the most recent event."""
        with self._lock:
            row = self.conn.execute(
                "SELECT hash FROM events ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def append_event(
        self,
        kind: str,
        content: str,
        meta: Dict[str, Any],
        hsh: str,
        prev: Optional[str],
        *,
        summary: Optional[str] = None,
        keywords: Optional[list] = None,
        embedding: Optional[bytes] = None,
    ):
        """Append new event to the chain.

        summary/keywords/embedding are optional and persisted when available.
        keywords will be JSON-encoded for storage.
        """
        kw_json = json.dumps(keywords or [], ensure_ascii=False)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO events(ts,kind,content,meta,prev_hash,hash,summary,keywords,embedding)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    ts,
                    kind,
                    content,
                    json.dumps(meta, ensure_ascii=False),
                    prev,
                    hsh,
                    summary,
                    kw_json,
                    embedding,
                ),
            )
            self.conn.commit()

    def all_events(self):
        """Get all events in chronological order."""
        with self._lock:
            rows = list(
                self.conn.execute(
                    "SELECT id,ts,kind,content,meta,prev_hash,hash,summary,keywords,embedding FROM events ORDER BY id"
                )
            )
        return [self._row_to_dict(r) for r in rows]

    def recent_events(self, limit: int = 10):
        """Get recent events for context."""
        with self._lock:
            rows = list(
                self.conn.execute(
                    """
                SELECT id,ts,kind,content,meta,prev_hash,hash,summary,keywords,embedding
                FROM events ORDER BY id DESC LIMIT ?
                """,
                    (limit,),
                )
            )
        return [self._row_to_dict(r) for r in rows]

    def _row_to_dict(self, r: sqlite3.Row) -> Dict[str, Any]:
        try:
            meta = json.loads(r["meta"]) if r["meta"] else {}
        except Exception:
            meta = {}
        try:
            keywords = json.loads(r["keywords"]) if r["keywords"] else []
        except Exception:
            keywords = []
        return {
            "id": r["id"],
            "ts": r["ts"],
            "kind": r["kind"],
            "content": r["content"],
            "meta": meta,
            "prev_hash": r["prev_hash"],
            "hash": r["hash"],
            "summary": r["summary"],
            "keywords": keywords,
            "embedding": r["embedding"],
        }

    def semantic_search(
        self, query_embedding: bytes, limit: int = 10, kind_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find events most similar to query embedding using cosine similarity."""
        import numpy as np

        if not query_embedding:
            return []

        query_vec = np.frombuffer(query_embedding, dtype=np.float32)

        # Get events with embeddings
        with self._lock:
            if kind_filter:
                rows = list(
                    self.conn.execute(
                        "SELECT id,ts,kind,content,meta,prev_hash,hash,summary,keywords,embedding FROM events WHERE embedding IS NOT NULL AND kind=? ORDER BY id DESC",
                        (kind_filter,),
                    )
                )
            else:
                rows = list(
                    self.conn.execute(
                        "SELECT id,ts,kind,content,meta,prev_hash,hash,summary,keywords,embedding FROM events WHERE embedding IS NOT NULL ORDER BY id DESC"
                    )
                )

        # Calculate similarities
        similarities = []
        for row in rows:
            try:
                event_embedding = row["embedding"]
                if not event_embedding:
                    continue

                event_vec = np.frombuffer(event_embedding, dtype=np.float32)

                # Cosine similarity
                dot_product = np.dot(query_vec, event_vec)
                norm_query = np.linalg.norm(query_vec)
                norm_event = np.linalg.norm(event_vec)

                if norm_query == 0 or norm_event == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm_query * norm_event)

                similarities.append((similarity, self._row_to_dict(row)))

            except Exception:
                continue

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [event for _, event in similarities[:limit]]

    def get_events_with_embeddings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events that have embeddings for semantic analysis."""
        with self._lock:
            rows = list(
                self.conn.execute(
                    """
                SELECT id,ts,kind,content,meta,prev_hash,hash,summary,keywords,embedding
                FROM events WHERE embedding IS NOT NULL ORDER BY id DESC LIMIT ?
                """,
                    (limit,),
                )
            )
        return [self._row_to_dict(r) for r in rows]

    def verify_chain(self, rehasher) -> bool:
        """Recompute each row's hash with `rehasher(row_dict)` and verify linkage."""
        rows = self.all_events()
        prev = None
        for row in rows:
            if row["prev_hash"] != prev:
                return False
            if rehasher(row) != row["hash"]:
                return False
            prev = row["hash"]
        return True

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
