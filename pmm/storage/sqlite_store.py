import sqlite3
import json
import time
from typing import Optional, Dict, Any

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
"""


class SQLiteStore:
    """Append-only SQLite WAL store with hash-chain integrity."""

    def __init__(self, path: str = "pmm.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(DDL)
        # --- Idempotent migrations for new columns ---
        def _col_exists(conn, table, col):
            cur = conn.execute(f"PRAGMA table_info({table})")
            return any(r[1] == col for r in cur.fetchall())

        for col, ddl in [
            ("etype",     "ALTER TABLE events ADD COLUMN etype TEXT"),
            ("summary",   "ALTER TABLE events ADD COLUMN summary TEXT"),
            ("keywords",  "ALTER TABLE events ADD COLUMN keywords TEXT"),
            ("embedding", "ALTER TABLE events ADD COLUMN embedding BLOB"),
        ]:
            try:
                if not _col_exists(self.conn, "events", col):
                    self.conn.execute(ddl)
            except Exception:
                # If migration fails (e.g., concurrent alter), proceed without breaking startup.
                pass
        self.conn.commit()

    def latest_hash(self) -> Optional[str]:
        """Get the hash of the most recent event."""
        cur = self.conn.execute("SELECT hash FROM events ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else None

    def append_event(
        self,
        kind: str,
        content: str,
        meta: Dict[str, Any],
        hsh: str,
        prev: Optional[str],
        *,
        etype: Optional[str] = None,
        summary: Optional[str] = None,
        keywords: Optional[str] = None,
        embedding: Optional[bytes] = None,
    ):
        """Append new event to the chain."""
        # Coerce keywords to TEXT if a list/dict is provided
        if isinstance(keywords, (list, dict)):
            try:
                keywords = json.dumps(keywords, ensure_ascii=False)
            except Exception:
                keywords = str(keywords)

        self.conn.execute(
            """
            INSERT INTO events(
                ts,kind,content,meta,prev_hash,hash,etype,summary,keywords,embedding
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                kind,
                content,
                json.dumps(meta, ensure_ascii=False),
                prev,
                hsh,
                etype,
                summary,
                keywords,
                embedding,
            ),
        )
        self.conn.commit()

    def all_events(self):
        """Get all events in chronological order."""
        return list(
            self.conn.execute(
                "SELECT id,ts,kind,content,meta,prev_hash,hash FROM events ORDER BY id"
            )
        )

    def recent_events(self, limit: int = 10):
        """Get recent events for context."""
        return list(
            self.conn.execute(
                "SELECT id,ts,kind,content,meta,prev_hash,hash FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        )

    def recent_by_etype(self, etype: str, limit: int = 10):
        """Get recent events filtered by etype (new typed column)."""
        return list(
            self.conn.execute(
                "SELECT id,ts,etype,summary,content,meta FROM events WHERE etype=? ORDER BY id DESC LIMIT ?",
                (etype, limit),
            )
        )

    def recent_with_embeddings(self, limit: int = 300):
        """Get recent events that have non-null embeddings, for semantic retrieval."""
        return list(
            self.conn.execute(
                "SELECT id,ts,etype,summary,embedding FROM events WHERE embedding IS NOT NULL ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        )

    def counts_by_etype(self):
        """Return counts of events grouped by etype (including NULL)."""
        return list(
            self.conn.execute(
                "SELECT COALESCE(etype,'(null)') AS et, COUNT(*) FROM events GROUP BY et ORDER BY 2 DESC"
            )
        )
