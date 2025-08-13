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
    ):
        """Append new event to the chain."""
        self.conn.execute(
            "INSERT INTO events(ts,kind,content,meta,prev_hash,hash) VALUES(?,?,?,?,?,?)",
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                kind,
                content,
                json.dumps(meta, ensure_ascii=False),
                prev,
                hsh,
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
