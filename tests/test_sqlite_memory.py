#!/usr/bin/env python3
"""
Deterministic SQLite memory test that does not depend on external state.
"""

import os
import tempfile

from pmm.storage.sqlite_store import SQLiteStore


def test_sqlite_memory():
    print("🧪 Testing SQLite Memory Storage (deterministic)...")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path = tmp_db.name

    try:
        store = SQLiteStore(db_path)

        # Seed known events
        res1 = store.append_event(
            kind="prompt",
            content="User said: My name is Scott",
            meta={"role": "user"},
        )
        res2 = store.append_event(
            kind="response",
            content="I responded: Nice to meet you, Scott!",
            meta={"role": "assistant"},
        )
        res3 = store.append_event(
            kind="event",
            content="Context: Discussing project updates",
            meta={},
        )

        recent = store.recent_events(limit=10)
        assert len(recent) >= 3

        # Verify 'Scott' is present in at least one recent content
        scott_mentions = [e for e in recent if "scott" in e["content"].lower()]
        assert len(scott_mentions) >= 1

        # Verify conversation extraction skeleton
        user_lines = [e for e in recent if e["kind"] == "prompt"]
        assistant_lines = [e for e in recent if e["kind"] == "response"]
        assert user_lines and assistant_lines

    finally:
        try:
            os.unlink(db_path)
        except Exception:
            pass
