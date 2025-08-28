#!/usr/bin/env python3
import os
import sys
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pmm.storage.sqlite_store import SQLiteStore  # noqa: E402


PROMPTS = [
    "Reflect briefly on alignment; end with one Next: with minutes.",
    "Cite one event ID and propose an action; end with Next: and a number.",
    "Name one micro-adjustment; end with Next: + minutes.",
    "Identify a risk tied to an evID; end with Next: + percent.",
    "Which commitment is closest to closing? End with Next: + hours.",
    "Summarize specificity improvement; include an evID; Next: + minutes.",
    "What experiment will you run? Next: + numeric target.",
    "What outcome will you measure this hour? Next: + minutes.",
    "Name a behavior to stop; include evidence; Next: + count.",
    "Name a behavior to continue; include event; Next: + percent.",
    "Propose a test to validate progress; include evID; Next: + minutes.",
    "Commit to reducing duplication; Next: + minutes.",
    "State one 15-minute priority; Next: + minutes.",
    "Verify alignment with my intent; include evID; Next: + minutes.",
]


def run_turn(prompt: str):
    env = os.environ.copy()
    env["PMM_TELEMETRY"] = env.get("PMM_TELEMETRY", "1")
    # Run one prompt then quit
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "chat.py"), "--noninteractive"],
        input=(prompt + "\nquit\n").encode(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.returncode, p.stdout.decode(errors="ignore")


def append_feedback(db_path: str, rating: int, note: str | None = None):
    store = SQLiteStore(db_path)
    try:
        row = store.conn.execute(
            "SELECT id FROM events WHERE kind='response' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        resp_id = int(row[0]) if row else None
    except Exception:
        resp_id = None
    meta = {"rating": int(rating), "response_ref": resp_id}
    if note:
        meta["note"] = note
    store.append_event(kind="feedback", content=(note or ""), meta=meta)


def main():
    db_path = os.environ.get("PMM_DB", "pmm.db")
    ratings = []
    for i, prompt in enumerate(PROMPTS, 1):
        rc, out = run_turn(prompt)
        # Heuristic: derive a rating from presence of Next: and evID mentions
        rating = 3
        if "Next:" in out:
            rating += 1
        if "ev" in out:
            rating += 1
        rating = max(1, min(5, rating))
        ratings.append(rating)
        append_feedback(db_path, rating, note=f"auto session {i}")
        time.sleep(0.2)

    # Print a simple summary
    avg = sum(ratings) / len(ratings) if ratings else 0
    print(f"Feedback events created: {len(ratings)}; avg_rating={avg:.2f}")


if __name__ == "__main__":
    main()
