import pytest

from pmm.emergence import compute_emergence_scores


@pytest.mark.usefixtures("reset_analyzer")
def test_metrics_parity_commit_close_rate(sqlite_store):
    store = sqlite_store

    # Seed a few turns
    for i in range(3):
        store.append_event(kind="conversation", content=f"u{i}", meta={})
        store.append_event(kind="self_expression", content=f"a{i}", meta={})

    # Seed 5 commitments
    commits = [
        store.append_event(kind="commitment", content=f"c{i}", meta={})["hash"]
        for i in range(5)
    ]

    # Close 3 of them via evidence
    for h in commits[:2]:
        store.append_event(kind="evidence", content="done", meta={"commit_ref": h})
    # Use short-hash prefix for third
    store.append_event(
        kind="evidence", content="done", meta={"commit_ref": commits[2][:10]}
    )

    scores = compute_emergence_scores(storage_manager=store)

    # Analyzer value (already rounded to 3 decimals per implementation)
    close_rate = float(scores.get("commit_close_rate", 0.0) or 0.0)

    # Recompute directly from DB state
    # Fetch recent commitments and evidence
    conn = store.conn
    cur = conn.cursor()
    cur.execute("SELECT hash FROM events WHERE kind='commitment'")
    commit_hashes = [row[0] for row in cur.fetchall()]

    cur.execute(
        "SELECT meta FROM events WHERE kind='evidence' OR kind LIKE 'evidence:%'"
    )
    evidence_meta_rows = [row[0] for row in cur.fetchall()]

    import json

    closed = set()
    for m in evidence_meta_rows:
        meta = json.loads(m or "{}")
        ref = meta.get("commit_ref")
        if not ref:
            continue
        # exact or short-prefix match
        if ref in commit_hashes or any(h.startswith(ref) for h in commit_hashes):
            closed.add(ref)

    parity = (len(closed) / len(commit_hashes)) if commit_hashes else 0.0

    assert round(close_rate, 3) == round(parity, 3)
