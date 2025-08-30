from pmm.storage.sqlite_store import SQLiteStore
from pmm.emergence import EmergenceAnalyzer


def _make_store(path):
    return SQLiteStore(str(path))


def _seed_session(store, n_turns=3, n_commit=2, n_close=1):
    for i in range(n_turns):
        store.append_event(kind="conversation", content=f"U{i}", meta={})
        store.append_event(kind="self_expression", content=f"A{i}", meta={})
    commits = [
        store.append_event(kind="commitment", content=f"C{i}", meta={})["hash"]
        for i in range(n_commit)
    ]
    for h in commits[:n_close]:
        store.append_event(kind="evidence", content="done", meta={"commit_ref": h})
    return commits


def test_ab_minimal(tmp_path, monkeypatch):
    # Keep analyzer deterministic
    monkeypatch.setenv(
        "PMM_EMERGENCE_TYPES",
        "self_expression,response,reflection,commitment,evidence",
    )
    monkeypatch.setenv("PMM_EMERGENCE_WINDOW", "50")

    # A: fewer closes
    db_a = tmp_path / "a.sqlite"
    store_a = _make_store(db_a)
    _seed_session(store_a, n_turns=4, n_commit=4, n_close=1)
    # Use analyzer's direct close-rate computation (adaptive path omits it)
    analyzer_a = EmergenceAnalyzer(storage_manager=store_a)
    close_a = analyzer_a.commitment_close_rate(window=50)

    # B: more closes
    db_b = tmp_path / "b.sqlite"
    store_b = _make_store(db_b)
    _seed_session(store_b, n_turns=4, n_commit=4, n_close=3)
    analyzer_b = EmergenceAnalyzer(storage_manager=store_b)
    close_b = analyzer_b.commitment_close_rate(window=50)

    # Analyzer deltas
    delta_close = float(close_b) - float(close_a)

    # Recompute close rates directly from DBs and compare deltas
    def _close_rate(store: SQLiteStore) -> float:
        cur = store.conn.cursor()
        cur.execute("SELECT hash FROM events WHERE kind='commitment'")
        commits = [r[0] for r in cur.fetchall()]
        cur.execute(
            "SELECT meta FROM events WHERE kind='evidence' OR kind LIKE 'evidence:%'"
        )
        metas = [r[0] for r in cur.fetchall()]
        import json

        closed = set()
        for m in metas:
            meta = json.loads(m or "{}")
            ref = meta.get("commit_ref")
            if not ref:
                continue
            if ref in commits or any(h.startswith(ref) for h in commits):
                closed.add(ref)
        return round((len(closed) / len(commits)) if commits else 0.0, 3)

    delta_parity = _close_rate(store_b) - _close_rate(store_a)

    assert round(delta_close, 3) == round(delta_parity, 3)
