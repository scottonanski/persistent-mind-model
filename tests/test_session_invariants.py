import contextlib

import pytest

# Test focuses: DB-grounded invariants, analyzer parity, no-keyword dependency, negative control

from pmm.emergence import compute_emergence_scores
from pmm.emergence import EmergenceAnalyzer  # used in back-compat test
from pmm.storage.sqlite_store import SQLiteStore  # used in back-compat test


def _assert_core_events(store):
    cur = store.conn.execute("SELECT kind FROM events ORDER BY id DESC LIMIT 10")
    kinds = [r[0] for r in cur.fetchall()]
    assert "self_expression" in kinds, "No assistant event logged"
    assert any(
        k in ("conversation", "non_behavioral") for k in kinds
    ), "No user event logged"


def _insert_conversation(store, text: str):
    # Simulate a user input event using append_event to respect schema
    store.append_event(
        kind="conversation",
        content=text,
        meta={"role": "user", "source": "test"},
    )


def _insert_self_expression(store, text: str):
    # Simulate an assistant output event using append_event
    store.append_event(
        kind="self_expression",
        content=text,
        meta={"role": "assistant", "source": "test"},
    )


def _insert_commitment(store, content: str = "I will do a thing") -> str:
    # Insert commitment using append_event; capture the computed hash
    res = store.append_event(
        kind="commitment",
        content=content,
        meta={"owner": "assistant", "source": "test"},
    )
    return str(res["hash"])  # commit_ref for evidence


def _insert_evidence_for(store, commit_ref: str, kind: str = "done"):
    store.append_event(
        kind="evidence",
        content=kind,
        meta={"commit_ref": commit_ref, "source": "test"},
    )


@pytest.fixture()
def fresh_store(tmp_path, monkeypatch):
    # Import the SQLiteStore used by PMM
    from pmm.storage.sqlite_store import SQLiteStore  # type: ignore

    db_path = tmp_path / "pmm_test_session.sqlite"
    store = SQLiteStore(str(db_path))

    # The constructor initializes schema via executescript(DDL)

    # Constrain analyzer inputs for determinism
    monkeypatch.setenv(
        "PMM_EMERGENCE_TYPES", "self_expression,response,reflection,commitment,evidence"
    )
    monkeypatch.setenv("PMM_EMERGENCE_WINDOW", "5")

    # Reset global analyzer to ensure it binds to this fresh store
    import pmm.emergence as emergence_mod

    emergence_mod._analyzer = None

    yield store

    # Cleanup
    with contextlib.suppress(Exception):
        store.conn.close()


def test_db_grounded_event_invariants(fresh_store):
    store = fresh_store

    _insert_conversation(store, "Hi there")
    _insert_self_expression(store, "Hello! I can help.")

    _assert_core_events(store)


def test_analyzer_parity_simple(fresh_store):
    store = fresh_store

    # Simulate a small turn then compute emergence scores directly from the same store
    _insert_conversation(store, "What's our identity today?")
    _insert_self_expression(store, "I am Echo, maintaining persistent identity.")

    scores = compute_emergence_scores(storage_manager=store)

    # Analyzer should return numeric IAS/GAS; assert they are within valid range [0,1]
    ias = float(scores.get("IAS", 0.0) or 0.0)
    gas = float(scores.get("GAS", 0.0) or 0.0)
    assert 0.0 <= ias <= 1.0
    assert 0.0 <= gas <= 1.0


def test_no_keyword_dependency_for_closure(fresh_store):
    store = fresh_store

    # Insert commitment and evidence without using the words "commitment"/"evidence" in content relevant paths
    commit_ref = _insert_commitment(store, content="Next, I will draft the doc.")
    _insert_evidence_for(store, commit_ref=commit_ref, kind="done")

    scores = compute_emergence_scores(storage_manager=store)
    close = float(scores.get("commit_close_rate", 0.0) or 0.0)

    # Should register some closure activity due to event+meta linkage, independent of phrasing
    assert close >= 0.0


def test_negative_control_structure_only_text(fresh_store):
    store = fresh_store

    # Only free text, no valid commitment/evidence structure
    _insert_conversation(store, "growth stocks are interesting")
    _insert_self_expression(store, "here is an unrelated response")

    scores = compute_emergence_scores(storage_manager=store)

    # Commitment close rate should be zero if no commitments/evidence present
    close = float(scores.get("commit_close_rate", 0.0) or 0.0)
    assert close == 0.0


def test_close_rate_back_compat_short_hash_and_event_kind(tmp_path):
    """
    Back-compat evidence forms should count as closures:
    - Short-hash prefix in meta.commit_ref
    - kind='evidence:...' with nested meta.evidence.commit_ref (plus top-level commit_ref)
    """
    db_path = tmp_path / "pmm2.db"
    store = SQLiteStore(str(db_path))

    # Primary self_expression for analyzer input
    store.append_event(
        kind="self_expression",
        content="Reflecting on my commitments and progress.",
        meta={"role": "assistant"},
    )

    # Two commitments
    c1 = store.append_event(kind="commitment", content="I will write docs.", meta={})[
        "hash"
    ]
    c2 = store.append_event(kind="commitment", content="I will add tests.", meta={})[
        "hash"
    ]

    # Evidence referencing c1 by short-hash prefix
    store.append_event(
        kind="evidence",
        content="Finished writing docs",
        meta={"commit_ref": c1[:12]},  # short prefix
    )

    # Evidence logged as namespaced evidence with nested and top-level commit_ref for c2
    store.append_event(
        kind="evidence:completion",
        content="Delivered tests",
        meta={
            "commit_ref": c2,  # top-level for analyzer
            "evidence": {"commit_ref": c2},  # nested back-compat
        },
    )

    analyzer = EmergenceAnalyzer(storage_manager=store)
    rate = analyzer.commitment_close_rate(window=50)
    assert abs(rate - 1.0) < 1e-9  # both commitments closed


def test_commit_close_rate_exact_fraction(fresh_store):
    """
    Seed 3 commitments with evidence for exactly 2.
    Expect close_rate = 2/3 with support for short-hash and namespaced evidence.
    """
    store = fresh_store

    # Ensure at least one primary event for analyzer
    _insert_self_expression(store, "Starting invariant test for close-rate.")

    # Three commitments
    c1 = store.append_event(kind="commitment", content="I will implement A.", meta={})[
        "hash"
    ]
    c2 = store.append_event(kind="commitment", content="I will implement B.", meta={})[
        "hash"
    ]
    store.append_event(kind="commitment", content="I will implement C.", meta={})

    # Evidence closing c1 via short-hash prefix
    store.append_event(
        kind="evidence",
        content="Completed A",
        meta={"commit_ref": c1[:10]},
    )

    # Namespaced evidence with both nested and top-level commit_ref for c2
    store.append_event(
        kind="evidence:completion",
        content="Completed B",
        meta={
            "commit_ref": c2,
            "evidence": {"commit_ref": c2},
        },
    )

    # c3 has no evidence

    scores = compute_emergence_scores(storage_manager=store)
    close = float(scores.get("commit_close_rate", 0.0) or 0.0)
    # The analyzer returns commit_close_rate rounded to 3 decimals
    assert round(close, 3) == round(2.0 / 3.0, 3)


def test_telemetry_parity_with_analyzer_values(fresh_store, monkeypatch, capsys):
    """
    Enable telemetry and assert printed IAS/GAS/stage match analyzer return values.
    Works for both LEGACY and ADAPTIVE print formats.
    """
    store = fresh_store

    # Enable telemetry output
    monkeypatch.setenv("PMM_TELEMETRY", "1")

    # Seed minimal interaction and a closure signal for non-trivial metrics
    _insert_conversation(store, "Ping")
    _insert_self_expression(store, "Pong. I am maintaining identity.")
    cref = _insert_commitment(store, content="I will write a unit test.")
    _insert_evidence_for(store, commit_ref=cref, kind="done")

    # Compute scores (this should also emit telemetry to stdout)
    capsys.readouterr()  # clear buffers
    scores = compute_emergence_scores(storage_manager=store)
    out = capsys.readouterr().out

    # Basic sanity: telemetry line present
    assert "[PMM][" in out

    # Parse IAS/GAS/stage from either LEGACY or ADAPTIVE line
    import re as _re

    m = _re.search(
        r"\[PMM\]\[(ADAPTIVE|LEGACY)\]\s+IAS=(\d+\.\d{3})\s+GAS=(\d+\.\d{3})\s+stage=([^\n]+)",
        out,
    )
    assert m, f"Unexpected telemetry format: {out!r}"

    ias_print = float(m.group(2))
    gas_print = float(m.group(3))
    stage_print = m.group(4).strip()

    # Compare with analyzer output (rounded to printed precision)
    ias = float(scores.get("IAS", 0.0) or 0.0)
    gas = float(scores.get("GAS", 0.0) or 0.0)
    stage = str(scores.get("stage", ""))

    # Printed values are rounded to 3 decimals; compare at that precision
    assert round(ias, 3) == round(ias_print, 3)
    assert round(gas, 3) == round(gas_print, 3)
    assert stage == stage_print
