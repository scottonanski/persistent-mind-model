import os
import sys
from pathlib import Path
import contextlib
import pytest

# Ensure project root is on sys.path so `import pmm` works when running plain `pytest`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Default to fast test mode unless explicitly disabled
os.environ.setdefault("PMM_TEST_MODE", "1")

# Shared DB-grounded fixtures and helpers


@pytest.fixture()
def sqlite_store(tmp_path, monkeypatch):
    from pmm.storage.sqlite_store import SQLiteStore

    db_path = tmp_path / "pmm_test.sqlite"
    store = SQLiteStore(str(db_path))

    # Deterministic analyzer window/types
    monkeypatch.setenv(
        "PMM_EMERGENCE_TYPES",
        "self_expression,response,reflection,commitment,evidence",
    )
    monkeypatch.setenv("PMM_EMERGENCE_WINDOW", "10")

    # Reset global analyzer between tests
    import pmm.emergence as emergence_mod

    emergence_mod._analyzer = None

    yield store

    with contextlib.suppress(Exception):
        store.conn.close()


@pytest.fixture()
def reset_analyzer():
    import pmm.emergence as emergence_mod

    emergence_mod._analyzer = None
    yield
    emergence_mod._analyzer = None


def seed_turn(store, user_text: str, ai_text: str):
    store.append_event(
        kind="conversation", content=user_text, meta={"role": "user", "source": "test"}
    )
    store.append_event(
        kind="self_expression",
        content=ai_text,
        meta={"role": "assistant", "source": "test"},
    )


def seed_commit(store, text: str = "I will do X.") -> str:
    return store.append_event(kind="commitment", content=text, meta={}).get("hash")  # type: ignore[return-value]


def seed_evidence(
    store,
    commit_hash: str,
    *,
    namespaced: bool = False,
    short: bool = False,
    nested: bool = False,
):
    ref = commit_hash[:10] if short else commit_hash
    if namespaced:
        meta = {"commit_ref": ref}
        if nested:
            meta["evidence"] = {"commit_ref": ref}
        store.append_event(
            kind="evidence:completion",
            content="done",
            meta=meta,
        )
    else:
        store.append_event(
            kind="evidence",
            content="done",
            meta={"commit_ref": ref},
        )
