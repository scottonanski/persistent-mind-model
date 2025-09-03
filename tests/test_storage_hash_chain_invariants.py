import importlib, inspect, pytest
from pathlib import Path

# --- utilities ---------------------------------------------------------------

def _first_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"None of {names} on {obj}")

def _get_or_key(rec, *names):
    for name in names:
        if hasattr(rec, name):
            return getattr(rec, name)
        if isinstance(rec, dict) and name in rec:
            return rec[name]
    raise AttributeError(f"record has no attr/key in {names}: {rec!r}")

def _find_storage_class():
    """Prefer the repo's SQLiteStore if present, else fall back to discovery."""
    try:
        mod = importlib.import_module("pmm.storage.sqlite_store")
        cls = getattr(mod, "SQLiteStore", None)
        if cls:
            return cls
    except Exception:
        pass

    # Fallback discovery (legacy)
    candidates = ["pmm.storage.store", "pmm.storage"]
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for name, cls in inspect.getmembers(mod, inspect.isclass):
            members = dir(cls)
            if any(n in members for n in ("append_event", "add_event", "insert_event")):
                return cls
    pytest.skip("No storage class with required methods found under pmm.storage.*")

# --- tests -------------------------------------------------------------------

def test_hash_chain_integrity(tmp_path):
    Store = _find_storage_class()
    dbpath = tmp_path / "pmm.db"

    # allow either Store(Path) or Store(str) or Store(db_path=...)
    try:
        store = Store(dbpath)
    except TypeError:
        try:
            store = Store(str(dbpath))
        except TypeError:
            store = Store(db_path=str(dbpath))

    # The repo's SQLiteStore exposes append_event(...) and all_events()/recent_events()
    add_event = _first_attr(store, "append_event", "add_event", "insert_event")
    get_events = _first_attr(store, "all_events", "get_events", "get_event_chain")

    # Append events directly; SQLiteStore manages chain hashes internally
    add_event(kind="user", content="hi", meta={"role": "user"})
    add_event(kind="assistant", content="hello", meta={"role": "assistant"})
    add_event(kind="user", content="how are you?", meta={"role": "user"})
    chain = list(get_events())

    assert len(chain) == 3

    # Each event after the first must link to the previous hash
    for prev, cur in zip(chain, chain[1:]):
        prev_hash = _get_or_key(prev, "hash", "event_hash", "digest")
        cur_prev  = _get_or_key(cur, "prev_hash", "previous_hash", "parent_hash")
        assert cur_prev == prev_hash, "Hash chain broken between consecutive events"

def test_event_order_is_stable(tmp_path):
    Store = _find_storage_class()
    dbpath = tmp_path / "pmm2.db"
    try:
        store = Store(dbpath)
    except TypeError:
        store = Store(str(dbpath))

    add_event = _first_attr(store, "append_event", "add_event", "insert_event")
    get_events = _first_attr(store, "all_events", "get_events", "get_event_chain")

    texts = ["a", "b", "c", "d"]
    for i, t in enumerate(texts):
        add_event(kind=("user" if i % 2 == 0 else "assistant"), content=t, meta={})

    chain = list(get_events())
    got = [_get_or_key(e, "content", "text", "body") for e in chain[-4:]]
    assert got == texts, "Event order changed between write and read"
