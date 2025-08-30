#!/usr/bin/env python3
import uuid


def test_hot_strength_computation_monotonic():
    from pmm.policy.bandit import compute_hot_strength

    assert compute_hot_strength(0.0, 0.0) == 0.0
    mid = compute_hot_strength(0.8, 0.7)
    hi = compute_hot_strength(1.0, 1.0)
    assert 0.0 <= mid <= 1.0
    assert hi == 1.0
    assert hi >= mid


def test_bandit_context_building_contains_hot_strength():
    from pmm.policy.bandit import build_context

    ctx = build_context(gas=0.8, ias=0.6, close=0.7)
    assert set(["gas", "ias", "close", "hot_strength", "hot"]).issubset(ctx.keys())
    assert 0.0 <= float(ctx["hot_strength"]) <= 1.0
    assert ctx["hot"] in (0.0, 1.0)


def test_reward_shaping_and_reflection_id_tracking(tmp_path, monkeypatch):
    from pmm.policy.bandit import _BanditCore
    from pmm.storage.sqlite_store import SQLiteStore

    # Avoid stdout noise
    monkeypatch.setenv("PMM_TELEMETRY", "0")
    monkeypatch.setenv("PMM_BANDIT_HOT_REFLECT_BOOST", "0.3")
    monkeypatch.setenv("PMM_BANDIT_HOT_CONTINUE_PENALTY", "0.1")

    store = SQLiteStore(str(tmp_path / "bandit.sqlite"))
    bandit = _BanditCore(store)

    ctx_hot = {"hot_strength": 0.8, "reflect_id": str(uuid.uuid4())[:8]}

    # Should not raise and should write to policy table
    bandit.record_outcome(ctx_hot, "reflect_now", 0.5, 10, "test")
    bandit.record_outcome(ctx_hot, "continue", 0.2, 10, "test")

    # Verify policy rows exist
    rows = list(store.conn.execute("SELECT action, value, pulls FROM bandit_policy"))
    actions = {r[0] for r in rows}
    assert {"reflect_now", "continue"}.issubset(actions)
