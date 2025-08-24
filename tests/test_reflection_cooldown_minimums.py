from datetime import datetime, timedelta, timezone

import pytest

from pmm.reflection_cooldown import ReflectionCooldownManager


def test_time_gate_denies_then_allows(monkeypatch):
    # Configure manager with 2s min time, 0 turns requirement to isolate time gate
    mgr = ReflectionCooldownManager(
        min_turns=0, min_wall_time_seconds=2, novelty_threshold=0.99
    )

    # Simulate a reflection just occurred now
    mgr.state.last_reflection_time = datetime.now(timezone.utc)

    should, reason = mgr.should_reflect(current_context="ctx")
    assert should is False
    assert "time_gate" in reason

    # Advance time by 3 seconds and try again
    mgr.state.last_reflection_time = datetime.now(timezone.utc) - timedelta(seconds=3)
    should, reason = mgr.should_reflect(current_context="ctx2")
    assert should is True
    assert "all_gates_passed" in reason or reason.startswith("forced:")


def test_turns_gate_denies_then_allows():
    # Configure manager with 3 turns and 0s time requirement to isolate turns gate
    mgr = ReflectionCooldownManager(
        min_turns=3, min_wall_time_seconds=0, novelty_threshold=0.99
    )

    # No reflections yet; should be denied for insufficient turns
    should, reason = mgr.should_reflect(current_context="hello")
    assert should is False
    assert "turns_gate" in reason

    # Increment turns below threshold
    mgr.increment_turn()
    mgr.increment_turn()
    should, reason = mgr.should_reflect(current_context="hello again")
    assert should is False
    assert "turns_gate" in reason

    # One more turn meets threshold
    mgr.increment_turn()
    should, reason = mgr.should_reflect(current_context="novel context")
    assert should is True


@pytest.mark.parametrize(
    "ctx1, ctx2, expect_pass",
    [
        ("abc def ghi", "abc def ghi", False),  # identical -> blocked
        ("abc def ghi", "abc def xyz", False),  # high overlap -> blocked
        ("abc def ghi", "uvw xyz qrs", True),  # disjoint -> allowed
    ],
)
def test_novelty_gate(ctx1, ctx2, expect_pass):
    mgr = ReflectionCooldownManager(
        min_turns=0, min_wall_time_seconds=0, novelty_threshold=0.6
    )

    # Seed recent contexts (as would happen per-turn)
    mgr.add_context(ctx1)

    should, reason = mgr.should_reflect(current_context=ctx2)
    assert should is expect_pass
    if expect_pass:
        assert "all_gates_passed" in reason or reason.startswith("forced:")
    else:
        assert "novelty_gate" in reason
