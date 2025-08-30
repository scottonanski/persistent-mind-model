import pytest

from pmm.emergence import compute_emergence_scores
from pmm.production_bandit_patch import enhance_reflection_telemetry


@pytest.mark.usefixtures("reset_analyzer")
def test_hot_context_telemetry_and_close_rate(sqlite_store, monkeypatch):
    store = sqlite_store

    # Ensure hot window requires targeting
    monkeypatch.setenv("PMM_BANDIT_REQUIRE_TARGETING_IN_HOT", "true")
    monkeypatch.setenv("PMM_BANDIT_HOT_STRENGTH", "0.5")

    # Seed base turns
    for i in range(2):
        store.append_event(kind="conversation", content=f"u{i}", meta={})
        store.append_event(kind="self_expression", content=f"a{i}", meta={})

    # Two commitments
    c1 = store.append_event(kind="commitment", content="commit A", meta={})["hash"]
    c2 = store.append_event(kind="commitment", content="commit B", meta={})["hash"]

    # Untargeted reflection in hot context should be rejected
    active = [{"id": c1, "title": "A"}, {"id": c2, "title": "B"}]
    meta_untargeted = enhance_reflection_telemetry(
        reflection_text="General reflection without targeting.",
        bandit_context={"hot_strength": 0.6},
        active_commits=active,
    )
    assert meta_untargeted["targeted_commit_id"] is None
    assert meta_untargeted["rejection_reason"] == "untargeted"

    # Targeted reflection should pass in hot
    targeted = f"""
Commit: {c1}
Thought: brief
Next: add evidence
"""
    meta_targeted = enhance_reflection_telemetry(
        reflection_text=targeted,
        bandit_context={"hot_strength": 0.6},
        active_commits=active,
    )
    assert meta_targeted["targeted_commit_id"] == c1
    assert meta_targeted["has_next_directive"] is True
    assert meta_targeted["rejection_reason"] is None

    # Verify analyzer close rate changes only with evidence
    scores_before = compute_emergence_scores(storage_manager=store)
    close_before = float(scores_before.get("commit_close_rate", 0.0) or 0.0)

    # Add evidence for c1
    store.append_event(kind="evidence", content="done", meta={"commit_ref": c1})

    scores_after = compute_emergence_scores(storage_manager=store)
    close_after = float(scores_after.get("commit_close_rate", 0.0) or 0.0)

    assert round(close_after, 3) > round(close_before, 3)
