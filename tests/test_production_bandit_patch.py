from pmm.production_bandit_patch import (
    reward_from_reflection,
    parse_targeted_reflection,
    enhance_reflection_telemetry,
    backfill_close_attribution,
)


def test_reward_from_reflection_scenarios(monkeypatch):
    monkeypatch.setenv("PMM_BANDIT_CREDIT_HORIZON_TURNS", "7")

    # Closed within horizon dominates
    meta = {"closed_within_horizon": True}
    r = reward_from_reflection(meta)
    assert isinstance(r, float)
    assert r >= 0.7  # default PMM_BANDIT_POS_CLOSE

    # Targeted + accepted + has next
    meta = {
        "targeted_commit_id": "c1",
        "has_next_directive": True,
        "rejection_reason": None,
    }
    r = reward_from_reflection(meta)
    assert 0.0 < r <= 0.7

    # Accepted but generic → neutral (0.0)
    meta = {"rejection_reason": None}
    assert reward_from_reflection(meta) == 0.0

    # Rejected → small negative
    meta = {"rejection_reason": "bad"}
    assert reward_from_reflection(meta) < 0.0


def test_parse_and_telemetry_and_rejection(monkeypatch):
    # Simulate hot window where targeting is required
    monkeypatch.setenv("PMM_BANDIT_REQUIRE_TARGETING_IN_HOT", "true")
    monkeypatch.setenv("PMM_BANDIT_HOT_STRENGTH", "0.5")

    active = [
        {"id": "commit_1", "title": "Improve communication skills"},
        {"id": "commit_2", "title": "Learn Python"},
    ]

    reflection = """
Commit: commit_1
Thought: brief
Next: add evidence
"""
    cid, title, has_next = parse_targeted_reflection(reflection, active)
    assert cid == "commit_1"
    assert has_next is True

    # Hot context → should pass when targeted + next
    meta = enhance_reflection_telemetry(
        reflection_text=reflection,
        bandit_context={"hot_strength": 0.6},
        active_commits=active,
    )
    assert meta["targeted_commit_id"] == "commit_1"
    assert meta["has_next_directive"] is True
    assert meta["rejection_reason"] is None

    # Untargeted during hot → rejected
    untargeted = "I will generally reflect."
    meta2 = enhance_reflection_telemetry(
        reflection_text=untargeted,
        bandit_context={"hot_strength": 0.6},
        active_commits=active,
    )
    assert meta2["targeted_commit_id"] is None
    assert meta2["rejection_reason"] == "untargeted"


def test_backfill_close_attribution_within_horizon():
    # One reflection targeted at commit_1 on turn 1
    reflections = [
        {
            "meta": {"targeted_commit_id": "commit_1", "closed_within_horizon": False},
            "turn": 1,
        }
    ]
    # Evidence for commit_1 arrives on turn 5
    evidences = [
        {"meta": {"commit_ref": "commit_1"}, "turn": 5},
    ]

    backfill_close_attribution(reflections, evidences, horizon=7)

    assert reflections[0]["meta"]["closed_within_horizon"] is True
    assert reflections[0]["meta"]["turns_to_close"] == 4
