
from pmm.name_detect import extract_agent_name_command
from pmm.atomic_reflection import AtomicReflectionManager


def test_negation_does_not_trigger_agent_rename():
    # Should not accept "not Scott" as a valid agent rename
    assert extract_agent_name_command("No, your name is not Scott.", "user") is None
    assert extract_agent_name_command("Your name is NOT Scott", "user") is None
    assert extract_agent_name_command("From now on, you are not Scott", "user") is None


def test_atomic_reflection_blocks_coach_like_commitments():
    # Create manager with a minimal stub pmm; this test only calls validation
    class _Stub:
        pass

    arm = AtomicReflectionManager(_Stub())

    banned_cases = [
        "I should ask a question every turn to deepen conversations.",
        "Ask a probing question every message.",
        "Always ask a question.",
        "We will deepen the conversation each reply by asking more questions.",
    ]
    for text in banned_cases:
        assert arm._passes_basic_validation(text) is False

    allowed_cases = [
        "I realized the user prefers concrete examples when discussing APIs.",
        "Tracking commitment close rate over time may reveal plateaus.",
        "Recent insights suggest my summaries are too verbose; I should be more concise.",
    ]
    for text in allowed_cases:
        assert arm._passes_basic_validation(text) is True
