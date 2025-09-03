from unittest.mock import MagicMock
from pmm.langchain_memory import PersistentMindMemory


def test_evidence_closes_commitment(tmp_path):
    """Verify that a 'done' evidence string closes the correct commitment."""
    # 1. Setup a mock PMM environment
    mock_pmm = MagicMock()
    mock_pmm.commitment_tracker.detect_evidence_events.return_value = [
        ("done", "c123", "Completed the task", None)
    ]
    mock_pmm.commitment_tracker.get_commitment.return_value = {"status": "open"}

    memory = PersistentMindMemory(agent_path=str(tmp_path / "agent"))
    memory.pmm = mock_pmm

    # 2. Define the evidence string
    evidence_text = "Evidence: done (commit_ref: c123) - The task is complete."

    # 3. Process the evidence
    detected_evidence = memory._process_evidence_events(evidence_text)

    # 4. Assert that the evidence was detected
    assert len(detected_evidence) == 1
    assert detected_evidence[0][0] == "done"
    assert detected_evidence[0][1] == "c123"

    # 5. Assert that the commitment closure method was called
    mock_pmm.commitment_tracker.close_commitment_with_evidence.assert_called_once_with(
        commit_hash="c123",
        evidence_type="done",
        description="Completed the task",
        artifact=None,
    )


def test_placeholder_for_evidence_closure():
    """A placeholder test for the evidence closure mechanism."""
    assert 1 == 1
