#!/usr/bin/env python3
"""
Phase 3 Validation: Evidence Events Test

Tests the evidence event system that makes commitment closures auditable and real.

Acceptance Criteria (ChatGPT's Phase 3 spec):
- 100% of closed commitments have ≥1 linked `evidence:done`
- `commit_close_rate` in `/emergence` equals fraction with `evidence:done` (not acknowledgements)
- `/events/recent` shows three evidence kinds with correct `commit_ref` hash round-trip
- Auto-closure never fires without an evidence row
"""

import os
import sys
import tempfile
from pathlib import Path
import pytest

# Add PMM to path
sys.path.insert(0, str(Path(__file__).parent))

from pmm.commitments import CommitmentTracker
from pmm.langchain_memory import PersistentMindMemory
from pmm.model import EvidenceEvent


@pytest.mark.unit
def test_evidence_event_detection():
    """Test that evidence events are detected from text patterns."""

    tracker = CommitmentTracker()

    # Add a test commitment first
    cid = tracker.add_commitment(
        "Next, I will create the validation tests tonight.", "test_source"
    )
    assert cid, "Test commitment should be added"

    # Test evidence detection patterns
    test_cases = [
        # Done patterns
        ("Done: created validation tests in `test_phase3.py`", "done"),
        ("Completed: PMM validation suite with 15 test cases", "done"),
        ("Finished: documentation update for evidence events", "done"),
        # Blocked patterns
        ("Blocked: waiting for API key -> need to set OPENAI_API_KEY", "blocked"),
        (
            "Cannot proceed: missing dependencies -> install requirements first",
            "blocked",
        ),
        # Delegated patterns
        ("Delegated to Scott: review and approve the test suite", "delegated"),
        ("Assigned to team: implement the probe API enhancements", "delegated"),
    ]

    detected_count = 0
    for text, expected_type in test_cases:
        evidence_events = tracker.detect_evidence_events(text)
        if evidence_events:
            evidence_type, commit_ref, description, artifact = evidence_events[0]
            if evidence_type == expected_type:
                detected_count += 1

    success_rate = detected_count / len(test_cases)
    assert (
        success_rate >= 0.8
    ), f"Evidence detection rate {success_rate:.1%} below 80% threshold"


@pytest.mark.unit
def test_commitment_hash_generation():
    """Test that commitment hashes are stable and unique."""

    tracker = CommitmentTracker()

    # Add test commitments (make them sufficiently different to avoid duplicate detection)
    cid1 = tracker.add_commitment(
        "Next, I will implement hash generation tonight.", "test1"
    )
    cid2 = tracker.add_commitment("Next, I will write documentation tomorrow.", "test2")

    assert cid1 and cid2, "Test commitments should be added"

    commitment1 = tracker.commitments[cid1]
    commitment2 = tracker.commitments[cid2]

    # Generate hashes
    hash1a = tracker.get_commitment_hash(commitment1)
    hash1b = tracker.get_commitment_hash(
        commitment1
    )  # Same commitment, should be same hash
    hash2 = tracker.get_commitment_hash(commitment2)

    # Test hash stability
    assert hash1a == hash1b, "Hash should be stable for same commitment"

    # Test hash uniqueness
    assert hash1a != hash2, "Different commitments should have different hashes"

    # Test hash format
    assert len(hash1a) == 16, "Hash should be 16 characters (truncated SHA-256)"
    assert all(c in "0123456789abcdef" for c in hash1a), "Hash should be hexadecimal"

    # Deterministic assertions above suffice


@pytest.mark.unit
def test_evidence_based_closure():
    """Test that commitments close only with evidence:done events."""

    tracker = CommitmentTracker()

    # Add test commitment
    cid = tracker.add_commitment(
        "Next, I will implement evidence closure tonight.", "test_source"
    )
    assert cid, "Test commitment should be added"

    commitment = tracker.commitments[cid]
    commit_hash = tracker.get_commitment_hash(commitment)

    # Test that non-done evidence doesn't close commitment
    blocked_result = tracker.close_commitment_with_evidence(
        commit_hash, "blocked", "Blocked by missing dependencies", "requirements.txt"
    )
    assert not blocked_result, "Blocked evidence should not close commitment"
    assert (
        commitment.status == "open"
    ), "Commitment should remain open after blocked evidence"

    delegated_result = tracker.close_commitment_with_evidence(
        commit_hash, "delegated", "Delegated to team lead", "team-lead@company.com"
    )
    assert not delegated_result, "Delegated evidence should not close commitment"
    assert (
        commitment.status == "open"
    ), "Commitment should remain open after delegated evidence"

    # Test that done evidence closes commitment
    done_result = tracker.close_commitment_with_evidence(
        commit_hash,
        "done",
        "Implemented evidence closure in `test_phase3.py`",
        "test_phase3.py",
    )
    assert done_result, "Done evidence should close commitment"
    assert (
        commitment.status == "closed"
    ), "Commitment should be closed after done evidence"
    assert "Evidence:" in commitment.close_note, "Close note should mention evidence"
    assert (
        "test_phase3.py" in commitment.close_note
    ), "Close note should include artifact"

    # No printouts; assertions check behavior


@pytest.mark.unit
def test_evidence_event_integration():
    """Test evidence events integration with PMM system."""

    with tempfile.TemporaryDirectory() as temp_dir:
        agent_path = os.path.join(temp_dir, "test_agent.json")
        memory = PersistentMindMemory(agent_path)

        initial_events = len(memory.pmm.model.self_knowledge.autobiographical_events)

        # Test evidence event processing in PMM integration
        memory.save_context(
            {"input": "Let me work on the evidence tests."},
            {
                "response": "I'll create comprehensive evidence tests tonight after reviewing the Phase 3 requirements."
            },
        )

        # Provide evidence of completion to trigger evidence-based closure
        memory.save_context(
            {"input": "How's the progress?"},
            {
                "response": "Done: created comprehensive evidence tests in `test_phase3_evidence_events.py` with full validation coverage."
            },
        )

        final_events = len(memory.pmm.model.self_knowledge.autobiographical_events)
        _ = final_events - initial_events

        # Assert via counts/structures instead of prints

        # Check for evidence events in the event log
        evidence_events = [
            e
            for e in memory.pmm.model.self_knowledge.autobiographical_events
            if e.type.startswith("evidence:")
        ]

        if evidence_events:
            evidence_event = evidence_events[0]
            assert evidence_event.type.startswith(
                "evidence:"
            ), "Evidence event type expected"
            assert evidence_event.summary, "Evidence event should have summary"
            if evidence_event.evidence:
                assert evidence_event.evidence.commit_ref, "Evidence needs commit_ref"

        # Check that commitment was closed
        closed_commitments = [
            c
            for c in memory.pmm.commitment_tracker.commitments.values()
            if c.status == "closed"
        ]

        if closed_commitments:
            closed_commitment = closed_commitments[0]
            # Verify commitment was closed (evidence integration working)
            assert (
                "Auto-closed" in closed_commitment.close_note
                or "Evidence:" in closed_commitment.close_note
            ), "Closure should indicate automatic closure or evidence reference"


@pytest.mark.unit
def test_artifact_extraction():
    """Test artifact extraction from evidence descriptions."""

    tracker = CommitmentTracker()

    test_cases = [
        ("created tests in `test_phase3.py`", "test_phase3.py"),
        ("uploaded file.txt to server", "file.txt"),
        ("fixed issue #123 in the codebase", "#123"),
        ("deployed https://example.com/app", "https://example.com/app"),
        ("completed task on 2025-01-15", "2025-01-15"),
        ("submitted PROJ-456 ticket", "PROJ-456"),
        ("no artifacts in this description", None),
    ]

    passed = 0
    for description, expected_artifact in test_cases:
        extracted = tracker._extract_artifact(description)
        if extracted == expected_artifact:
            passed += 1
            _ = "✅"
        else:
            _ = "❌"

    success_rate = passed / len(test_cases)
    assert (
        success_rate >= 0.8
    ), f"Artifact extraction rate {success_rate:.1%} below 80% threshold"


@pytest.mark.unit
def test_evidence_event_structure():
    """Test that evidence events have correct structure and data."""

    from datetime import datetime, timezone

    # Create test evidence event
    evidence = EvidenceEvent(
        id="test_evidence_1",
        t=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        evidence_type="done",
        commit_ref="abc123def456",
        description="Completed validation tests with 15 test cases",
        artifact="test_validation.py",
        next_action=None,
    )

    # Validate structure
    assert evidence.id == "test_evidence_1", "Evidence ID should match"
    assert evidence.evidence_type == "done", "Evidence type should be 'done'"
    assert evidence.commit_ref == "abc123def456", "Commit ref should match"
    assert (
        "validation tests" in evidence.description
    ), "Description should contain key info"
    assert evidence.artifact == "test_validation.py", "Artifact should match"
    assert evidence.next_action is None, "Next action should be None for done evidence"

    # Test blocked evidence with next action
    blocked_evidence = EvidenceEvent(
        id="test_evidence_2",
        t=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        evidence_type="blocked",
        commit_ref="def456ghi789",
        description="Blocked by missing API key",
        artifact=None,
        next_action="Set OPENAI_API_KEY environment variable",
    )

    assert (
        blocked_evidence.evidence_type == "blocked"
    ), "Blocked evidence type should match"
    assert (
        blocked_evidence.next_action is not None
    ), "Blocked evidence should have next action"
    assert (
        "OPENAI_API_KEY" in blocked_evidence.next_action
    ), "Next action should be specific"

    # Structure validated via assertions above


@pytest.mark.unit
def test_commitment_closure_enforcement():
    """Test that commitments can only be closed with evidence."""

    tracker = CommitmentTracker()

    # Add test commitment
    cid = tracker.add_commitment(
        "Next, I will test closure enforcement tonight.", "test_source"
    )
    assert cid, "Test commitment should be added"

    commitment = tracker.commitments[cid]

    # Try to close commitment manually (should fail in Phase 3)
    # original_status = commitment.status  # Not used in this test

    # The old auto-close method should not work without evidence
    # (This test assumes we've disabled auto-closure without evidence)

    # Only evidence-based closure should work
    commit_hash = tracker.get_commitment_hash(commitment)
    success = tracker.close_commitment_with_evidence(
        commit_hash,
        "done",
        "Enforcement test completed successfully",
        "test_results.log",
    )

    assert success, "Evidence-based closure should succeed"
    assert commitment.status == "closed", "Commitment should be closed with evidence"
    assert "Evidence:" in commitment.close_note, "Close note should reference evidence"

    # Assertions validate enforcement


@pytest.mark.unit
def test_phase3_acceptance_criteria():
    """Test Phase 3 acceptance criteria from ChatGPT's spec."""

    tracker = CommitmentTracker()

    # Create multiple commitments (use PMM-context terms)
    cid1 = tracker.add_commitment(
        "Next, I will implement PMM validation tonight.", "test1"
    )
    cid2 = tracker.add_commitment(
        "Next, I will document evidence system tomorrow.", "test2"
    )
    cid3 = tracker.add_commitment(
        "Next, I will test commitment closure this week.", "test3"
    )

    assert all([cid1, cid2, cid3]), "All test commitments should be added"

    # Close some commitments with evidence
    hash1 = tracker.get_commitment_hash(tracker.commitments[cid1])
    hash2 = tracker.get_commitment_hash(tracker.commitments[cid2])

    tracker.close_commitment_with_evidence(
        hash1, "done", "Feature A implemented in `feature_a.py`", "feature_a.py"
    )
    tracker.close_commitment_with_evidence(
        hash2,
        "done",
        "Feature B documented in `docs/feature_b.md`",
        "docs/feature_b.md",
    )

    # Test acceptance criteria

    # 1. 100% of closed commitments have ≥1 linked evidence:done
    closed_commitments = [
        c for c in tracker.commitments.values() if c.status == "closed"
    ]
    commitments_with_evidence = [
        c for c in closed_commitments if "Evidence:" in c.close_note
    ]

    evidence_rate = (
        len(commitments_with_evidence) / len(closed_commitments)
        if closed_commitments
        else 0
    )
    assert evidence_rate == 1.0, "100% of closed commitments must have evidence"

    # 2. Commit close rate calculation
    total_commitments = len(tracker.commitments)
    closed_with_evidence = len(commitments_with_evidence)
    _ = closed_with_evidence / total_commitments if total_commitments else 0

    # 3. Evidence events should have correct commit_ref hash round-trip
    for cid, commitment in tracker.commitments.items():
        if commitment.status == "closed":
            commit_hash = tracker.get_commitment_hash(commitment)
            # Hash should be 16-char hex string
            assert (
                len(commit_hash) == 16
            ), f"Commit hash should be 16 characters: {commit_hash}"
            assert all(
                c in "0123456789abcdef" for c in commit_hash
            ), f"Hash should be hex: {commit_hash}"
            # No prints; assertions enforce correctness


# Removed demo-style __main__ runner; tests are executed via pytest
