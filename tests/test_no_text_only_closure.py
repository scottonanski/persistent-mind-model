"""
Test to prevent text-only commitment closures (LARP prevention).
"""

import pytest
import sys
import os

# Add pmm to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmm.commitments import CommitmentTracker, Commitment
from datetime import datetime, timezone


def test_text_reply_does_not_close_without_artifact():
    """Text-only evidence must not close commitments."""
    tracker = CommitmentTracker()

    # Add a test commitment
    commitment = Commitment(
        cid="test1",
        text="I will implement the feature",
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_insight_id="insight1",
        status="open",
    )
    tracker.commitments["test1"] = commitment
    commit_hash = tracker.get_commitment_hash(commitment)

    # Text-only evidence should NOT close
    result = tracker.close_commitment_with_evidence(
        commit_hash=commit_hash,
        evidence_type="done",
        description="Feature implemented successfully",
        artifact=None,
    )

    assert result is False, "Text-only evidence must not close commitment"
    assert commitment.status == "open", "Commitment must remain open"


def test_artifact_closes():
    """Real artifacts should close commitments."""
    tracker = CommitmentTracker()

    # Add a test commitment
    commitment = Commitment(
        cid="test2",
        text="I will create the documentation",
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_insight_id="insight2",
        status="open",
    )
    tracker.commitments["test2"] = commitment
    commit_hash = tracker.get_commitment_hash(commitment)

    # Artifact evidence should close
    result = tracker.close_commitment_with_evidence(
        commit_hash=commit_hash,
        evidence_type="done",
        description="Documentation created",
        artifact="docs/feature.md",
    )

    assert result is True, "Artifact evidence must close commitment"
    assert commitment.status == "closed", "Commitment must be closed"


def test_url_artifact_closes():
    """URL artifacts should close commitments."""
    tracker = CommitmentTracker()

    # Add a test commitment
    commitment = Commitment(
        cid="test3",
        text="I will deploy the service",
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_insight_id="insight3",
        status="open",
    )
    tracker.commitments["test3"] = commitment
    commit_hash = tracker.get_commitment_hash(commitment)

    # URL artifact should close
    result = tracker.close_commitment_with_evidence(
        commit_hash=commit_hash,
        evidence_type="done",
        description="Service deployed",
        artifact="https://myservice.com/health",
    )

    assert result is True, "URL artifact must close commitment"
    assert commitment.status == "closed", "Commitment must be closed"


def test_hash_artifact_closes():
    """Hash artifacts should close commitments."""
    tracker = CommitmentTracker()

    # Add a test commitment
    commitment = Commitment(
        cid="test4",
        text="I will commit the changes",
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_insight_id="insight4",
        status="open",
    )
    tracker.commitments["test4"] = commitment
    commit_hash = tracker.get_commitment_hash(commitment)

    # Hash artifact should close
    print(f"DEBUG: Before closure - commitment status: {commitment.status}")
    print(f"DEBUG: Testing hash artifact: abc123def456")
    
    result = tracker.close_commitment_with_evidence(
        commit_hash=commit_hash,
        evidence_type="done",
        description="Changes committed",
        artifact="abc123def456",
    )
    
    print(f"DEBUG: After closure - result: {result}, commitment status: {commitment.status}")
    assert result is True, f"Hash artifact must close commitment. Got result={result}, status={commitment.status}"
    assert commitment.status == "closed", f"Commitment must be closed. Got status={commitment.status}"


def test_delivered_with_id_closes():
    """Delivered evidence with artifact ID should close commitments."""
    tracker = CommitmentTracker()

    # Add a test commitment
    commitment = Commitment(
        cid="test5",
        text="I will deliver the package",
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_insight_id="insight5",
        status="open",
    )
    tracker.commitments["test5"] = commitment
    commit_hash = tracker.get_commitment_hash(commitment)

    # Delivered with artifact ID should close (use "done" evidence type)
    result = tracker.close_commitment_with_evidence(
        commit_hash=commit_hash,
        evidence_type="done",
        description="Package delivered",
        artifact="PKG-12345",
    )

    assert result is True, "Delivered with artifact ID must close commitment"
    assert commitment.status == "closed", "Commitment must be closed"


if __name__ == "__main__":
    pytest.main([__file__])
