#!/usr/bin/env python3
"""
Test suite for the surgical fixes to PMM metrics and reflection system.
Validates that close rates come from events, reflection attempts are logged,
and duplicate detection works correctly.
"""

import os
import sys
import tempfile
import pytest
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmm.metrics import compute_close_rate
from pmm.atomic_reflection import AtomicReflectionManager
from pmm.commitments import CommitmentTracker


class TestEventBasedCloseRate:
    """Test that close rate is computed from actual events, not vibes."""

    def test_close_from_events_only(self):
        """Close metric should be event-based."""
        # No events → zero
        assert compute_close_rate([]) == 0.0

        # No opens → zero (avoid division by zero)
        events = [{"type": "commit_close"} for _ in range(3)]
        assert compute_close_rate(events) == 0.0

        # 4 opens, 1 close → 0.25
        events = [{"type": "commit_open"} for _ in range(4)] + [
            {"type": "commit_close"}
        ]
        assert compute_close_rate(events) == 0.25

        # 2 opens, 2 closes → 1.0
        events = [
            {"type": "commit_open"},
            {"type": "commit_close"},
            {"type": "commit_open"},
            {"type": "commit_close"},
        ]
        assert compute_close_rate(events) == 1.0


class TestReflectionAttemptLogging:
    """Test that reflection attempts are logged when ready."""

    def test_reflection_attempt_logged(self):
        """run_once should log reflection_attempt event."""
        # Mock PMM manager with sqlite_store
        mock_pmm = Mock()
        mock_store = Mock()
        mock_pmm.sqlite_store = mock_store

        # Create reflection manager
        manager = AtomicReflectionManager(mock_pmm)

        # Mock reflect_once to return None (no insight)
        with patch("pmm.reflection.reflect_once", return_value=None):
            result = manager.run_once("test input")

        # Should return False (no insight generated)
        assert result is False

        # Should have logged reflection_attempt
        calls = mock_store.add_event.call_args_list
        attempt_calls = [
            call for call in calls if call[0][0]["type"] == "reflection_attempt"
        ]
        assert len(attempt_calls) >= 1

        # Should have logged reflection_skip
        skip_calls = [call for call in calls if call[0][0]["type"] == "reflection_skip"]
        assert len(skip_calls) >= 1


class TestTextOnlyEvidenceBlocking:
    """Test that text-only evidence never closes commitments."""

    def test_text_only_evidence_never_closes(self):
        """Text-only evidence should not close commitments."""
        # Mock storage
        mock_storage = Mock()

        # Create commitment tracker with proper initialization
        tracker = CommitmentTracker()
        tracker.storage = mock_storage

        # Add a test commitment
        cid = tracker.add_commitment("I will test this", "test_insight")
        commitment = tracker.commitments[cid]

        # Try to close with text-only evidence
        result = tracker.close_commitment_with_evidence(
            tracker.get_commitment_hash(commitment),
            "done",
            "I completed the test",
            artifact=None,  # No artifact = text-only
        )

        # Should return False (not closed)
        assert result is False

        # Commitment should still be open
        assert commitment.status == "open"


class TestDuplicateDetectionThresholds:
    """Test that duplicate detection has proper thresholds."""

    def test_duplicate_detection_configurable(self):
        """Duplicate detection should use configurable thresholds."""
        with tempfile.TemporaryDirectory():
            mock_storage = Mock()
            tracker = CommitmentTracker()
            tracker.storage = mock_storage

            # Add first commitment
            cid1 = tracker.add_commitment("I will write tests", "insight1")

            # Try to add very similar commitment with default strict threshold
            with patch.dict(os.environ, {"PMM_DUPLICATE_SIM_THRESHOLD": "0.98"}):
                # This should NOT be detected as duplicate (below 98% threshold)
                cid2 = tracker.add_commitment("I will write more tests", "insight2")
                assert cid2 is not None
                assert cid1 != cid2

            # Try with lenient threshold
            with patch.dict(os.environ, {"PMM_DUPLICATE_SIM_THRESHOLD": "0.5"}):
                # This should be detected as duplicate (above 50% threshold)
                duplicate_cid = tracker._is_duplicate_commitment(
                    "I will write tests again"
                )
                # Should find the original commitment as duplicate
                assert duplicate_cid is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
