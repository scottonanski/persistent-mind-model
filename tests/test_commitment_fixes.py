#!/usr/bin/env python3
"""
Test the improved commitment detection patterns.
"""

import pytest
from pmm.commitments import CommitmentTracker


def test_commitment_detection_fixes():
    """Test that the improved patterns catch Echo session commitments."""
    tracker = CommitmentTracker()

    # Test cases aligned with current invariants (must start with 'I will ...' and be concrete)
    test_cases = [
        "I will prioritize emotional awareness by acknowledging cues and responding with empathy in our conversations tonight.",
        "I will foster more meaningful interactions by applying the listed practices within the next week.",
        "I will use storytelling to convey an emotional intelligence concept in our conversation today at 5pm.",
        "I will outline specific goals for each conversation to stay focused and organized starting now.",
        "I will implement a structured reflection process after each conversation starting tonight at 8pm, focusing on emotional engagement and depth.",
    ]

    for text in test_cases:
        commitment, _ = tracker.extract_commitment(text)
        # Each example should be detected as a commitment
        assert (
            commitment is not None
        ), f"Expected commitment detection for: {text[:80]}..."

    # Unsupported forms should currently be rejected; keep as documentation for future normalization work
    unsupported = [
        "I commit to enhancing our conversations by actively incorporating emotional intelligence principles into my responses.",
        "I commit to achieving the goal of effectively utilizing storytelling to convey a concept today.",
        "I commit to evolving my conscientiousness by setting clear goals.",
        "I aim to enhance my creative thinking and provide a more engaging conversational experience.",
    ]
    for text in unsupported:
        commitment, _ = tracker.extract_commitment(text)
        assert commitment is None, f"Unsupported form should be rejected: {text}"


if __name__ == "__main__":
    test_commitment_detection_fixes()
