#!/usr/bin/env python3
"""
Enhanced commitment validation with tiered approach and relaxed filtering.
Addresses the over-strict filtering that throttles PMM growth.
"""

from typing import List, Dict
from dataclasses import dataclass
from enum import Enum


class CommitmentTier(Enum):
    TENTATIVE = "tentative"  # Borderline, needs confirmation
    CONFIRMED = "confirmed"  # Clear commitment
    PERMANENT = "permanent"  # Explicitly marked as permanent


@dataclass
class CommitmentAnalysis:
    """Analysis result for commitment validation."""

    is_valid: bool
    tier: CommitmentTier
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]
    structural_elements: Dict[str, bool]


class EnhancedCommitmentValidator:
    """
    Enhanced validator that uses structural logic instead of brittle patterns.
    Implements tiered validation with relaxed similarity thresholds.
    """

    def __init__(self):
        # Relaxed similarity threshold (was 0.92, now 0.95)
        self.similarity_threshold = 0.95

        # Structural indicators (more flexible than hardcoded patterns)
        self.ownership_indicators = {
            "first_person": [
                "i will",
                "i shall",
                "i commit",
                "i acknowledge",
                "i aim",
                "i strive",
                "my goal",
            ],
            "identity_statements": ["i am", "i operate", "i function", "i serve as"],
            "capability_claims": ["i can", "i have the capacity", "i am designed to"],
        }

        self.action_indicators = [
            "will",
            "shall",
            "aim to",
            "strive to",
            "commit to",
            "ensure",
            "provide",
            "maintain",
            "adapt",
            "learn",
            "reflect",
            "analyze",
            "identify",
            "foster",
        ]

        self.temporal_indicators = [
            "over time",
            "in the future",
            "going forward",
            "from now on",
            "continuously",
            "ongoing",
            "always",
            "whenever",
            "as needed",
        ]

        # Permanent commitment markers
        self.permanence_markers = [
            "permanent",
            "always",
            "core principle",
            "fundamental",
            "guiding principle",
            "meta-principle",
            "baseline",
            "foundation",
            "identity",
            "define",
        ]

    def validate_commitment(
        self, text: str, existing_commitments: List[str] = None
    ) -> CommitmentAnalysis:
        """
        Enhanced validation using structural analysis instead of brittle patterns.

        Args:
            text: Text to analyze for commitment
            existing_commitments: List of existing commitment texts for similarity check

        Returns:
            CommitmentAnalysis with tier, confidence, and reasoning
        """
        text_lower = text.lower().strip()

        # Skip very short texts
        if len(text) < 20:
            return CommitmentAnalysis(
                is_valid=False,
                tier=CommitmentTier.TENTATIVE,
                confidence=0.0,
                reasoning=["Text too short to be meaningful commitment"],
                structural_elements={},
            )

        # Analyze structural elements
        elements = self._analyze_structural_elements(text_lower)
        confidence = self._calculate_confidence(elements)
        reasoning = self._generate_reasoning(elements)

        # Determine tier based on confidence and permanence markers
        tier = self._determine_tier(text_lower, confidence)

        # Check similarity only for high-confidence commitments
        if confidence > 0.7 and existing_commitments:
            similarity_score = self._check_similarity(text, existing_commitments)
            if similarity_score > self.similarity_threshold:
                return CommitmentAnalysis(
                    is_valid=False,
                    tier=CommitmentTier.TENTATIVE,
                    confidence=confidence * 0.5,  # Reduce confidence but don't reject
                    reasoning=reasoning
                    + [
                        f"High similarity to existing commitment: {similarity_score:.3f}"
                    ],
                    structural_elements=elements,
                )

        # Accept if confidence is reasonable
        is_valid = confidence >= 0.3  # Lowered threshold from typical 0.5

        return CommitmentAnalysis(
            is_valid=is_valid,
            tier=tier,
            confidence=confidence,
            reasoning=reasoning,
            structural_elements=elements,
        )

    def _analyze_structural_elements(self, text_lower: str) -> Dict[str, bool]:
        """Analyze structural elements that indicate commitment."""
        elements = {
            "has_ownership": False,
            "has_action": False,
            "has_temporal": False,
            "has_context": False,
            "is_identity_statement": False,
            "is_capability_claim": False,
            "has_permanence_marker": False,
        }

        # Check ownership (first person indicators)
        for category, indicators in self.ownership_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                elements["has_ownership"] = True
                if category == "identity_statements":
                    elements["is_identity_statement"] = True
                elif category == "capability_claims":
                    elements["is_capability_claim"] = True
                break

        # Check action indicators
        elements["has_action"] = any(
            action in text_lower for action in self.action_indicators
        )

        # Check temporal indicators
        elements["has_temporal"] = any(
            temporal in text_lower for temporal in self.temporal_indicators
        )

        # Check context (meaningful content beyond just patterns)
        elements["has_context"] = len(text_lower.split()) > 8  # More than 8 words

        # Check permanence markers
        elements["has_permanence_marker"] = any(
            marker in text_lower for marker in self.permanence_markers
        )

        return elements

    def _calculate_confidence(self, elements: Dict[str, bool]) -> float:
        """Calculate confidence score based on structural elements."""
        score = 0.0

        # Core elements (required for basic commitment)
        if elements["has_ownership"]:
            score += 0.3
        if elements["has_action"]:
            score += 0.2
        if elements["has_context"]:
            score += 0.2

        # Bonus elements
        if elements["has_temporal"]:
            score += 0.1
        if elements["is_identity_statement"]:
            score += 0.15  # Identity statements are strong commitments
        if elements["is_capability_claim"]:
            score += 0.1
        if elements["has_permanence_marker"]:
            score += 0.2  # Permanent commitments get bonus

        return min(1.0, score)

    def _determine_tier(self, text_lower: str, confidence: float) -> CommitmentTier:
        """Determine commitment tier based on content and confidence."""

        # Permanent tier for explicit permanence markers
        if any(marker in text_lower for marker in self.permanence_markers):
            return CommitmentTier.PERMANENT

        # Confirmed tier for high confidence
        if confidence >= 0.7:
            return CommitmentTier.CONFIRMED

        # Tentative for lower confidence
        return CommitmentTier.TENTATIVE

    def _generate_reasoning(self, elements: Dict[str, bool]) -> List[str]:
        """Generate human-readable reasoning for the validation decision."""
        reasoning = []

        if elements["has_ownership"]:
            reasoning.append("Contains first-person ownership indicators")
        if elements["has_action"]:
            reasoning.append("Contains action/intention indicators")
        if elements["has_temporal"]:
            reasoning.append("Contains temporal/future indicators")
        if elements["has_context"]:
            reasoning.append("Has sufficient contextual content")
        if elements["is_identity_statement"]:
            reasoning.append("Contains identity/self-definition statements")
        if elements["has_permanence_marker"]:
            reasoning.append("Contains permanence/principle markers")

        if not reasoning:
            reasoning.append("No clear commitment indicators found")

        return reasoning

    def _check_similarity(self, text: str, existing_commitments: List[str]) -> float:
        """
        Simple similarity check using word overlap.
        In production, this could use embeddings for better accuracy.
        """
        if not existing_commitments:
            return 0.0

        text_words = set(text.lower().split())
        max_similarity = 0.0

        for existing in existing_commitments:
            existing_words = set(existing.lower().split())

            if not text_words or not existing_words:
                continue

            intersection = len(text_words & existing_words)
            union = len(text_words | existing_words)

            if union > 0:
                similarity = intersection / union
                max_similarity = max(max_similarity, similarity)

        return max_similarity

    def should_revisit_tentative(self, commitment_age_hours: float) -> bool:
        """
        Determine if a tentative commitment should be revisited for promotion.
        """
        # Revisit tentative commitments after 24 hours
        return commitment_age_hours >= 24.0


# Example usage and test cases
def test_enhanced_validator():
    """Test the enhanced validator with various commitment types."""

    validator = EnhancedCommitmentValidator()

    test_cases = [
        # Clear commitments
        "I will analyze past conversations to identify patterns and improve my responses.",
        "I commit to being more proactive by asking follow-up questions.",
        # Identity statements (should be captured)
        "I am a reflective assistant designed to provide thoughtful dialogue.",
        "I operate under a framework that allows me to remember past interactions.",
        # Permanence markers
        "I acknowledge honesty as a permanent guiding principle.",
        "This meta-principle will guide how I form future commitments.",
        # Borderline cases
        "Thank you for the feedback.",  # Should be rejected
        "I understand your point about recursive reflection.",  # Borderline
        # Complex self-reflection (should be captured)
        "Based on our interactions, I can summarize what I am now as a supportive entity that aims to foster positive outcomes.",
    ]

    print("Enhanced Commitment Validator Test Results:")
    print("=" * 60)

    for i, text in enumerate(test_cases, 1):
        analysis = validator.validate_commitment(text)

        print(f"\nTest {i}: {text[:50]}...")
        print(f"Valid: {analysis.is_valid}")
        print(f"Tier: {analysis.tier.value}")
        print(f"Confidence: {analysis.confidence:.3f}")
        print(f"Reasoning: {', '.join(analysis.reasoning)}")


if __name__ == "__main__":
    test_enhanced_validator()
