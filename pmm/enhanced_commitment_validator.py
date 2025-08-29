#!/usr/bin/env python3
"""
Enhanced commitment validator - validates and tiers commitments with semantic analysis.
Provides multi-tier validation with duplicate detection and structural analysis.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
import difflib


class ValidationTier(Enum):
    """Validation tiers for commitments."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    PERMANENT = "permanent"


@dataclass
class ValidationResult:
    """Result of commitment validation."""

    is_valid: bool
    confidence: float
    tier: ValidationTier
    reasons: List[str]
    similarity_score: Optional[float] = None
    duplicate_of: Optional[str] = None


class EnhancedCommitmentValidator:
    """
    Enhanced commitment validator with semantic analysis and duplicate detection.

    Validates commitments using:
    - Structural analysis (actionability, specificity)
    - Semantic similarity detection for duplicates
    - Context-aware tiering (tentative/confirmed/permanent)
    - Pattern-based quality assessment
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.validation_history = []
        self.similarity_threshold = similarity_threshold

        # Patterns for commitment quality assessment
        self.actionable_patterns = [
            r"\b(?:will|shall|commit to|promise to|aim to)\b",
            r"\b(?:deliver|complete|finish|accomplish|achieve)\b",
            r"\b(?:create|build|develop|implement|design)\b",
            r"\b(?:analyze|review|evaluate|assess|examine)\b",
        ]

        self.specific_patterns = [
            r"\b(?:by|before|within|until|on)\s+\w+",  # Time specificity
            r"\b(?:\d+|one|two|three|four|five)\b",  # Quantity specificity
            r"\b(?:exactly|specifically|precisely)\b",  # Explicit specificity
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Proper nouns (specific entities)
        ]

        self.vague_patterns = [
            r"\b(?:maybe|perhaps|might|could|possibly)\b",
            r"\b(?:try to|attempt to|hope to|wish to)\b",
            r"\b(?:generally|usually|typically|often)\b",
            r"\b(?:something|anything|whatever|somehow)\b",
        ]

    def validate_commitment(
        self, text: str, existing_commitments: List[str]
    ) -> ValidationResult:
        """
        Validate a commitment text against quality criteria and existing commitments.

        Args:
            text: Commitment text to validate
            existing_commitments: List of existing commitment texts

        Returns:
            ValidationResult with detailed validation analysis
        """
        text = text.strip()
        reasons = []

        # Basic length and structure checks
        if len(text) < 15:
            return ValidationResult(
                is_valid=False,
                confidence=0.1,
                tier=ValidationTier.TENTATIVE,
                reasons=["Text too short (minimum 15 characters)"],
            )

        if len(text) > 500:
            reasons.append("Text very long - may need refinement")

        # Check for actionability
        actionable_score = self._calculate_pattern_score(text, self.actionable_patterns)
        if actionable_score == 0:
            reasons.append("No clear actionable language detected")
        else:
            reasons.append(
                f"Actionable language detected (score: {actionable_score:.2f})"
            )

        # Check for specificity
        specific_score = self._calculate_pattern_score(text, self.specific_patterns)
        vague_score = self._calculate_pattern_score(text, self.vague_patterns)

        if specific_score > 0:
            reasons.append(f"Specific elements detected (score: {specific_score:.2f})")
        if vague_score > 0:
            reasons.append(f"Vague language detected (score: {vague_score:.2f})")

        # Check for duplicates
        duplicate_info = self._check_duplicates(text, existing_commitments)
        if duplicate_info["is_duplicate"]:
            return ValidationResult(
                is_valid=False,
                confidence=0.2,
                tier=ValidationTier.TENTATIVE,
                reasons=[
                    f"Duplicate of existing commitment (similarity: {duplicate_info['similarity']:.2f})"
                ],
                similarity_score=duplicate_info["similarity"],
                duplicate_of=duplicate_info["duplicate_text"],
            )

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            actionable_score, specific_score, vague_score, len(text)
        )

        # Determine tier and validity
        if quality_score >= 0.8:
            tier = ValidationTier.PERMANENT
            is_valid = True
            confidence = min(0.95, quality_score)
        elif quality_score >= 0.6:
            tier = ValidationTier.CONFIRMED
            is_valid = True
            confidence = quality_score
        elif quality_score >= 0.4:
            tier = ValidationTier.TENTATIVE
            is_valid = True
            confidence = quality_score
        else:
            tier = ValidationTier.TENTATIVE
            is_valid = False
            confidence = quality_score

        reasons.append(f"Overall quality score: {quality_score:.2f}")

        result = ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            tier=tier,
            reasons=reasons,
            similarity_score=duplicate_info.get("max_similarity", 0.0),
        )

        # Record validation for learning
        self.validation_history.append(
            {
                "text": text,
                "result": result,
                "existing_count": len(existing_commitments),
                "quality_score": quality_score,
                "actionable_score": actionable_score,
                "specific_score": specific_score,
                "vague_score": vague_score,
            }
        )

        return result

    def _calculate_pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate score based on pattern matches."""
        score = 0.0
        text_lower = text.lower()

        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
            score += matches * 0.2  # Each match adds 0.2

        return min(1.0, score)  # Cap at 1.0

    def _check_duplicates(
        self, text: str, existing_commitments: List[str]
    ) -> Dict[str, Any]:
        """Check for duplicate commitments using similarity analysis."""
        if not existing_commitments:
            return {"is_duplicate": False, "similarity": 0.0, "max_similarity": 0.0}

        text_normalized = self._normalize_text(text)
        max_similarity = 0.0
        most_similar_text = None

        for existing in existing_commitments:
            existing_normalized = self._normalize_text(existing)
            similarity = difflib.SequenceMatcher(
                None, text_normalized, existing_normalized
            ).ratio()

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_text = existing

        is_duplicate = max_similarity >= self.similarity_threshold

        return {
            "is_duplicate": is_duplicate,
            "similarity": max_similarity,
            "duplicate_text": most_similar_text if is_duplicate else None,
            "max_similarity": max_similarity,
        }

    def _normalize_text(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r"\s+", " ", text.lower().strip())

        # Remove common commitment prefixes that don't affect meaning
        prefixes = [
            r"^i\s+(?:will|shall|commit\s+to|promise\s+to|aim\s+to)\s+",
            r"^next,?\s+i\s+will\s+",
            r"^i\s+acknowledge\s+that\s+i\s+will\s+",
        ]

        for prefix in prefixes:
            normalized = re.sub(prefix, "", normalized)

        return normalized.strip()

    def _calculate_quality_score(
        self, actionable: float, specific: float, vague: float, length: int
    ) -> float:
        """Calculate overall quality score for commitment."""
        # Base score from actionability and specificity
        base_score = (actionable * 0.6) + (specific * 0.3)

        # Penalty for vague language
        vague_penalty = vague * 0.2

        # Length bonus/penalty
        length_factor = 1.0
        if length < 30:
            length_factor = 0.8  # Too short
        elif length > 200:
            length_factor = 0.9  # Too long
        elif 50 <= length <= 150:
            length_factor = 1.1  # Good length

        # Calculate final score
        final_score = (base_score - vague_penalty) * length_factor

        return max(0.0, min(1.0, final_score))

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about validation performance."""
        if not self.validation_history:
            return {"total_validations": 0}

        valid_count = sum(
            1 for item in self.validation_history if item["result"].is_valid
        )
        tier_counts = {}

        for item in self.validation_history:
            tier = item["result"].tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            "total_validations": len(self.validation_history),
            "validation_rate": valid_count / len(self.validation_history),
            "average_confidence": sum(
                item["result"].confidence for item in self.validation_history
            )
            / len(self.validation_history),
            "average_quality_score": sum(
                item["quality_score"] for item in self.validation_history
            )
            / len(self.validation_history),
            "tier_distribution": tier_counts,
            "duplicate_detection_rate": sum(
                1 for item in self.validation_history if item["result"].duplicate_of
            )
            / len(self.validation_history),
        }
