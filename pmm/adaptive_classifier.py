#!/usr/bin/env python3
"""
Adaptive directive classifier - classifies directives into hierarchical tiers.
Uses pattern matching and context analysis to determine directive type.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class ConversationContext:
    """Context information for directive classification."""

    user_message: str
    ai_response: str
    event_id: str
    metadata: Optional[Dict[str, Any]] = None
    # Legacy parameters for backward compatibility
    preceding_user_message: Optional[str] = None
    preceding_ai_response: Optional[str] = None
    directive_position: Optional[int] = None
    conversation_phase: Optional[str] = None
    user_intent_signal: Optional[str] = None


class AdaptiveDirectiveClassifier:
    """
    Classifies directives into hierarchical tiers based on content and context.

    Uses pattern matching, linguistic analysis, and conversation context
    to determine whether a directive is a meta-principle, principle, or commitment.
    """

    def __init__(self):
        self.classification_history = []

        # Pattern definitions for classification
        self.meta_principle_patterns = [
            r"\b(?:always|never|fundamental|core|essential|invariant)\b",
            r"\b(?:across all|in all|consistently|universally)\b",
            r"\b(?:identity|character|nature|essence)\b",
            r"\b(?:meta|overarching|governing|foundational)\b",
        ]

        self.principle_patterns = [
            r"\b(?:should|ought|generally|typically|usually)\b",
            r"\b(?:aim to|strive to|seek to|tend to)\b",
            r"\b(?:prefer|favor|prioritize|emphasize)\b",
            r"\b(?:approach|method|strategy|guideline)\b",
        ]

        self.commitment_patterns = [
            r"\b(?:will|shall|commit to|promise to)\b",
            r"\b(?:next|today|tomorrow|this week|by)\b",
            r"\b(?:deliver|complete|finish|accomplish)\b",
            r"\b(?:specific|concrete|actionable|measurable)\b",
        ]

    def classify_with_context(
        self, text: str, context: ConversationContext
    ) -> Tuple[str, float]:
        """
        Classify directive text with conversation context.

        Args:
            text: Directive text to classify
            context: Conversation context

        Returns:
            Tuple of (classification, confidence)
        """
        text_lower = text.lower()

        # Calculate pattern scores
        meta_score = self._calculate_pattern_score(
            text_lower, self.meta_principle_patterns
        )
        principle_score = self._calculate_pattern_score(
            text_lower, self.principle_patterns
        )
        commitment_score = self._calculate_pattern_score(
            text_lower, self.commitment_patterns
        )

        # Apply context modifiers
        meta_score += self._get_context_modifier(context, "meta-principle")
        principle_score += self._get_context_modifier(context, "principle")
        commitment_score += self._get_context_modifier(context, "commitment")

        # Determine classification
        scores = {
            "meta-principle": meta_score,
            "principle": principle_score,
            "commitment": commitment_score,
        }

        classification = max(scores, key=scores.get)
        confidence = min(0.95, max(0.5, scores[classification]))

        # Record classification for learning
        self.classification_history.append(
            {
                "text": text,
                "classification": classification,
                "confidence": confidence,
                "scores": scores,
            }
        )

        return classification, confidence

    def _calculate_pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate score based on pattern matches."""
        score = 0.0
        for pattern in patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 0.2  # Each match adds 0.2 to score
        return min(1.0, score)  # Cap at 1.0

    def _get_context_modifier(
        self, context: ConversationContext, directive_type: str
    ) -> float:
        """Get context-based score modifier."""
        modifier = 0.0

        if not context.user_message:
            return modifier

        user_lower = context.user_message.lower()

        # User intent signals
        if directive_type == "meta-principle":
            if any(
                word in user_lower
                for word in ["permanent", "always", "identity", "core"]
            ):
                modifier += 0.3
        elif directive_type == "principle":
            if any(
                word in user_lower for word in ["guideline", "approach", "generally"]
            ):
                modifier += 0.3
        elif directive_type == "commitment":
            if any(
                word in user_lower
                for word in ["commit", "will do", "promise", "deliver"]
            ):
                modifier += 0.3

        # Conversation phase
        if hasattr(context, "conversation_phase"):
            if (
                context.conversation_phase == "commitment"
                and directive_type == "commitment"
            ):
                modifier += 0.2
            elif (
                context.conversation_phase == "evolution"
                and directive_type == "meta-principle"
            ):
                modifier += 0.2

        return modifier

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about classification performance."""
        if not self.classification_history:
            return {"total_classifications": 0}

        classifications = [
            item["classification"] for item in self.classification_history
        ]
        confidences = [item["confidence"] for item in self.classification_history]

        return {
            "total_classifications": len(self.classification_history),
            "average_confidence": sum(confidences) / len(confidences),
            "classification_distribution": {
                "meta-principle": classifications.count("meta-principle"),
                "principle": classifications.count("principle"),
                "commitment": classifications.count("commitment"),
            },
        }
