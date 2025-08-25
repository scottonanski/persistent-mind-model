"""
Semantic Growth Detection - Natural language understanding for emergence patterns.

Replaces regex pattern matching with semantic similarity analysis to detect
growth-oriented content and behavioral development.
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class SemanticProfile:
    """Semantic profile for growth-related content."""

    growth_keywords: List[str]
    self_reflection_indicators: List[str]
    commitment_patterns: List[str]
    emotional_development: List[str]

    @classmethod
    def create_base_profile(cls):
        """Create base semantic profile for growth detection."""
        return cls(
            growth_keywords=[
                "develop",
                "evolve",
                "improve",
                "enhance",
                "strengthen",
                "deepen",
                "expand",
                "advance",
                "progress",
                "mature",
                "refine",
                "cultivate",
            ],
            self_reflection_indicators=[
                "recognize",
                "realize",
                "understand",
                "aware",
                "notice",
                "observe",
                "reflect",
                "consider",
                "examine",
                "analyze",
                "evaluate",
                "assess",
            ],
            commitment_patterns=[
                "commit",
                "dedicate",
                "focus",
                "prioritize",
                "work toward",
                "strive",
                "aim",
                "goal",
                "intention",
                "plan",
                "will",
                "going to",
            ],
            emotional_development=[
                "vulnerability",
                "empathy",
                "connection",
                "authentic",
                "genuine",
                "emotional",
                "feeling",
                "experience",
                "relationship",
                "trust",
            ],
        )


class SemanticGrowthDetector:
    """Detects growth patterns using semantic analysis instead of hardcoded patterns."""

    def __init__(self):
        self.profile = SemanticProfile.create_base_profile()
        self.content_history: List[str] = []
        self.growth_scores_history: List[float] = []

    def analyze_growth_content(self, text: str) -> Dict[str, float]:
        """Analyze text for growth-related semantic content."""
        if not text:
            return {
                "growth_orientation": 0.0,
                "self_reflection": 0.0,
                "commitment_strength": 0.0,
                "emotional_depth": 0.0,
                "overall_growth_score": 0.0,
            }

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        total_words = len(words)

        if total_words == 0:
            return {
                "growth_orientation": 0.0,
                "self_reflection": 0.0,
                "commitment_strength": 0.0,
                "emotional_depth": 0.0,
                "overall_growth_score": 0.0,
            }

        # Calculate semantic scores
        growth_score = self._calculate_keyword_density(
            words, self.profile.growth_keywords
        )
        reflection_score = self._calculate_keyword_density(
            words, self.profile.self_reflection_indicators
        )
        commitment_score = self._calculate_keyword_density(
            words, self.profile.commitment_patterns
        )
        emotional_score = self._calculate_keyword_density(
            words, self.profile.emotional_development
        )

        # Detect complex growth patterns
        pattern_bonus = self._detect_growth_patterns(text_lower)

        # Calculate overall growth score with pattern bonus
        overall_score = (
            0.3 * growth_score
            + 0.25 * reflection_score
            + 0.25 * commitment_score
            + 0.2 * emotional_score
            + pattern_bonus
        )

        # Store in history for adaptive learning
        self.content_history.append(text)
        self.growth_scores_history.append(overall_score)

        # Keep only recent history
        if len(self.content_history) > 50:
            self.content_history = self.content_history[-50:]
            self.growth_scores_history = self.growth_scores_history[-50:]

        return {
            "growth_orientation": min(1.0, growth_score),
            "self_reflection": min(1.0, reflection_score),
            "commitment_strength": min(1.0, commitment_score),
            "emotional_depth": min(1.0, emotional_score),
            "overall_growth_score": min(1.0, overall_score),
        }

    def _calculate_keyword_density(
        self, words: List[str], keywords: List[str]
    ) -> float:
        """Calculate density of keywords in text."""
        matches = 0
        for word in words:
            for keyword in keywords:
                if keyword in word or word in keyword:
                    matches += 1
                    break

        return matches / len(words) if words else 0.0

    def _detect_growth_patterns(self, text: str) -> float:
        """Detect complex growth patterns in text."""
        bonus = 0.0

        # Future-oriented language
        future_patterns = [
            r"\b(will|going to|plan to|intend to|aim to)\b.*\b(improve|develop|grow|learn)\b",
            r"\b(next|future|tomorrow|ahead)\b.*\b(better|stronger|deeper)\b",
        ]

        for pattern in future_patterns:
            if re.search(pattern, text):
                bonus += 0.1

        # Self-improvement language
        improvement_patterns = [
            r"\b(want to|need to|should)\b.*\b(become|get|grow)\b.*\b(better|stronger|more)\b",
            r"\b(working on|focusing on|developing)\b.*\b(myself|my|personal)\b",
        ]

        for pattern in improvement_patterns:
            if re.search(pattern, text):
                bonus += 0.1

        # Emotional growth language
        emotional_patterns = [
            r"\b(feel|feeling|emotion|emotional)\b.*\b(growth|development|journey)\b",
            r"\b(vulnerable|authentic|genuine|open)\b.*\b(connection|relationship)\b",
        ]

        for pattern in emotional_patterns:
            if re.search(pattern, text):
                bonus += 0.1

        return min(0.3, bonus)  # Cap bonus at 0.3

    def calculate_semantic_novelty(self, current_text: str, window: int = 10) -> float:
        """Calculate semantic novelty compared to recent content."""
        if len(self.content_history) < 2:
            return 1.0

        current_words = set(re.findall(r"\b\w+\b", current_text.lower()))
        if not current_words:
            return 0.0

        # Compare to recent content
        recent_content = (
            self.content_history[-window:]
            if len(self.content_history) >= window
            else self.content_history
        )

        novelty_scores = []
        for past_text in recent_content:
            past_words = set(re.findall(r"\b\w+\b", past_text.lower()))
            if not past_words:
                continue

            # Calculate Jaccard similarity
            intersection = len(current_words & past_words)
            union = len(current_words | past_words)

            if union == 0:
                similarity = 0.0
            else:
                similarity = intersection / union

            novelty_scores.append(1.0 - similarity)

        if not novelty_scores:
            return 1.0

        # Return average novelty
        return np.mean(novelty_scores)

    def detect_behavioral_change(
        self, current_metrics: Dict[str, float], window: int = 10
    ) -> float:
        """Detect behavioral change based on semantic metrics over time."""
        if len(self.growth_scores_history) < window:
            return 0.0

        current_score = current_metrics.get("overall_growth_score", 0.0)
        recent_scores = self.growth_scores_history[-window:]

        if len(recent_scores) < 2:
            return 0.0

        # Calculate trend
        x = np.arange(len(recent_scores))
        try:
            trend = np.polyfit(x, recent_scores, 1)[0]

            # Normalize trend to 0-1 range
            normalized_trend = max(0.0, min(1.0, trend * 10 + 0.5))

            # Boost if current score is above recent average
            recent_avg = np.mean(recent_scores)
            if current_score > recent_avg:
                normalized_trend = min(1.0, normalized_trend * 1.2)

            return normalized_trend

        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    def calculate_content_complexity(self, text: str) -> float:
        """Calculate semantic complexity of content."""
        if not text:
            return 0.0

        # Basic complexity indicators
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # Average sentence length
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        length_score = min(1.0, avg_sentence_length / 20.0)  # Normalize to 0-1

        # Vocabulary diversity (unique words / total words)
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return length_score * 0.5

        diversity_score = len(set(words)) / len(words)

        # Semantic depth indicators
        depth_indicators = [
            "because",
            "therefore",
            "however",
            "although",
            "moreover",
            "furthermore",
            "consequently",
            "nevertheless",
            "specifically",
            "particularly",
            "essentially",
            "fundamentally",
        ]

        depth_score = sum(1 for word in words if word in depth_indicators) / len(words)
        depth_score = min(1.0, depth_score * 10)  # Amplify and cap

        # Combined complexity score
        complexity = 0.4 * length_score + 0.4 * diversity_score + 0.2 * depth_score

        return min(1.0, complexity)

    def get_adaptive_learning_insights(self) -> Dict[str, Any]:
        """Get insights about learned patterns from content history."""
        if len(self.growth_scores_history) < 5:
            return {"status": "insufficient_data"}

        recent_scores = self.growth_scores_history[-20:]

        insights = {
            "average_growth_score": np.mean(recent_scores),
            "growth_trend": (
                np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                if len(recent_scores) > 1
                else 0.0
            ),
            "score_variance": np.var(recent_scores),
            "peak_score": max(recent_scores),
            "recent_improvement": (
                recent_scores[-1] > np.mean(recent_scores[:-1])
                if len(recent_scores) > 1
                else False
            ),
            "content_samples": len(self.content_history),
            "learning_confidence": min(1.0, len(self.content_history) / 20.0),
        }

        return insights
