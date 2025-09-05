"""
Semantic Growth Detection - Natural language understanding for emergence patterns.

Refactored to remove all regex usage. Uses structural tokenization, sentence
segmentation, and co-occurrence proximity checks instead of regex.
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

# Regex-free structural helpers
from .struct_semantics import split_sentences, normalize_whitespace


def _strip_punct(tok: str) -> str:
    """Strip leading/trailing punctuation without regex."""
    if not tok:
        return tok
    return tok.strip("\"'\t\r\n.,;:!?()[]{}")


def _tokenize_words(text: str) -> List[str]:
    """Lowercase, whitespace-split, and strip punctuation from tokens."""
    if not text:
        return []
    low = normalize_whitespace(text.lower())
    toks = []
    for raw in low.split():
        t = _strip_punct(raw)
        if t:
            toks.append(t)
    return toks


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
        words = _tokenize_words(text_lower)
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

        toks = _tokenize_words(text)

        def has_pair_within(
            a_set: List[str], b_set: List[str], window: int = 8
        ) -> bool:
            """Return True if any token from a_set appears within `window` tokens of any token from b_set."""
            a_pos = [i for i, t in enumerate(toks) if any(a in t for a in a_set)]
            if not a_pos:
                return False
            b_pos = [i for i, t in enumerate(toks) if any(b in t for b in b_set)]
            if not b_pos:
                return False
            for i in a_pos:
                for j in b_pos:
                    if abs(i - j) <= window:
                        return True
            return False

        # Future-oriented language
        future_a = [
            "will",
            "going",
            "plan",
            "intend",
            "aim",
            "next",
            "future",
            "tomorrow",
            "ahead",
        ]
        future_b = [
            "improve",
            "develop",
            "grow",
            "learn",
            "better",
            "stronger",
            "deeper",
        ]
        if has_pair_within(future_a, future_b):
            bonus += 0.1

        # Self-improvement language
        improve_a = ["want", "need", "should", "working", "focusing", "developing"]
        improve_b = [
            "become",
            "get",
            "grow",
            "better",
            "stronger",
            "more",
            "myself",
            "personal",
            "my",
        ]
        if has_pair_within(improve_a, improve_b):
            bonus += 0.1

        # Emotional growth language
        emo_a = [
            "feel",
            "feeling",
            "emotion",
            "emotional",
            "vulnerable",
            "authentic",
            "genuine",
            "open",
        ]
        emo_b = ["growth", "development", "journey", "connection", "relationship"]
        if has_pair_within(emo_a, emo_b):
            bonus += 0.1

        return min(0.3, bonus)  # Cap bonus at 0.3

    def calculate_semantic_novelty(self, current_text: str, window: int = 10) -> float:
        """Calculate semantic novelty compared to recent content."""
        if len(self.content_history) < 2:
            return 1.0

        current_words = set(_tokenize_words(current_text.lower()))
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
            past_words = set(_tokenize_words(past_text.lower()))
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

        # Basic complexity indicators (regex-free)
        # Light preprocessing to help split on ! and ? without regex
        pre = text.replace("!", ".!").replace("?", ".?")
        sentences = [s.strip() for s in split_sentences(pre) if s.strip()]

        if not sentences:
            return 0.0

        # Average sentence length
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        length_score = min(1.0, avg_sentence_length / 20.0)  # Normalize to 0-1

        # Vocabulary diversity (unique words / total words)
        words = _tokenize_words(text.lower())
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
