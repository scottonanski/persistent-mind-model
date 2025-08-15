# pmm/meta_reflection.py
"""
Meta-reflection module for Phase 3C self-awareness capabilities.
Enables AI awareness of its own reflection patterns and quality.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, UTC
from pmm.semantic_analysis import get_semantic_analyzer


class MetaReflectionAnalyzer:
    """Analyzes reflection patterns and quality for self-awareness."""

    def __init__(self):
        self.semantic_analyzer = get_semantic_analyzer()

    def analyze_reflection_patterns(
        self, recent_reflections: List[Dict[str, Any]], window_days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze patterns in recent reflections for meta-cognitive awareness.

        Args:
            recent_reflections: List of reflection dicts with 'content', 'timestamp', 'meta'
            window_days: Analysis window in days

        Returns:
            Dict with pattern analysis results
        """
        if not recent_reflections:
            return {
                "total_reflections": 0,
                "avg_quality": 0.0,
                "novelty_trend": 0.0,
                "patterns": [],
                "recommendations": [],
            }

        # Filter reflections within window
        cutoff = datetime.now(UTC) - timedelta(days=window_days)
        windowed_reflections = []

        for reflection in recent_reflections:
            try:
                ts = datetime.fromisoformat(reflection.get("timestamp", ""))
                if ts >= cutoff:
                    windowed_reflections.append(reflection)
            except (ValueError, TypeError, AttributeError):
                # Include reflections without valid timestamps
                windowed_reflections.append(reflection)

        if not windowed_reflections:
            return {
                "total_reflections": 0,
                "avg_quality": 0.0,
                "novelty_trend": 0.0,
                "patterns": [],
                "recommendations": [],
            }

        # Extract reflection contents
        reflection_texts = [r.get("content", "") for r in windowed_reflections]

        # Calculate quality metrics
        quality_scores = []
        novelty_scores = []

        for i, text in enumerate(reflection_texts):
            # Compare against previous reflections for novelty
            previous_texts = reflection_texts[:i] if i > 0 else []
            novelty = self.semantic_analyzer.semantic_novelty_score(
                text, previous_texts
            )
            novelty_scores.append(novelty)

            # Basic quality assessment (length, complexity, specificity)
            quality = self._assess_reflection_quality(text)
            quality_scores.append(quality)

        # Identify semantic clusters (repeated themes)
        clusters = self.semantic_analyzer.cluster_similar_texts(
            reflection_texts, similarity_threshold=0.7
        )

        # Generate insights
        patterns = self._identify_patterns(reflection_texts, clusters)
        recommendations = self._generate_recommendations(
            quality_scores, novelty_scores, patterns
        )

        return {
            "total_reflections": len(windowed_reflections),
            "avg_quality": (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            ),
            "novelty_trend": (
                sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
            ),
            "duplicate_rate": self._calculate_duplicate_rate(novelty_scores),
            "patterns": patterns,
            "recommendations": recommendations,
            "quality_scores": quality_scores,
            "novelty_scores": novelty_scores,
        }

    def _assess_reflection_quality(self, reflection_text: str) -> float:
        """Assess individual reflection quality based on content analysis."""
        if not reflection_text.strip():
            return 0.0

        score = 0.0

        # Length factor (optimal range 50-300 words)
        word_count = len(reflection_text.split())
        if 50 <= word_count <= 300:
            score += 0.3
        elif word_count > 20:
            score += 0.1

        # Specificity indicators
        specific_indicators = [
            "event",
            "commitment",
            "evidence",
            "pattern",
            "behavior",
            "goal",
            "strategy",
            "learning",
            "improvement",
            "challenge",
        ]
        specificity = sum(
            1
            for indicator in specific_indicators
            if indicator in reflection_text.lower()
        )
        score += min(0.3, specificity * 0.05)

        # Self-reference indicators (meta-cognitive awareness)
        self_ref_indicators = [
            "i noticed",
            "i realized",
            "i learned",
            "my approach",
            "my pattern",
        ]
        self_refs = sum(
            1
            for indicator in self_ref_indicators
            if indicator in reflection_text.lower()
        )
        score += min(0.2, self_refs * 0.1)

        # Future-oriented planning
        future_indicators = ["next", "will", "plan", "intend", "goal", "aim"]
        future_refs = sum(
            1 for indicator in future_indicators if indicator in reflection_text.lower()
        )
        score += min(0.2, future_refs * 0.05)

        return min(1.0, score)

    def _calculate_duplicate_rate(
        self, novelty_scores: List[float], threshold: float = 0.2
    ) -> float:
        """Calculate rate of low-novelty (duplicate) reflections."""
        if not novelty_scores:
            return 0.0

        duplicates = sum(1 for score in novelty_scores if score < threshold)
        return duplicates / len(novelty_scores)

    def _identify_patterns(
        self, reflection_texts: List[str], clusters: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """Identify recurring patterns in reflections."""
        patterns = []

        # Cluster-based patterns (repeated themes)
        for cluster in clusters:
            if len(cluster) > 1:  # Only clusters with multiple reflections
                cluster_texts = [reflection_texts[i] for i in cluster]
                # Extract common themes (simplified)
                common_words = self._extract_common_themes(cluster_texts)
                patterns.append(
                    {
                        "type": "repeated_theme",
                        "frequency": len(cluster),
                        "theme": common_words[:5],  # Top 5 common words
                        "reflections": cluster,
                    }
                )

        # Temporal patterns (frequency analysis)
        if len(reflection_texts) >= 3:
            patterns.append(
                {
                    "type": "reflection_frequency",
                    "total_reflections": len(reflection_texts),
                    "avg_per_week": len(reflection_texts)
                    * 7
                    / 7,  # Assuming 7-day window
                }
            )

        return patterns

    def _extract_common_themes(self, texts: List[str]) -> List[str]:
        """Extract common themes from a cluster of similar texts."""
        # Simple word frequency analysis
        word_freq = {}
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
        }

        for text in texts:
            words = text.lower().split()
            for word in words:
                word = word.strip('.,!?;:"()[]{}')
                if len(word) > 3 and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Return most frequent words
        return sorted(word_freq.keys(), key=lambda w: word_freq[w], reverse=True)

    def _generate_recommendations(
        self,
        quality_scores: List[float],
        novelty_scores: List[float],
        patterns: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations for improving reflection quality."""
        recommendations = []

        # Quality-based recommendations
        avg_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )
        if avg_quality < 0.5:
            recommendations.append(
                "Focus on more specific, detailed reflections with concrete examples"
            )

        # Novelty-based recommendations
        avg_novelty = (
            sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
        )
        if avg_novelty < 0.3:
            recommendations.append(
                "Explore new perspectives and avoid repeating similar insights"
            )

        # Pattern-based recommendations
        repeated_themes = [p for p in patterns if p.get("type") == "repeated_theme"]
        if len(repeated_themes) > 2:
            recommendations.append(
                "Consider diversifying reflection topics to avoid repetitive themes"
            )

        # Frequency recommendations
        if len(quality_scores) < 2:
            recommendations.append(
                "Increase reflection frequency for better self-awareness development"
            )
        elif len(quality_scores) > 10:
            recommendations.append(
                "Consider focusing on quality over quantity in reflections"
            )

        return recommendations

    def generate_meta_insight(self, pattern_analysis: Dict[str, Any]) -> Optional[str]:
        """Generate a meta-cognitive insight about reflection patterns."""
        if pattern_analysis["total_reflections"] == 0:
            return None

        insights = []

        # Quality insight
        avg_quality = pattern_analysis["avg_quality"]
        if avg_quality > 0.7:
            insights.append(
                "My reflections show high quality with specific, actionable insights"
            )
        elif avg_quality < 0.4:
            insights.append("My reflections could be more detailed and specific")

        # Novelty insight
        novelty_trend = pattern_analysis["novelty_trend"]
        if novelty_trend > 0.7:
            insights.append(
                "I'm consistently generating novel insights and avoiding repetition"
            )
        elif novelty_trend < 0.3:
            insights.append(
                "I notice I'm repeating similar themes - I should explore new perspectives"
            )

        # Pattern insight
        if pattern_analysis["patterns"]:
            repeated_themes = [
                p
                for p in pattern_analysis["patterns"]
                if p.get("type") == "repeated_theme"
            ]
            if repeated_themes:
                insights.append(
                    f"I tend to focus on {len(repeated_themes)} recurring themes in my reflections"
                )

        if not insights:
            return None

        return "Meta-reflection: " + ". ".join(insights) + "."


# Global analyzer instance
_meta_analyzer = None


def get_meta_reflection_analyzer() -> MetaReflectionAnalyzer:
    """Get global meta-reflection analyzer instance."""
    global _meta_analyzer
    if _meta_analyzer is None:
        _meta_analyzer = MetaReflectionAnalyzer()
    return _meta_analyzer
