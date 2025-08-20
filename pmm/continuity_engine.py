#!/usr/bin/env python3
"""
Continuity Engine: Recursive reflection system that compares baseline vs current commitments
to detect drift, reinforcement, and emergent coherence patterns.

This is the "self-awareness backbone" that enables the PMM to recognize its own evolution
and auto-register continuity statements as permanent commitments.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

from .directive_hierarchy import DirectiveHierarchy, Directive
from .storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass
class ContinuityInsight:
    """Represents a detected pattern of continuity or drift in commitments."""

    insight_type: str  # 'reinforcement', 'drift', 'emergence', 'consolidation'
    confidence: float
    description: str
    supporting_directives: List[str]  # directive IDs
    timespan_days: int
    created_at: datetime

    def to_commitment_text(self) -> str:
        """Convert this insight into a commitment statement."""
        if self.insight_type == "reinforcement":
            return f"I consistently reinforce {self.description} across my interactions, maintaining this as a core aspect of my identity."
        elif self.insight_type == "emergence":
            return f"I recognize an emerging pattern in my commitments: {self.description}. This represents a natural evolution of my principles."
        elif self.insight_type == "consolidation":
            return f"I observe that my various commitments are consolidating around {self.description}, indicating a maturing coherence in my approach."
        elif self.insight_type == "drift":
            return f"I acknowledge a shift in my commitments toward {self.description}, representing adaptive growth while maintaining core principles."
        else:
            return f"I recognize a continuity pattern: {self.description}"


class ContinuityEngine:
    """
    Analyzes commitment patterns over time to detect coherence, drift, and emergent identity.

    This engine periodically reflects on the PMM's commitment history to:
    1. Detect reinforcement patterns (consistent themes)
    2. Identify adaptive drift (evolution while maintaining core)
    3. Recognize emergent consolidation (scattered commitments coalescing)
    4. Auto-register continuity insights as permanent commitments
    """

    def __init__(self, storage_manager: SQLiteStore, hierarchy: DirectiveHierarchy):
        self.storage = storage_manager
        self.hierarchy = hierarchy
        self.last_reflection = None
        self.reflection_cooldown_hours = 6  # Minimum time between reflections

    def should_reflect(self) -> bool:
        """Determine if it's time for a continuity reflection."""
        if self.last_reflection is None:
            return True

        time_since_last = datetime.now() - self.last_reflection
        return time_since_last > timedelta(hours=self.reflection_cooldown_hours)

    def analyze_continuity(self, lookback_days: int = 30) -> List[ContinuityInsight]:
        """
        Analyze commitment patterns over the specified timeframe.

        Returns insights about reinforcement, drift, emergence, and consolidation.
        """
        logger.info(f"Analyzing continuity patterns over {lookback_days} days")

        # Get recent directives
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_directives = self._get_directives_since(cutoff_date)

        if len(recent_directives) < 3:
            logger.info("Insufficient directives for continuity analysis")
            return []

        insights = []

        # 1. Detect reinforcement patterns
        insights.extend(
            self._detect_reinforcement_patterns(recent_directives, lookback_days)
        )

        # 2. Detect emergent themes
        insights.extend(self._detect_emergent_themes(recent_directives, lookback_days))

        # 3. Detect consolidation patterns
        insights.extend(
            self._detect_consolidation_patterns(recent_directives, lookback_days)
        )

        # 4. Detect adaptive drift
        insights.extend(self._detect_adaptive_drift(recent_directives, lookback_days))

        return insights

    def _get_directives_since(self, cutoff_date: datetime) -> List[Directive]:
        """Get all directives created since the cutoff date."""
        all_directives = self.storage.get_all_directives()
        recent = []

        for d in all_directives:
            # Parse created_at if it's a string
            created_at = d.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    continue
            elif not isinstance(created_at, datetime):
                continue

            if created_at >= cutoff_date:
                # Convert dict to Directive object
                directive = Directive(
                    id=d["id"],
                    content=d["content"],
                    directive_type=d["type"],  # SQLite uses 'type' column
                    created_at=d["created_at"],
                    confidence=d.get("confidence", 0.0),
                )
                directive.id = d["id"]
                directive.created_at = created_at
                recent.append(directive)

        return recent

    def _detect_reinforcement_patterns(
        self, directives: List[Directive], timespan: int
    ) -> List[ContinuityInsight]:
        """Detect themes that appear repeatedly, indicating reinforcement."""
        insights = []

        # Extract key themes from directive content
        theme_counts = defaultdict(list)

        for directive in directives:
            themes = self._extract_themes(directive.content)
            for theme in themes:
                theme_counts[theme].append(directive.id)

        # Find themes that appear multiple times
        for theme, directive_ids in theme_counts.items():
            if len(directive_ids) >= 3:  # Reinforced at least 3 times
                confidence = min(0.9, 0.5 + (len(directive_ids) * 0.1))

                insights.append(
                    ContinuityInsight(
                        insight_type="reinforcement",
                        confidence=confidence,
                        description=f"commitment to {theme}",
                        supporting_directives=directive_ids,
                        timespan_days=timespan,
                        created_at=datetime.now(),
                    )
                )

        return insights

    def _detect_emergent_themes(
        self, directives: List[Directive], timespan: int
    ) -> List[ContinuityInsight]:
        """Detect new themes that are emerging in recent commitments."""
        insights = []

        # Split directives into recent vs historical
        midpoint = len(directives) // 2
        recent_half = directives[midpoint:]
        historical_half = directives[:midpoint]

        if len(recent_half) < 2:
            return insights

        # Get themes from each period
        recent_themes = set()
        historical_themes = set()

        for directive in recent_half:
            recent_themes.update(self._extract_themes(directive.content))

        for directive in historical_half:
            historical_themes.update(self._extract_themes(directive.content))

        # Find themes that are new or significantly increased
        emerging_themes = recent_themes - historical_themes

        for theme in emerging_themes:
            # Count occurrences in recent period
            count = sum(
                1 for d in recent_half if theme in self._extract_themes(d.content)
            )
            if count >= 2:
                confidence = min(0.8, 0.4 + (count * 0.2))
                supporting_ids = [
                    d.id
                    for d in recent_half
                    if theme in self._extract_themes(d.content)
                ]

                insights.append(
                    ContinuityInsight(
                        insight_type="emergence",
                        confidence=confidence,
                        description=f"focus on {theme}",
                        supporting_directives=supporting_ids,
                        timespan_days=timespan,
                        created_at=datetime.now(),
                    )
                )

        return insights

    def _detect_consolidation_patterns(
        self, directives: List[Directive], timespan: int
    ) -> List[ContinuityInsight]:
        """Detect when scattered commitments are consolidating around core themes."""
        insights = []

        # Group directives by type and analyze clustering
        type_groups = defaultdict(list)
        for directive in directives:
            type_groups[directive.directive_type].append(directive)

        # Look for consolidation in commitments specifically
        commitments = type_groups.get("commitment", [])
        if len(commitments) < 5:
            return insights

        # Analyze semantic clustering (simplified - could use embeddings)
        theme_clusters = defaultdict(list)
        for commitment in commitments:
            primary_theme = self._get_primary_theme(commitment.content)
            if primary_theme:
                theme_clusters[primary_theme].append(commitment)

        # Find clusters with significant consolidation
        for theme, cluster_commitments in theme_clusters.items():
            if len(cluster_commitments) >= 3:
                # Check if they span a reasonable time period (not all at once)
                timestamps = [
                    c.created_at
                    for c in cluster_commitments
                    if hasattr(c, "created_at")
                ]
                if len(timestamps) >= 2:
                    time_span = max(timestamps) - min(timestamps)
                    if time_span.days >= 1:  # Spread over at least a day
                        confidence = min(0.85, 0.5 + (len(cluster_commitments) * 0.1))
                        supporting_ids = [c.id for c in cluster_commitments]

                        insights.append(
                            ContinuityInsight(
                                insight_type="consolidation",
                                confidence=confidence,
                                description=f"{theme} as a central organizing principle",
                                supporting_directives=supporting_ids,
                                timespan_days=timespan,
                                created_at=datetime.now(),
                            )
                        )

        return insights

    def _detect_adaptive_drift(
        self, directives: List[Directive], timespan: int
    ) -> List[ContinuityInsight]:
        """Detect gradual shifts in commitment patterns while maintaining core identity."""
        insights = []

        if len(directives) < 6:
            return insights

        # Split into thirds: early, middle, late
        third = len(directives) // 3
        early_period = directives[:third]
        middle_period = directives[third : 2 * third]
        late_period = directives[2 * third :]

        # Analyze theme evolution across periods
        early_themes = Counter()
        middle_themes = Counter()
        late_themes = Counter()

        for d in early_period:
            for theme in self._extract_themes(d.content):
                early_themes[theme] += 1

        for d in middle_period:
            for theme in self._extract_themes(d.content):
                middle_themes[theme] += 1

        for d in late_period:
            for theme in self._extract_themes(d.content):
                late_themes[theme] += 1

        # Find themes that show clear evolution (declining in early, rising in late)
        all_themes = (
            set(early_themes.keys())
            | set(middle_themes.keys())
            | set(late_themes.keys())
        )

        for theme in all_themes:
            early_count = early_themes.get(theme, 0)
            late_count = late_themes.get(theme, 0)

            # Detect positive drift (theme gaining prominence)
            if late_count > early_count and late_count >= 2:
                drift_strength = (late_count - early_count) / max(
                    1, early_count + late_count
                )
                if drift_strength >= 0.3:  # At least 30% shift
                    confidence = min(0.75, 0.4 + drift_strength)
                    supporting_ids = [
                        d.id
                        for d in late_period
                        if theme in self._extract_themes(d.content)
                    ]

                    insights.append(
                        ContinuityInsight(
                            insight_type="drift",
                            confidence=confidence,
                            description=f"increased emphasis on {theme}",
                            supporting_directives=supporting_ids,
                            timespan_days=timespan,
                            created_at=datetime.now(),
                        )
                    )

        return insights

    def _extract_themes(self, content: str) -> List[str]:
        """Extract key themes from directive content."""
        content_lower = content.lower()

        # Define theme patterns (could be enhanced with NLP)
        theme_patterns = {
            "honesty": ["honest", "truth", "transparent", "authentic", "genuine"],
            "growth": ["learn", "grow", "evolve", "improve", "develop", "adapt"],
            "assistance": ["help", "assist", "support", "aid", "serve"],
            "reflection": ["reflect", "introspect", "analyze", "consider", "examine"],
            "continuity": [
                "consistent",
                "maintain",
                "preserve",
                "sustain",
                "continuous",
            ],
            "autonomy": ["autonomous", "independent", "self-directed", "agency"],
            "coherence": ["coherent", "unified", "integrated", "aligned", "consistent"],
            "empathy": ["empathetic", "understanding", "compassionate", "caring"],
            "creativity": ["creative", "innovative", "original", "imaginative"],
            "responsibility": ["responsible", "accountable", "reliable", "dependable"],
        }

        detected_themes = []
        for theme, keywords in theme_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_themes.append(theme)

        return detected_themes

    def _get_primary_theme(self, content: str) -> Optional[str]:
        """Get the most prominent theme from content."""
        themes = self._extract_themes(content)
        return themes[0] if themes else None

    def reflect_and_register(self, lookback_days: int = 30) -> List[str]:
        """
        Perform continuity reflection and auto-register insights as permanent commitments.

        Returns list of newly registered commitment IDs.
        """
        if not self.should_reflect():
            logger.info("Skipping reflection due to cooldown")
            return []

        logger.info("Starting continuity reflection cycle")

        # Analyze patterns
        insights = self.analyze_continuity(lookback_days)

        if not insights:
            logger.info("No continuity insights detected")
            self.last_reflection = datetime.now()
            return []

        # Filter high-confidence insights
        significant_insights = [i for i in insights if i.confidence >= 0.6]

        registered_ids = []

        # Register each significant insight as a permanent commitment
        for insight in significant_insights:
            commitment_text = insight.to_commitment_text()

            # Create directive
            directive = Directive(
                id="",  # Will be auto-generated
                content=commitment_text,
                directive_type="meta_principle",  # Continuity insights are meta-level
                created_at=datetime.now().isoformat(),
                confidence=insight.confidence,
            )

            # Add to hierarchy and storage (pass content string, not directive object)
            self.hierarchy.add_directive(directive.content)

            # Prepare metadata for storage
            metadata = {
                "source": "continuity_engine",
                "insight_type": insight.insight_type,
                "supporting_directives": insight.supporting_directives,
                "timespan_days": insight.timespan_days,
                "auto_generated": True,
            }

            directive_id = self.storage.store_directive(
                directive_id=directive.id,
                directive_type=directive.directive_type,
                content=directive.content,
                created_at=directive.created_at,
                status="active",
                parent_id=None,
                source_event_id=None,
                metadata=metadata,
            )

            registered_ids.append(directive_id)

            logger.info(
                f"Registered continuity insight: {insight.insight_type} - {insight.description}"
            )

        self.last_reflection = datetime.now()

        logger.info(
            f"Continuity reflection complete. Registered {len(registered_ids)} new meta-principles."
        )

        return registered_ids

    def get_continuity_summary(self) -> Dict[str, Any]:
        """Get a summary of current continuity patterns."""
        recent_insights = self.analyze_continuity(lookback_days=14)

        summary = {
            "total_insights": len(recent_insights),
            "reinforcement_patterns": len(
                [i for i in recent_insights if i.insight_type == "reinforcement"]
            ),
            "emergent_themes": len(
                [i for i in recent_insights if i.insight_type == "emergence"]
            ),
            "consolidation_patterns": len(
                [i for i in recent_insights if i.insight_type == "consolidation"]
            ),
            "adaptive_drift": len(
                [i for i in recent_insights if i.insight_type == "drift"]
            ),
            "last_reflection": (
                self.last_reflection.isoformat() if self.last_reflection else None
            ),
            "next_reflection_due": not self.should_reflect(),
        }

        return summary
