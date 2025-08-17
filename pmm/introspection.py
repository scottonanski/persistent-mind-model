#!/usr/bin/env python3
"""
PMM Introspection Engine - Hybrid Meta-Cognitive Analysis System

Provides both user-prompted and automatic introspection capabilities with full transparency.
Users can invoke introspection commands explicitly or receive notifications about automatic analysis.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum


class IntrospectionType(Enum):
    """Types of introspection analysis available."""

    PATTERNS = "patterns"  # Behavioral pattern analysis
    DECISIONS = "decisions"  # Decision-making quality review
    GROWTH = "growth"  # Personality evolution assessment
    COMMITMENTS = "commitments"  # Commitment success/failure analysis
    CONFLICTS = "conflicts"  # Internal contradictions/tensions
    GOALS = "goals"  # Goal alignment and progress
    EMERGENCE = "emergence"  # Emergence score analysis
    MEMORY = "memory"  # Memory retrieval effectiveness
    REFLECTION = "reflection"  # Reflection quality assessment


class TriggerReason(Enum):
    """Reasons for automatic introspection triggers."""

    USER_COMMAND = "user_command"
    FAILED_COMMITMENT = "failed_commitment"
    TRAIT_DRIFT = "trait_drift"
    REFLECTION_LOOP = "reflection_loop"
    EMERGENCE_PLATEAU = "emergence_plateau"
    PATTERN_CONFLICT = "pattern_conflict"
    IDENTITY_EVOLUTION = "identity_evolution"
    PERIODIC_REVIEW = "periodic_review"


@dataclass
class IntrospectionResult:
    """Result of an introspection analysis."""

    type: IntrospectionType
    trigger_reason: TriggerReason
    timestamp: str
    analysis: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence: float
    data_sources: List[str]
    user_visible: bool = True


@dataclass
class IntrospectionConfig:
    """Configuration for introspection system."""

    # User notification settings
    notify_automatic: bool = True
    notify_threshold: float = 0.7  # Only notify if confidence >= threshold

    # Automatic trigger settings
    enable_automatic: bool = True
    trait_drift_threshold: float = 0.1
    emergence_plateau_days: int = 7
    reflection_loop_threshold: int = 3

    # Analysis settings
    lookback_days: int = 30
    min_events_for_analysis: int = 10

    # Command aliases
    command_aliases: Dict[str, IntrospectionType] = field(
        default_factory=lambda: {
            "patterns": IntrospectionType.PATTERNS,
            "behavior": IntrospectionType.PATTERNS,
            "decisions": IntrospectionType.DECISIONS,
            "choices": IntrospectionType.DECISIONS,
            "growth": IntrospectionType.GROWTH,
            "evolution": IntrospectionType.GROWTH,
            "commitments": IntrospectionType.COMMITMENTS,
            "promises": IntrospectionType.COMMITMENTS,
            "conflicts": IntrospectionType.CONFLICTS,
            "tensions": IntrospectionType.CONFLICTS,
            "goals": IntrospectionType.GOALS,
            "objectives": IntrospectionType.GOALS,
            "emergence": IntrospectionType.EMERGENCE,
            "scores": IntrospectionType.EMERGENCE,
            "memory": IntrospectionType.MEMORY,
            "recall": IntrospectionType.MEMORY,
            "reflection": IntrospectionType.REFLECTION,
            "insights": IntrospectionType.REFLECTION,
            "help": IntrospectionType.PATTERNS,  # Special case handled separately
        }
    )


class IntrospectionEngine:
    """
    Hybrid introspection system providing both user-prompted and automatic meta-cognitive analysis.

    Features:
    - User-prompted commands (@introspect patterns, @introspect growth, etc.)
    - Automatic triggers based on behavioral patterns and system state
    - Full transparency with user notifications
    - Comprehensive analysis across all PMM subsystems
    """

    def __init__(self, storage_manager, config: Optional[IntrospectionConfig] = None):
        self.storage = storage_manager
        self.config = config or IntrospectionConfig()
        self.last_automatic_check = datetime.now(timezone.utc)
        self.analysis_cache: Dict[str, IntrospectionResult] = {}

    def parse_user_command(self, user_input: str) -> Optional[IntrospectionType]:
        """Parse user input for introspection commands."""
        if not user_input.lower().startswith("@introspect"):
            return None

        # Extract command after @introspect
        parts = user_input.lower().replace("@introspect", "").strip().split()
        if not parts:
            return None

        command = parts[0]
        return self.config.command_aliases.get(command)

    def get_available_commands(self) -> Dict[str, str]:
        """Get all available introspection commands with descriptions."""
        return {
            "@introspect patterns": "Analyze recent behavioral patterns and trends",
            "@introspect decisions": "Review decision-making quality and outcomes",
            "@introspect growth": "Assess personality evolution and development",
            "@introspect commitments": "Analyze commitment success/failure patterns",
            "@introspect conflicts": "Identify internal contradictions or tensions",
            "@introspect goals": "Evaluate goal alignment and progress",
            "@introspect emergence": "Review emergence scores and development stage",
            "@introspect memory": "Assess memory retrieval effectiveness",
            "@introspect reflection": "Analyze reflection quality and insights",
            "@introspect help": "Show all available introspection commands",
        }

    def analyze_patterns(self, lookback_days: int = None) -> IntrospectionResult:
        """Analyze behavioral patterns and trends."""
        lookback = lookback_days or self.config.lookback_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback)

        # Get recent events
        events = self.storage.recent_events(limit=100)
        recent_events = [
            e
            for e in events
            if e.get("ts", "") >= cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
        ]

        # Analyze pattern evolution
        pattern_analysis = {
            "total_events": len(recent_events),
            "event_types": {},
            "temporal_distribution": {},
            "content_themes": [],
        }

        # Count event types
        for event in recent_events:
            kind = event.get("kind", "unknown")
            pattern_analysis["event_types"][kind] = (
                pattern_analysis["event_types"].get(kind, 0) + 1
            )

        # Generate insights
        insights = []
        if pattern_analysis["event_types"].get("reflection", 0) > 5:
            insights.append("High reflection activity indicates active self-analysis")
        if pattern_analysis["event_types"].get("commitment", 0) > 3:
            insights.append("Multiple commitments show goal-oriented behavior")
        if pattern_analysis["event_types"].get("evidence", 0) > 2:
            insights.append("Evidence events suggest commitment follow-through")

        recommendations = []
        if pattern_analysis["event_types"].get("reflection", 0) == 0:
            recommendations.append("Consider more frequent self-reflection")
        if pattern_analysis["total_events"] < self.config.min_events_for_analysis:
            recommendations.append("Increase interaction frequency for better analysis")

        return IntrospectionResult(
            type=IntrospectionType.PATTERNS,
            trigger_reason=TriggerReason.USER_COMMAND,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            analysis=pattern_analysis,
            insights=insights,
            recommendations=recommendations,
            confidence=(
                0.8
                if len(recent_events) >= self.config.min_events_for_analysis
                else 0.5
            ),
            data_sources=[f"{len(recent_events)} events from last {lookback} days"],
        )

    def analyze_growth(self) -> IntrospectionResult:
        """Analyze personality evolution and development trends."""
        # This would integrate with the traits endpoint data
        analysis = {
            "trait_changes": "Analysis would integrate with /traits/drift endpoint",
            "development_stage": "Would use emergence analyzer data",
            "growth_velocity": "Rate of personality change over time",
        }

        insights = [
            "Personality development analysis requires trait drift data",
            "Growth patterns can be tracked through emergence scores",
        ]

        recommendations = [
            "Monitor trait drift patterns for development insights",
            "Use emergence scores to guide growth strategies",
        ]

        return IntrospectionResult(
            type=IntrospectionType.GROWTH,
            trigger_reason=TriggerReason.USER_COMMAND,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            analysis=analysis,
            insights=insights,
            recommendations=recommendations,
            confidence=0.7,
            data_sources=["Trait drift analysis", "Emergence scores"],
        )

    def analyze_commitments(self) -> IntrospectionResult:
        """Analyze commitment success/failure patterns."""
        events = self.storage.recent_events(limit=50)

        commitment_events = [e for e in events if e.get("kind") == "commitment"]
        evidence_events = [e for e in events if e.get("kind") == "evidence"]

        analysis = {
            "total_commitments": len(commitment_events),
            "evidence_events": len(evidence_events),
            "completion_rate": len(evidence_events) / max(1, len(commitment_events)),
            "recent_commitments": [
                e.get("content", "")[:100] for e in commitment_events[:3]
            ],
        }

        insights = []
        completion_rate = analysis["completion_rate"]
        if completion_rate > 0.8:
            insights.append(
                "High commitment completion rate indicates strong follow-through"
            )
        elif completion_rate < 0.3:
            insights.append(
                "Low completion rate suggests need for better commitment management"
            )
        else:
            insights.append("Moderate completion rate shows room for improvement")

        recommendations = []
        if completion_rate < 0.5:
            recommendations.append("Consider smaller, more achievable commitments")
            recommendations.append("Set up better tracking and reminder systems")
        if len(commitment_events) == 0:
            recommendations.append(
                "Consider making specific commitments to drive growth"
            )

        return IntrospectionResult(
            type=IntrospectionType.COMMITMENTS,
            trigger_reason=TriggerReason.USER_COMMAND,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            analysis=analysis,
            insights=insights,
            recommendations=recommendations,
            confidence=0.9 if len(commitment_events) > 0 else 0.3,
            data_sources=[
                f"{len(commitment_events)} commitments, {len(evidence_events)} evidence events"
            ],
        )

    def user_introspect(self, command_type: IntrospectionType) -> IntrospectionResult:
        """Handle user-prompted introspection commands."""
        if command_type == IntrospectionType.PATTERNS:
            return self.analyze_patterns()
        elif command_type == IntrospectionType.GROWTH:
            return self.analyze_growth()
        elif command_type == IntrospectionType.COMMITMENTS:
            return self.analyze_commitments()
        else:
            # Placeholder for other analysis types
            return IntrospectionResult(
                type=command_type,
                trigger_reason=TriggerReason.USER_COMMAND,
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                analysis={
                    "status": f"Analysis for {command_type.value} not yet implemented"
                },
                insights=[
                    f"{command_type.value.title()} analysis is planned for future implementation"
                ],
                recommendations=[
                    "This introspection type will be available in upcoming versions"
                ],
                confidence=0.0,
                data_sources=["Placeholder implementation"],
            )

    def check_automatic_triggers(self) -> List[IntrospectionResult]:
        """Check for conditions that should trigger automatic introspection."""
        if not self.config.enable_automatic:
            return []

        results = []
        now = datetime.now(timezone.utc)

        # Check if enough time has passed since last check
        if (now - self.last_automatic_check).total_seconds() < 3600:  # 1 hour minimum
            return []

        self.last_automatic_check = now

        # Check for failed commitments
        recent_events = self.storage.recent_events(limit=20)
        commitment_events = [e for e in recent_events if e.get("kind") == "commitment"]
        evidence_events = [e for e in recent_events if e.get("kind") == "evidence"]

        if len(commitment_events) > 0 and len(evidence_events) == 0:
            # Commitments without evidence might indicate failures
            result = IntrospectionResult(
                type=IntrospectionType.COMMITMENTS,
                trigger_reason=TriggerReason.FAILED_COMMITMENT,
                timestamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                analysis={
                    "trigger": "Commitments detected without corresponding evidence"
                },
                insights=["Recent commitments may need follow-up or completion"],
                recommendations=[
                    "Review open commitments and provide evidence of completion"
                ],
                confidence=0.6,
                data_sources=[
                    f"{len(commitment_events)} commitments, {len(evidence_events)} evidence"
                ],
                user_visible=self.config.notify_automatic,
            )
            results.append(result)

        return results

    def format_result_for_user(self, result: IntrospectionResult) -> str:
        """Format introspection result for user display."""
        trigger_emoji = (
            "🔍" if result.trigger_reason == TriggerReason.USER_COMMAND else "🤖"
        )
        confidence_bar = "█" * int(result.confidence * 10) + "░" * (
            10 - int(result.confidence * 10)
        )

        output = f"{trigger_emoji} **{result.type.value.title()} Introspection**\n"
        output += f"📊 Confidence: {confidence_bar} ({result.confidence:.1%})\n"
        output += f"⏰ Timestamp: {result.timestamp}\n\n"

        if result.insights:
            output += "💡 **Key Insights:**\n"
            for insight in result.insights:
                output += f"• {insight}\n"
            output += "\n"

        if result.recommendations:
            output += "🎯 **Recommendations:**\n"
            for rec in result.recommendations:
                output += f"• {rec}\n"
            output += "\n"

        if result.trigger_reason != TriggerReason.USER_COMMAND:
            output += f"🔔 *Automatic analysis triggered by: {result.trigger_reason.value.replace('_', ' ')}*\n"

        output += f"📋 Data sources: {', '.join(result.data_sources)}\n"

        return output
