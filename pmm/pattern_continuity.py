#!/usr/bin/env python3
"""
Pattern continuity manager - tracks and analyzes behavioral patterns for continuity.
Detects pattern reinforcement, breaks, and emergent themes across interactions.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import re
import difflib


@dataclass
class PatternEvent:
    """Represents a pattern event in the continuity system."""
    timestamp: datetime
    pattern_type: str
    content: str
    metadata: Dict[str, Any]
    confidence: float = 0.0
    source_event_id: Optional[str] = None


@dataclass
class ContinuityPattern:
    """Represents a detected continuity pattern."""
    pattern_id: str
    pattern_type: str
    occurrences: List[PatternEvent]
    strength: float
    trend: str  # "reinforcing", "weakening", "stable"
    first_seen: datetime
    last_seen: datetime


class PatternContinuityManager:
    """
    Tracks and analyzes behavioral patterns for continuity detection.
    
    Identifies:
    - Pattern reinforcement (repeated behaviors)
    - Pattern breaks (discontinuities)
    - Emergent themes (new patterns forming)
    - Behavioral drift (gradual changes)
    """
    
    def __init__(self, storage_manager=None, min_reference_count=3, similarity_threshold=0.8, min_references=None):
        self.storage = storage_manager
        self.pattern_history = []
        self.detected_patterns = {}
        self.continuity_threshold = 0.7
        
        # Configuration parameters
        self.min_reference_count = min_reference_count
        self.similarity_threshold = similarity_threshold
        self.min_references = min_references or min_reference_count  # Legacy compatibility
        self.max_lookback_days = 30
        
        # Pattern detection settings
        self.pattern_types = [
            "commitment_style",
            "reflection_depth", 
            "response_tone",
            "topic_focus",
            "decision_making",
            "value_expression"
        ]
    
    def track_pattern(self, pattern_type: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track a new pattern event and analyze for continuity.
        
        Args:
            pattern_type: Type of pattern (e.g., "commitment", "reflection", "behavior")
            content: Pattern content
            metadata: Optional metadata including confidence and source
        """
        event = PatternEvent(
            timestamp=datetime.now(timezone.utc),
            pattern_type=pattern_type,
            content=content,
            metadata=metadata or {},
            confidence=metadata.get("confidence", 0.5) if metadata else 0.5,
            source_event_id=metadata.get("source_event_id") if metadata else None
        )
        
        self.pattern_history.append(event)
        
        # Update detected patterns
        self._update_pattern_detection(event)
        
        # Persist to storage if available
        if self.storage:
            self._persist_pattern_event(event)
    
    def analyze_continuity(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Analyze pattern continuity over recent events with detailed metrics.
        
        Args:
            window_size: Number of recent events to analyze
            
        Returns:
            Comprehensive continuity analysis
        """
        recent_events = self.pattern_history[-window_size:] if self.pattern_history else []
        
        if not recent_events:
            return {
                "continuity_score": 0.0,
                "dominant_patterns": [],
                "pattern_diversity": 0.0,
                "reinforcement_strength": 0.0,
                "trend_analysis": {}
            }
        
        # Analyze pattern frequencies
        pattern_counts = {}
        pattern_confidences = {}
        
        for event in recent_events:
            pattern_type = event.pattern_type
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            if pattern_type not in pattern_confidences:
                pattern_confidences[pattern_type] = []
            pattern_confidences[pattern_type].append(event.confidence)
        
        # Calculate continuity metrics
        total_events = len(recent_events)
        dominant_pattern_count = max(pattern_counts.values()) if pattern_counts else 0
        continuity_score = dominant_pattern_count / total_events if total_events > 0 else 0.0
        
        # Pattern diversity (Shannon entropy-like measure)
        pattern_diversity = len(pattern_counts) / total_events if total_events > 0 else 0.0
        
        # Reinforcement strength (weighted by confidence)
        reinforcement_strength = 0.0
        if pattern_confidences:
            for pattern_type, confidences in pattern_confidences.items():
                avg_confidence = sum(confidences) / len(confidences)
                frequency_weight = pattern_counts[pattern_type] / total_events
                reinforcement_strength += avg_confidence * frequency_weight
        
        # Trend analysis
        trend_analysis = self._analyze_trends(recent_events)
        
        return {
            "continuity_score": continuity_score,
            "dominant_patterns": sorted(pattern_counts.keys(), key=lambda x: pattern_counts[x], reverse=True),
            "pattern_diversity": pattern_diversity,
            "pattern_counts": pattern_counts,
            "total_events": total_events,
            "reinforcement_strength": reinforcement_strength,
            "trend_analysis": trend_analysis,
            "average_confidence": sum(e.confidence for e in recent_events) / len(recent_events)
        }
    
    def detect_pattern_breaks(self) -> List[Dict[str, Any]]:
        """
        Detect breaks in pattern continuity using multiple detection methods.
        
        Returns:
            List of detected pattern breaks with detailed analysis
        """
        breaks = []
        
        if len(self.pattern_history) < self.min_reference_count:
            return breaks
        
        # Method 1: Sudden diversity increase
        if len(self.pattern_history) >= 10:
            recent_window = self.pattern_history[-5:]
            previous_window = self.pattern_history[-10:-5]
            
            recent_types = set(e.pattern_type for e in recent_window)
            previous_types = set(e.pattern_type for e in previous_window)
            
            diversity_increase = len(recent_types) - len(previous_types)
            if diversity_increase >= 2:
                breaks.append({
                    "type": "diversity_spike",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "description": f"Pattern diversity increased by {diversity_increase}",
                    "severity": "medium" if diversity_increase == 2 else "high"
                })
        
        # Method 2: Confidence drop
        if len(self.pattern_history) >= 6:
            recent_confidences = [e.confidence for e in self.pattern_history[-3:]]
            previous_confidences = [e.confidence for e in self.pattern_history[-6:-3]]
            
            recent_avg = sum(recent_confidences) / len(recent_confidences)
            previous_avg = sum(previous_confidences) / len(previous_confidences)
            
            confidence_drop = previous_avg - recent_avg
            if confidence_drop > 0.3:
                breaks.append({
                    "type": "confidence_drop",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "description": f"Pattern confidence dropped by {confidence_drop:.2f}",
                    "severity": "medium" if confidence_drop < 0.5 else "high"
                })
        
        # Method 3: Pattern absence (expected pattern missing)
        dominant_patterns = self._get_dominant_patterns()
        if dominant_patterns:
            recent_patterns = set(e.pattern_type for e in self.pattern_history[-5:])
            missing_patterns = set(dominant_patterns) - recent_patterns
            
            if missing_patterns:
                breaks.append({
                    "type": "pattern_absence",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "description": f"Expected patterns missing: {', '.join(missing_patterns)}",
                    "severity": "low"
                })
        
        return breaks
    
    def _update_pattern_detection(self, event: PatternEvent) -> None:
        """Update detected patterns with new event."""
        pattern_key = f"{event.pattern_type}_{self._get_content_signature(event.content)}"
        
        if pattern_key not in self.detected_patterns:
            self.detected_patterns[pattern_key] = ContinuityPattern(
                pattern_id=pattern_key,
                pattern_type=event.pattern_type,
                occurrences=[event],
                strength=event.confidence,
                trend="new",
                first_seen=event.timestamp,
                last_seen=event.timestamp
            )
        else:
            pattern = self.detected_patterns[pattern_key]
            pattern.occurrences.append(event)
            pattern.last_seen = event.timestamp
            
            # Update strength (moving average)
            pattern.strength = (pattern.strength * 0.7) + (event.confidence * 0.3)
            
            # Update trend
            if len(pattern.occurrences) >= 3:
                recent_confidences = [e.confidence for e in pattern.occurrences[-3:]]
                if recent_confidences[-1] > recent_confidences[0]:
                    pattern.trend = "reinforcing"
                elif recent_confidences[-1] < recent_confidences[0]:
                    pattern.trend = "weakening"
                else:
                    pattern.trend = "stable"
    
    def _get_content_signature(self, content: str) -> str:
        """Generate a signature for content similarity."""
        # Simple word-based signature
        words = re.findall(r'\w+', content.lower())
        return "_".join(sorted(set(words))[:5])  # Top 5 unique words
    
    def _analyze_trends(self, events: List[PatternEvent]) -> Dict[str, Any]:
        """Analyze trends in recent events."""
        if len(events) < 3:
            return {"insufficient_data": True}
        
        # Confidence trend
        confidences = [e.confidence for e in events]
        confidence_trend = "stable"
        if confidences[-1] > confidences[0] + 0.1:
            confidence_trend = "increasing"
        elif confidences[-1] < confidences[0] - 0.1:
            confidence_trend = "decreasing"
        
        # Pattern type stability
        pattern_types = [e.pattern_type for e in events]
        unique_types = len(set(pattern_types))
        stability = "stable" if unique_types <= 2 else "diverse"
        
        return {
            "confidence_trend": confidence_trend,
            "pattern_stability": stability,
            "unique_pattern_types": unique_types,
            "confidence_range": max(confidences) - min(confidences)
        }
    
    def _get_dominant_patterns(self) -> List[str]:
        """Get list of dominant pattern types from history."""
        if len(self.pattern_history) < self.min_reference_count:
            return []
        
        pattern_counts = {}
        for event in self.pattern_history[-20:]:  # Look at recent history
            pattern_counts[event.pattern_type] = pattern_counts.get(event.pattern_type, 0) + 1
        
        # Return patterns that appear at least min_reference_count times
        return [p for p, count in pattern_counts.items() if count >= self.min_reference_count]
    
    def _persist_pattern_event(self, event: PatternEvent) -> None:
        """Persist pattern event to storage."""
        if not self.storage:
            return
        
        try:
            # This would integrate with PMM's storage system
            # For now, just log the attempt
            pass
        except Exception as e:
            print(f"Warning: Failed to persist pattern event: {e}")
    
    def get_continuity_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about pattern continuity."""
        analysis = self.analyze_continuity()
        breaks = self.detect_pattern_breaks()
        
        return {
            "total_patterns": len(self.pattern_history),
            "detected_pattern_types": len(self.detected_patterns),
            "continuity_threshold": self.continuity_threshold,
            "recent_analysis": analysis,
            "pattern_breaks": len(breaks),
            "break_details": breaks,
            "strongest_patterns": self._get_strongest_patterns(),
            "configuration": {
                "min_reference_count": self.min_reference_count,
                "similarity_threshold": self.similarity_threshold,
                "max_lookback_days": self.max_lookback_days
            }
        }
    
    def _get_strongest_patterns(self) -> List[Dict[str, Any]]:
        """Get the strongest detected patterns."""
        patterns = list(self.detected_patterns.values())
        patterns.sort(key=lambda p: p.strength, reverse=True)
        
        return [{
            "pattern_id": p.pattern_id,
            "pattern_type": p.pattern_type,
            "strength": p.strength,
            "trend": p.trend,
            "occurrences": len(p.occurrences),
            "duration_days": (p.last_seen - p.first_seen).days
        } for p in patterns[:5]]
    
    def enhance_context_with_continuity(self, original_context: str) -> str:
        """
        Enhance context with pattern continuity information.
        
        Args:
            original_context: The original context string
            
        Returns:
            Enhanced context with continuity patterns
        """
        # Get recent events for pattern analysis
        recent_events = self.storage.recent_events(limit=20) if self.storage else []
        
        if not recent_events:
            return original_context
        
        # Analyze patterns in recent events
        continuity_insights = []
        
        # Look for commitment patterns
        commitments = [e for e in recent_events if e.get("kind") == "commitment"]
        if commitments:
            commitment_refs = ", ".join([f"Commitment {e['id']}" for e in commitments])
            continuity_insights.append(f"Recent commitments: {len(commitments)} active ({commitment_refs})")
        
        # Look for reflection patterns
        reflections = [e for e in recent_events if e.get("kind") == "reflection"]
        if reflections:
            insight_refs = ", ".join([f"Insight {e['id']}" for e in reflections])
            continuity_insights.append(f"Reflection frequency: {len(reflections)} insights ({insight_refs})")
        
        # Look for evidence patterns
        evidence = [e for e in recent_events if e.get("kind") == "evidence"]
        if evidence:
            evidence_refs = ", ".join([f"Evidence {e['id']}" for e in evidence])
            continuity_insights.append(f"Evidence tracking: {len(evidence)} completions ({evidence_refs})")
        
        # Get strongest patterns
        strong_patterns = self._get_strongest_patterns()
        if strong_patterns:
            pattern_summary = ", ".join([p["pattern_type"] for p in strong_patterns[:3]])
            continuity_insights.append(f"Dominant patterns: {pattern_summary}")
        
        # Enhance context if we have insights
        if continuity_insights:
            enhanced = f"{original_context}\n\nCONTINUITY CONTEXT:\n"
            enhanced += "\n".join(f"- {insight}" for insight in continuity_insights)
            return enhanced
        
        return original_context
    
    def calculate_pattern_reuse_score(self, current_text: str, historical_events: List[Dict]) -> float:
        """
        Calculate how much the current text reuses patterns from historical events.
        
        Args:
            current_text: Text to analyze for pattern reuse
            historical_events: List of historical events to compare against
            
        Returns:
            Pattern reuse score between 0.0 and 1.0
        """
        if not historical_events:
            return 0.0
        
        current_words = set(current_text.lower().split())
        
        # Calculate overlap with historical content
        total_overlap = 0
        total_words = 0
        
        for event in historical_events:
            content = event.get("content", "")
            if content:
                event_words = set(content.lower().split())
                overlap = len(current_words.intersection(event_words))
                total_overlap += overlap
                total_words += len(event_words)
        
        if total_words == 0:
            return 0.0
        
        # Normalize by total historical vocabulary
        reuse_score = min(1.0, total_overlap / len(current_words))
        return reuse_score
    
    def apply_novelty_decay(self, base_novelty: float, reuse_score: float) -> float:
        """
        Apply novelty decay based on pattern reuse.
        
        Args:
            base_novelty: Base novelty score (0.0-1.0)
            reuse_score: Pattern reuse score (0.0-1.0)
            
        Returns:
            Adjusted novelty score
        """
        # Higher reuse means lower novelty
        decay_factor = 1.0 - (reuse_score * 0.5)  # Max 50% decay
        adjusted_novelty = base_novelty * decay_factor
        return max(0.0, min(1.0, adjusted_novelty))
