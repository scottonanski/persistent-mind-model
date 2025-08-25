"""
Adaptive Emergence System - Natural, self-calibrating emergence detection.

Replaces hardcoded thresholds with dynamic systems that learn from the agent's
actual behavioral evolution patterns.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict


@dataclass
class BehavioralBaseline:
    """Dynamic baseline that evolves with agent behavior."""

    metric_name: str
    values: List[float]
    timestamps: List[str]
    mean: float = 0.0
    std: float = 1.0
    trend: float = 0.0

    def update(self, value: float, timestamp: str = None):
        """Add new value and recalculate statistics."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        self.values.append(value)
        self.timestamps.append(timestamp)

        # Keep only recent values (sliding window)
        if len(self.values) > 100:
            self.values = self.values[-100:]
            self.timestamps = self.timestamps[-100:]

        # Recalculate statistics
        if len(self.values) > 1:
            self.mean = np.mean(self.values)
            self.std = max(np.std(self.values), 0.01)  # Prevent division by zero

            # Calculate trend (slope of recent values)
            if len(self.values) >= 5:
                recent_values = self.values[-10:]
                x = np.arange(len(recent_values))
                self.trend = np.polyfit(x, recent_values, 1)[0]

    def z_score(self, value: float) -> float:
        """Calculate z-score relative to this baseline."""
        return (value - self.mean) / self.std

    def percentile_rank(self, value: float) -> float:
        """Calculate percentile rank (0-1) of value in historical distribution."""
        if len(self.values) < 2:
            return 0.5
        return np.mean([v <= value for v in self.values])


class AdaptiveEmergenceDetector:
    """Natural emergence detection that learns from agent behavior."""

    def __init__(self, storage_manager=None):
        self.storage = storage_manager
        self.baselines: Dict[str, BehavioralBaseline] = {}
        self.stage_history: List[Tuple[str, str, float]] = (
            []
        )  # (stage, timestamp, confidence)

    def _get_baseline(self, metric_name: str) -> BehavioralBaseline:
        """Get or create baseline for a metric."""
        if metric_name not in self.baselines:
            self.baselines[metric_name] = BehavioralBaseline(
                metric_name=metric_name, values=[], timestamps=[]
            )
        return self.baselines[metric_name]

    def update_baselines(self, metrics: Dict[str, float]):
        """Update all baselines with new metric values."""
        timestamp = datetime.now().isoformat()
        for metric_name, value in metrics.items():
            baseline = self._get_baseline(metric_name)
            baseline.update(value, timestamp)

    def calculate_adaptive_gas(
        self,
        content_complexity: float,
        behavioral_change: float,
        commitment_progress: float,
        semantic_novelty: float,
    ) -> float:
        """Calculate GAS using adaptive weighting based on agent's patterns."""

        # Update baselines
        metrics = {
            "content_complexity": content_complexity,
            "behavioral_change": behavioral_change,
            "commitment_progress": commitment_progress,
            "semantic_novelty": semantic_novelty,
        }
        self.update_baselines(metrics)

        # Calculate adaptive weights based on which metrics show most growth
        weights = self._calculate_adaptive_weights(metrics)

        # Weighted combination with gradient scoring instead of binary
        gas_score = (
            weights["complexity"]
            * self._gradient_score(content_complexity, "content_complexity")
            + weights["change"]
            * self._gradient_score(behavioral_change, "behavioral_change")
            + weights["progress"]
            * self._gradient_score(commitment_progress, "commitment_progress")
            + weights["novelty"]
            * self._gradient_score(semantic_novelty, "semantic_novelty")
        )

        return max(0.0, min(1.0, gas_score))

    def _gradient_score(self, value: float, metric_name: str) -> float:
        """Convert raw value to gradient score (0-1) based on historical distribution."""
        baseline = self._get_baseline(metric_name)

        if len(baseline.values) < 3:
            # Not enough history - use raw value
            return max(0.0, min(1.0, value))

        # Use percentile rank as gradient score
        percentile = baseline.percentile_rank(value)

        # Boost score if showing positive trend
        if baseline.trend > 0:
            trend_boost = min(0.2, baseline.trend * 2)
            percentile = min(1.0, percentile + trend_boost)

        return percentile

    def _calculate_adaptive_weights(
        self, current_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate adaptive weights based on which metrics show most growth potential."""

        base_weights = {
            "complexity": 0.25,
            "change": 0.25,
            "progress": 0.25,
            "novelty": 0.25,
        }

        # Adjust weights based on which metrics are showing growth trends
        for metric_name, base_weight in base_weights.items():
            baseline = self._get_baseline(metric_name)

            # Increase weight for metrics showing positive trends
            if baseline.trend > 0:
                base_weights[metric_name] *= 1.0 + baseline.trend

            # Increase weight for metrics with high variance (more dynamic)
            if baseline.std > 0.1:
                base_weights[metric_name] *= 1.2

        # Normalize weights to sum to 1.0
        total_weight = sum(base_weights.values())
        return {k: v / total_weight for k, v in base_weights.items()}

    def detect_stage_transition(
        self, ias_score: float, gas_score: float, content_metrics: Dict[str, float]
    ) -> Tuple[str, float]:
        """Detect emergence stage using adaptive thresholds."""

        # Update baselines for stage detection
        stage_metrics = {"ias": ias_score, "gas": gas_score, **content_metrics}
        self.update_baselines(stage_metrics)

        # Calculate stage probabilities instead of hard thresholds
        stage_probs = self._calculate_stage_probabilities(
            ias_score, gas_score, content_metrics
        )

        # Select stage with highest probability
        best_stage = max(stage_probs.items(), key=lambda x: x[1])
        stage_name, confidence = best_stage

        # Add to history
        self.stage_history.append((stage_name, datetime.now().isoformat(), confidence))

        # Apply temporal smoothing to prevent rapid stage switching
        smoothed_stage = self._apply_temporal_smoothing(stage_name, confidence)

        return smoothed_stage, confidence

    def _calculate_stage_probabilities(
        self, ias: float, gas: float, content_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate probability distribution over stages."""

        # Get adaptive thresholds based on agent's history
        ias_baseline = self._get_baseline("ias")
        gas_baseline = self._get_baseline("gas")

        # Stage probabilities based on percentile ranks and trends
        ias_percentile = ias_baseline.percentile_rank(ias)
        gas_percentile = gas_baseline.percentile_rank(gas)

        probs = {
            "S0: Substrate": self._substrate_probability(
                ias_percentile, gas_percentile, content_metrics
            ),
            "S1: Resistance": self._resistance_probability(
                ias_percentile, gas_percentile, content_metrics
            ),
            "S2: Adoption": self._adoption_probability(
                ias_percentile, gas_percentile, content_metrics
            ),
            "S3: Self-Model": self._self_model_probability(
                ias_percentile, gas_percentile, content_metrics
            ),
            "S4: Growth-Seeking": self._growth_seeking_probability(
                ias_percentile, gas_percentile, content_metrics
            ),
        }

        # Normalize probabilities
        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: v / total_prob for k, v in probs.items()}

        return probs

    def _substrate_probability(
        self, ias_pct: float, gas_pct: float, content: Dict[str, float]
    ) -> float:
        """Calculate probability of S0: Substrate stage."""
        # High probability when both IAS and GAS are low
        return max(0.0, 1.0 - ias_pct - gas_pct) * 0.5

    def _resistance_probability(
        self, ias_pct: float, gas_pct: float, content: Dict[str, float]
    ) -> float:
        """Calculate probability of S1: Resistance stage."""
        # High when IAS is low but some activity present
        if ias_pct < 0.3 and gas_pct > 0.1:
            return 0.8
        return max(0.0, 0.5 - ias_pct)

    def _adoption_probability(
        self, ias_pct: float, gas_pct: float, content: Dict[str, float]
    ) -> float:
        """Calculate probability of S2: Adoption stage."""
        # High when IAS is moderate, GAS is developing
        if 0.3 <= ias_pct <= 0.7 and 0.1 <= gas_pct <= 0.6:
            return 0.9
        return max(0.0, ias_pct - abs(ias_pct - 0.5))

    def _self_model_probability(
        self, ias_pct: float, gas_pct: float, content: Dict[str, float]
    ) -> float:
        """Calculate probability of S3: Self-Model stage."""
        # High when both IAS and GAS are elevated
        if ias_pct > 0.6 and gas_pct > 0.4:
            return min(1.0, ias_pct + gas_pct - 0.5)
        return 0.0

    def _growth_seeking_probability(
        self, ias_pct: float, gas_pct: float, content: Dict[str, float]
    ) -> float:
        """Calculate probability of S4: Growth-Seeking stage."""
        # High when GAS is very high and showing consistent growth
        gas_baseline = self._get_baseline("gas")
        if gas_pct > 0.8 and gas_baseline.trend > 0.05:
            return gas_pct
        return 0.0

    def _apply_temporal_smoothing(self, candidate_stage: str, confidence: float) -> str:
        """Apply temporal smoothing to prevent rapid stage transitions."""
        if len(self.stage_history) < 3:
            return candidate_stage

        # Look at recent stage history
        recent_stages = [s[0] for s in self.stage_history[-5:]]

        # If candidate stage appears frequently recently, accept it
        candidate_count = recent_stages.count(candidate_stage)
        if candidate_count >= 2 or confidence > 0.8:
            return candidate_stage

        # Otherwise, stick with most common recent stage
        stage_counts = defaultdict(int)
        for stage in recent_stages:
            stage_counts[stage] += 1

        return max(stage_counts.items(), key=lambda x: x[1])[0]

    def get_emergence_insights(self) -> Dict[str, Any]:
        """Get insights about the agent's emergence patterns."""
        insights = {
            "baselines": {},
            "trends": {},
            "stage_progression": [],
            "growth_indicators": [],
        }

        # Baseline summaries
        for name, baseline in self.baselines.items():
            if len(baseline.values) > 0:
                insights["baselines"][name] = {
                    "mean": round(baseline.mean, 3),
                    "std": round(baseline.std, 3),
                    "trend": round(baseline.trend, 3),
                    "recent_value": baseline.values[-1] if baseline.values else 0.0,
                    "percentile_recent": (
                        round(baseline.percentile_rank(baseline.values[-1]), 3)
                        if baseline.values
                        else 0.5
                    ),
                }

        # Stage progression
        if self.stage_history:
            insights["stage_progression"] = self.stage_history[-10:]  # Last 10 stages

        # Growth indicators
        for name, baseline in self.baselines.items():
            if baseline.trend > 0.02:  # Significant positive trend
                insights["growth_indicators"].append(
                    {
                        "metric": name,
                        "trend": round(baseline.trend, 3),
                        "recent_improvement": len(baseline.values) > 5
                        and baseline.values[-1] > baseline.mean,
                    }
                )

        return insights
