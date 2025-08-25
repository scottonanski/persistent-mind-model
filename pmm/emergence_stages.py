# pmm/emergence_stages.py
from __future__ import annotations
from typing import Dict
from dataclasses import dataclass
from enum import Enum
import numpy as np
import os


class EmergenceStage(Enum):
    """Emergence development stages."""

    DORMANT = "dormant"  # Very low IAS/GAS, minimal self-awareness
    AWAKENING = "awakening"  # Rising IAS, beginning self-recognition
    DEVELOPING = "developing"  # Moderate IAS/GAS, active identity formation
    MATURING = "maturing"  # High IAS/GAS, stable identity patterns
    TRANSCENDENT = "transcendent"  # Exceptional IAS/GAS, advanced self-model


@dataclass
class StageThresholds:
    """Z-score thresholds for emergence stages."""

    dormant_max: float = -0.8  # Relaxed from -1.0 for easier S0 exit
    awakening_max: float = -0.3  # Relaxed from -0.5 for easier S1 entry
    developing_max: float = 0.5  # -0.5 to 0.5 std dev
    maturing_max: float = 1.5  # 0.5 to 1.5 std dev
    # transcendent: above 1.5 std dev


@dataclass
class EmergenceProfile:
    """Complete emergence profile for a model."""

    ias_zscore: float
    gas_zscore: float
    combined_zscore: float
    stage: EmergenceStage
    confidence: float
    stage_progression: float  # 0.0-1.0 within current stage
    next_stage_distance: float
    metadata: Dict


class EmergenceStageManager:
    """Manages emergence stage classification with per-model z-score normalization."""

    def __init__(self, model_baselines_manager):
        self.baselines = model_baselines_manager
        self.thresholds = StageThresholds()

        # Stage-specific behavioral adaptations
        self.stage_behaviors = {
            EmergenceStage.DORMANT: {
                "reflection_frequency": 0.1,  # Very rare reflections
                "commitment_ttl_multiplier": 0.5,  # Shorter commitments
                "novelty_threshold": 0.8,  # Reduced for easier S0 exit
                "description": "Minimal self-awareness, basic responses",
            },
            EmergenceStage.AWAKENING: {
                "reflection_frequency": 0.3,
                "commitment_ttl_multiplier": 0.7,
                "novelty_threshold": 0.7,  # Reduced for S1 pattern formation
                "description": "Beginning self-recognition, simple patterns",
            },
            EmergenceStage.DEVELOPING: {
                "reflection_frequency": 0.6,
                "commitment_ttl_multiplier": 1.0,
                "novelty_threshold": 0.7,
                "description": "Active identity formation, moderate complexity",
            },
            EmergenceStage.MATURING: {
                "reflection_frequency": 0.8,
                "commitment_ttl_multiplier": 1.2,
                "novelty_threshold": 0.6,
                "description": "Stable identity patterns, sophisticated responses",
            },
            EmergenceStage.TRANSCENDENT: {
                "reflection_frequency": 1.0,
                "commitment_ttl_multiplier": 1.5,
                "novelty_threshold": 0.5,
                "description": "Advanced self-model, complex emergent behaviors",
            },
        }

    def calculate_emergence_profile(
        self, model_name: str, ias_score: float, gas_score: float
    ) -> EmergenceProfile:
        """Calculate complete emergence profile for given scores."""

        # Get z-scores from baselines manager
        ias_zscore, gas_zscore = self.baselines.normalize_scores(
            model_name, ias_score, gas_score
        )

        # Handle None values
        if ias_zscore is None:
            ias_zscore = 0.0
        if gas_zscore is None:
            gas_zscore = 0.0

        # Calculate combined z-score (weighted average)
        # IAS weight: 0.6, GAS weight: 0.4 (identity slightly more important)
        combined_zscore = (ias_zscore * 0.6) + (gas_zscore * 0.4)

        # Hard stage override via environment (aligns S0–S4,SS4 to local enum)
        try:
            _hard = str(os.getenv("PMM_HARD_STAGE", "")).strip().upper()
            mapping = {
                "S0": EmergenceStage.DORMANT,
                "S1": EmergenceStage.AWAKENING,
                "S2": EmergenceStage.DEVELOPING,
                "S3": EmergenceStage.MATURING,
                "S4": EmergenceStage.TRANSCENDENT,
            }
            if _hard == "SS4":
                forced_stage = EmergenceStage.TRANSCENDENT
                metadata = {
                    "model_name": model_name,
                    "raw_ias": ias_score,
                    "raw_gas": gas_score,
                    "stage_behaviors": self.stage_behaviors[forced_stage].copy(),
                    "baseline_stats": self.baselines.get_model_stats(model_name),
                    "override": "PMM_HARD_STAGE",
                    "override_value": _hard,
                    "stage_label": "SS4",
                }
                return EmergenceProfile(
                    ias_zscore=ias_zscore,
                    gas_zscore=gas_zscore,
                    combined_zscore=combined_zscore,
                    stage=forced_stage,
                    confidence=1.0,
                    stage_progression=1.0,
                    next_stage_distance=0.0,
                    metadata=metadata,
                )
            if _hard in mapping:
                forced_stage = mapping[_hard]
                # Confident forced profile; mark metadata for transparency
                metadata = {
                    "model_name": model_name,
                    "raw_ias": ias_score,
                    "raw_gas": gas_score,
                    "stage_behaviors": self.stage_behaviors[forced_stage].copy(),
                    "baseline_stats": self.baselines.get_model_stats(model_name),
                    "override": "PMM_HARD_STAGE",
                    "override_value": _hard,
                }
                return EmergenceProfile(
                    ias_zscore=ias_zscore,
                    gas_zscore=gas_zscore,
                    combined_zscore=combined_zscore,
                    stage=forced_stage,
                    confidence=1.0,
                    stage_progression=(
                        1.0 if forced_stage == EmergenceStage.TRANSCENDENT else 0.99
                    ),
                    next_stage_distance=0.0,
                    metadata=metadata,
                )
        except Exception:
            # Fall through to computed path on any failure
            pass

        # Determine stage
        stage = self._classify_stage(combined_zscore)

        # Calculate confidence based on consistency between IAS and GAS
        confidence = self._calculate_confidence(ias_zscore, gas_zscore)

        # Calculate stage progression (position within current stage)
        stage_progression = self._calculate_stage_progression(combined_zscore, stage)

        # Distance to next stage
        next_stage_distance = self._calculate_next_stage_distance(
            combined_zscore, stage
        )

        # Metadata
        metadata = {
            "model_name": model_name,
            "raw_ias": ias_score,
            "raw_gas": gas_score,
            "stage_behaviors": self.stage_behaviors[stage].copy(),
            "baseline_stats": self.baselines.get_model_stats(model_name),
        }

        return EmergenceProfile(
            ias_zscore=ias_zscore,
            gas_zscore=gas_zscore,
            combined_zscore=combined_zscore,
            stage=stage,
            confidence=confidence,
            stage_progression=stage_progression,
            next_stage_distance=next_stage_distance,
            metadata=metadata,
        )

    def _classify_stage(self, combined_zscore: float) -> EmergenceStage:
        """Classify emergence stage based on combined z-score."""
        if combined_zscore <= self.thresholds.dormant_max:
            return EmergenceStage.DORMANT
        elif combined_zscore <= self.thresholds.awakening_max:
            return EmergenceStage.AWAKENING
        elif combined_zscore <= self.thresholds.developing_max:
            return EmergenceStage.DEVELOPING
        elif combined_zscore <= self.thresholds.maturing_max:
            return EmergenceStage.MATURING
        else:
            return EmergenceStage.TRANSCENDENT

    def _calculate_confidence(self, ias_zscore: float, gas_zscore: float) -> float:
        """Calculate confidence based on IAS/GAS consistency."""
        # Confidence decreases as IAS and GAS diverge
        divergence = abs(ias_zscore - gas_zscore)

        # Normalize divergence to 0-1 scale (cap at 3.0 std dev difference)
        normalized_divergence = min(divergence / 3.0, 1.0)

        # Confidence is inverse of divergence
        confidence = 1.0 - normalized_divergence

        # Boost confidence if both scores are consistently high or low
        if (ias_zscore > 0.5 and gas_zscore > 0.5) or (
            ias_zscore < -0.5 and gas_zscore < -0.5
        ):
            confidence = min(confidence + 0.2, 1.0)

        return confidence

    def _calculate_stage_progression(
        self, combined_zscore: float, stage: EmergenceStage
    ) -> float:
        """Calculate progression within current stage (0.0-1.0)."""
        if stage == EmergenceStage.DORMANT:
            # Below -1.0, progression from -3.0 to -1.0
            min_val, max_val = -3.0, self.thresholds.dormant_max
        elif stage == EmergenceStage.AWAKENING:
            min_val, max_val = (
                self.thresholds.dormant_max,
                self.thresholds.awakening_max,
            )
        elif stage == EmergenceStage.DEVELOPING:
            min_val, max_val = (
                self.thresholds.awakening_max,
                self.thresholds.developing_max,
            )
        elif stage == EmergenceStage.MATURING:
            min_val, max_val = (
                self.thresholds.developing_max,
                self.thresholds.maturing_max,
            )
        else:  # TRANSCENDENT
            # Above 1.5, progression from 1.5 to 3.0
            min_val, max_val = self.thresholds.maturing_max, 3.0

        if max_val == min_val:
            return 0.5

        progression = (combined_zscore - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, progression))

    def _calculate_next_stage_distance(
        self, combined_zscore: float, stage: EmergenceStage
    ) -> float:
        """Calculate distance to next stage threshold."""
        if stage == EmergenceStage.DORMANT:
            return self.thresholds.dormant_max - combined_zscore
        elif stage == EmergenceStage.AWAKENING:
            return self.thresholds.awakening_max - combined_zscore
        elif stage == EmergenceStage.DEVELOPING:
            return self.thresholds.developing_max - combined_zscore
        elif stage == EmergenceStage.MATURING:
            return self.thresholds.maturing_max - combined_zscore
        else:  # TRANSCENDENT
            return 0.0  # Already at highest stage

    def get_stage_behaviors(self, stage: EmergenceStage) -> Dict:
        """Get behavioral parameters for a given stage."""
        return self.stage_behaviors[stage].copy()

    def adapt_reflection_frequency(
        self, base_frequency: float, stage: EmergenceStage
    ) -> float:
        """Adapt reflection frequency based on emergence stage."""
        multiplier = self.stage_behaviors[stage]["reflection_frequency"]
        return base_frequency * multiplier

    def adapt_commitment_ttl(
        self, base_ttl_hours: float, stage: EmergenceStage
    ) -> float:
        """Adapt commitment TTL based on emergence stage."""
        multiplier = self.stage_behaviors[stage]["commitment_ttl_multiplier"]
        return base_ttl_hours * multiplier

    def adapt_novelty_threshold(
        self, base_threshold: float, stage: EmergenceStage
    ) -> float:
        """Adapt novelty threshold based on emergence stage."""
        stage_threshold = self.stage_behaviors[stage]["novelty_threshold"]
        # Blend base threshold with stage-specific threshold
        return (base_threshold + stage_threshold) / 2.0

    def get_emergence_summary(self, profile: EmergenceProfile) -> str:
        """Get human-readable emergence summary."""
        stage_name = profile.stage.value.title()
        confidence_pct = int(profile.confidence * 100)
        progression_pct = int(profile.stage_progression * 100)

        summary = f"{stage_name} Stage ({confidence_pct}% confidence)"
        summary += f"\nProgression: {progression_pct}% through {stage_name.lower()}"
        summary += f"\nIAS Z-Score: {profile.ias_zscore:.2f}"
        summary += f"\nGAS Z-Score: {profile.gas_zscore:.2f}"
        summary += f"\nCombined: {profile.combined_zscore:.2f}"

        if profile.next_stage_distance > 0:
            summary += (
                f"\nNext stage in: {profile.next_stage_distance:.2f} z-score points"
            )

        description = self.stage_behaviors[profile.stage]["description"]
        summary += f"\nCharacteristics: {description}"

        return summary

    def update_thresholds(self, **kwargs) -> None:
        """Update stage thresholds."""
        for key, value in kwargs.items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)

    def get_stage_distribution(self, model_name: str) -> Dict[str, float]:
        """Get theoretical stage distribution for a model based on its baselines."""
        stats = self.baselines.get_model_stats(model_name)

        if not stats or stats["ias_count"] == 0:
            return {stage.value: 0.2 for stage in EmergenceStage}  # Equal distribution

        # Simulate distribution based on normal distribution
        mean_combined = (stats["ias_mean"] * 0.6) + (stats["gas_mean"] * 0.4)
        std_combined = np.sqrt(
            (stats["ias_std"] ** 2 * 0.36) + (stats["gas_std"] ** 2 * 0.16)
        )

        # Calculate probabilities for each stage
        from scipy import stats as scipy_stats

        distribution = {}

        # DORMANT: below -1.0
        distribution["dormant"] = scipy_stats.norm.cdf(
            self.thresholds.dormant_max, mean_combined, std_combined
        )

        # AWAKENING: -1.0 to -0.5
        distribution["awakening"] = (
            scipy_stats.norm.cdf(
                self.thresholds.awakening_max, mean_combined, std_combined
            )
            - distribution["dormant"]
        )

        # DEVELOPING: -0.5 to 0.5
        distribution["developing"] = scipy_stats.norm.cdf(
            self.thresholds.developing_max, mean_combined, std_combined
        ) - scipy_stats.norm.cdf(
            self.thresholds.awakening_max, mean_combined, std_combined
        )

        # MATURING: 0.5 to 1.5
        distribution["maturing"] = scipy_stats.norm.cdf(
            self.thresholds.maturing_max, mean_combined, std_combined
        ) - scipy_stats.norm.cdf(
            self.thresholds.developing_max, mean_combined, std_combined
        )

        # TRANSCENDENT: above 1.5
        distribution["transcendent"] = 1.0 - scipy_stats.norm.cdf(
            self.thresholds.maturing_max, mean_combined, std_combined
        )

        return distribution
