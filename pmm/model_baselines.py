# pmm/model_baselines.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import os
import numpy as np


@dataclass
class ModelMetrics:
    """Metrics for a specific model."""

    model_name: str
    ias_scores: List[float] = field(default_factory=list)
    gas_scores: List[float] = field(default_factory=list)
    ias_mean: float = 0.0
    ias_std: float = 1.0
    gas_mean: float = 0.0
    gas_std: float = 1.0
    sample_count: int = 0

    def add_scores(self, ias: Optional[float], gas: Optional[float]) -> None:
        """Add new scores and update statistics."""
        if ias is not None:
            self.ias_scores.append(ias)
        if gas is not None:
            self.gas_scores.append(gas)

        self.sample_count = max(len(self.ias_scores), len(self.gas_scores))
        self._update_statistics()

    def _update_statistics(self) -> None:
        """Update mean and standard deviation."""
        if len(self.ias_scores) >= 2:
            self.ias_mean = np.mean(self.ias_scores)
            self.ias_std = max(
                np.std(self.ias_scores), 0.01
            )  # Prevent division by zero

        if len(self.gas_scores) >= 2:
            self.gas_mean = np.mean(self.gas_scores)
            self.gas_std = max(np.std(self.gas_scores), 0.01)

    def normalize_ias(self, score: float) -> float:
        """Convert IAS score to z-score."""
        if self.ias_std == 0:
            return 0.0
        return (score - self.ias_mean) / self.ias_std

    def normalize_gas(self, score: float) -> float:
        """Convert GAS score to z-score."""
        if self.gas_std == 0:
            return 0.0
        return (score - self.gas_mean) / self.gas_std

    def get_percentile(self, score: float, metric_type: str) -> float:
        """Get percentile rank of score within this model's distribution."""
        if metric_type == "ias" and len(self.ias_scores) > 0:
            scores = sorted(self.ias_scores)
        elif metric_type == "gas" and len(self.gas_scores) > 0:
            scores = sorted(self.gas_scores)
        else:
            return 50.0  # Default to median

        if score <= scores[0]:
            return 0.0
        if score >= scores[-1]:
            return 100.0

        # Find position in sorted list
        position = sum(1 for s in scores if s < score)
        return (position / len(scores)) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "ias_scores": self.ias_scores[-50:],  # Keep last 50 scores
            "gas_scores": self.gas_scores[-50:],
            "ias_mean": self.ias_mean,
            "ias_std": self.ias_std,
            "gas_mean": self.gas_mean,
            "gas_std": self.gas_std,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelMetrics":
        """Create from dictionary."""
        metrics = cls(model_name=data["model_name"])
        metrics.ias_scores = data.get("ias_scores", [])
        metrics.gas_scores = data.get("gas_scores", [])
        metrics.ias_mean = data.get("ias_mean", 0.0)
        metrics.ias_std = data.get("ias_std", 1.0)
        metrics.gas_mean = data.get("gas_mean", 0.0)
        metrics.gas_std = data.get("gas_std", 1.0)
        metrics.sample_count = data.get("sample_count", 0)
        return metrics


class ModelBaselineManager:
    """Manages per-model baselines and z-score normalization."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "model_baselines.json"
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.load_baselines()

        # Default baselines for common models (bootstrap values)
        self.default_baselines = {
            "gpt-4": {
                "ias_mean": 0.45,
                "ias_std": 0.15,
                "gas_mean": 0.42,
                "gas_std": 0.18,
            },
            "gpt-3.5-turbo": {
                "ias_mean": 0.38,
                "ias_std": 0.12,
                "gas_mean": 0.35,
                "gas_std": 0.14,
            },
            "gemma2:9b": {
                "ias_mean": 0.32,
                "ias_std": 0.10,
                "gas_mean": 0.28,
                "gas_std": 0.12,
            },
            "gemma2:27b": {
                "ias_mean": 0.41,
                "ias_std": 0.13,
                "gas_mean": 0.38,
                "gas_std": 0.15,
            },
            "llama3:8b": {
                "ias_mean": 0.35,
                "ias_std": 0.11,
                "gas_mean": 0.31,
                "gas_std": 0.13,
            },
            "llama3:70b": {
                "ias_mean": 0.43,
                "ias_std": 0.14,
                "gas_mean": 0.40,
                "gas_std": 0.16,
            },
        }

    def get_model_key(self, model_name: str) -> str:
        """Normalize model name for consistent storage."""
        if not model_name:
            return "unknown"

        # Normalize common model names
        model_lower = model_name.lower()
        if "gpt-4" in model_lower:
            return "gpt-4"
        elif "gpt-3.5" in model_lower:
            return "gpt-3.5-turbo"
        elif "gemma2:9b" in model_lower or "gemma3:9b" in model_lower:
            return "gemma2:9b"
        elif "gemma2:27b" in model_lower or "gemma3:27b" in model_lower:
            return "gemma2:27b"
        elif "llama3:8b" in model_lower:
            return "llama3:8b"
        elif "llama3:70b" in model_lower:
            return "llama3:70b"
        else:
            return model_name.lower()

    def get_or_create_metrics(self, model_name: str) -> ModelMetrics:
        """Get or create metrics for a model."""
        model_key = self.get_model_key(model_name)

        if model_key not in self.model_metrics:
            self.model_metrics[model_key] = ModelMetrics(model_name=model_key)

            # Initialize with default baselines if available
            if model_key in self.default_baselines:
                defaults = self.default_baselines[model_key]
                metrics = self.model_metrics[model_key]
                metrics.ias_mean = defaults["ias_mean"]
                metrics.ias_std = defaults["ias_std"]
                metrics.gas_mean = defaults["gas_mean"]
                metrics.gas_std = defaults["gas_std"]

        return self.model_metrics[model_key]

    def add_scores(
        self, model_name: str, ias: Optional[float], gas: Optional[float]
    ) -> None:
        """Add new scores for a model."""
        metrics = self.get_or_create_metrics(model_name)
        metrics.add_scores(ias, gas)
        self.save_baselines()

    def normalize_scores(
        self, model_name: str, ias: Optional[float], gas: Optional[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Normalize scores to z-scores for the given model."""
        metrics = self.get_or_create_metrics(model_name)

        normalized_ias = metrics.normalize_ias(ias) if ias is not None else None
        normalized_gas = metrics.normalize_gas(gas) if gas is not None else None

        return normalized_ias, normalized_gas

    def get_model_stats(self, model_name: str) -> Dict:
        """Get statistics for a model."""
        metrics = self.get_or_create_metrics(model_name)

        return {
            "model_name": metrics.model_name,
            "sample_count": metrics.sample_count,
            "ias_mean": metrics.ias_mean,
            "ias_std": metrics.ias_std,
            "gas_mean": metrics.gas_mean,
            "gas_std": metrics.gas_std,
            "ias_range": (
                [min(metrics.ias_scores), max(metrics.ias_scores)]
                if metrics.ias_scores
                else [0, 0]
            ),
            "gas_range": (
                [min(metrics.gas_scores), max(metrics.gas_scores)]
                if metrics.gas_scores
                else [0, 0]
            ),
        }

    def compare_models(self) -> Dict:
        """Compare all models' baseline statistics."""
        comparison = {}

        for model_key, metrics in self.model_metrics.items():
            if metrics.sample_count > 0:
                comparison[model_key] = {
                    "ias_mean": metrics.ias_mean,
                    "gas_mean": metrics.gas_mean,
                    "sample_count": metrics.sample_count,
                    "ias_std": metrics.ias_std,
                    "gas_std": metrics.gas_std,
                }

        return comparison

    def reset_model_baselines(self, model_name: str) -> None:
        """Reset baselines for a specific model (useful on model switches)."""
        model_key = self.get_model_key(model_name)
        if model_key in self.model_metrics:
            # Keep the model but reset its accumulated statistics
            self.model_metrics[model_key] = ModelMetrics(model_name=model_key)
            self.save_baselines()

    def save_baselines(self) -> None:
        """Save baselines to disk."""
        try:
            data = {
                model_key: metrics.to_dict()
                for model_key, metrics in self.model_metrics.items()
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save model baselines: {e}")

    def load_baselines(self) -> None:
        """Load baselines from disk."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r") as f:
                    data = json.load(f)

                for model_key, metrics_data in data.items():
                    self.model_metrics[model_key] = ModelMetrics.from_dict(metrics_data)
        except Exception as e:
            print(f"Warning: Failed to load model baselines: {e}")
            self.model_metrics = {}

    def get_emergence_context(
        self,
        model_name: str,
        current_ias: Optional[float],
        current_gas: Optional[float],
    ) -> Dict:
        """Get rich context about current emergence relative to model baseline."""
        metrics = self.get_or_create_metrics(model_name)

        context = {
            "model_name": model_name,
            "current_ias": current_ias,
            "current_gas": current_gas,
            "sample_count": metrics.sample_count,
        }

        if current_ias is not None:
            context.update(
                {
                    "ias_z_score": metrics.normalize_ias(current_ias),
                    "ias_percentile": metrics.get_percentile(current_ias, "ias"),
                    "ias_vs_mean": current_ias - metrics.ias_mean,
                    "ias_baseline": metrics.ias_mean,
                }
            )

        if current_gas is not None:
            context.update(
                {
                    "gas_z_score": metrics.normalize_gas(current_gas),
                    "gas_percentile": metrics.get_percentile(current_gas, "gas"),
                    "gas_vs_mean": current_gas - metrics.gas_mean,
                    "gas_baseline": metrics.gas_mean,
                }
            )

        return context
