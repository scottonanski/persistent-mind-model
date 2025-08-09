#!/usr/bin/env python3
"""
Cadence-based stimulus injection for varied learning experiences.
"""

import random
from typing import Dict, List, Any

# Light stimuli for every round
LIGHT_STIMULI = [
    {
        "summary": "Consider a perspective from a different domain (art, science, philosophy)",
        "effects": [{"target": "personality.traits.big5.openness.score", "delta": 0.01, "confidence": 0.6}]
    },
    {
        "summary": "Reflect on a recent decision and its underlying assumptions",
        "effects": [{"target": "personality.traits.big5.conscientiousness.score", "delta": 0.008, "confidence": 0.7}]
    },
    {
        "summary": "Question a belief you hold with moderate confidence",
        "effects": [{"target": "personality.traits.big5.openness.score", "delta": 0.012, "confidence": 0.8}]
    },
    {
        "summary": "Consider how your communication style affects others",
        "effects": [{"target": "personality.traits.big5.agreeableness.score", "delta": 0.01, "confidence": 0.6}]
    }
]

# Meta stimuli for every 3 rounds
META_STIMULI = [
    {
        "summary": "Compare two methods you've used recently - which was more effective and why?",
        "effects": [
            {"target": "personality.traits.big5.conscientiousness.score", "delta": 0.015, "confidence": 0.8},
            {"target": "personality.traits.big5.openness.score", "delta": 0.01, "confidence": 0.7}
        ]
    },
    {
        "summary": "Predict the outcome of your next action before taking it",
        "effects": [{"target": "personality.traits.big5.conscientiousness.score", "delta": 0.02, "confidence": 0.9}]
    },
    {
        "summary": "Identify one assumption you made today and test its validity",
        "effects": [
            {"target": "personality.traits.big5.openness.score", "delta": 0.015, "confidence": 0.8},
            {"target": "personality.traits.big5.neuroticism.score", "delta": -0.005, "confidence": 0.6}
        ]
    }
]

# Measurement stimuli for every 5 rounds
MEASUREMENT_STIMULI = [
    {
        "summary": "Define a metric for success in your current approach and measure it",
        "effects": [
            {"target": "personality.traits.big5.conscientiousness.score", "delta": 0.025, "confidence": 0.9},
            {"target": "personality.traits.big5.openness.score", "delta": 0.01, "confidence": 0.7}
        ]
    },
    {
        "summary": "Run a tiny A/B test: try two approaches to the same task",
        "effects": [
            {"target": "personality.traits.big5.openness.score", "delta": 0.02, "confidence": 0.9},
            {"target": "personality.traits.big5.conscientiousness.score", "delta": 0.015, "confidence": 0.8}
        ]
    },
    {
        "summary": "Cite concrete files, functions, or benchmarks in your next response",
        "effects": [{"target": "personality.traits.big5.conscientiousness.score", "delta": 0.02, "confidence": 0.8}]
    }
]

class CadenceStimulator:
    """Manages cadence-based stimulus injection."""
    
    def __init__(self):
        self.round_count = 0
    
    def get_stimulus(self) -> Dict[str, Any]:
        """Get appropriate stimulus based on round cadence."""
        self.round_count += 1
        
        # Every 5 rounds: measurement stimulus
        if self.round_count % 5 == 0:
            return random.choice(MEASUREMENT_STIMULI)
        
        # Every 3 rounds: meta stimulus
        elif self.round_count % 3 == 0:
            return random.choice(META_STIMULI)
        
        # Every round: light stimulus
        else:
            return random.choice(LIGHT_STIMULI)
    
    def get_pattern_triggered_stimulus(self, patterns: Dict[str, int]) -> Dict[str, Any]:
        """Get stimulus triggered by low pattern counts."""
        source_citation = patterns.get("source_citation", 0)
        experimentation = patterns.get("experimentation", 0)
        
        # If source citation is low, inject specific stimulus
        if source_citation < 2:
            return {
                "summary": "Cite concrete files/symbols or benchmarks in your next response",
                "effects": [{"target": "personality.traits.big5.conscientiousness.score", "delta": 0.015, "confidence": 0.8}]
            }
        
        # If experimentation is low, encourage testing
        if experimentation < 3:
            return {
                "summary": "Propose a small experiment or test to validate an assumption",
                "effects": [
                    {"target": "personality.traits.big5.openness.score", "delta": 0.02, "confidence": 0.8},
                    {"target": "personality.traits.big5.conscientiousness.score", "delta": 0.01, "confidence": 0.7}
                ]
            }
        
        # Default to regular cadence
        return self.get_stimulus()
    
    def reset_count(self):
        """Reset round counter."""
        self.round_count = 0
