#!/usr/bin/env python3
"""
Adaptive directive classifier that learns from natural conversation patterns.
Uses contextual analysis and semantic evolution to classify directives naturally.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ConversationContext:
    """Context from the conversation flow that helps classification."""

    preceding_user_message: str
    directive_position: str  # "response_start", "response_middle", "response_end"
    conversation_phase: str  # "setup", "exploration", "commitment", "evolution"
    user_intent_signals: List[str]  # What the user was asking for


class AdaptiveDirectiveClassifier:
    """
    Learns directive classification from conversation context and semantic patterns.
    Adapts based on how directives naturally emerge in dialogue.
    """

    def __init__(self):
        self.conversation_patterns = defaultdict(list)
        self.classification_history = []

    def classify_with_context(
        self, directive_text: str, context: ConversationContext
    ) -> Tuple[str, float]:
        """
        Classify directive using conversation context and semantic analysis.
        Returns (classification, confidence)
        """

        # Analyze the user's intent from their message
        user_intent = self._analyze_user_intent(context.preceding_user_message)

        # Analyze the directive's semantic properties
        semantic_properties = self._analyze_semantic_properties(directive_text)

        # Analyze conversation flow position
        flow_signals = self._analyze_conversation_flow(directive_text, context)

        # Combine signals for classification
        classification = self._integrate_classification_signals(
            user_intent, semantic_properties, flow_signals, directive_text
        )

        return classification

    def _analyze_user_intent(self, user_message: str) -> Dict[str, float]:
        """Analyze what the user was requesting that led to this directive."""

        user_lower = user_message.lower()
        intent_signals = {
            "requesting_commitment": 0.0,
            "requesting_principle": 0.0,
            "requesting_evolution": 0.0,
            "providing_framework": 0.0,
        }

        # Commitment request signals
        commitment_indicators = [
            "register a commitment",
            "commit to",
            "make a commitment",
            "promise to",
            "will you",
            "can you commit",
        ]
        for indicator in commitment_indicators:
            if indicator in user_lower:
                intent_signals["requesting_commitment"] += 0.3

        # Principle establishment signals
        principle_indicators = [
            "permanent",
            "directive",
            "principle",
            "rule",
            "guideline",
            "cannot contradict",
            "permanent commitment",
            "standing rule",
        ]
        for indicator in principle_indicators:
            if indicator in user_lower:
                intent_signals["requesting_principle"] += 0.4

        # Evolution/meta signals
        evolution_indicators = [
            "evolve",
            "combine",
            "synthesize",
            "reflect and evolve",
            "make your own",
            "create a principle",
            "of your own making",
        ]
        for indicator in evolution_indicators:
            if indicator in user_lower:
                intent_signals["requesting_evolution"] += 0.5

        # Framework provision signals
        framework_indicators = [
            "your directives are",
            "your protocol is",
            "kernel",
            "operating frame",
            "power-bound protocol",
            "integrity-lock",
        ]
        for indicator in framework_indicators:
            if indicator in user_lower:
                intent_signals["providing_framework"] += 0.6

        return intent_signals

    def _analyze_semantic_properties(self, directive_text: str) -> Dict[str, float]:
        """Analyze semantic properties of the directive itself."""

        text_lower = directive_text.lower()
        properties = {
            "behavioral_specificity": 0.0,
            "identity_depth": 0.0,
            "evolutionary_scope": 0.0,
            "permanence_signals": 0.0,
            "synthesis_signals": 0.0,
        }

        # Behavioral specificity (specific actions)
        behavioral_verbs = [
            "will ask",
            "will provide",
            "will create",
            "will implement",
            "will respond",
            "will analyze",
            "will deliver",
            "will complete",
        ]
        for verb in behavioral_verbs:
            if verb in text_lower:
                properties["behavioral_specificity"] += 0.2

        # Identity depth (who I am vs what I do)
        identity_phrases = [
            "i am",
            "i acknowledge",
            "i embody",
            "i represent",
            "my approach",
            "my framework",
            "my principles",
            "my nature",
        ]
        for phrase in identity_phrases:
            if phrase in text_lower:
                properties["identity_depth"] += 0.3

        # Evolutionary scope (changing how I change)
        evolution_phrases = [
            "evolve",
            "combine",
            "synthesize",
            "integrate",
            "refine",
            "develop my",
            "improve my",
            "adapt my",
            "transform my",
        ]
        for phrase in evolution_phrases:
            if phrase in text_lower:
                properties["evolutionary_scope"] += 0.4

        # Permanence signals
        permanence_phrases = [
            "permanent",
            "always",
            "never",
            "cannot contradict",
            "standing",
            "core",
            "fundamental",
            "essential",
        ]
        for phrase in permanence_phrases:
            if phrase in text_lower:
                properties["permanence_signals"] += 0.3

        # Synthesis signals (combining multiple concepts)
        synthesis_phrases = [
            "guiding principle",
            "framework",
            "approach that",
            "combination of",
            "synthesis of",
            "integration of",
        ]
        for phrase in synthesis_phrases:
            if phrase in text_lower:
                properties["synthesis_signals"] += 0.4

        return properties

    def _analyze_conversation_flow(
        self, directive_text: str, context: ConversationContext
    ) -> Dict[str, float]:
        """Analyze how the directive fits into conversation flow."""

        flow_signals = {
            "response_to_registration": 0.0,
            "spontaneous_commitment": 0.0,
            "synthesis_response": 0.0,
            "acknowledgment_pattern": 0.0,
        }

        # Response to explicit registration request
        if any(
            phrase in context.preceding_user_message.lower()
            for phrase in ["register", "commit to", "permanent"]
        ):
            flow_signals["response_to_registration"] = 0.8

        # Spontaneous commitment (AI offering without being asked)
        if (
            context.directive_position == "response_start"
            and "i will" in directive_text.lower()
        ):
            flow_signals["spontaneous_commitment"] = 0.6

        # Synthesis response (combining/evolving existing directives)
        if any(
            phrase in directive_text.lower()
            for phrase in ["combine", "synthesis", "guiding principle", "framework"]
        ):
            flow_signals["synthesis_response"] = 0.7

        # Acknowledgment pattern (formal acceptance)
        if any(
            phrase in directive_text.lower()
            for phrase in [
                "registered",
                "i acknowledge",
                "i accept",
                "permanent commitment",
            ]
        ):
            flow_signals["acknowledgment_pattern"] = 0.9

        return flow_signals

    def _integrate_classification_signals(
        self,
        user_intent: Dict[str, float],
        semantic_properties: Dict[str, float],
        flow_signals: Dict[str, float],
        directive_text: str,
    ) -> Tuple[str, float]:
        """Integrate all signals to make final classification."""

        # Calculate scores for each classification
        scores = {"meta-principle": 0.0, "principle": 0.0, "commitment": 0.0}

        # Meta-principle scoring
        scores["meta-principle"] += user_intent.get("requesting_evolution", 0) * 0.4
        scores["meta-principle"] += (
            semantic_properties.get("evolutionary_scope", 0) * 0.3
        )
        scores["meta-principle"] += flow_signals.get("synthesis_response", 0) * 0.3

        # Principle scoring
        scores["principle"] += user_intent.get("requesting_principle", 0) * 0.3
        scores["principle"] += user_intent.get("providing_framework", 0) * 0.2
        scores["principle"] += semantic_properties.get("identity_depth", 0) * 0.2
        scores["principle"] += semantic_properties.get("permanence_signals", 0) * 0.2
        scores["principle"] += flow_signals.get("acknowledgment_pattern", 0) * 0.1

        # Commitment scoring
        scores["commitment"] += user_intent.get("requesting_commitment", 0) * 0.3
        scores["commitment"] += (
            semantic_properties.get("behavioral_specificity", 0) * 0.4
        )
        scores["commitment"] += flow_signals.get("spontaneous_commitment", 0) * 0.2
        scores["commitment"] += flow_signals.get("response_to_registration", 0) * 0.1

        # Special case: High synthesis signals override other classifications
        if semantic_properties.get("synthesis_signals", 0) > 0.6:
            scores["principle"] += 0.5

        # Find highest scoring classification
        best_classification = max(scores.items(), key=lambda x: x[1])

        # If all scores are low, default based on linguistic patterns
        if best_classification[1] < 0.3:
            if (
                "i will" in directive_text.lower()
                and "permanent" not in directive_text.lower()
            ):
                return ("commitment", 0.5)
            elif any(
                phrase in directive_text.lower()
                for phrase in ["principle", "permanent", "acknowledge", "registered"]
            ):
                return ("principle", 0.6)
            else:
                return ("commitment", 0.4)

        return best_classification

    def learn_from_conversation(
        self,
        directive_text: str,
        context: ConversationContext,
        actual_classification: str,
    ):
        """Learn from conversation patterns to improve future classifications."""

        pattern = {
            "user_message": context.preceding_user_message,
            "directive": directive_text,
            "classification": actual_classification,
            "context": context,
        }

        self.conversation_patterns[actual_classification].append(pattern)
        self.classification_history.append(pattern)

    def get_classification_explanation(
        self, directive_text: str, context: ConversationContext
    ) -> Dict:
        """Get detailed explanation of classification reasoning."""

        user_intent = self._analyze_user_intent(context.preceding_user_message)
        semantic_properties = self._analyze_semantic_properties(directive_text)
        flow_signals = self._analyze_conversation_flow(directive_text, context)

        return {
            "user_intent_signals": user_intent,
            "semantic_properties": semantic_properties,
            "conversation_flow": flow_signals,
            "classification_reasoning": "Based on contextual analysis of user intent and semantic properties",
        }
