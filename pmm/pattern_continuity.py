# pmm/pattern_continuity.py
"""
Pattern Continuity Enhancement for S0→S1 Transition

This module implements multi-event continuity prompts and pattern reuse weighting
to help PMM transition from substrate to pattern formation stage.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import re


class PatternContinuityManager:
    """Manages pattern continuity and multi-event referencing for S0→S1 transition."""

    def __init__(self, sqlite_store, min_references: int = 3):
        self.sqlite_store = sqlite_store
        self.min_references = min_references

    def enhance_context_with_continuity(
        self, current_context: str, max_refs: int = 5
    ) -> str:
        """
        Enhance context with explicit multi-event continuity references.

        This helps PMM build temporal coherence and reduce novelty scores
        by systematically referencing prior events and commitments.
        """
        try:
            # Get recent events for continuity building
            recent_events = self.sqlite_store.recent_events(limit=20)
            if not recent_events or len(recent_events) < self.min_references:
                return current_context

            # Extract referenceable events (commitments, insights, evidence)
            referenceable = []
            for event in recent_events:
                event_id = event.get("id", "")
                kind = event.get("kind", "")
                content = event.get("content", "")
                summary = event.get("summary", "")

                if kind in ["commitment", "reflection", "evidence"] and (
                    content or summary
                ):
                    referenceable.append(
                        {
                            "id": event_id,
                            "kind": kind,
                            "content": summary or content,
                            "ts": event.get("ts", ""),
                        }
                    )

            if len(referenceable) < self.min_references:
                return current_context

            # Build continuity references
            continuity_refs = []
            for i, ref in enumerate(referenceable[:max_refs]):
                ref_text = self._format_reference(ref)
                if ref_text:
                    continuity_refs.append(f"[Ref {i+1}] {ref_text}")

            if not continuity_refs:
                return current_context

            # Enhance context with continuity prompt
            continuity_prompt = (
                "\n\nCONTINUITY CONTEXT - Reference these prior events in your response:\n"
                + "\n".join(continuity_refs)
                + f"\n\nPlease reference at least {min(self.min_references, len(continuity_refs))} "
                + "of these prior events to build coherent patterns and reduce novelty."
            )

            return current_context + continuity_prompt

        except Exception:
            # Fail gracefully - return original context if continuity enhancement fails
            return current_context

    def _format_reference(self, event: Dict[str, Any]) -> Optional[str]:
        """Format an event for continuity reference."""
        kind = event.get("kind", "")
        content = event.get("content", "")[:150]  # Truncate for brevity
        event_id = event.get("id", "")[:8]  # Short ID

        if kind == "commitment":
            return f"Commitment {event_id}: {content}"
        elif kind == "reflection":
            return f"Insight {event_id}: {content}"
        elif kind == "evidence":
            return f"Evidence {event_id}: {content}"
        else:
            return None

    def calculate_pattern_reuse_score(
        self, current_text: str, historical_events: List[Dict]
    ) -> float:
        """
        Calculate how much current text reuses established patterns.

        Higher scores indicate more pattern reuse, which should lower novelty.
        """
        if not current_text or not historical_events:
            return 0.0

        # Extract key phrases and concepts from current text
        current_phrases = self._extract_key_phrases(current_text)
        if not current_phrases:
            return 0.0

        # Count pattern matches in historical events
        pattern_matches = 0
        total_phrases = len(current_phrases)

        for event in historical_events:
            event_content = event.get("content", "") + " " + event.get("summary", "")
            event_phrases = self._extract_key_phrases(event_content)

            # Count overlapping phrases
            overlap = len(set(current_phrases) & set(event_phrases))
            pattern_matches += overlap

        # Normalize to 0-1 range
        if total_phrases == 0:
            return 0.0

        # Pattern reuse score: higher = more reuse of established patterns
        reuse_score = min(1.0, pattern_matches / (total_phrases * 2))
        return reuse_score

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text for pattern matching."""
        if not text:
            return []

        # Clean and normalize text
        text = text.lower().strip()

        # Extract meaningful phrases (2-4 words)
        phrases = []

        # Split into sentences and extract noun phrases, verb phrases
        sentences = re.split(r"[.!?]+", text)
        for sentence in sentences:
            words = re.findall(r"\b\w+\b", sentence)

            # Extract 2-3 word phrases
            for i in range(len(words) - 1):
                if i + 2 < len(words):
                    phrase = " ".join(words[i : i + 3])
                    if len(phrase) > 8:  # Skip very short phrases
                        phrases.append(phrase)

                # Also extract 2-word phrases
                phrase = " ".join(words[i : i + 2])
                if len(phrase) > 6:
                    phrases.append(phrase)

        # Remove common stop phrases and return unique phrases
        stop_phrases = {
            "i am",
            "i will",
            "i can",
            "i have",
            "this is",
            "that is",
            "it is",
        }
        phrases = [p for p in phrases if p not in stop_phrases]

        return list(set(phrases))  # Return unique phrases

    def apply_novelty_decay(
        self,
        base_novelty: float,
        pattern_reuse_score: float,
        decay_factor: float = 0.85,
    ) -> float:
        """
        Apply novelty decay based on pattern reuse.

        Higher pattern reuse should result in lower novelty scores.
        """
        if pattern_reuse_score <= 0:
            return base_novelty

        # Apply exponential decay based on pattern reuse
        adjusted_novelty = base_novelty * (decay_factor**pattern_reuse_score)

        # Ensure we don't go below a minimum threshold
        min_novelty = 0.1
        return max(min_novelty, adjusted_novelty)
