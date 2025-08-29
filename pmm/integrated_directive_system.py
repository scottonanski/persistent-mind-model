#!/usr/bin/env python3
"""
Integrated directive system that replaces the old commitment-only approach
with natural hierarchical classification and evolution.
"""

from typing import Dict, List, Any
import json

from .adaptive_classifier import AdaptiveDirectiveClassifier, ConversationContext
from .directive_hierarchy import (
    DirectiveHierarchy,
    Directive,
    MetaPrinciple,
    Principle,
    Commitment,
)
from .enhanced_commitment_validator import EnhancedCommitmentValidator
from .continuity_engine import ContinuityEngine


class IntegratedDirectiveSystem:
    """
    Replaces the old CommitmentTracker with a natural hierarchical system.
    Integrates seamlessly with existing PMM architecture while providing
    semantic classification and natural evolution.
    """

    def __init__(self, storage_manager=None):
        self.classifier = AdaptiveDirectiveClassifier()
        self.hierarchy = DirectiveHierarchy()
        self.validator = EnhancedCommitmentValidator()
        self.storage = storage_manager  # SQLite storage for persistence

        # Initialize continuity engine for recursive reflection
        self.continuity_engine = (
            ContinuityEngine(storage_manager, self.hierarchy)
            if storage_manager
            else None
        )

        # Maintain compatibility with existing PMM interfaces
        self.commitments = {}  # Legacy interface

        # Load existing directives from storage if available
        if self.storage:
            self._load_from_storage()

    def process_response(
        self, user_message: str, ai_response: str, event_id: str
    ) -> List[Directive]:
        """
        Process an AI response for directives, replacing the old extract_commitment logic.
        Returns list of detected directives with proper classification.
        """
        detected_directives = []

        # Create conversation context
        ConversationContext(
            user_message=user_message,
            ai_response=ai_response,
            event_id=event_id,
            preceding_user_message=user_message,
            directive_position=self._determine_position(ai_response),
            conversation_phase=self._determine_phase(user_message),
            user_intent_signal=", ".join(self._extract_user_intents(user_message)),
        )

        # Look for directive patterns in AI response
        directive_candidates = self._extract_directive_candidates(ai_response)

        for candidate_text in directive_candidates:
            # For directive detection, be more permissive than strict commitment validation
            # Use the hierarchy's classifier directly for consistency
            directive = self.hierarchy.add_directive(
                content=candidate_text, source_event_id=event_id
            )

            if directive:
                detected_directives.append(directive)

                # Persist to storage if available
                if self.storage:
                    self._persist_directive(directive, event_id)

                # Update legacy interface for backward compatibility
                if directive.directive_type == "commitment":
                    self.commitments[directive.id] = {
                        "cid": directive.id,
                        "text": directive.content,
                        "created_at": directive.created_at,
                        "status": directive.status,
                        "confidence": directive.confidence,
                    }

        return detected_directives

    def _extract_directive_candidates(self, ai_response: str) -> List[str]:
        """Extract potential directive statements from AI response."""
        candidates = []

        # Look for explicit acknowledgment patterns
        acknowledgment_patterns = [
            r"I acknowledge (?:that )?(.+?)(?:\.|$)",
            r"I (?:have )?registered (?:the )?(?:commitment|directive)[:\s]+(.+?)(?:\.|$)",
            r"I commit to (.+?)(?:\.|$)",
            r"I will (.+?)(?:\.|$)",
            r"(?:This means )?I (?:will|shall|aim to) (.+?)(?:\.|$)",
        ]

        import re

        for pattern in acknowledgment_patterns:
            matches = re.finditer(pattern, ai_response, re.IGNORECASE | re.DOTALL)
            for match in matches:
                candidate = match.group(0).strip()
                if len(candidate) > 20:  # Filter out very short matches
                    candidates.append(candidate)

        # Also check for full sentences that might be directives
        sentences = re.split(r"[.!?]+", ai_response)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30 and any(
                indicator in sentence.lower()
                for indicator in [
                    "i will",
                    "i commit",
                    "i acknowledge",
                    "i shall",
                    "i aim",
                ]
            ):
                candidates.append(sentence)

        return candidates

    def _determine_position(self, ai_response: str) -> str:
        """Determine where in the response the directive appears."""
        # Simple heuristic - could be enhanced
        return "response_start"

    def _determine_phase(self, user_message: str) -> str:
        """Determine what phase of conversation this is."""
        user_lower = user_message.lower()

        if any(phrase in user_lower for phrase in ["register", "commit", "directive"]):
            return "commitment"
        elif any(
            phrase in user_lower for phrase in ["evolve", "combine", "synthesize"]
        ):
            return "evolution"
        else:
            return "exploration"

    def _extract_user_intents(self, user_message: str) -> List[str]:
        """Extract user intent signals from their message."""
        intents = []
        user_lower = user_message.lower()

        if "register" in user_lower:
            intents.append("registration_request")
        if "permanent" in user_lower:
            intents.append("permanence_request")
        if "evolve" in user_lower or "combine" in user_lower:
            intents.append("evolution_request")

        return intents

    def get_active_principles(self) -> List[Dict]:
        """Get all active principles for system prompt generation."""
        principles = []

        for principle in self.hierarchy.principles.values():
            if principle.status == "active":
                principles.append(
                    {
                        "id": principle.id,
                        "content": principle.content,
                        "type": "principle",
                        "permanence": getattr(principle, "permanence_level", "high"),
                    }
                )

        return principles

    def get_active_commitments(self) -> List[Dict]:
        """Get all active commitments for legacy compatibility."""
        commitments = []

        for commitment in self.hierarchy.commitments.values():
            if commitment.status == "active":
                commitments.append(
                    {
                        "cid": commitment.id,
                        "text": commitment.content,
                        "created_at": commitment.created_at,
                        "status": commitment.status,
                        "type": "commitment",
                    }
                )

        return commitments

    def get_meta_principles(self) -> List[Dict]:
        """Get all active meta-principles."""
        meta_principles = []

        for mp in self.hierarchy.meta_principles.values():
            if mp.status == "active":
                meta_principles.append(
                    {
                        "id": mp.id,
                        "content": mp.content,
                        "type": "meta-principle",
                        "triggers_evolution": getattr(mp, "triggers_evolution", False),
                    }
                )

        return meta_principles

    def get_directive_summary(self) -> Dict:
        """Get comprehensive summary of all directives."""
        summary = self.hierarchy.get_hierarchy_summary()

        # Add classification statistics
        total_directives = (
            summary["meta_principles"]["count"]
            + summary["principles"]["count"]
            + summary["commitments"]["count"]
        )

        summary["statistics"] = {
            "total_directives": total_directives,
            "classification_accuracy": self._calculate_classification_accuracy(),
            "evolution_events": len(
                [
                    mp
                    for mp in self.hierarchy.meta_principles.values()
                    if getattr(mp, "triggers_evolution", False)
                ]
            ),
        }

        return summary

    def _calculate_classification_accuracy(self) -> float:
        """Calculate classification accuracy based on confidence scores."""
        if not self.classifier.classification_history:
            return 0.0

        # This would be enhanced with actual validation data
        return 0.85  # Placeholder

    def trigger_evolution_if_needed(self) -> bool:
        """Check if any meta-principles should trigger evolution."""
        evolution_triggered = False

        for mp in self.hierarchy.meta_principles.values():
            if mp.status == "active" and getattr(mp, "triggers_evolution", False):

                # Trigger natural evolution
                self.hierarchy._trigger_evolution(mp)
                evolution_triggered = True

        return evolution_triggered

    def export_directives(self) -> Dict:
        """Export all directives for persistence."""
        return {
            "meta_principles": {
                mp_id: {
                    "content": mp.content,
                    "created_at": mp.created_at,
                    "status": mp.status,
                    "triggers_evolution": getattr(mp, "triggers_evolution", False),
                }
                for mp_id, mp in self.hierarchy.meta_principles.items()
            },
            "principles": {
                p_id: {
                    "content": p.content,
                    "created_at": p.created_at,
                    "status": p.status,
                    "parent_meta_principle": getattr(p, "parent_meta_principle", None),
                    "permanence_level": getattr(p, "permanence_level", "high"),
                }
                for p_id, p in self.hierarchy.principles.items()
            },
            "commitments": {
                c_id: {
                    "content": c.content,
                    "created_at": c.created_at,
                    "status": c.status,
                    "parent_principle": getattr(c, "parent_principle", None),
                    "due_date": getattr(c, "due_date", None),
                }
                for c_id, c in self.hierarchy.commitments.items()
            },
        }

    def import_directives(self, data: Dict):
        """Import directives from persistence."""
        # Import meta-principles
        for mp_id, mp_data in data.get("meta_principles", {}).items():
            mp = MetaPrinciple(
                id=mp_id,
                content=mp_data["content"],
                created_at=mp_data["created_at"],
                status=mp_data.get("status", "active"),
            )
            mp.triggers_evolution = mp_data.get("triggers_evolution", False)
            self.hierarchy.meta_principles[mp_id] = mp

        # Import principles
        for p_id, p_data in data.get("principles", {}).items():
            p = Principle(
                id=p_id,
                content=p_data["content"],
                created_at=p_data["created_at"],
                status=p_data.get("status", "active"),
            )
            p.parent_meta_principle = p_data.get("parent_meta_principle")
            p.permanence_level = p_data.get("permanence_level", "high")
            self.hierarchy.principles[p_id] = p

        # Import commitments
        for c_id, c_data in data.get("commitments", {}).items():
            c = Commitment(
                id=c_id,
                content=c_data["content"],
                created_at=c_data["created_at"],
                status=c_data.get("status", "active"),
            )
            c.parent_principle = c_data.get("parent_principle")
            c.due_date = c_data.get("due_date")
            self.hierarchy.commitments[c_id] = c

            # Update legacy interface
            self.commitments[c_id] = {
                "cid": c_id,
                "text": c.content,
                "created_at": c.created_at,
                "status": c.status,
            }

    def _load_from_storage(self):
        """Load existing directives from persistent storage."""
        try:
            # Load meta-principles
            meta_principles = self.storage.get_directives_by_type("meta-principle")
            for mp_data in meta_principles:
                mp = MetaPrinciple(
                    id=mp_data["id"],
                    content=mp_data["content"],
                    created_at=mp_data["created_at"],
                    status=mp_data["status"],
                )
                metadata = (
                    json.loads(mp_data["metadata"]) if mp_data["metadata"] else {}
                )
                mp.triggers_evolution = metadata.get("triggers_evolution", False)
                self.hierarchy.meta_principles[mp.id] = mp

            # Load principles
            principles = self.storage.get_directives_by_type("principle")
            for p_data in principles:
                p = Principle(
                    id=p_data["id"],
                    content=p_data["content"],
                    created_at=p_data["created_at"],
                    status=p_data["status"],
                )
                metadata = json.loads(p_data["metadata"]) if p_data["metadata"] else {}
                p.parent_meta_principle = p_data["parent_id"]
                p.permanence_level = metadata.get("permanence_level", "high")
                self.hierarchy.principles[p.id] = p

            # Load commitments
            commitments = self.storage.get_directives_by_type("commitment")
            for c_data in commitments:
                c = Commitment(
                    id=c_data["id"],
                    content=c_data["content"],
                    created_at=c_data["created_at"],
                    status=c_data["status"],
                )
                metadata = json.loads(c_data["metadata"]) if c_data["metadata"] else {}
                c.parent_principle = c_data["parent_id"]
                c.due_date = metadata.get("due_date")
                self.hierarchy.commitments[c.id] = c

                # Update legacy interface
                self.commitments[c.id] = {
                    "cid": c.id,
                    "text": c.content,
                    "created_at": c.created_at,
                    "status": c.status,
                }

        except Exception as e:
            print(f"Warning: Failed to load directives from storage: {e}")

    def _persist_directive(self, directive: Directive, source_event_id: str):
        """Persist a directive to storage."""
        try:
            directive_type = directive.__class__.__name__.lower()
            if directive_type == "metaprinciple":
                directive_type = "meta-principle"

            # Prepare metadata
            metadata = {}
            if hasattr(directive, "triggers_evolution"):
                metadata["triggers_evolution"] = directive.triggers_evolution
            if hasattr(directive, "permanence_level"):
                metadata["permanence_level"] = directive.permanence_level
            if hasattr(directive, "due_date"):
                metadata["due_date"] = directive.due_date

            # Determine parent ID
            parent_id = None
            if hasattr(directive, "parent_meta_principle"):
                parent_id = directive.parent_meta_principle
            elif hasattr(directive, "parent_principle"):
                parent_id = directive.parent_principle

            self.storage.store_directive(
                directive_id=directive.id,
                directive_type=directive_type,
                content=directive.content,
                created_at=directive.created_at,
                status=directive.status,
                parent_id=parent_id,
                source_event_id=None,  # Skip foreign key constraint for now
                metadata=metadata if metadata else None,
            )

        except Exception as e:
            print(f"Warning: Failed to persist directive: {e}")

    def reflect_on_continuity(self, lookback_days: int = 30) -> List[str]:
        """
        Trigger continuity reflection to detect patterns and auto-register insights.

        This is the recursive backbone that enables emergent self-awareness.
        Returns list of newly registered meta-principle IDs.
        """
        if not self.continuity_engine:
            return []

        return self.continuity_engine.reflect_and_register(lookback_days)

    def get_continuity_summary(self) -> Dict[str, Any]:
        """
        Get summary of current continuity patterns for system transparency.
        """
        if not self.continuity_engine:
            return {"error": "Continuity engine not initialized"}

        return self.continuity_engine.get_continuity_summary()

    def should_trigger_reflection(self) -> bool:
        """
        Check if conditions are met for triggering continuity reflection.
        """
        if not self.continuity_engine:
            return False

        return self.continuity_engine.should_reflect()
