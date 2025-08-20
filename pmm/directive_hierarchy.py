#!/usr/bin/env python3
"""
Hierarchical directive system for PMM.
Manages the three-tier hierarchy: Meta-principles → Principles → Commitments
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib

from .directive_classifier import SemanticDirectiveClassifier


@dataclass
class Directive:
    """Base class for all directive types."""

    id: str
    content: str
    directive_type: str  # 'meta-principle', 'principle', 'commitment'
    created_at: str
    source_event_id: Optional[str] = None
    status: str = "active"  # active, archived, superseded
    confidence: float = 0.0

    def __post_init__(self):
        if not self.id:
            # Generate ID from content hash
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:12]
            prefix = {"meta-principle": "mp", "principle": "pr", "commitment": "cm"}
            self.id = f"{prefix.get(self.directive_type, 'dir')}_{content_hash}"


@dataclass
class MetaPrinciple(Directive):
    """Rules about how to evolve principles and commitments."""

    directive_type: str = field(default="meta-principle", init=False)
    triggers_evolution: bool = True
    evolution_scope: str = "framework"  # framework, principles, commitments


@dataclass
class Principle(Directive):
    """Identity-defining guidelines that govern behavior."""

    directive_type: str = field(default="principle", init=False)
    parent_meta_principle: Optional[str] = None
    derived_from_commitments: List[str] = field(default_factory=list)
    permanence_level: str = "high"  # high, medium, low


@dataclass
class Commitment(Directive):
    """Specific behavioral intentions."""

    directive_type: str = field(default="commitment", init=False)
    parent_principle: Optional[str] = None
    due_date: Optional[str] = None
    completion_evidence: List[str] = field(default_factory=list)
    behavioral_scope: str = "specific"  # specific, general, ongoing


class DirectiveHierarchy:
    """
    Manages the hierarchical relationship between directives.
    Handles natural evolution from commitments → principles → meta-principles.
    """

    def __init__(self):
        self.classifier = SemanticDirectiveClassifier()
        self.meta_principles: Dict[str, MetaPrinciple] = {}
        self.principles: Dict[str, Principle] = {}
        self.commitments: Dict[str, Commitment] = {}

    def add_directive(
        self, content: str, source_event_id: Optional[str] = None
    ) -> Optional[Directive]:
        """
        Add a directive, automatically classifying its type and establishing relationships.
        """
        # Classify the directive semantically
        directive_type = self.classifier.classify_directive(content)
        confidence_scores = self.classifier.get_classification_confidence(content)
        confidence = confidence_scores[directive_type]

        # Create timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create appropriate directive object
        if directive_type == "meta-principle":
            directive = MetaPrinciple(
                id="",  # Will be generated in __post_init__
                content=content,
                created_at=timestamp,
                source_event_id=source_event_id,
                confidence=confidence,
            )
            self.meta_principles[directive.id] = directive

            # Meta-principles may trigger evolution
            if directive.triggers_evolution:
                self._trigger_evolution(directive)

        elif directive_type == "principle":
            directive = Principle(
                id="",
                content=content,
                created_at=timestamp,
                source_event_id=source_event_id,
                confidence=confidence,
            )

            # Try to link to parent meta-principle
            directive.parent_meta_principle = self._find_parent_meta_principle(content)

            self.principles[directive.id] = directive

        else:  # commitment
            directive = Commitment(
                id="",
                content=content,
                created_at=timestamp,
                source_event_id=source_event_id,
                confidence=confidence,
            )

            # Try to link to parent principle
            directive.parent_principle = self._find_parent_principle(content)

            self.commitments[directive.id] = directive

        return directive

    def _find_parent_meta_principle(self, content: str) -> Optional[str]:
        """Find the most relevant meta-principle for a principle."""
        # Simple semantic matching - could be enhanced with embeddings
        content_lower = content.lower()

        best_match = None
        best_score = 0.0

        for mp_id, meta_principle in self.meta_principles.items():
            if meta_principle.status != "active":
                continue

            # Calculate semantic overlap
            mp_words = set(meta_principle.content.lower().split())
            content_words = set(content_lower.split())

            overlap = len(mp_words & content_words)
            union = len(mp_words | content_words)

            if union > 0:
                score = overlap / union
                if score > best_score and score > 0.2:  # Minimum threshold
                    best_score = score
                    best_match = mp_id

        return best_match

    def _find_parent_principle(self, content: str) -> Optional[str]:
        """Find the most relevant principle for a commitment."""
        content_lower = content.lower()

        best_match = None
        best_score = 0.0

        for pr_id, principle in self.principles.items():
            if principle.status != "active":
                continue

            # Calculate semantic overlap
            pr_words = set(principle.content.lower().split())
            content_words = set(content_lower.split())

            overlap = len(pr_words & content_words)
            union = len(pr_words | content_words)

            if union > 0:
                score = overlap / union
                if (
                    score > best_score and score > 0.15
                ):  # Lower threshold for commitments
                    best_score = score
                    best_match = pr_id

        return best_match

    def _trigger_evolution(self, meta_principle: MetaPrinciple):
        """
        Trigger evolution based on meta-principle.
        This is where natural evolution happens - meta-principles reshape the hierarchy.
        """
        content_lower = meta_principle.content.lower()

        # Evolution patterns (learned from natural language, not hardcoded)
        if any(
            phrase in content_lower
            for phrase in ["combine", "synthesize", "integrate", "merge"]
        ):
            self._synthesize_principles()

        elif any(
            phrase in content_lower
            for phrase in ["refine", "improve", "enhance", "develop"]
        ):
            self._refine_existing_directives()

        elif any(
            phrase in content_lower
            for phrase in ["evolve", "adapt", "transform", "grow"]
        ):
            self._evolve_framework()

    def _synthesize_principles(self):
        """Combine related commitments into new principles."""
        # Group commitments by semantic similarity
        commitment_groups = self._group_similar_commitments()

        for group in commitment_groups:
            if len(group) >= 3:  # Minimum threshold for synthesis
                # Create new principle from commitment group
                synthesized_content = self._synthesize_content(group)
                if synthesized_content:
                    new_principle = Principle(
                        id="",
                        content=synthesized_content,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        derived_from_commitments=[c.id for c in group],
                        confidence=0.8,  # High confidence for synthesized principles
                    )
                    self.principles[new_principle.id] = new_principle

    def _group_similar_commitments(self) -> List[List[Commitment]]:
        """Group commitments by semantic similarity."""
        active_commitments = [
            c for c in self.commitments.values() if c.status == "active"
        ]
        groups = []
        processed = set()

        for commitment in active_commitments:
            if commitment.id in processed:
                continue

            # Find similar commitments
            similar_group = [commitment]
            processed.add(commitment.id)

            for other in active_commitments:
                if other.id in processed:
                    continue

                # Calculate similarity
                similarity = self._calculate_semantic_similarity(
                    commitment.content, other.content
                )

                if similarity > 0.3:  # Similarity threshold
                    similar_group.append(other)
                    processed.add(other.id)

            if len(similar_group) > 1:
                groups.append(similar_group)

        return groups

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Simple word overlap - could be enhanced with embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _synthesize_content(self, commitments: List[Commitment]) -> Optional[str]:
        """Synthesize principle content from a group of commitments."""
        # Extract common themes and create principle statement
        all_words = []
        for commitment in commitments:
            all_words.extend(commitment.content.lower().split())

        # Find most common meaningful words
        word_freq = {}
        stop_words = {
            "i",
            "will",
            "to",
            "and",
            "the",
            "in",
            "with",
            "for",
            "of",
            "a",
            "an",
        }

        for word in all_words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top themes
        top_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        if not top_themes:
            return None

        # Create principle statement
        theme_words = [theme[0] for theme in top_themes]
        return (
            f"Maintain {', '.join(theme_words[:3])} as core values in all interactions"
        )

    def _refine_existing_directives(self):
        """Refine existing directives based on patterns."""
        # This would analyze directive effectiveness and refine them
        pass

    def _evolve_framework(self):
        """Evolve the overall framework structure."""
        # This would modify how directives relate to each other
        pass

    def get_hierarchy_summary(self) -> Dict:
        """Get summary of the current directive hierarchy."""
        return {
            "meta_principles": {
                "count": len(
                    [
                        mp
                        for mp in self.meta_principles.values()
                        if mp.status == "active"
                    ]
                ),
                "items": [
                    {"id": mp.id, "content": mp.content[:100]}
                    for mp in self.meta_principles.values()
                    if mp.status == "active"
                ],
            },
            "principles": {
                "count": len(
                    [p for p in self.principles.values() if p.status == "active"]
                ),
                "items": [
                    {"id": p.id, "content": p.content[:100]}
                    for p in self.principles.values()
                    if p.status == "active"
                ],
            },
            "commitments": {
                "count": len(
                    [c for c in self.commitments.values() if c.status == "active"]
                ),
                "items": [
                    {"id": c.id, "content": c.content[:100]}
                    for c in self.commitments.values()
                    if c.status == "active"
                ],
            },
        }

    def get_directive_relationships(self) -> Dict:
        """Get the hierarchical relationships between directives."""
        relationships = {
            "meta_principle_to_principles": {},
            "principle_to_commitments": {},
        }

        # Map principles to their meta-principles
        for principle in self.principles.values():
            if principle.parent_meta_principle:
                if (
                    principle.parent_meta_principle
                    not in relationships["meta_principle_to_principles"]
                ):
                    relationships["meta_principle_to_principles"][
                        principle.parent_meta_principle
                    ] = []
                relationships["meta_principle_to_principles"][
                    principle.parent_meta_principle
                ].append(principle.id)

        # Map commitments to their principles
        for commitment in self.commitments.values():
            if commitment.parent_principle:
                if (
                    commitment.parent_principle
                    not in relationships["principle_to_commitments"]
                ):
                    relationships["principle_to_commitments"][
                        commitment.parent_principle
                    ] = []
                relationships["principle_to_commitments"][
                    commitment.parent_principle
                ].append(commitment.id)

        return relationships
