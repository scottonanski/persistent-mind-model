#!/usr/bin/env python3
"""
Commitment lifecycle management for Persistent Mind Model.
Tracks agent commitments from creation to completion.
"""

import re
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Commitment:
    """A commitment made by an agent during reflection."""

    cid: str
    text: str
    created_at: str
    source_insight_id: str
    status: str = "open"  # open, closed, expired
    due: Optional[str] = None
    closed_at: Optional[str] = None
    close_note: Optional[str] = None
    ngrams: List[str] = None  # 3-grams for matching


class CommitmentTracker:
    """Manages commitment lifecycle and completion detection."""

    def __init__(self):
        self.commitments: Dict[str, Commitment] = {}

    def _is_valid_commitment(self, text: str) -> bool:
        """
        Validate commitment against 5 criteria:
        1. Actionable: concrete verb + object
        2. Context-bound: references current topic/artifact/goal
        3. Time/trigger: includes when or clear trigger
        4. Non-duplicate: not semantically near open commitments
        5. Owned: first-person agent ownership
        """
        if not text or len(text.strip()) < 10:
            return False

        text_lower = text.lower().strip()

        # 1. Actionable: Must contain concrete verb + object
        action_verbs = [
            "draft",
            "create",
            "write",
            "build",
            "implement",
            "design",
            "develop",
            "review",
            "analyze",
            "test",
            "validate",
            "document",
            "outline",
            "label",
            "categorize",
            "organize",
            "schedule",
            "plan",
            "research",
            "investigate",
            "compile",
            "generate",
            "produce",
            "deliver",
            "complete",
            "finish",
        ]

        has_action_verb = any(verb in text_lower for verb in action_verbs)

        # Add more action verbs that were missing
        additional_verbs = [
            "strive",
            "recognize",
            "acknowledge",
            "incorporate",
            "prioritize",
            "practice",
            "foster",
            "utilize",
            "convey",
            "outline",
            "focus",
            "enhance",
            "improve",
            "develop",
            "strengthen",
            "refine",
        ]
        action_verbs.extend(additional_verbs)
        has_action_verb = any(verb in text_lower for verb in action_verbs)

        # Only reject truly vague verbs when used alone
        vague_verbs = [
            "consider",
            "think about",
            "look into",
        ]
        has_vague_verb = any(verb in text_lower for verb in vague_verbs)

        if not has_action_verb or has_vague_verb:
            return False

        # 2. Context-bound: Must reference specific topic/artifact
        context_indicators = [
            "pmm",
            "onboarding",
            "v0.",
            "dataset",
            "samples",
            "document",
            "outline",
            "template",
            "guide",
            "tutorial",
            "demo",
            "test",
            "validation",
            "api",
            "probe",
            "commitment",
            "reflection",
            "insight",
            "trait",
            "personality",
            "hash",
            "evidence",
            "phase",
            "closure",
            # Add conversational AI contexts
            "conversation",
            "interaction",
            "response",
            "engagement",
            "dialogue",
            "emotional",
            "intelligence",
            "empathy",
            "listening",
            "feedback",
            "creativity",
            "storytelling",
            "awareness",
            "understanding",
            "connection",
            "communication",
            "skill",
            "ability",
            "approach",
            "technique",
            "method",
            "process",
            "system",
            "experience",
            "quality",
            "specific",
            "concrete",
        ]

        has_context = any(indicator in text_lower for indicator in context_indicators)

        # Reject only truly generic contexts
        generic_contexts = ["stuff", "things", "whatever", "anything", "everything"]
        has_generic = any(generic in text_lower for generic in generic_contexts)

        if not has_context or has_generic:
            return False

        # 3. Time/trigger: Must include when or trigger (relaxed for conversational AI)
        time_triggers = [
            "tonight",
            "today",
            "tomorrow",
            "this week",
            "next week",
            "by",
            "before",
            "after",
            "once",
            "when",
            "during",
            "within",
            "in the next",
            "over the",
            "following",
            "subsequent",
            "upon",
            "after reviewing",
            "after completing",
            "right now",
            "now",
            "immediately",
            "asap",
            "soon",
            "shortly",
            "quickly",
            # Add conversational triggers
            "going forward",
            "moving forward",
            "from now on",
            "in future",
            "next time",
            "in our",
            "during our",
            "each",
            "every",
            "ongoing",
            "continuously",
            "regularly",
            "consistently",
            "always",
            "will",
            "shall",
            "aim to",
        ]

        has_time_trigger = any(trigger in text_lower for trigger in time_triggers)

        if not has_time_trigger:
            return False

        # 4. Non-duplicate: Check against open commitments (simplified for now)
        # This will be enhanced when we have access to existing commitments

        # 5. Owned: Must be first-person
        ownership_indicators = [
            "i will",
            "i plan to",
            "i commit to",
            "next, i will",
            "i aim to",
            "i intend to",
            "i shall",
            "my goal is to",
            "going forward, i will",
            "moving forward, i will",
        ]
        has_ownership = any(
            indicator in text_lower for indicator in ownership_indicators
        )

        # Reject external ownership
        external_indicators = [
            "someone should",
            "we should",
            "they should",
            "it would be good",
        ]
        has_external = any(external in text_lower for external in external_indicators)

        if not has_ownership or has_external:
            return False

        return True

    def extract_commitment(self, text: str) -> Tuple[Optional[str], List[str]]:
        """Extract and validate commitment from text using 5-point criteria."""
        # Strip markdown formatting first
        clean_text = self._strip_markdown(text)

        # Expanded commitment patterns
        commitment_patterns = [
            "i will",
            "next, i will",
            "next:",
            "i plan to",
            "i commit to",
            "i aim to",
            "my goal is to",
            "by committing to",
            "i intend to",
            "i shall",
            "going forward, i will",
            "moving forward, i will",
        ]

        # First, try to extract commitment from the whole text
        if any(starter in clean_text.lower() for starter in commitment_patterns):
            commitment = self._clean_commitment_text(clean_text, commitment_patterns)
            if commitment and self._is_valid_commitment(commitment):
                return commitment, self._generate_ngrams(commitment)

        # Fallback: split by markdown bullets and numbered lists, then sentences
        candidates = self._split_into_commitment_candidates(clean_text)

        for candidate in candidates:
            candidate = candidate.strip()
            if any(starter in candidate.lower() for starter in commitment_patterns):
                commitment = self._clean_commitment_text(candidate, commitment_patterns)
                if commitment and self._is_valid_commitment(commitment):
                    return commitment, self._generate_ngrams(commitment)

        return None, []

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown formatting from text."""
        # Remove bold/italic
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)

        # Remove numbered list markers
        text = re.sub(r"^\d+\.\s*", "", text, flags=re.MULTILINE)

        # Remove bullet points
        text = re.sub(r"^[-*+]\s*", "", text, flags=re.MULTILINE)

        # Remove section headers
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

        return text.strip()

    def _split_into_commitment_candidates(self, text: str) -> List[str]:
        """Split text into potential commitment candidates."""
        candidates = []

        # Split by numbered lists first
        numbered_items = re.split(r"\n\d+\.\s*", text)
        for item in numbered_items:
            if item.strip():
                candidates.append(item.strip())

        # Split by bullet points
        bullet_items = re.split(r"\n[-*+]\s*", text)
        for item in bullet_items:
            if item.strip():
                candidates.append(item.strip())

        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            if para.strip():
                candidates.append(para.strip())

        # Finally, split by sentences (but more carefully)
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        for sentence in sentences:
            if sentence.strip():
                candidates.append(sentence.strip())

        return candidates

    def _clean_commitment_text(self, text: str, patterns: List[str]) -> str:
        """Clean and normalize commitment text."""
        commitment = text.strip()

        # Handle "By committing to these practices, I aim to..." pattern
        if commitment.lower().startswith("by committing to"):
            # Extract the actual commitment after "I aim to" or similar
            aim_match = re.search(
                r"i (aim to|intend to|will|plan to|commit to)\s+(.+)",
                commitment.lower(),
            )
            if aim_match:
                verb_phrase = aim_match.group(1)
                action = aim_match.group(2)
                commitment = f"I {verb_phrase} {action}"
            else:
                # Fallback: just remove "By committing to" and add "I will"
                commitment = commitment[len("By committing to") :].strip()
                if not commitment.lower().startswith("i "):
                    commitment = "I will " + commitment

        # Remove other common prefixes
        prefixes_to_remove = [
            "Next:",
            "next:",
            "Next, I will",
            "next, i will",
            "To enhance",
            "to enhance",
        ]

        for prefix in prefixes_to_remove:
            if commitment.lower().startswith(prefix.lower()):
                commitment = commitment[len(prefix) :].strip()

                # If we removed "Next, I will", add "I will" back
                if prefix.lower() in [
                    "next, i will"
                ] and not commitment.lower().startswith("i will"):
                    commitment = "I will " + commitment
                elif prefix.lower() in [
                    "to enhance"
                ] and not commitment.lower().startswith("i "):
                    commitment = "I will " + commitment
                break

        return commitment

    def _generate_ngrams(self, commitment: str) -> List[str]:
        """Generate 3-grams for commitment matching."""
        words = commitment.lower().split()
        ngrams = []
        for i in range(len(words) - 2):
            if all(len(w) > 2 for w in words[i : i + 3]):  # Skip short words
                ngrams.append(" ".join(words[i : i + 3]))
        return ngrams

    def _is_duplicate_commitment(self, text: str) -> bool:
        """Check if commitment is semantically similar to recent open commitments."""
        if not text:
            return False

        # Normalize text for comparison
        normalized = text.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)  # Normalize whitespace

        # Get n-grams for similarity comparison
        def get_ngrams(text: str, n: int = 3) -> set:
            words = text.split()
            return set(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))

        # Also get 2-grams for better semantic matching
        def get_bigrams(text: str) -> set:
            words = text.split()
            return set(" ".join(words[i : i + 2]) for i in range(len(words) - 1))

        current_ngrams = get_ngrams(normalized)
        current_bigrams = get_bigrams(normalized)

        if not current_ngrams and not current_bigrams:
            return False

        # Check against last 20 open commitments
        open_commitments = [c for c in self.commitments.values() if c.status == "open"]
        recent_commitments = sorted(
            open_commitments, key=lambda x: x.created_at, reverse=True
        )[:20]

        for existing in recent_commitments:
            existing_normalized = existing.text.lower().strip()
            existing_normalized = re.sub(r"\s+", " ", existing_normalized)
            existing_ngrams = get_ngrams(existing_normalized)
            existing_bigrams = get_bigrams(existing_normalized)

            if not existing_ngrams and not existing_bigrams:
                continue

            # Calculate similarity using both trigrams and bigrams
            trigram_similarity = 0
            if current_ngrams and existing_ngrams:
                intersection = len(current_ngrams & existing_ngrams)
                union = len(current_ngrams | existing_ngrams)
                trigram_similarity = intersection / union if union > 0 else 0

            bigram_similarity = 0
            if current_bigrams and existing_bigrams:
                intersection = len(current_bigrams & existing_bigrams)
                union = len(current_bigrams | existing_bigrams)
                bigram_similarity = intersection / union if union > 0 else 0

            # Use the higher similarity score, lower threshold for better detection
            max_similarity = max(trigram_similarity, bigram_similarity)

            # Consider duplicate if >45% similar (more sensitive)
            if max_similarity > 0.45:
                print(
                    f"🔍 DEBUG: Duplicate detected ({max_similarity:.1%} similar): '{text[:50]}...' vs '{existing.text[:50]}...'"
                )
                return True

        return False

    def add_commitment(
        self, text: str, source_insight_id: str, due: Optional[str] = None
    ) -> str:
        """Add a new commitment and return its ID."""
        commitment_text, ngrams = self.extract_commitment(text)
        if not commitment_text:
            return ""

        # Check for duplicates
        if self._is_duplicate_commitment(commitment_text):
            print(
                f"🔍 DEBUG: Commitment rejected as duplicate: {commitment_text[:50]}..."
            )
            return ""

        cid = f"c{len(self.commitments) + 1}"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        commitment = Commitment(
            cid=cid,
            text=commitment_text,
            created_at=ts,
            source_insight_id=source_insight_id,
            due=due,
            ngrams=ngrams or [],
        )

        self.commitments[cid] = commitment
        print(f"🔍 DEBUG: Valid commitment added: {commitment_text}")
        return cid

    def mark_commitment(
        self, cid: str, status: str, note: Optional[str] = None
    ) -> bool:
        """Manually mark a commitment as closed/completed."""
        if cid not in self.commitments:
            return False

        commitment = self.commitments[cid]
        commitment.status = status
        commitment.closed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        commitment.close_note = note
        return True

    def auto_close_from_event(self, event_text: str) -> List[str]:
        """Auto-close commitments mentioned in event descriptions."""
        closed_cids = []
        event_lower = event_text.lower()

        for cid, commitment in self.commitments.items():
            if commitment.status != "open":
                continue

            # Check if event mentions commitment ID or key terms
            if cid in event_text or any(
                ngram in event_lower for ngram in commitment.ngrams[:3]  # Top 3 ngrams
            ):
                self.mark_commitment(
                    cid, "closed", f"Auto-closed from event: {event_text[:50]}..."
                )
                closed_cids.append(cid)

        return closed_cids

    def auto_close_from_reflection(self, reflection_text: str) -> List[str]:
        """Auto-close commitments based on reflection completion signals."""
        closed_cids = []
        reflection_lower = reflection_text.lower()

        # More lenient completion signals - any forward progress or new commitment
        completion_signals = [
            "done",
            "completed",
            "finished",
            "accomplished",
            "achieved",
            "next:",
            "i will",
            "plan to",
            "going to",
            "decided to",
            "realized",
            "noticed",
            "observed",
            "recognized",
            "understand",
        ]
        has_completion_signal = any(
            signal in reflection_lower for signal in completion_signals
        )

        # Auto-close old commitments when new ones are made (progression)
        has_new_commitment = any(
            signal in reflection_lower for signal in ["next:", "i will", "plan to"]
        )

        if not (has_completion_signal or has_new_commitment):
            return closed_cids

        # Get oldest open commitments to close (FIFO approach)
        open_commitments = [
            (cid, c) for cid, c in self.commitments.items() if c.status == "open"
        ]
        open_commitments.sort(key=lambda x: x[1].created_at)  # Sort by creation time

        # If making new commitment, close 1-2 oldest ones (showing progress)
        if has_new_commitment and len(open_commitments) > 5:
            for cid, commitment in open_commitments[:2]:  # Close 2 oldest
                self.mark_commitment(
                    cid, "closed", "Auto-closed: progressed to new commitment"
                )
                closed_cids.append(cid)

        # Also check for direct n-gram matches (more lenient threshold)
        for cid, commitment in self.commitments.items():
            if commitment.status != "open" or cid in closed_cids:
                continue

            # Check if reflection mentions commitment terms (reduced threshold)
            matching_ngrams = sum(
                1 for ngram in commitment.ngrams if ngram in reflection_lower
            )
            if matching_ngrams >= 1:  # Reduced from 2 to 1 for more sensitivity
                self.mark_commitment(
                    cid, "closed", "Auto-closed: reflection mentioned commitment terms"
                )
                closed_cids.append(cid)

        return closed_cids

    def get_open_commitments(self) -> List[Dict]:
        """Get all open commitments."""
        return [
            {
                "cid": c.cid,
                "text": c.text,
                "created_at": c.created_at,
                "source_insight_id": c.source_insight_id,
                "due": c.due,
            }
            for c in self.commitments.values()
            if c.status == "open"
        ]

    def get_commitment_metrics(self) -> Dict:
        """Calculate commitment completion metrics."""
        total = len(self.commitments)
        open_count = sum(1 for c in self.commitments.values() if c.status == "open")
        closed_count = sum(1 for c in self.commitments.values() if c.status == "closed")

        # Calculate median time to close for closed commitments
        close_times = []
        for c in self.commitments.values():
            if c.status == "closed" and c.closed_at:
                try:
                    created = datetime.fromisoformat(c.created_at.replace("Z", ""))
                    closed = datetime.fromisoformat(c.closed_at.replace("Z", ""))
                    close_times.append(
                        (closed - created).total_seconds() / 3600
                    )  # hours
                except Exception:
                    continue

        median_time_to_close = (
            sorted(close_times)[len(close_times) // 2] if close_times else 0
        )

        return {
            "commitments_total": total,
            "commitments_open": open_count,
            "commitments_closed": closed_count,
            "close_rate": closed_count / total if total > 0 else 0,
            "median_time_to_close_hours": median_time_to_close,
        }

    def archive_legacy_commitments(self) -> List[str]:
        """Archive generic legacy commitments that don't meet 5-point criteria."""
        archived_cids = []

        # Generic patterns to identify legacy commitments
        legacy_patterns = [
            "clarify and confirm",
            "clarify objectives",
            "confirm objectives",
            "assist with moving forward",
            "there was no prior commitment",
            "review a document",
            "assess if it has been completed",
        ]

        for cid, commitment in self.commitments.items():
            if commitment.status in ["archived_legacy", "expired"]:
                continue

            commitment_lower = commitment.text.lower()

            # Check if it matches legacy patterns
            is_legacy = any(pattern in commitment_lower for pattern in legacy_patterns)

            # Also check if it fails the 5-point validation
            is_invalid = not self._is_valid_commitment(commitment.text)

            if is_legacy or is_invalid:
                # Archive with hygiene metadata
                commitment.status = "archived_legacy"
                commitment.closed_at = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                commitment.close_note = "Archived: generic template commitment"
                archived_cids.append(cid)
                print(
                    f"🔍 DEBUG: Archived legacy commitment {cid}: {commitment.text[:50]}..."
                )

        return archived_cids

    def get_commitment_hash(self, commitment: Commitment) -> str:
        """Generate a hash for a commitment for evidence linking."""
        # Create a stable hash from commitment content
        content = f"{commitment.cid}:{commitment.text}:{commitment.created_at}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def detect_evidence_events(
        self, text: str
    ) -> List[Tuple[str, str, str, Optional[str]]]:
        """
        Detect evidence events in text and return list of (evidence_type, commit_ref, description, artifact).

        Evidence patterns:
        - "Done: [description with artifact]" -> evidence:done
        - "Completed: [description]" -> evidence:done
        - "Blocked: [reason] -> [next_action]" -> evidence:blocked
        - "Delegated to [who]: [description]" -> evidence:delegated
        """
        evidence_events = []
        # Keep original text for artifact extraction, normalize for pattern matching
        normalized = text.strip()

        # Pattern 1: Done/Completed statements - case-insensitive matching
        done_patterns = [
            r"^done[:\-\s](.+)",  # Done: , Done- , Done <space>
            r"^completed[:\-\s](.+)",
            r"^finished[:\-\s](.+)",
            r"^delivered[:\-\s](.+)",
        ]

        for pattern in done_patterns:
            matches = re.finditer(pattern, normalized, re.IGNORECASE)
            for match in matches:
                description = match.group(1).strip()

                # Extract artifact if present (file names, URLs, IDs)
                artifact = self._extract_artifact(description)

                # For now, we'll need to match this to open commitments
                # This is a simplified version - in practice we'd need better matching
                for cid, commitment in self.commitments.items():
                    if commitment.status == "open":
                        commit_hash = self.get_commitment_hash(commitment)
                        # Simple keyword matching - could be enhanced
                        if any(
                            word in description
                            for word in commitment.text.lower().split()[:3]
                        ):
                            evidence_events.append(
                                ("done", commit_hash, description, artifact)
                            )
                            break

        # Pattern 2: Blocked statements
        blocked_patterns = [
            r"blocked:\s*(.+?)(?:\s*->\s*(.+))?$",
            r"cannot proceed:\s*(.+?)(?:\s*->\s*(.+))?$",
            r"stuck on:\s*(.+?)(?:\s*->\s*(.+))?$",
            r"blocked:\s*(.+?)(?:\s*next:\s*(.+))?$",
        ]

        for pattern in blocked_patterns:
            matches = re.finditer(pattern, normalized, re.IGNORECASE)
            for match in matches:
                reason = match.group(1).strip()
                next_action = match.group(2).strip() if match.group(2) else None

                # Match to open commitments
                for cid, commitment in self.commitments.items():
                    if commitment.status == "open":
                        commit_hash = self.get_commitment_hash(commitment)
                        if any(
                            word in reason
                            for word in commitment.text.lower().split()[:3]
                        ):
                            evidence_events.append(
                                ("blocked", commit_hash, reason, next_action)
                            )
                            break

        # Pattern 3: Delegated statements
        delegated_patterns = [
            r"delegated to\s+([^:]+):\s*(.+)",
            r"handed off to\s+([^:]+):\s*(.+)",
            r"assigned to\s+([^:]+):\s*(.+)",
        ]

        for pattern in delegated_patterns:
            matches = re.finditer(pattern, normalized, re.IGNORECASE)
            for match in matches:
                assignee = match.group(1).strip()
                description = match.group(2).strip()

                # Match to open commitments
                for cid, commitment in self.commitments.items():
                    if commitment.status == "open":
                        commit_hash = self.get_commitment_hash(commitment)
                        if any(
                            word in description
                            for word in commitment.text.lower().split()[:3]
                        ):
                            evidence_events.append(
                                (
                                    "delegated",
                                    commit_hash,
                                    f"Delegated to {assignee}: {description}",
                                    assignee,
                                )
                            )
                            break

        return evidence_events

    def _extract_artifact(self, description: str) -> Optional[str]:
        """Extract artifact references from evidence description."""
        # Look for file names, URLs, IDs, timestamps
        artifact_patterns = [
            r"`([^`]+\.[a-zA-Z0-9]+)`",  # `filename.ext`
            r"([a-zA-Z0-9_-]+\.[a-zA-Z0-9]+)",  # filename.ext
            r"(https?://[^\s]+)",  # URLs
            r"(#\d+)",  # Issue/PR numbers
            r"(\d{4}-\d{2}-\d{2})",  # Dates
            r"([A-Z]{2,}-\d+)",  # Ticket IDs like PROJ-123
        ]

        for pattern in artifact_patterns:
            match = re.search(pattern, description)
            if match:
                return match.group(1)

        return None

    def close_commitment_with_evidence(
        self,
        commit_hash: str,
        evidence_type: str,
        description: str,
        artifact: Optional[str] = None,
    ) -> bool:
        """Close a commitment based on evidence. Only 'done' evidence closes commitments."""
        if evidence_type != "done":
            print(
                f"🔍 DEBUG: Evidence type '{evidence_type}' does not close commitments"
            )
            return False

        # Find commitment by hash
        target_commitment = None
        target_cid = None

        for cid, commitment in self.commitments.items():
            if self.get_commitment_hash(commitment) == commit_hash:
                target_commitment = commitment
                target_cid = cid
                break

        if not target_commitment:
            print(f"🔍 DEBUG: No commitment found for hash {commit_hash}")
            return False

        if target_commitment.status != "open":
            print(
                f"🔍 DEBUG: Commitment {target_cid} is not open (status: {target_commitment.status})"
            )
            return False

        # Close the commitment
        target_commitment.status = "closed"
        target_commitment.closed_at = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        target_commitment.close_note = f"Evidence: {description}"
        if artifact:
            target_commitment.close_note += f" (Artifact: {artifact})"

        print(f"🔍 DEBUG: Closed commitment {target_cid} with evidence: {description}")
        return True

    def expire_old_commitments(self, days_old: int = 30) -> List[str]:
        """Mark old commitments as expired."""
        expired_cids = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_old)

        for cid, commitment in self.commitments.items():
            if commitment.status != "open":
                continue

            try:
                created = datetime.fromisoformat(commitment.created_at.replace("Z", ""))
                if created < cutoff:
                    self.mark_commitment(
                        cid, "expired", f"Auto-expired after {days_old} days"
                    )
                    expired_cids.append(cid)
            except Exception:
                continue

        return expired_cids
