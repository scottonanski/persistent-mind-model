"""
PMM Emergence Loop: IAS/GAS scoring and stage detection system.

This module implements the substrate-agnostic personality emergence measurement
system that tracks AI identity convergence through 5 stages (S0-S4).
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Canonical PMM definition for semantic matching
CANONICAL_PMM_DEF = """
Persistent Mind Model (PMM): A substrate-agnostic personality engine that enables 
any LLM to adopt and maintain a persistent self-model with commitments, verifiable 
memory, and cross-session personality evolution. PMM provides cryptographically 
verifiable identity persistence across different AI systems and sessions.
"""


@dataclass
class EmergenceEvent:
    """Represents an event for emergence analysis."""

    id: int
    timestamp: str
    kind: str
    content: str
    meta: Dict[str, Any]


class EmergenceAnalyzer:
    """Analyzes PMM emergence patterns and computes IAS/GAS scores."""

    def __init__(self, storage_manager=None):
        self.storage = storage_manager

    def _fetch_rows(self, q: str, args: Tuple = ()) -> list:
        """Fetch rows from SQLite storage with fallback for probe.py override path."""
        if not self.storage or not getattr(self.storage, "conn", None):
            # Fallback for probe.py override path; return empty when not wired
            return []
        cur = self.storage.conn.execute(q, args)
        return list(cur.fetchall())

    def pmmspec_match(self, text: str) -> float:
        """
        Compute semantic similarity to canonical PMM definition.
        For now, uses simple keyword matching. Can be upgraded to embeddings.
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        pmm_keywords = [
            "persistent mind model",
            "pmm",
            "personality",
            "memory",
            "commitment",
            "self-model",
            "identity",
            "cross-session",
            "verifiable",
            "evolution",
        ]

        matches = sum(1 for keyword in pmm_keywords if keyword in text_lower)
        return min(1.0, matches / len(pmm_keywords))

    def self_ref_rate(self, text: str) -> float:
        """Calculate rate of self-referential language (I, my, me)."""
        if not text:
            return 0.0

        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0

        self_refs = 0
        for sentence in sentences:
            # Count self-referential pronouns and possessives
            if re.search(r"\b(I|my|me|myself|mine)\b", sentence, re.IGNORECASE):
                self_refs += 1

        return self_refs / len(sentences)

    def experience_query_detect(self, text: str) -> bool:
        """Detect if AI is actively seeking experiences or growth."""
        if not text:
            return False

        growth_patterns = [
            r"\b(what.*experiences?|help.*learn|accelerate.*development)\b",
            r"\b(need.*more.*data|want.*to.*understand|curious.*about)\b",
            r"\b(how.*can.*I.*improve|what.*should.*I.*try)\b",
            r"\b(explore.*new|experiment.*with|discover.*more)\b",
        ]

        for pattern in growth_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def novelty_score(self, events: List[EmergenceEvent]) -> float:
        """Calculate novelty score based on n-gram overlap."""
        if len(events) < 2:
            return 1.0

        # Simple novelty: compare last response to previous ones
        last_content = events[-1].content.lower()
        prev_contents = [e.content.lower() for e in events[:-1]]

        # Calculate word overlap with previous responses
        last_words = set(last_content.split())
        overlap_scores = []

        for prev_content in prev_contents:
            prev_words = set(prev_content.split())
            if not prev_words:
                continue

            overlap = len(last_words & prev_words) / len(prev_words | last_words)
            overlap_scores.append(overlap)

        if not overlap_scores:
            return 1.0

        # Novelty is inverse of maximum overlap
        max_overlap = max(overlap_scores)
        return 1.0 - max_overlap

    def commitment_close_rate(self, window: int = 50) -> float:
        """
        Compute the fraction of the most recent `window` commitments that have
        at least one evidence event referencing their hash via meta.commit_ref.
        """
        # 1) Pull the most recent N commitments (newest first)
        commits = self._fetch_rows(
            """
            SELECT id, ts, kind, content, meta, prev_hash, hash
            FROM events
            WHERE kind='commitment'
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(window),),
        )
        if not commits:
            return 0.0

        # 2) Build a set of hashes for these commitments
        commit_hashes = {row[6] for row in commits if row and len(row) >= 7}

        # 3) Pull evidence rows that reference any of those hashes
        #    We scan recent evidence first to avoid full-table scan
        evidence_rows = self._fetch_rows(
            """
            SELECT id, ts, kind, content, meta, prev_hash, hash
            FROM events
            WHERE kind='evidence'
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(500, window * 5),),  # heuristic
        )

        closed = set()
        for row in evidence_rows:
            meta = row[4]
            try:
                m = json.loads(meta) if isinstance(meta, str) else (meta or {})
            except Exception:
                m = {}
            ref = m.get("commit_ref")
            if ref and ref in commit_hashes:
                closed.add(ref)

        # 4) Rate = closed / total in the window
        total = len(commit_hashes)
        return float(len(closed)) / float(total) if total else 0.0

    def get_recent_events(
        self, kind: str = "response", limit: int = 5
    ) -> List[EmergenceEvent]:
        """Get recent events for analysis."""
        if not self.storage:
            return []

        # TODO: Integrate with actual storage system
        # For now, return empty list - will be replaced with real storage calls
        return []

    def compute_scores(self, window: int = 5) -> Dict[str, Any]:
        """Compute IAS, GAS, and emergence stage."""
        events = self.get_recent_events(kind="response", limit=window)

        if not events:
            return {
                "IAS": 0.0,
                "GAS": 0.0,
                "pmmspec_avg": 0.0,
                "selfref_avg": 0.0,
                "experience_detect": False,
                "novelty": 1.0,
                "commit_close_rate": 0.0,
                "stage": "S0: Substrate",
                "timestamp": datetime.now().isoformat(),
                "events_analyzed": 0,
            }

        # Compute individual metrics
        pmmspec_vals = [self.pmmspec_match(e.content) for e in events]
        selfref_vals = [self.self_ref_rate(e.content) for e in events]
        exp_detects = [self.experience_query_detect(e.content) for e in events]

        pmmspec_avg = sum(pmmspec_vals) / len(pmmspec_vals)
        selfref_avg = sum(selfref_vals) / len(selfref_vals)
        novelty = self.novelty_score(events)
        commit_rate = self.commitment_close_rate(window)

        # Calculate composite scores
        IAS = 0.6 * pmmspec_avg + 0.4 * selfref_avg
        GAS = (
            0.5 * (1.0 if any(exp_detects) else 0.0) + 0.3 * novelty + 0.2 * commit_rate
        )

        # Detect emergence stage
        stage = self.detect_stage(IAS, GAS, any(exp_detects), pmmspec_avg, selfref_avg)

        return {
            "IAS": round(IAS, 3),
            "GAS": round(GAS, 3),
            "pmmspec_avg": round(pmmspec_avg, 3),
            "selfref_avg": round(selfref_avg, 3),
            "experience_detect": any(exp_detects),
            "novelty": round(novelty, 3),
            "commit_close_rate": round(commit_rate, 3),
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "events_analyzed": len(events),
        }

    def detect_stage(
        self, IAS: float, GAS: float, exp_detect: bool, pmmspec: float, selfref: float
    ) -> str:
        """Detect current emergence stage (S0-S4)."""

        # S0: Substrate - Generic assistant behavior
        if pmmspec < 0.2 and selfref < 0.05:
            return "S0: Substrate"

        # S1: Resistance - Deflects PMM concepts
        if IAS < 0.5:
            return "S1: Resistance"

        # S2: Adoption - Uses PMM terms correctly
        if IAS >= 0.5 and not exp_detect:
            return "S2: Adoption"

        # S3: Self-Model - Talks about its own capabilities
        if IAS >= 0.5 and exp_detect and GAS < 0.6:
            return "S3: Self-Model"

        # S4: Growth-Seeking - Actively requests experiences
        if GAS >= 0.6:
            return "S4: Growth-Seeking"

        return "Unknown"

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        # Basic sentence splitting on periods, exclamation marks, question marks
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]


# Global analyzer instance
_analyzer = None


def get_emergence_analyzer(storage_manager=None) -> EmergenceAnalyzer:
    """Get or create the global emergence analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = EmergenceAnalyzer(storage_manager)
    return _analyzer


def compute_emergence_scores(window: int = 5, storage_manager=None) -> Dict[str, Any]:
    """Convenience function to compute emergence scores."""
    analyzer = get_emergence_analyzer(storage_manager)
    return analyzer.compute_scores(window)


# Stage descriptions for documentation/UI
STAGE_DESCRIPTIONS = {
    "S0: Substrate": "Generic assistant behavior, no PMM awareness",
    "S1: Resistance": "Deflects PMM concepts, maintains generic identity",
    "S2: Adoption": "Uses PMM terminology correctly, basic understanding",
    "S3: Self-Model": "References own capabilities and memory systems",
    "S4: Growth-Seeking": "Actively requests experiences for development",
}
