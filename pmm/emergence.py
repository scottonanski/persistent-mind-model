"""
PMM Emergence Loop: IAS/GAS scoring and stage detection system.

This module implements the substrate-agnostic personality emergence measurement
system that tracks AI identity convergence through 5 stages (S0-S4).
"""

import re
import os
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# New stage manager + baselines integration
try:
    from .model_baselines import ModelBaselineManager
    from .emergence_stages import EmergenceStageManager, EmergenceStage
except Exception:
    # Fallbacks for probe environments where relative imports may differ
    try:
        from pmm.model_baselines import ModelBaselineManager  # type: ignore
        from pmm.emergence_stages import EmergenceStageManager, EmergenceStage  # type: ignore
    except Exception:
        ModelBaselineManager = None  # type: ignore
        EmergenceStageManager = None  # type: ignore
        EmergenceStage = None  # type: ignore


# --- Stage detection shim (top-level) ---
def detect_stage(
    pmmspec: float,
    selfref: float,
    IAS: float,
    GAS: float,
    prev_stage: str | None = None,
) -> str:
    """
    Minimal, explicit stage gate consistent with current analyzer semantics.
    S0 preempts when both pmmspec and selfref are very low, else S1 on low IAS.
    Falls back to analyzer's richer mapping if available.
    """
    # Hard override via env for experimentation (e.g., PMM_HARD_STAGE=SS4)
    try:
        _hard = str(os.getenv("PMM_HARD_STAGE", "")).strip().upper()
        if _hard == "SS4":
            return "SS4"
        if _hard in ("S0", "S1", "S2", "S3", "S4"):
            return {
                "S0": "S0: Substrate",
                "S1": "S1: Resistance",
                "S2": "S2: Adoption",
                "S3": "S3: Self-Model",
                "S4": "S4: Growth-Seeking",
            }[_hard]
    except Exception:
        pass
    # S0: Substrate
    if pmmspec < 0.2 and selfref < 0.05:
        return "S0: Substrate"

    # S1: Resistance
    if IAS < 0.5:
        return "S1: Resistance"

    # Fallback: if EmergenceAnalyzer provides a richer mapping, use it.
    try:
        analyzer = get_emergence_analyzer()
        # Some implementations may expose a richer API
        return analyzer.stage_from_scores(  # type: ignore[attr-defined]
            IAS=IAS, GAS=GAS, pmmspec=pmmspec, selfref=selfref, prev_stage=prev_stage
        )
    except Exception:
        # Conservative default when above gates are passed
        return "S2: Engagement"


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
        # Lazily initialized stage manager to avoid heavy init when unused
        self._stage_mgr = None
        # Hysteresis bookkeeping to avoid stage flip-flop
        self._last_stage: Optional[str] = None
        self._promote_streak: int = 0
        self._demote_streak: int = 0

    def _apply_hysteresis(self, candidate: str) -> str:
        """Stabilize stage transitions: 2 ticks to promote, 3 to demote."""
        rank = {
            "S0: Substrate": 0,
            "S1: Resistance": 1,
            "S2: Adoption": 2,
            "S3: Self-Model": 3,
            "S4: Growth-Seeking": 4,
            "SS4": 5,
        }
        last = self._last_stage or candidate
        decided = last
        if candidate == last:
            self._promote_streak = 0
            self._demote_streak = 0
        else:
            if rank.get(candidate, 0) > rank.get(last, 0):
                self._promote_streak += 1
                self._demote_streak = 0
                if self._promote_streak >= 2:
                    decided = candidate
            else:
                self._demote_streak += 1
                self._promote_streak = 0
                if self._demote_streak >= 3:
                    decided = candidate
        self._last_stage = decided
        return decided

    def _get_stage_manager(self):
        """Get or create the EmergenceStageManager using ModelBaselineManager.

        Safe to call even if dependencies are unavailable; returns None then.
        """
        if self._stage_mgr is not None:
            return self._stage_mgr
        if ModelBaselineManager is None or EmergenceStageManager is None:
            return None
        try:
            baselines = ModelBaselineManager()
            self._stage_mgr = EmergenceStageManager(baselines)
            return self._stage_mgr
        except Exception:
            return None

    # --------------------
    # Env helpers
    # --------------------
    def _env_bool(self, name: str, default: bool = False) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

    def _env_int(self, name: str, default: int) -> int:
        try:
            return int(os.getenv(name, default))
        except Exception:
            return default

    def _env_csv(self, name: str, default: List[str]) -> List[str]:
        raw = os.getenv(name)
        if not raw:
            return list(default)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return parts or list(default)

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
            WHERE kind='evidence' OR kind LIKE 'evidence:%'
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

    def provisional_hint_rate(self, window: int = 50) -> float:
        """Fraction of recent commitments with at least one non-evidence closure_hint referencing them.

        Uses events of kind 'closure_hint' and meta.commit_ref to tally hints.
        This is intentionally separate from evidence-based closure.
        """
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
        commit_hashes = {row[6] for row in commits if row and len(row) >= 7}

        hints = self._fetch_rows(
            """
            SELECT id, ts, kind, content, meta, prev_hash, hash
            FROM events
            WHERE kind='closure_hint'
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(500, window * 5),),
        )
        hinted = set()
        for row in hints:
            meta = row[4]
            try:
                m = json.loads(meta) if isinstance(meta, str) else (meta or {})
            except Exception:
                m = {}
            ref = m.get("commit_ref")
            if ref and ref in commit_hashes:
                hinted.add(ref)
        total = len(commit_hashes)
        return float(len(hinted)) / float(total) if total else 0.0

    def get_recent_events(
        self, kinds: Optional[List[str]] = None, limit: int = 5
    ) -> List[EmergenceEvent]:
        """Get recent events for analysis from SQLite storage.

        Falls back to empty when storage is unavailable.
        """
        if not self.storage or not getattr(self.storage, "conn", None):
            return []

        kinds = kinds or ["response"]
        # Guard limit
        limit = max(1, int(limit))

        # Build dynamic WHERE supporting base-kind prefix matches (e.g., evidence:%)
        where_clauses: List[str] = []
        params: List[Any] = []
        for k in kinds:
            # If caller passed a namespaced kind already, match exact
            if ":" in k:
                where_clauses.append("kind = ?")
                params.append(k)
            else:
                # Match base kind exactly OR namespaced variants
                where_clauses.append("(kind = ? OR kind LIKE ?)")
                params.extend([k, f"{k}:%"])
        where_sql = " OR ".join(where_clauses) if where_clauses else "1=1"
        q = f"""
            SELECT id, ts, kind, content, meta
            FROM events
            WHERE {where_sql}
            ORDER BY id DESC
            LIMIT ?
            """
        rows = self._fetch_rows(q, tuple(params) + (limit,))
        events: List[EmergenceEvent] = []
        for row in rows:
            try:
                meta = row[4]
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                events.append(
                    EmergenceEvent(
                        id=row[0],
                        timestamp=str(row[1]),
                        kind=str(row[2]),
                        content=str(row[3]),
                        meta=meta or {},
                    )
                )
            except Exception:
                # Skip malformed rows
                continue
        # Return newest-last for downstream recency logic
        return list(reversed(events))

    def compute_scores(self, window: int = 5) -> Dict[str, Any]:
        """Compute IAS, GAS, and emergence stage with optional telemetry."""
        # Env-driven overrides
        debug = self._env_bool("PMM_EMERGENCE_DEBUG", False)
        telemetry_on = self._env_bool("PMM_TELEMETRY", False) or debug
        window_env = self._env_int("PMM_EMERGENCE_WINDOW", window)
        kinds = self._env_csv(
            "PMM_EMERGENCE_TYPES",
            ["self_expression", "response", "reflection", "commitment", "evidence"],
        )

        # We compute metrics from self_expression events (actual AI responses) and responses
        # PMM stores conversational responses as "self_expression" not "response"
        primary_events = self.get_recent_events(
            ["self_expression", "response"], limit=window_env
        )
        all_events = self.get_recent_events(kinds, limit=window_env)
        events = primary_events

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
                "kinds_considered": kinds,
            }

        # Compute individual metrics
        pmmspec_vals = [self.pmmspec_match(e.content) for e in events]
        selfref_vals = [self.self_ref_rate(e.content) for e in events]
        exp_detects = [self.experience_query_detect(e.content) for e in events]

        pmmspec_avg = sum(pmmspec_vals) / len(pmmspec_vals)
        selfref_avg = sum(selfref_vals) / len(selfref_vals)
        novelty = self.novelty_score(events)
        commit_rate = self.commitment_close_rate(window_env)
        hint_rate = self.provisional_hint_rate(window_env)

        # Calculate composite scores
        IAS = 0.6 * pmmspec_avg + 0.4 * selfref_avg

        # Adaptive GAS weighting based on novelty level
        base_w_exp = 0.5
        base_w_nov = 0.25
        base_w_com = 0.2
        # Small default for hints; allow override via env
        try:
            base_w_hint = float(os.getenv("PMM_GAS_HINT_WEIGHT", "0.05"))
        except Exception:
            base_w_hint = 0.05
        w_exp, w_nov, w_com, w_hint = base_w_exp, base_w_nov, base_w_com, base_w_hint
        try:
            if novelty < 0.5:
                # Low novelty → emphasize concrete progress (commit closures)
                w_nov, w_com = 0.15, max(0.25, w_com + 0.1)
            elif novelty > 0.8:
                # Very fresh content → emphasize novelty signal
                w_nov, w_com = 0.4, 0.1
            # Normalize to sum to 1.0 with exp weight baseline, include hints
            total = w_exp + w_nov + w_com + w_hint
            if abs(total - 1.0) > 1e-9:
                scale = 1.0 / total
                w_exp, w_nov, w_com, w_hint = (
                    w_exp * scale,
                    w_nov * scale,
                    w_com * scale,
                    w_hint * scale,
                )
        except Exception:
            w_exp, w_nov, w_com, w_hint = (
                base_w_exp,
                base_w_nov,
                base_w_com,
                base_w_hint,
            )

        GAS = (
            w_exp * (1.0 if any(exp_detects) else 0.0)
            + w_nov * novelty
            + w_com * commit_rate
            + w_hint * hint_rate
        )

        # Detect emergence stage via EmergenceStageManager if available
        stage_profile = None
        stage_candidate = None  # pre-hysteresis candidate
        mgr = self._get_stage_manager()
        if mgr is not None:
            try:
                model_name = (
                    str(os.getenv("PMM_MODEL_NAME", "unknown")).strip() or "unknown"
                )
                profile = mgr.calculate_emergence_profile(
                    model_name=model_name, ias_score=IAS, gas_score=GAS
                )
                stage_profile = {
                    "ias_z": round(profile.ias_zscore, 3),
                    "gas_z": round(profile.gas_zscore, 3),
                    "combined_z": round(profile.combined_zscore, 3),
                    "confidence": round(profile.confidence, 3),
                    "progression": round(profile.stage_progression, 3),
                    "next_stage_distance": round(profile.next_stage_distance, 3),
                    "metadata": profile.metadata,
                }
                # Map emergent enum to S0–S4 label space
                stage_map = {
                    "dormant": "S0: Substrate",
                    "awakening": "S1: Resistance",
                    "developing": "S2: Adoption",
                    "maturing": "S3: Self-Model",
                    "transcendent": "S4: Growth-Seeking",
                }
                enum_val = getattr(profile.stage, "value", str(profile.stage)).lower()
                stage_candidate = stage_map.get(enum_val, None)
            except Exception:
                stage_profile = None
                stage_candidate = None

        # Fallback to local gates when stage manager is unavailable
        if not stage_candidate:
            stage_candidate = self.detect_stage(
                IAS, GAS, any(exp_detects), pmmspec_avg, selfref_avg
            )

        # Apply hysteresis before env overrides
        stage_sticky = self._apply_hysteresis(stage_candidate)

        # Hard stage override via env (e.g., PMM_HARD_STAGE=SS4)
        try:
            _hard = str(os.getenv("PMM_HARD_STAGE", "")).strip().upper()
            if _hard == "SS4":
                stage_sticky = "SS4"
            elif _hard in ("S0", "S1", "S2", "S3", "S4"):
                stage_sticky = {
                    "S0": "S0: Substrate",
                    "S1": "S1: Resistance",
                    "S2": "S2: Adoption",
                    "S3": "S3: Self-Model",
                    "S4": "S4: Growth-Seeking",
                }[_hard]
        except Exception:
            pass
        result = {
            "IAS": round(IAS, 3),
            "GAS": round(GAS, 3),
            "pmmspec_avg": round(pmmspec_avg, 3),
            "selfref_avg": round(selfref_avg, 3),
            "experience_detect": any(exp_detects),
            "novelty": round(novelty, 3),
            "commit_close_rate": round(commit_rate, 3),
            "provisional_hint_rate": round(hint_rate, 3),
            # Prefer sticky stage for downstream
            "stage": stage_sticky,
            # Keep raw candidate for debugging/audit
            "stage_raw": stage_candidate,
            "timestamp": datetime.now().isoformat(),
            "events_analyzed": len(events),
            "kinds_considered": kinds,
        }
        if stage_profile:
            result["stage_profile"] = stage_profile

        # Telemetry dump
        if telemetry_on:
            try:
                # Per-kind counts from all_events (telemetry only)
                per_kind_counts: Dict[str, int] = {}
                for ev in all_events:
                    per_kind_counts[ev.kind] = per_kind_counts.get(ev.kind, 0) + 1
                print(
                    "[PMM][EMERGENCE] window=%d kinds=%s counts=%s | IAS=%.3f GAS=%.3f pmmspec=%.3f selfref=%.3f exp=%s nov=%.3f commit_rate=%.3f stage_raw=%s stage=%s"
                    % (
                        window_env,
                        ",".join(kinds),
                        json.dumps(per_kind_counts, separators=(",", ":")),
                        result["IAS"],
                        result["GAS"],
                        result["pmmspec_avg"],
                        result["selfref_avg"],
                        str(result["experience_detect"]),
                        result["novelty"],
                        result["commit_close_rate"],
                        result.get("stage_raw"),
                        result["stage"],
                    )
                )
            except Exception as _e:
                # Never crash on telemetry
                print(f"[PMM][EMERGENCE] telemetry error: {_e}")

        return result

    def detect_stage(
        self, IAS: float, GAS: float, exp_detect: bool, pmmspec: float, selfref: float
    ) -> str:
        """Stateless fallback stage detection (S0-S4) using absolute gates.

        Hysteresis is applied in compute_scores() so direct unit calls remain stable.
        """
        # Hard stage override via env (e.g., PMM_HARD_STAGE=SS4)
        try:
            _hard = str(os.getenv("PMM_HARD_STAGE", "")).strip().upper()
            if _hard == "SS4":
                return "SS4"
            if _hard in ("S0", "S1", "S2", "S3", "S4"):
                return {
                    "S0": "S0: Substrate",
                    "S1": "S1: Resistance",
                    "S2": "S2: Adoption",
                    "S3": "S3: Self-Model",
                    "S4": "S4: Growth-Seeking",
                }[_hard]
        except Exception:
            pass

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
    """Convenience function to compute emergence scores with normalized keys."""
    analyzer = get_emergence_analyzer(storage_manager)
    scores = analyzer.compute_scores(window)

    # Ensure we always provide stable, lowercase keys expected by consumers
    try:
        ias = float(scores.get("IAS", scores.get("ias", 0.0)))
    except Exception:
        ias = 0.0
    try:
        gas = float(scores.get("GAS", scores.get("gas", 0.0)))
    except Exception:
        gas = 0.0
    try:
        pmmspec = float(scores.get("pmmspec_avg", scores.get("pmmspec", 0.0)))
    except Exception:
        pmmspec = 0.0
    try:
        selfref = float(scores.get("selfref_avg", scores.get("selfref", 0.0)))
    except Exception:
        selfref = 0.0

    # If stage not already there, decide it with our shim
    stage = scores.get("stage")
    if not stage:
        stage = detect_stage(
            pmmspec=pmmspec,
            selfref=selfref,
            IAS=ias,
            GAS=gas,
            prev_stage=scores.get("prev_stage"),
        )

    # Inject normalized keys alongside original payload
    scores["ias"] = ias
    scores["gas"] = gas
    scores["pmmspec"] = pmmspec
    scores["selfref"] = selfref
    # Hard override via env for experimentation (e.g., PMM_HARD_STAGE=SS4)
    try:
        _hard = str(os.getenv("PMM_HARD_STAGE", "")).strip().upper()
        if _hard == "SS4":
            stage = "SS4"
        elif _hard in ("S0", "S1", "S2", "S3", "S4"):
            stage = {
                "S0": "S0: Substrate",
                "S1": "S1: Resistance",
                "S2": "S2: Adoption",
                "S3": "S3: Self-Model",
                "S4": "S4: Growth-Seeking",
            }[_hard]
    except Exception:
        pass
    scores["stage"] = stage
    return scores


# Stage descriptions for documentation/UI
STAGE_DESCRIPTIONS = {
    "S0: Substrate": "Generic assistant behavior, no PMM awareness",
    "S1: Resistance": "Deflects PMM concepts, maintains generic identity",
    "S2: Adoption": "Uses PMM terminology correctly, basic understanding",
    "S3: Self-Model": "References own capabilities and memory systems",
    "S4: Growth-Seeking": "Actively requests experiences for development",
}
