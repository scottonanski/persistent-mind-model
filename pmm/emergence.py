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

# Import adaptive emergence systems
try:
    from .adaptive_emergence import AdaptiveEmergenceDetector
    from .semantic_growth import SemanticGrowthDetector
except ImportError:
    # Fallback for environments where these modules aren't available
    AdaptiveEmergenceDetector = None
    SemanticGrowthDetector = None

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

# --- Embedding helpers (paraphrase-robust scoring) ---
_PMM_DEF_EMB = None
_INTENT_SEED_EMBS = None


def _get_analyzer():
    """Lazily import the shared semantic analyzer used elsewhere in PMM."""
    try:
        from pmm.semantic_analysis import get_semantic_analyzer  # local import

        return get_semantic_analyzer()
    except Exception:
        return None


def _embed(text: str):
    """Return embedding vector (list[float]) or None on failure."""
    if not text or not str(text).strip():
        return None
    analyzer = _get_analyzer()
    if analyzer is None:
        return None
    try:
        return analyzer._get_embedding(str(text).strip())
    except Exception:
        return None


def _cosine(a, b):
    """Cosine similarity for two dense vectors (lists)."""
    try:
        import math

        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)
    except Exception:
        return 0.0


def _norm_01(x: float, lo: float = 0.20, hi: float = 0.80):
    """Clamp+scale similarity to [0,1] with a soft band."""
    try:
        xv = float(x)
    except Exception:
        return 0.0
    if xv <= lo:
        return 0.0
    if xv >= hi:
        return 1.0
    return (xv - lo) / (hi - lo)


# Seed prompts representing growth/experience-seeking intent
_INTENT_SEEDS = [
    "How can I improve my abilities over time?",
    "What experiments should I try to learn faster?",
    "Help me discover new experiences to grow.",
    "I want to understand my behavior and evolve it.",
    "What should I practice next to develop my skills?",
    "How do I accelerate my development?",
]


def _intent_centroid():
    """Build (and cache) the centroid embedding for growth-seeking intent."""
    global _INTENT_SEED_EMBS
    if _INTENT_SEED_EMBS is None:
        embs = []
        for s in _INTENT_SEEDS:
            e = _embed(s)
            if e is not None:
                embs.append(e)
        _INTENT_SEED_EMBS = embs
    if not _INTENT_SEED_EMBS:
        return None
    dim = len(_INTENT_SEED_EMBS[0])
    centroid = [0.0] * dim
    for v in _INTENT_SEED_EMBS:
        if len(v) != dim:
            continue
        for i in range(dim):
            centroid[i] += v[i]
    n = float(len(_INTENT_SEED_EMBS))
    return [x / n for x in centroid] if n > 0 else None


def experience_intent_score(text: str) -> float:
    """
    Continuous [0,1] growth/experience-seeking score via cosine similarity
    to a centroid of canonical seed prompts.
    """
    if not text:
        return 0.0
    c = _intent_centroid()
    if c is None:
        return 0.0
    emb = _embed(text)
    if emb is None:
        return 0.0
    sim = _cosine(emb, c)
    return float(round(_norm_01(sim, lo=0.20, hi=0.75), 3))


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

        # Initialize adaptive systems if available
        self._adaptive_detector = None
        self._semantic_detector = None
        if AdaptiveEmergenceDetector is not None:
            self._adaptive_detector = AdaptiveEmergenceDetector(storage_manager)
        if SemanticGrowthDetector is not None:
            self._semantic_detector = SemanticGrowthDetector()

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
        Paraphrase-robust PMM-specificness score via embedding similarity
        to the canonical PMM definition.
        """
        global _PMM_DEF_EMB
        if not text:
            return 0.0

        # Ensure reference embedding is cached
        if _PMM_DEF_EMB is None:
            ref_emb = _embed(CANONICAL_PMM_DEF)
            _PMM_DEF_EMB = ref_emb if ref_emb is not None else []

        if not _PMM_DEF_EMB:
            # Fallback if embeddings unavailable
            return 0.0

        emb = _embed(text)
        if emb is None:
            return 0.0
        sim = _cosine(emb, _PMM_DEF_EMB)
        return float(round(_norm_01(sim, lo=0.20, hi=0.80), 3))

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
        """Back-compat boolean growth/experience detector using embeddings."""
        if not text:
            return False
        try:
            return experience_intent_score(text) >= 0.5
        except Exception:
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
        # 1) Pull the most recent N commitments (newest first). Include both
        #    generic commitments and identity-style 'commitment.open' rows.
        commits_generic = self._fetch_rows(
            """
            SELECT id, ts, kind, content, meta, prev_hash, hash
            FROM events
            WHERE kind='commitment'
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(window),),
        )
        commits_identity = self._fetch_rows(
            """
            SELECT id, ts, kind, content, meta, prev_hash, hash
            FROM events
            WHERE kind='commitment.open'
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(window),),
        )
        # Merge by recency and cap to window
        commits_all = sorted(
            list(commits_generic) + list(commits_identity),
            key=lambda r: r[0],
            reverse=True,
        )[: int(window)]
        if not commits_all:
            return 0.0

        # 2) Build a set/list of hashes for these commitments
        commit_hashes = {row[6] for row in commits_all if row and len(row) >= 7}
        commit_hash_list = list(commit_hashes)

        # 3) Pull evidence rows that reference any of those hashes
        #    We scan recent evidence first to avoid full-table scan
        scan_limit = max(500, window * 5)
        evidence_rows = self._fetch_rows(
            """
            SELECT id, ts, kind, content, meta, prev_hash, hash
            FROM events
            WHERE kind='evidence' OR kind LIKE 'evidence:%'
            ORDER BY id DESC
            LIMIT ?
            """,
            (scan_limit,),
        )

        closed = set()
        # A) Canonical evidence rows with top-level meta.commit_ref
        for row in evidence_rows:
            meta = row[4]
            try:
                m = json.loads(meta) if isinstance(meta, str) else (meta or {})
            except Exception:
                m = {}
            ref = (m or {}).get("commit_ref")
            if not ref:
                continue
            if ref in commit_hashes:
                closed.add(ref)
                continue
            # Allow short-hash prefixes (e.g., first 8-16 hex chars)
            try:
                for h in commit_hash_list:
                    if (
                        isinstance(ref, str)
                        and isinstance(h, str)
                        and h.startswith(ref)
                    ):
                        closed.add(h)
                        break
            except Exception:
                pass

        # B) Back-compat: some evidence is logged as kind='event' with meta.type like 'evidence:*'
        #    and nested commit_ref under meta.evidence.commit_ref. Include those too.
        event_rows = self._fetch_rows(
            """
            SELECT id, ts, kind, content, meta, prev_hash, hash
            FROM events
            WHERE kind='event'
            ORDER BY id DESC
            LIMIT ?
            """,
            (scan_limit,),
        )
        for row in event_rows:
            meta = row[4]
            try:
                m = json.loads(meta) if isinstance(meta, str) else (meta or {})
            except Exception:
                m = {}
            t = (m or {}).get("type", "")
            if not (isinstance(t, str) and t.startswith("evidence")):
                continue
            # Prefer top-level commit_ref if present, else nested under 'evidence'
            ref = (m or {}).get("commit_ref")
            if not ref:
                ev = (m or {}).get("evidence")
                if isinstance(ev, dict):
                    ref = ev.get("commit_ref")
            if not ref:
                continue
            if ref in commit_hashes:
                closed.add(ref)
                continue
            try:
                for h in commit_hash_list:
                    if (
                        isinstance(ref, str)
                        and isinstance(h, str)
                        and h.startswith(ref)
                    ):
                        closed.add(h)
                        break
            except Exception:
                pass

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

        # Use adaptive emergence detection if available
        if self._adaptive_detector is not None and self._semantic_detector is not None:
            return self._compute_adaptive_scores(
                events, all_events, window_env, kinds, telemetry_on
            )

        # Fallback to legacy hardcoded system
        return self._compute_legacy_scores(
            events, all_events, window_env, kinds, telemetry_on
        )

    def _compute_adaptive_scores(
        self, events, all_events, window_env, kinds, telemetry_on
    ):
        """Compute scores using adaptive emergence detection."""
        # Get latest event content for analysis
        latest_content = events[-1].content if events else ""

        # Semantic analysis of growth patterns
        semantic_metrics = self._semantic_detector.analyze_growth_content(
            latest_content
        )

        # Calculate adaptive metrics
        content_complexity = self._semantic_detector.calculate_content_complexity(
            latest_content
        )
        behavioral_change = self._semantic_detector.detect_behavioral_change(
            semantic_metrics
        )
        commitment_progress = self.commitment_close_rate(window_env)
        semantic_novelty = self._semantic_detector.calculate_semantic_novelty(
            latest_content
        )

        # Calculate adaptive GAS
        GAS = self._adaptive_detector.calculate_adaptive_gas(
            content_complexity=content_complexity,
            behavioral_change=behavioral_change,
            commitment_progress=commitment_progress,
            semantic_novelty=semantic_novelty,
        )

        # Calculate adaptive IAS using semantic metrics
        pmmspec_vals = [self.pmmspec_match(e.content) for e in events]
        selfref_vals = [self.self_ref_rate(e.content) for e in events]

        pmmspec_avg = sum(pmmspec_vals) / len(pmmspec_vals)
        selfref_avg = sum(selfref_vals) / len(selfref_vals)

        # Adaptive IAS weighting based on content quality
        growth_score = semantic_metrics.get("overall_growth_score", 0.0)
        if growth_score > 0.3:
            # High growth content - weight self-reference more heavily
            IAS = 0.4 * pmmspec_avg + 0.6 * selfref_avg
        else:
            # Standard weighting
            IAS = 0.6 * pmmspec_avg + 0.4 * selfref_avg

        # Optional IAS lift when identity evidence is present
        identity_boost = 0.0
        identity_signal_count = 0
        try:
            use_extended = str(os.getenv("PMM_IAS_IDENTITY_EXTENDED", "0")).lower() in (
                "1",
                "true",
                "yes",
            )
            scan = all_events if all_events else events
            if use_extended:
                # Per-signal accumulation with cap and optional commit bonus
                try:
                    mult = float(os.getenv("PMM_IAS_IDENTITY_EVIDENCE_MULT", "0.02"))
                except Exception:
                    mult = 0.02
                try:
                    cap = float(os.getenv("PMM_IAS_IDENTITY_MAX_BOOST", "0.12"))
                except Exception:
                    cap = 0.12
                try:
                    cbonus = float(os.getenv("PMM_IAS_ID_COMMIT_BONUS", "0.03"))
                except Exception:
                    cbonus = 0.03
                identity_signal_count = self._count_identity_signals(
                    scan, window=min(window_env, 15)
                )
                identity_boost = min(identity_signal_count * mult, cap)
                if self._has_open_identity_commit(scan, window=min(window_env, 15)):
                    identity_boost = min(identity_boost + cbonus, cap)
            else:
                # Conservative single-step boost
                identity_boost = float(self._compute_identity_boost(all_events))
        except Exception:
            identity_boost = 0.0
        if identity_boost > 0:
            IAS = min(1.0, IAS + identity_boost)

        # Detect stage using adaptive thresholds
        stage, stage_confidence = self._adaptive_detector.detect_stage_transition(
            ias_score=IAS, gas_score=GAS, content_metrics=semantic_metrics
        )

        result = {
            "IAS": round(IAS, 3),
            "GAS": round(GAS, 3),
            "pmmspec_avg": round(pmmspec_avg, 3),
            "selfref_avg": round(selfref_avg, 3),
            "experience_detect": growth_score > 0.2,  # Semantic growth detection
            "novelty": round(semantic_novelty, 3),
            "commit_close_rate": round(commitment_progress, 3),
            "stage": stage,
            "stage_confidence": round(stage_confidence, 3),
            "timestamp": datetime.now().isoformat(),
            "events_analyzed": len(events),
            "kinds_considered": kinds,
            "identity_evidence": bool(identity_boost > 0),
            "identity_signal_count": int(identity_signal_count),
            "identity_boost_applied": round(identity_boost, 3),
            "adaptive_metrics": {
                "content_complexity": round(content_complexity, 3),
                "behavioral_change": round(behavioral_change, 3),
                "semantic_growth": round(growth_score, 3),
                "growth_orientation": round(
                    semantic_metrics.get("growth_orientation", 0.0), 3
                ),
                "self_reflection": round(
                    semantic_metrics.get("self_reflection", 0.0), 3
                ),
                "commitment_strength": round(
                    semantic_metrics.get("commitment_strength", 0.0), 3
                ),
                "emotional_depth": round(
                    semantic_metrics.get("emotional_depth", 0.0), 3
                ),
            },
        }

        if telemetry_on:
            print(
                f"[PMM][ADAPTIVE] IAS={result['IAS']:.3f} GAS={result['GAS']:.3f} stage={stage} confidence={stage_confidence:.3f} growth={growth_score:.3f}"
            )

        return result

    def _compute_legacy_scores(
        self, events, all_events, window_env, kinds, telemetry_on
    ):
        """Fallback to legacy hardcoded scoring system."""
        # Compute individual metrics
        pmmspec_vals = [self.pmmspec_match(e.content) for e in events]
        selfref_vals = [self.self_ref_rate(e.content) for e in events]
        exp_detects = [self.experience_query_detect(e.content) for e in events]

        pmmspec_avg = sum(pmmspec_vals) / len(pmmspec_vals)
        selfref_avg = sum(selfref_vals) / len(selfref_vals)
        novelty = self.novelty_score(events)
        commit_rate = self.commitment_close_rate(window_env)
        hint_rate = self.provisional_hint_rate(window_env)

        # Calculate composite IAS
        base_IAS = 0.6 * pmmspec_avg + 0.4 * selfref_avg

        # Optional IAS lift when identity evidence is present (extended mode behind env)
        identity_boost = 0.0
        identity_signal_count = 0
        try:
            use_extended = str(os.getenv("PMM_IAS_IDENTITY_EXTENDED", "0")).lower() in (
                "1",
                "true",
                "yes",
            )
            scan = all_events if all_events else events
            if use_extended:
                try:
                    mult = float(os.getenv("PMM_IAS_IDENTITY_EVIDENCE_MULT", "0.02"))
                except Exception:
                    mult = 0.02
                try:
                    cap = float(os.getenv("PMM_IAS_IDENTITY_MAX_BOOST", "0.12"))
                except Exception:
                    cap = 0.12
                try:
                    cbonus = float(os.getenv("PMM_IAS_ID_COMMIT_BONUS", "0.03"))
                except Exception:
                    cbonus = 0.03
                identity_signal_count = self._count_identity_signals(
                    scan, window=min(window_env, 15)
                )
                identity_boost = min(identity_signal_count * mult, cap)
                if self._has_open_identity_commit(scan, window=min(window_env, 15)):
                    identity_boost = min(identity_boost + cbonus, cap)
            else:
                identity_boost = float(self._compute_identity_boost(all_events))
        except Exception:
            identity_boost = 0.0
        IAS = max(0.0, min(1.0, base_IAS + identity_boost))

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

        # Detect stage using legacy hardcoded thresholds
        stage = self.detect_stage(IAS, GAS, any(exp_detects), pmmspec_avg, selfref_avg)

        result = {
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
            "kinds_considered": kinds,
            "identity_evidence": bool(identity_boost > 0),
            "identity_signal_count": int(identity_signal_count),
            "identity_boost_applied": round(identity_boost, 3),
        }

        if telemetry_on:
            print(
                f"[PMM][LEGACY] IAS={result['IAS']:.3f} GAS={result['GAS']:.3f} stage={stage}"
            )

        return result

    # --------------------
    # Identity signal helpers (extended mode)
    # --------------------
    def _count_identity_signals(
        self, events: List[EmergenceEvent], window: int = 15
    ) -> int:
        """Count recent identity-like signals in the last `window` events.

        Signals include tags containing 'identity' or regex matches in content.
        """
        if not events:
            return 0
        id_rx = re.compile(
            r"\b(my name is|i am\s+\w+|identity confirm|identity:|i am quest|name is)\b",
            re.IGNORECASE,
        )
        tail = events[-window:]
        cnt = 0
        for e in tail:
            try:
                text = e.content or ""
                tags = (e.meta or {}).get("tags", [])
                kind = (e.kind or "").lower()
            except Exception:
                text, tags, kind = "", [], ""
            if kind in (
                "evidence",
                "event",
                "response",
                "self_expression",
                "commitment",
                "commitment.open",
            ):
                if ("identity" in tags) or id_rx.search(text):
                    cnt += 1
        return cnt

    def _has_open_identity_commit(
        self, events: List[EmergenceEvent], window: int = 15
    ) -> bool:
        """Heuristic: detect if a recent commitment.open mentions identity."""
        if not events:
            return False
        id_rx = re.compile(
            r"\b(identity|identity confirm|adopt(ed)? name|i am\s+\w+)\b", re.IGNORECASE
        )
        for e in events[-window:]:
            if (e.kind or "").lower() == "commitment.open":
                text = e.content or ""
                tags = (e.meta or {}).get("tags", [])
                if ("identity" in tags) or id_rx.search(text):
                    return True
        return False

    def _compute_identity_boost(self, events: List[EmergenceEvent]) -> float:
        """Detect identity evidence in recent events and return a small IAS boost.

        Heuristics:
        - Strong signals: recent 'identity_change' or 'identity_update' events.
        - Evidence signals: recent 'evidence' with JSON content summary like
          'Identity adopted: <name>' (from behavior_engine).

        The applied boost is controlled via env var PMM_IAS_IDENTITY_BOOST (default 0.06)
        and scaled by 1.0 for strong signals, 0.6 for evidence signals.
        """
        if not events:
            return 0.0

        try:
            base_boost = float(os.getenv("PMM_IAS_IDENTITY_BOOST", "0.06"))
        except Exception:
            base_boost = 0.06

        strong = False
        evidence = False

        for e in events:
            k = (e.kind or "").lower()
            if k in ("identity_change", "identity_update"):
                strong = True
                break
            if k.startswith("evidence"):
                # Try to parse JSON content; fallback to substring check
                s = (e.content or "").strip()
                summary = ""
                if s.startswith("{") and s.endswith("}"):
                    try:
                        obj = json.loads(s)
                        summary = str(obj.get("summary", ""))
                    except Exception:
                        summary = s
                else:
                    summary = s
                sl = summary.lower()
                if "identity adopted" in sl or (
                    ("name" in sl and ("adopt" in sl or "now" in sl))
                ):
                    evidence = True

        if strong:
            return max(0.0, min(0.2, base_boost * 1.0))
        if evidence:
            return max(0.0, min(0.2, base_boost * 0.6))
        return 0.0

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
