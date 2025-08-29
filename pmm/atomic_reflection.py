# pmm/atomic_reflection.py
from __future__ import annotations
from typing import List, Dict, Any
from collections import OrderedDict
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import uuid
from pmm.config.models import (
    get_min_embedding_threshold,
    get_threshold_cooldown_turns,
)
from pmm.logging_config import pmm_tlog, pmm_dlog


@dataclass
class ReflectionCandidate:
    """A reflection candidate for validation and persistence."""

    content: str
    timestamp: datetime
    source_events: List[str]
    model_config: Dict[str, Any]
    epoch: int


class AtomicReflectionManager:
    """Manages atomic reflection validation → dedup → persist pipeline."""

    def __init__(self, pmm_manager, embedding_threshold: float | None = None):
        self.pmm = pmm_manager
        # Allow override via env var; default to a more permissive 0.94
        # so that only very similar insights are rejected by embeddings.
        env_threshold = os.getenv("PMM_EMBEDDING_THRESHOLD")
        if env_threshold is not None:
            try:
                self.embedding_threshold = float(env_threshold)
            except ValueError:
                # Fallback if env var is malformed
                self.embedding_threshold = 0.94
        else:
            self.embedding_threshold = (
                float(embedding_threshold) if embedding_threshold is not None else 0.94
            )
        self._lock = threading.Lock()

        # Adaptive dedup configuration
        flag = os.getenv("PMM_ADAPTIVE_DEDUP")
        # Enable adaptive dedup by default; allow env to explicitly disable
        if flag is None:
            self._adaptive_enabled = True
        else:
            self._adaptive_enabled = flag.lower() in ("1", "true", "yes", "on")
        # Stage-adaptive blending (separate toggle; enabled by default)
        stage_flag = os.getenv("PMM_STAGE_ADAPTIVE_DEDUP")
        self._stage_adaptive_enabled = (
            True
            if stage_flag is None
            else stage_flag.lower() in ("1", "true", "yes", "on")
        )
        # Start with configured threshold and adapt within bounds
        self._effective_threshold = self.embedding_threshold
        # Min threshold is env-driven to avoid over-aggressive dedup
        self._min_thresh = float(get_min_embedding_threshold())
        self._max_thresh = 0.95
        self._step = 0.02
        self._accept_streak = 0
        self._reject_streak = 0
        # Cooldown turns between adjustments to prevent oscillation
        self._cooldown_turns = int(get_threshold_cooldown_turns())
        self._cooldown_remaining = 0

        # Cache recent insights for fast text similarity
        self._recent_insights_cache = []
        self._max_cache_size = 10

        # Lightweight LRU cache for embeddings to reduce API calls
        self._embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._embedding_cache_max = 64

        # Global switches (env-driven) for experimentation and debugging
        # - PMM_DISABLE_EMBEDDING_DEDUP: when truthy, skip embedding-based duplicate checks
        # - PMM_FORCE_ACCEPT_NEXT_INSIGHT: when truthy, accept the next insight bypassing
        #   dedup gates (one-shot). Cleared upon use.
        try:
            flag = os.getenv("PMM_DISABLE_EMBEDDING_DEDUP", "").lower()
            self._disable_embedding_dedup = flag in ("1", "true", "yes", "on")
        except Exception:
            self._disable_embedding_dedup = False

    def _get_emergence_context(self) -> dict:
        """
        Robust context fetcher that won't explode if a legacy accessor is missing.
        Tries self_model.get_context(); falls back to compute_emergence_scores().
        """
        ctx: dict = {}
        try:
            smm = getattr(self, "self_model", None)

            # Preferred modern accessor
            if smm and hasattr(smm, "get_context"):
                try:
                    maybe = smm.get_context()
                    ctx = maybe if isinstance(maybe, dict) else {}
                except Exception:
                    ctx = {}

            # Fallback to on-demand emergence snapshot
            if not ctx:
                try:
                    from pmm.emergence import compute_emergence_scores

                    ctx = compute_emergence_scores(
                        window=5,
                        storage_manager=getattr(self, "storage_manager", None),
                    )
                    if not isinstance(ctx, dict):
                        ctx = {}
                except Exception:
                    ctx = {}

            return ctx
        except Exception:
            return {}

    def add_insight(
        self, insight_content: str, model_config: Dict[str, Any], epoch: int
    ) -> bool:
        """
        Atomic insight addition: validate → dedup → persist.
        Returns True if insight was added, False if rejected.
        """
        with self._lock:
            # Step 3: Check if we're in a hot context and ease gates accordingly
            hot_context = False
            hot_strength = 0.0
            try:
                from pmm.emergence import compute_emergence_scores
                em = compute_emergence_scores(window=15, storage_manager=getattr(self.pmm, "sqlite_store", None)) or {}
                gas_now = float(em.get("GAS", 0.0) or 0.0)
                close_now = float(em.get("commit_close_rate", 0.0) or 0.0)
                
                # Use the same hot_strength computation as bandit
                from pmm.policy.bandit import compute_hot_strength
                hot_strength = compute_hot_strength(gas_now, close_now)
                hot_context = hot_strength >= 0.5
            except Exception:
                pass
            
            # Stage-adapt the effective threshold up front so subsequent dedup logic uses it
            try:
                self._apply_stage_adaptation()
            except Exception:
                # Never block on adaptation failures
                pass
            
            # Step 3: Ease dedup floor when hot
            if hot_context:
                dedup_floor_hot = float(os.getenv("PMM_REFLECT_DEDUP_FLOOR_HOT", "0.88"))
                if self._effective_threshold > dedup_floor_hot:
                    old_threshold = self._effective_threshold
                    self._effective_threshold = max(dedup_floor_hot, self._min_thresh)
                    if os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on"):
                        print(f"[PMM_TELEMETRY] dedup_hot_ease: threshold {old_threshold:.3f} -> {self._effective_threshold:.3f}, hot_strength={hot_strength:.3f}")
            
            # Step 3: Check minimum tokens when hot
            if hot_context:
                min_tokens_hot = int(os.getenv("PMM_REFLECT_MIN_TOKENS_HOT", "45"))
                token_count = len(insight_content.split())
                if token_count < min_tokens_hot:
                    if os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on"):
                        print(f"[PMM_TELEMETRY] reflection_attempt: decision=rejected, reason=min_tokens_hot, tokens={token_count}, min_required={min_tokens_hot}, hot_strength={hot_strength:.3f}")
                    return False
            # Snapshot emergence stage for decision logging
            try:
                _ctx = self._get_emergence_context()
                _stage_for_logs = str(_ctx.get("stage", "")).strip()
            except Exception:
                _stage_for_logs = ""
            # Step 1: Basic validation (with normalization to reduce style boilerplate)
            cleaned_content = self._clean_and_normalize(insight_content)
            if not self._passes_basic_validation(cleaned_content):
                pmm_dlog("🔍 DEBUG: Insight failed basic validation")
                # Step 6: Enhanced telemetry for rejection reasons
                if os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on"):
                    print(f"[PMM_TELEMETRY] reflection_attempt: decision=rejected, reason=basic_validation, hot_strength={hot_strength:.3f}")
                return False

            # Step 2: Fast text similarity check
            if self._is_duplicate_text(cleaned_content):
                pmm_dlog("🔍 DEBUG: Insight rejected - duplicate text similarity")
                # Step 6: Enhanced telemetry for duplicate rejection
                if os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on"):
                    print(f"[PMM_TELEMETRY] reflection_attempt: decision=rejected, reason=duplicate_text, hot_strength={hot_strength:.3f}")
                return False

            # Step 2.5: One-shot force-accept override (env-controlled)
            try:
                force_once = os.getenv("PMM_FORCE_ACCEPT_NEXT_INSIGHT", "").lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
            except Exception:
                force_once = False

            if force_once:
                # Clear the env flag so it's one-shot. Best-effort; ignore failures.
                try:
                    os.environ.pop("PMM_FORCE_ACCEPT_NEXT_INSIGHT", None)
                except Exception:
                    pass

                telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                # Proceed to epoch validation then persist without dedup gates
                if telemetry:
                    print(
                        "[PMM_TELEMETRY] dedup_override: reason=force_accept_once, gates=skipped(text_ok, embedding_skipped)"
                    )
                # Step 4: Epoch validation (ensure model hasn't changed)
                from pmm.llm_factory import get_llm_factory

                current_epoch = get_llm_factory().get_current_epoch()
                if epoch != current_epoch:
                    pmm_dlog(
                        f"🔍 DEBUG: Insight rejected - epoch mismatch {epoch} != {current_epoch}"
                    )
                    return False
                try:
                    success = self._persist_insight(cleaned_content, model_config)
                    if success:
                        self._update_cache(cleaned_content)
                        pmm_dlog(
                            "🔍 DEBUG: Insight successfully persisted (force-accept)"
                        )
                        self._on_decision(accepted=True)
                        return True
                    else:
                        pmm_dlog(
                            "🔍 DEBUG: Insight persistence failed (force-accept path)"
                        )
                        return False
                except Exception as e:
                    pmm_dlog(f"🔍 DEBUG: Insight persistence error (force-accept): {e}")
                    return False

            # Step 2.6: Automatic bootstrap acceptance (no env required)
            # If there are no existing insights, or we've had a short rejection streak with
            # no recent acceptances, allow a single acceptance to break stalemates.
            try:
                no_insights = not bool(self.pmm.model.self_knowledge.insights)
            except Exception:
                no_insights = False

            bootstrap_allowed = no_insights or (
                self._accept_streak == 0 and self._reject_streak >= 3
            )

            if bootstrap_allowed:
                telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                reason = (
                    "bootstrap_no_insights"
                    if no_insights
                    else "bootstrap_rejection_streak"
                )
                if telemetry:
                    print(
                        f"[PMM_TELEMETRY] dedup_override: reason={reason}, gates=skipped(text_ok, embedding_skipped)"
                    )
                # Step 4: Epoch validation
                from pmm.llm_factory import get_llm_factory

                current_epoch = get_llm_factory().get_current_epoch()
                if epoch != current_epoch:
                    print(
                        f"🔍 DEBUG: Insight rejected - epoch mismatch {epoch} != {current_epoch}"
                    )
                    return False

                try:
                    success = self._persist_insight(cleaned_content, model_config)
                    if success:
                        self._update_cache(cleaned_content)
                        pmm_dlog("🔍 DEBUG: Insight successfully persisted (bootstrap)")
                        self._on_decision(accepted=True)
                        return True
                    else:
                        pmm_dlog(
                            "🔍 DEBUG: Insight persistence failed (bootstrap path)"
                        )
                        return False
                except Exception as e:
                    pmm_dlog(f"🔍 DEBUG: Insight persistence error (bootstrap): {e}")
                    return False

            # Step 3: Embedding similarity check (more expensive), with first-hit pass for borderline cases
            # Allow global switch to disable embedding dedup for debugging/experiments
            if self._disable_embedding_dedup:
                dup_check = (False, -1.0)
                telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                if telemetry:
                    print(
                        f"[PMM_TELEMETRY] dedup_override: reason=embedding_dedup_disabled, threshold={self._effective_threshold:.3f}"
                    )
            else:
                dup_check = self._is_duplicate_embedding(
                    cleaned_content, return_best_sim=True
                )
            if isinstance(dup_check, tuple):
                is_dup, best_sim = dup_check
            else:
                is_dup, best_sim = bool(dup_check), None

            if is_dup:
                # Allow a single borderline near-duplicate if we've had no recent accepted insights
                if self._first_hit_pass_allowed(best_sim):
                    telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                    if telemetry:
                        print(
                            f"[PMM_TELEMETRY] dedup_override: reason=first_hit_pass, sim={(best_sim if best_sim is not None else -1):.3f}, threshold={self._effective_threshold:.3f}, stage={_stage_for_logs}"
                        )
                else:
                    # Before final rejection, if we're in early stages (S0/S1) and appear stuck, apply a one-shot threshold drop
                    try:
                        _stage_lower = (_stage_for_logs or "").lower()
                        _stuck = self._reject_streak >= 2
                        if (
                            (
                                _stage_lower.startswith("s0")
                                or _stage_lower.startswith("s1")
                            )
                            and _stuck
                            and best_sim is not None
                        ):
                            old_thr = self._effective_threshold
                            new_thr = max(self._min_thresh, round(old_thr * 0.95, 4))
                            if os.getenv("PMM_TELEMETRY", "").lower() in (
                                "1",
                                "true",
                                "yes",
                                "on",
                            ):
                                print(
                                    f"[PMM_TELEMETRY] dedup_adapt: reason=stuck_drop, stage={_stage_for_logs}, reject_streak={self._reject_streak}, threshold {old_thr:.3f} -> {new_thr:.3f}, sim={best_sim:.3f}"
                                )
                            self._effective_threshold = new_thr
                            # Re-evaluate duplicate after adaptive drop
                            if best_sim <= self._effective_threshold:
                                # Treat as not duplicate; proceed
                                if os.getenv("PMM_TELEMETRY", "").lower() in (
                                    "1",
                                    "true",
                                    "yes",
                                    "on",
                                ):
                                    print(
                                        f"[PMM_TELEMETRY] dedup_override: reason=post_drop_accept, sim={best_sim:.3f}, threshold={self._effective_threshold:.3f}, stage={_stage_for_logs}"
                                    )
                            else:
                                # Still duplicate; reject
                                pmm_dlog(
                                    f"🔍 DEBUG: Insight rejected - embedding similarity {best_sim:.3f} > threshold {self._effective_threshold:.3f} (stage={_stage_for_logs})"
                                )
                                self._on_decision(accepted=False)
                                return False
                        else:
                            # Regular rejection path
                            pmm_dlog(
                                f"🔍 DEBUG: Insight rejected - embedding similarity {best_sim:.3f} > threshold {self._effective_threshold:.3f} (stage={_stage_for_logs})"
                            )
                            # Adaptive: rejection event
                            self._on_decision(accepted=False)
                            return False
                    except Exception:
                        pmm_dlog(
                            f"🔍 DEBUG: Insight rejected - embedding similarity {best_sim if best_sim is not None else -1:.3f} > threshold {self._effective_threshold:.3f} (stage={_stage_for_logs})"
                        )
                        self._on_decision(accepted=False)
                        return False

            # Step 4: Epoch validation (ensure model hasn't changed)
            from pmm.llm_factory import get_llm_factory

            current_epoch = get_llm_factory().get_current_epoch()
            if epoch != current_epoch:
                pmm_dlog(
                    f"🔍 DEBUG: Insight rejected - epoch mismatch {epoch} != {current_epoch}"
                )
                return False

            # Step 5: Atomic persistence
            try:
                success = self._persist_insight(cleaned_content, model_config)
                if success:
                    self._update_cache(cleaned_content)
                    pmm_dlog(
                        f"🔍 DEBUG: Insight successfully persisted (stage={_stage_for_logs})"
                    )
                    # Adaptive: acceptance event
                    self._on_decision(accepted=True)
                    return True
                else:
                    pmm_dlog("🔍 DEBUG: Insight persistence failed")
                    return False
            except Exception as e:
                pmm_dlog(f"🔍 DEBUG: Insight persistence error: {e}")
                return False

    def _clean_and_normalize(self, content: str) -> str:
        """Clean and normalize insight content."""
        if not content:
            return ""

        # Apply stance filter
        from pmm.stance_filter import StanceFilter

        stance_filter = StanceFilter()
        filtered_content, _ = stance_filter.filter_response(content)

        # Normalize whitespace
        import re

        normalized = re.sub(r"\s+", " ", filtered_content.strip())

        # Light style normalization to reduce false dedup (remove stock openers/closers)
        lowered = normalized.lower()
        boilerplate_prefixes = (
            "reflection:",
            "micro-reflection:",
            "insight:",
            "i realize that ",
            "i've realized that ",
            "i am noticing that ",
            "i notice that ",
        )
        for pref in boilerplate_prefixes:
            if lowered.startswith(pref):
                normalized = normalized[len(pref) :].lstrip()
                lowered = normalized.lower()
                break

        boilerplate_suffixes = (
            " this suggests i should ",
            " therefore, i will ",
        )
        for suff in boilerplate_suffixes:
            if lowered.endswith(suff):
                normalized = normalized[: -len(suff)].rstrip()
                lowered = normalized.lower()
                break

        return normalized

    def _passes_basic_validation(self, content: str) -> bool:
        """Basic validation checks."""
        if not content or len(content.strip()) < 10:
            return False

        if len(content) > 2000:  # Too long
            return False

        # Check for obvious spam patterns
        words = content.lower().split()
        if len(set(words)) < len(words) * 0.3:  # Too repetitive
            return False

        # Hygiene: block coach-like meta-prescriptions that cause spammy commitments
        # Examples to block: "ask a probing question every (turn|message)",
        # "deepen conversations each turn", "i should ask a question every time"
        try:
            import re

            banned_patterns = (
                r"\bask (?:a|more) (?:probing|deeper) question(?:s)? (?:every|each) (?:turn|message|reply)\b",
                r"\bdeepen (?:the )?conversation(?:s)? (?:every|each) (?:turn|message|reply)\b",
                r"\bi should ask (?:a )?question (?:every|each) (?:turn|message|reply)\b",
                r"\bi will ask (?:a )?question (?:every|each) (?:turn|message|reply)\b",
                r"\balways ask (?:a )?question\b",
                r"\bask more questions each (?:turn|message|reply)\b",
            )

            lowered = content.lower()
            for pat in banned_patterns:
                if re.search(pat, lowered):
                    return False
        except Exception:
            # On regex failure, don't block
            pass

        return True

    def _is_duplicate_text(self, content: str) -> bool:
        """Fast text similarity check using cached insights."""
        if not self._recent_insights_cache:
            return False

        # Simple token-based similarity
        content_tokens = set(content.lower().split())

        for cached_insight in self._recent_insights_cache:
            cached_tokens = set(cached_insight.lower().split())

            if not cached_tokens:
                continue

            # Jaccard similarity
            intersection = len(content_tokens & cached_tokens)
            union = len(content_tokens | cached_tokens)

            if union > 0:
                similarity = intersection / union
                if similarity > 0.7:  # Fast rejection threshold
                    return True

        return False

    def _is_duplicate_embedding(
        self, content: str, return_best_sim: bool = False
    ) -> bool | tuple[bool, float]:
        """Check embedding similarity against recent insights.

        If return_best_sim=True, returns a tuple (is_duplicate, best_similarity).
        """
        try:
            # Get recent insights from PMM
            recent_insights = self.pmm.model.self_knowledge.insights[-8:]
            if not recent_insights:
                return (False, -1.0) if return_best_sim else False

            # Skip embedding check if OpenAI API not available or embeddings disabled
            try:
                from openai import OpenAI
                import os

                if not os.getenv("OPENAI_API_KEY"):
                    pmm_dlog(
                        "🔍 DEBUG: No OpenAI API key - skipping embedding deduplication"
                    )
                    return (False, -1.0) if return_best_sim else False

                client = OpenAI()
                # Use cached/modern embedding model
                new_embedding = self._get_embedding(client, content)

                # Compute similarity against recent insights and pick the best match
                import numpy as np

                best_sim = -1.0
                best_insight = None
                for insight in recent_insights:
                    if not insight.content:
                        continue

                    # Fast exact-text equivalence short-circuit to avoid confusing 1.000 embedding sims
                    cand_norm = content.strip().lower()
                    exist_norm = insight.content.strip().lower()
                    if cand_norm == exist_norm:
                        best_sim = 1.0
                        best_insight = insight
                        break

                    existing_embedding = self._get_embedding(client, insight.content)

                    sim = np.dot(new_embedding, existing_embedding) / (
                        np.linalg.norm(new_embedding)
                        * np.linalg.norm(existing_embedding)
                    )

                    if sim > best_sim:
                        best_sim = sim
                        best_insight = insight

                if best_insight is not None and (
                    best_sim >= 0.999 or best_sim > self._effective_threshold
                ):
                    pmm_dlog(f"🔍 DEBUG: High embedding similarity: {best_sim:.3f}")

                    # Structured override: allow near-duplicates if they reference NEW IDs
                    candidate_refs = self._extract_referenced_ids(content)
                    prior_refs = self._extract_referenced_ids(best_insight.content)
                    new_refs = candidate_refs - prior_refs

                    telemetry = os.getenv("PMM_TELEMETRY", "").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )

                    if new_refs:
                        # Always emit a plain stdout line so tests can capture it reliably
                        print(
                            f"dedup_override: reason=new_references, sim={best_sim:.3f}, threshold={self._effective_threshold:.3f}, added_refs={sorted(list(new_refs))}"
                        )
                        if telemetry:
                            print(
                                f"[PMM_TELEMETRY] dedup_override: reason=new_references, sim={best_sim:.3f}, threshold={self._effective_threshold:.3f}, added_refs={sorted(list(new_refs))}"
                            )
                        # Bypass duplicate rejection since candidate adds new references
                        return (False, best_sim) if return_best_sim else False

                    # No new references; check identity/evidence anchors before treating as duplicate
                    if self._should_accept_insight(
                        content, best_sim, candidate_refs=candidate_refs
                    ):
                        print(
                            f"dedup_override: reason=anchor_terms_or_evidence, sim={best_sim:.3f}, threshold={self._effective_threshold:.3f}"
                        )
                        if telemetry:
                            print(
                                f"[PMM_TELEMETRY] dedup_override: reason=anchor_terms_or_evidence, sim={best_sim:.3f}, threshold={self._effective_threshold:.3f}"
                            )
                        return (False, best_sim) if return_best_sim else False

                    # Still no anchors; treat as duplicate
                    if telemetry:
                        print(
                            f"[PMM_TELEMETRY] dedup_reject: reason=high_similarity, sim={best_sim:.3f}, threshold={self._effective_threshold:.3f}, gates=embedding"
                        )
                    return (True, best_sim) if return_best_sim else True

                # Not similar enough to consider duplicate
                return (False, best_sim) if return_best_sim else False
            except Exception as e:
                pmm_dlog(f"🔍 DEBUG: Embedding check failed, allowing insight: {e}")
                return (False, -1.0) if return_best_sim else False

        except Exception as e:
            pmm_dlog(f"🔍 DEBUG: Embedding similarity check failed: {e}")
            # Fallback to text similarity
            return False

    def _persist_insight(self, content: str, model_config: Dict[str, Any]) -> bool:
        """Atomically persist insight to PMM."""
        try:
            # Create insight object
            from pmm.model import Insight

            insight = Insight(
                id=str(uuid.uuid4()),
                t=datetime.now(timezone.utc).isoformat(),
                content=content,
            )

            # Add to PMM model
            self.pmm.model.self_knowledge.insights.append(insight)

            # Save to disk
            self.pmm.save_model()

            # Also append a 'reflection' event to SQLite for audit/analytics
            try:
                meta = {
                    "model": (
                        model_config.get("name")
                        if isinstance(model_config, dict)
                        else None
                    ),
                    "provider": (
                        model_config.get("provider")
                        if isinstance(model_config, dict)
                        else None
                    ),
                }
                store = getattr(self.pmm, "sqlite_store", None)
                if store is not None:
                    store.append_event(kind="reflection", content=content, meta=meta)
            except Exception:
                # Never fail persistence due to DB audit issues
                pass

            return True

        except Exception as e:
            pmm_dlog(f"🔍 DEBUG: Failed to persist insight: {e}")
            return False

    def _update_cache(self, content: str) -> None:
        """Update the recent insights cache."""
        self._recent_insights_cache.append(content)

        # Trim cache if too large
        if len(self._recent_insights_cache) > self._max_cache_size:
            self._recent_insights_cache = self._recent_insights_cache[
                -self._max_cache_size :
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about reflection processing."""
        return {
            "cache_size": len(self._recent_insights_cache),
            "embedding_threshold_configured": self.embedding_threshold,
            "embedding_threshold_effective": self._effective_threshold,
            "adaptive_enabled": self._adaptive_enabled,
            "min_threshold_cap": self._min_thresh,
            "max_threshold_cap": self._max_thresh,
            "cooldown_turns": self._cooldown_turns,
            "cooldown_remaining": self._cooldown_remaining,
            "accept_streak": self._accept_streak,
            "reject_streak": self._reject_streak,
            "total_insights": len(self.pmm.model.self_knowledge.insights),
            "recent_insights_preview": (
                self._recent_insights_cache[-3:] if self._recent_insights_cache else []
            ),
        }

    def _extract_referenced_ids(self, text: str) -> set[str]:
        """Extract referenced event IDs and short commitment hashes from text.

        - Event IDs: ev123 or forms that include ev followed by digits
        - Commitment hashes: 16-char lowercase hex substrings
        """
        if not text:
            return set()
        try:
            import re

            refs: set[str] = set()

            # Event IDs: match 'ev123' (case-insensitive), normalize to lowercase
            for m in re.findall(r"\bev\d+\b", text, flags=re.IGNORECASE):
                refs.add(m.lower())

            # 16-char hex (commitment short hashes)
            for m in re.findall(r"\b[a-f0-9]{16}\b", text.lower()):
                refs.add(m)

            return refs
        except Exception:
            # On any failure, do not block acceptance
            return set()

    def _first_hit_pass_allowed(self, best_sim: float | None) -> bool:
        """Decide if a borderline near-duplicate can pass once.

        Policy:
        - Only consider when we have a similarity value and it's within a small
          margin above the current effective threshold.
        - Never allow exact/near-exact duplicates (>= 0.999) via this path;
          those are handled by structured-reference override earlier.
        - Only allow when there have been no recent accepted insights
          (based on `_accept_streak == 0`).
        """
        try:
            if best_sim is None:
                return False

            # Block exact/near-exact duplicates
            if best_sim >= 0.999:
                return False

            # Allow only if within a small margin above the threshold
            margin = 0.01
            if not (
                self._effective_threshold
                <= best_sim
                <= self._effective_threshold + margin
            ):
                return False

            # Only if no recent acceptances
            return self._accept_streak == 0
        except Exception:
            return False

    def _should_accept_insight(
        self,
        content: str,
        best_sim: float | None,
        *,
        candidate_refs: set[str] | None = None,
    ) -> bool:
        """Allow useful near-duplicates when they cite evidence or PMM anchor terms.

        Rules:
        - Never accept exact/near-exact duplicates (>= 0.999).
        - Consider only within a narrow band above the effective threshold (<= +0.02).
        - Accept if content mentions PMM anchor terms OR cites enough evidence IDs/hashes.
        """
        try:
            if best_sim is None:
                return False

            # Hard cap: do not allow exact/near-exact duplicates here
            if best_sim >= 0.999:
                return False

            # Only consider near-threshold band to avoid flooding
            margin = 0.02
            upper_cap = min(self._effective_threshold + margin, 0.97)
            if not (self._effective_threshold <= best_sim <= upper_cap):
                return False

            text = (content or "").lower()
            # PMM anchor terms that improve identity/momentum signals
            anchor_terms = (
                "commitment",
                "commitments",
                "evidence",
                "memory",
                "memories",
                "identity",
                "drift",
                "emergence",
                "pmm",
            )

            mentions_anchor = any(term in text for term in anchor_terms)

            # Evidence citation: enough referenced IDs/hashes even if not NEW compared to prior
            refs = (
                candidate_refs
                if candidate_refs is not None
                else self._extract_referenced_ids(text)
            )
            cites_evidence = len(refs) >= 2 or (
                len(refs) == 1
                and (
                    "ev" in next(iter(refs), "")
                    or any(c.isdigit() for c in next(iter(refs), ""))
                )
            )

            decision = bool(mentions_anchor or cites_evidence)
            try:
                pmm_tlog(
                    f"[INSIGHT] sim={best_sim:.3f} refs={len(refs)} pmm={(1 if mentions_anchor else 0)} decision={'ACCEPT' if decision else 'REJECT'}"
                )
            except Exception:
                pass
            return decision
        except Exception:
            # Be conservative on any failure
            return False

    # --- Adaptive threshold helpers ---
    def _on_decision(self, accepted: bool) -> None:
        """Update adaptive state on accept/reject and adjust threshold if enabled."""
        if not self._adaptive_enabled:
            return

        telemetry = os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on")

        # decrement cooldown if active
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if accepted:
            self._accept_streak += 1
            self._reject_streak = 0
            # If we keep accepting, become a bit stricter to catch more dups
            if self._accept_streak >= 2 and self._cooldown_remaining == 0:
                old = self._effective_threshold
                self._effective_threshold = max(
                    self._min_thresh, round(self._effective_threshold - self._step, 4)
                )
                self._cooldown_remaining = self._cooldown_turns
                if telemetry:
                    print(
                        f"[PMM_TELEMETRY] dedup_adapt: accepted streak={self._accept_streak}, threshold {old:.3f} -> {self._effective_threshold:.3f}, cooldown={self._cooldown_turns}"
                    )
        else:
            self._reject_streak += 1
            self._accept_streak = 0
            # If we keep rejecting as duplicates, relax slightly to allow more acceptance
            if self._reject_streak >= 2 and self._cooldown_remaining == 0:
                old = self._effective_threshold
                self._effective_threshold = min(
                    self._max_thresh, round(self._effective_threshold + self._step, 4)
                )
                self._cooldown_remaining = self._cooldown_turns
                if telemetry:
                    print(
                        f"[PMM_TELEMETRY] dedup_adapt: rejected streak={self._reject_streak}, threshold {old:.3f} -> {self._effective_threshold:.3f}, cooldown={self._cooldown_turns}"
                    )

    def _get_embedding(self, client, text: str) -> List[float]:
        """Get embedding for text using an LRU cache and modern model.

        Use a lowercased cache key, but pass the original-cased text to the client
        to preserve deterministic behavior with tests/mocks that map exact strings.
        """
        original = (text or "").strip()
        cache_key = original.lower()
        if cache_key in self._embedding_cache:
            # Move to end (most recently used)
            self._embedding_cache.move_to_end(cache_key)
            return self._embedding_cache[cache_key]

        emb = (
            client.embeddings.create(input=original, model="text-embedding-3-small")
            .data[0]
            .embedding
        )
        # Insert and trim LRU
        self._embedding_cache[cache_key] = emb
        if len(self._embedding_cache) > self._embedding_cache_max:
            self._embedding_cache.popitem(last=False)
        return emb

    # --- Stage-adaptive dedup thresholding ---
    def _apply_stage_adaptation(self) -> None:
        """Blend a stage-targeted threshold into the current effective threshold.

        - Uses current S0–S4 stage from `compute_emergence_scores()`.
        - Early stages (S0/S1) get lenient thresholds to allow more acceptance.
        - Later stages (S3/S4) get stricter thresholds.
        - Respects min/max caps and emits telemetry when enabled.
        """
        if not self._stage_adaptive_enabled:
            return

        try:
            from pmm.emergence import compute_emergence_scores

            scores = compute_emergence_scores(
                window=5, storage_manager=getattr(self, "storage_manager", None)
            )
            stage_str = str(scores.get("stage", "")).strip()
            ias = float(scores.get("ias", scores.get("IAS", 0.0)) or 0.0)
            gas = float(scores.get("gas", scores.get("GAS", 0.0)) or 0.0)

            # Map S0–S4 to target thresholds (higher = more lenient)
            # Updated mapping per spec: S0/S1=0.88, S2=0.90, else=0.94
            # Bound everything to [min_thresh, max_thresh]
            def target_for(stage_label: str) -> float:
                s = stage_label.lower()
                if s.startswith("s0"):
                    return 0.88
                if s.startswith("s1"):
                    return 0.88
                if s.startswith("s2"):
                    return 0.90
                if s.startswith("s3"):
                    return 0.94
                if s.startswith("s4"):
                    return 0.94
                # Unknown/default
                return self._effective_threshold

            target = max(self._min_thresh, min(self._max_thresh, target_for(stage_str)))

            # Blend toward target softly to avoid oscillation
            old = self._effective_threshold
            alpha = 0.3  # 30% toward stage target each call
            blended = (1 - alpha) * old + alpha * target
            # Respect bounds and round
            self._effective_threshold = max(
                self._min_thresh, min(self._max_thresh, round(blended, 4))
            )

            if os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on"):
                print(
                    f"[PMM_TELEMETRY] stage_dedup_adapt: stage={stage_str} ias={ias:.3f} gas={gas:.3f} target={target:.3f} threshold {old:.3f} -> {self._effective_threshold:.3f}"
                )
        except Exception:
            # Silent fail; dedup continues with existing threshold
            return


# Backward compatibility alias for older imports
# Allows: from pmm.atomic_reflection import AtomicReflection
AtomicReflection = AtomicReflectionManager
