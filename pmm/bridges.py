# pmm/bridges.py
"""Bridge manager for embodiment-aware model switching."""

from __future__ import annotations
import time
from typing import Optional
from dataclasses import dataclass
from .model_config import ModelConfig
from .embodiment import get_adapter, create_family_adapters
from .stance_filter import StanceFilter


@dataclass
class PMMEvent:
    """PMM event with provenance tracking."""

    kind: str  # "insight", "commitment", "reflection", "evidence", "handover"
    canonical_text: str  # neutral phrasing, no vendor quirks
    ts: float  # timestamp
    origin_model_id: str  # "ollama:gemma3:4b@3.3"
    origin_family: str  # "gemma"
    epoch: int  # from unified factory
    style_fingerprint: Optional[str] = None  # optional small hash


class BridgeManager:
    """Manages embodiment-aware model switching and rendering."""

    def __init__(self, factory, storage, cooldown, ngram_ban, stages):
        self.factory = factory
        self.storage = storage
        self.cooldown = cooldown
        self.stages = stages
        self.ngram_ban = ngram_ban

        # Create family adapters
        self.adapters = create_family_adapters(ngram_ban)

        # Initialize stance filter
        self.stance_filter = StanceFilter()

        # Track state
        self._last_family: Optional[str] = None
        self._switch_count = 0
        self._continuity_turns_remaining = 0

    def on_switch(self, prev: Optional[ModelConfig], curr: ModelConfig):
        """Handle model switch with proper handover."""
        # Reset gates to avoid stale triggers
        if hasattr(self.cooldown, "reset_on_model_switch"):
            self.cooldown.reset_on_model_switch()
        elif hasattr(self.cooldown, "reset"):
            self.cooldown.reset(reason="model_switch")

        # Reset emergence stages for new model
        if hasattr(self.stages, "reset_on_model_switch"):
            self.stages.reset_on_model_switch(curr.name)

        # Create handover event
        handover_event = PMMEvent(
            kind="handover",
            canonical_text=f"Switch {prev.family if prev else 'none'} -> {curr.family}",
            ts=time.time(),
            origin_model_id=f"{curr.provider}:{curr.name}@{curr.version}",
            origin_family=curr.family,
            epoch=curr.epoch,
            style_fingerprint=None,
        )

        # Store handover event
        if hasattr(self.storage, "add_event"):
            self.storage.add_event(handover_event)

        # Update tracking
        self._last_family = curr.family
        self._switch_count += 1
        self._continuity_turns_remaining = 2  # Show preface for 2 turns

        print(
            f"🔄 Bridge: Switched {prev.family if prev else 'none'} -> {curr.family} (#{self._switch_count})"
        )

    def continuity_preface(self, curr: ModelConfig) -> Optional[str]:
        """Generate continuity preface for post-switch turns."""
        if self._continuity_turns_remaining > 0:
            self._continuity_turns_remaining -= 1
            return f"[continuity] Persistent memory active; rendering for {curr.name} ({curr.family})."
        return None

    def speak(self, canonical_text: str) -> str:
        """Render canonical text through current embodiment."""
        try:
            curr = self.factory.get_active_config()
        except (RuntimeError, AttributeError):
            # Fallback if no active config
            return self._apply_stance_filter(canonical_text)

        # Get adapter for current family
        adapter = get_adapter(curr.family, self.adapters)

        # Render through family adapter
        styled = adapter.render(canonical_text)

        # Apply stance filter after styling
        styled = self._apply_stance_filter(styled)

        return styled

    def _apply_stance_filter(self, text: str) -> str:
        """Apply anthropomorphic stance filtering."""
        try:
            filtered_text, _ = self.stance_filter.filter_response(text)
            return filtered_text
        except Exception:
            # Fallback if stance filter fails
            return text

    def create_event(self, kind: str, canonical_text: str) -> PMMEvent:
        """Create a new PMM event with proper provenance."""
        try:
            curr = self.factory.get_active_config()
            model_id = f"{curr.provider}:{curr.name}@{curr.version}"
            family = curr.family
            epoch = curr.epoch
        except (RuntimeError, AttributeError):
            # Fallback values
            model_id = "unknown:unknown@unknown"
            family = "unknown"
            epoch = 0

        return PMMEvent(
            kind=kind,
            canonical_text=canonical_text,
            ts=time.time(),
            origin_model_id=model_id,
            origin_family=family,
            epoch=epoch,
            style_fingerprint=self._compute_style_fingerprint(canonical_text),
        )

    def _compute_style_fingerprint(self, text: str) -> Optional[str]:
        """Compute simple style fingerprint for cross-family safety."""
        if not text:
            return None

        # Simple hash of top n-grams and sentence length
        import hashlib

        words = text.lower().split()
        avg_sentence_len = len(words) / max(
            1, text.count(".") + text.count("!") + text.count("?")
        )

        # Create fingerprint from style metrics
        style_data = f"{avg_sentence_len:.1f}:{len(words)}"
        return hashlib.md5(style_data.encode()).hexdigest()[:8]

    def just_switched(self) -> bool:
        """Check if we just switched models (for continuity preface)."""
        return self._continuity_turns_remaining > 0

    def get_switch_count(self) -> int:
        """Get number of model switches."""
        return self._switch_count
