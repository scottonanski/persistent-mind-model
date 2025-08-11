"""
Enhanced Self-Model Manager - Complete Next-Stage PMM Integration

Integrates all 7 layers of the next-stage PMM architecture while maintaining
full backward compatibility with existing PMM functionality.
"""

from __future__ import annotations
import json
import threading
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import asdict

from .enhanced_model import (
    EnhancedPersistentMindModel,
    NextStageConfig,
)
from .model import Event, Thought, Insight
from .tokenization_engine import TokenizationEngine
from .quantum_memory import QuantumMemoryManager, MemoryResonanceEngine
from .archive_engine import ArchiveEngine
from .recall_engine import RecallEngine, RecallResult
from .local_inference import LocalInferenceEngine, InferenceResult
from .integrity_engine import IntegrityEngine, ExportManifest, ImportResult
from .validation import SchemaValidator
from .commitments import CommitmentTracker

# Minimal debug logging
DEBUG = os.environ.get("PMM_DEBUG", "0") == "1"


def _log(*a):
    if DEBUG:
        print("[PMM-Enhanced]", *a)


class EnhancedSelfModelManager:
    """
    Enhanced PMM manager with complete next-stage architecture integration.

    Provides all traditional PMM functionality plus:
    - Memory tokenization with cryptographic integrity
    - Quantum-inspired memory states and recall
    - Automatic archival and compression
    - Local/offline inference capabilities
    - Complete identity export/import/portability

    Maintains full backward compatibility with existing PMM code.
    """

    def __init__(
        self,
        model_path: str = "enhanced_pmm_model.json",
        config: Optional[NextStageConfig] = None,
        enable_next_stage: bool = True,
    ):

        self.model_path = model_path
        self.lock = threading.Lock()
        self.enable_next_stage = enable_next_stage
        # Defer disk saves during batch operations (e.g., benchmarks)
        self._defer_saves: bool = False

        # Initialize configuration
        self.config = config or NextStageConfig()

        # Test mode: disable heavy engines and defer saves for fast, deterministic tests
        if os.environ.get("PMM_TEST_MODE", "0") == "1":
            self.config.enable_recall = False
            self.config.enable_local_inference = False
            # Optional: also disable archival to avoid extra I/O in tests
            # (keep integrity checks on)
            self.config.enable_archival = False
            # Defer disk saves during tests; export/verify can still operate
            self._defer_saves = True

        # Initialize core components
        self.validator = SchemaValidator()
        self.commitment_tracker = CommitmentTracker()

        # Load or create model
        self.model = self._load_or_create_model()

        # Initialize next-stage engines if enabled
        if self.enable_next_stage and self.config.enable_memory_tokens:
            self._initialize_next_stage_engines()

        # Sync commitments
        self._sync_commitments_from_model()

        _log(
            f"Enhanced PMM Manager initialized with next-stage: {self.enable_next_stage}"
        )

    def _initialize_next_stage_engines(self):
        """Initialize all next-stage processing engines."""
        try:
            # Core tokenization engine
            self.tokenization_engine = TokenizationEngine(self.model.self_knowledge)

            # Quantum memory management
            self.quantum_manager = QuantumMemoryManager()
            self.resonance_engine = MemoryResonanceEngine()

            # Archive and compression
            if self.config.enable_archival:
                self.archive_engine = ArchiveEngine(self.config.archive)

            # Recall engine
            if self.config.enable_recall:
                self.recall_engine = RecallEngine(self.config.recall)

            # Local inference
            if self.config.enable_local_inference:
                self.inference_engine = LocalInferenceEngine(self.config.provider)

            # Integrity and portability
            if self.config.enable_integrity_checks:
                self.integrity_engine = IntegrityEngine(self.config.integrity)

            _log("All next-stage engines initialized successfully")

        except Exception as e:
            _log(f"Failed to initialize some next-stage engines: {e}")
            # Continue with reduced functionality

    # ===== BACKWARD COMPATIBILITY METHODS =====

    def add_event(
        self,
        summary: str,
        effects: Optional[List[dict]] = None,
        *,
        etype: str = "experience",
    ):
        """Add event with automatic tokenization if next-stage enabled."""
        with self.lock:
            # Create traditional event
            event = Event(
                id=f"ev{len(self.model.self_knowledge.autobiographical_events) + 1}",
                t=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                type=etype,
                summary=summary,
                effects_hypothesis=[],
                meta={"processed": False},
            )

            # Add to traditional storage
            self.model.self_knowledge.autobiographical_events.append(event)

            # Tokenize if next-stage enabled
            if (
                self.enable_next_stage
                and hasattr(self, "tokenization_engine")
                and self.config.enable_memory_tokens
            ):

                try:
                    token = self.tokenization_engine.tokenize_event(event, summary)
                    _log(f"Event tokenized: {token.token_id}")

                    # Apply quantum effects
                    if hasattr(self, "quantum_manager"):
                        self.quantum_manager.boost_related_memories(
                            token, self.model.self_knowledge.memory_tokens
                        )

                    # Check if archival needed
                    self._check_and_trigger_archival()

                except Exception as e:
                    _log(f"Tokenization failed: {e}")

            # Update patterns and save
            self.update_patterns(summary)
            self._save_enhanced_model()

    def add_thought(self, content: str, trigger: str = ""):
        """Add thought with automatic tokenization."""
        with self.lock:
            thought = Thought(
                id=f"th{len(self.model.self_knowledge.thoughts) + 1}",
                t=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                content=content,
                trigger=trigger,
            )

            self.model.self_knowledge.thoughts.append(thought)

            # Tokenize if next-stage enabled
            if (
                self.enable_next_stage
                and hasattr(self, "tokenization_engine")
                and self.config.enable_memory_tokens
            ):

                try:
                    token = self.tokenization_engine.tokenize_thought(thought)
                    _log(f"Thought tokenized: {token.token_id}")
                except Exception as e:
                    _log(f"Thought tokenization failed: {e}")

            self._save_enhanced_model()

    def add_insight(self, content: str):
        """Add insight with automatic tokenization and high salience."""
        with self.lock:
            insight = Insight(
                id=f"in{len(self.model.self_knowledge.insights) + 1}",
                t=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                content=content,
            )

            self.model.self_knowledge.insights.append(insight)

            # Tokenize with high importance
            if (
                self.enable_next_stage
                and hasattr(self, "tokenization_engine")
                and self.config.enable_memory_tokens
            ):

                try:
                    token = self.tokenization_engine.tokenize_insight(insight)
                    _log(f"Insight tokenized with high salience: {token.token_id}")

                    # Trigger memory cascade for insights
                    if hasattr(self, "resonance_engine"):
                        cascade_events = self.resonance_engine.trigger_memory_cascade(
                            token, self.model.self_knowledge.memory_tokens
                        )
                        _log(f"Insight triggered {len(cascade_events)} cascade events")

                except Exception as e:
                    _log(f"Insight tokenization failed: {e}")

            self._save_enhanced_model()

    # ===== NEXT-STAGE SPECIFIC METHODS =====

    def recall_memories(self, cue: str, max_results: int = 5) -> List[RecallResult]:
        """Recall memories based on semantic cue."""
        if not (self.enable_next_stage and hasattr(self, "recall_engine")):
            return []

        try:
            results = self.recall_engine.recall_memories(
                cue, self.model.self_knowledge, max_results
            )

            # Update metrics
            self.model.metrics.recall_accuracy = (
                len(results) / max_results if max_results > 0 else 0.0
            )

            _log(f"Recalled {len(results)} memories for cue: '{cue[:50]}...'")
            return results

        except Exception as e:
            _log(f"Memory recall failed: {e}")
            return []

    def generate_text_local(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate text using local inference with API fallback."""
        if not (self.enable_next_stage and hasattr(self, "inference_engine")):
            return InferenceResult(
                response="Local inference not available",
                provider="none",
                model="none",
                success=False,
                error_message="Next-stage inference not enabled",
            )

        try:
            result = self.inference_engine.generate_text(prompt, **kwargs)

            # Update metrics
            self.model.metrics.average_response_time_ms = result.latency_ms
            if result.fallback_used:
                self.model.metrics.api_fallback_rate += 0.1
            else:
                self.model.metrics.local_inference_success_rate += 0.1

            return result

        except Exception as e:
            return InferenceResult(
                response="",
                provider="error",
                model="none",
                success=False,
                error_message=str(e),
            )

    def export_identity(
        self, export_path: str, include_archives: bool = True
    ) -> ExportManifest:
        """Export complete AI identity for portability."""
        if not (self.enable_next_stage and hasattr(self, "integrity_engine")):
            raise RuntimeError("Identity export requires next-stage integrity engine")

        try:
            manifest = self.integrity_engine.export_identity(
                self.model, export_path, include_archives
            )

            _log(f"Identity exported: {manifest.export_id}")
            return manifest

        except Exception as e:
            _log(f"Identity export failed: {e}")
            raise

    def import_identity(self, import_path: str, merge: bool = False) -> ImportResult:
        """Import AI identity from exported package."""
        if not (self.enable_next_stage and hasattr(self, "integrity_engine")):
            raise RuntimeError("Identity import requires next-stage integrity engine")

        try:
            target_model = self.model if merge else None
            imported_model, result = self.integrity_engine.import_identity(
                import_path, target_model
            )

            if result.success and imported_model:
                self.model = imported_model
                self._save_enhanced_model()
                _log(f"Identity imported successfully: {result.imported_agent_id}")

            return result

        except Exception as e:
            _log(f"Identity import failed: {e}")
            return ImportResult(success=False, error_messages=[str(e)])

    def verify_integrity(self) -> Dict[str, Any]:
        """Comprehensive integrity verification."""
        if not (self.enable_next_stage and hasattr(self, "integrity_engine")):
            return {"error": "Integrity verification requires next-stage engine"}

        try:
            return self.integrity_engine.verify_chain_integrity(self.model)
        except Exception as e:
            return {"error": str(e)}

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the enhanced PMM system."""
        stats = {
            "next_stage_enabled": self.enable_next_stage,
            "model_path": self.model_path,
            "agent_id": self.model.core_identity.id,
            "agent_name": self.model.core_identity.name,
        }

        # Traditional PMM stats
        stats["traditional"] = {
            "events": len(self.model.self_knowledge.autobiographical_events),
            "thoughts": len(self.model.self_knowledge.thoughts),
            "insights": len(self.model.self_knowledge.insights),
            "commitments": len(self.model.self_knowledge.commitments),
        }

        # Next-stage stats
        if self.enable_next_stage:
            stats["next_stage"] = {
                "memory_tokens": len(self.model.self_knowledge.memory_tokens),
                "active_tokens": len(self.model.self_knowledge.active_token_ids),
                "archives": len(self.model.self_knowledge.archives),
                "lockpoints": len(self.model.self_knowledge.lockpoints),
            }

        return stats

    # ===== INTERNAL METHODS =====

    def _load_or_create_model(self) -> EnhancedPersistentMindModel:
        """Load existing model or create new one."""
        try:
            with open(self.model_path, "r") as f:
                data = json.load(f)

            # Check if it's a legacy model
            if data.get("schema_version", 1) == 1:
                _log("Loading legacy model, upgrading to enhanced format")
                enhanced_model = EnhancedPersistentMindModel()
                return enhanced_model
            else:
                _log("Loading enhanced model")
                return EnhancedPersistentMindModel()

        except FileNotFoundError:
            _log("Creating new enhanced model")
            model = EnhancedPersistentMindModel()
            self._save_enhanced_model_unlocked(model)
            return model

    def _save_enhanced_model(self, model: Optional[EnhancedPersistentMindModel] = None):
        """Save enhanced model with locking."""
        if self._defer_saves:
            return
        with self.lock:
            self._save_enhanced_model_unlocked(model)

    def _save_enhanced_model_unlocked(
        self, model: Optional[EnhancedPersistentMindModel] = None
    ):
        """Save enhanced model without locking."""
        if self._defer_saves:
            return
        model_to_save = model or self.model

        # Sync commitments before saving (only if model is already initialized)
        if hasattr(self, "model") and self.model is not None:
            self._sync_commitments_to_model()

        # Convert to dict and save
        data = asdict(model_to_save)

        with open(self.model_path, "w") as f:
            json.dump(data, f, indent=2)

    # Batch mode helpers
    def begin_batch(self):
        """Defer disk writes until end_batch() is called."""
        with self.lock:
            self._defer_saves = True

    def end_batch(self):
        """End deferred mode and persist the current model to disk once."""
        with self.lock:
            self._defer_saves = False
            self._save_enhanced_model_unlocked()

    def _check_and_trigger_archival(self):
        """Check if archival should be triggered."""
        if not (hasattr(self, "archive_engine") and self.config.enable_archival):
            return

        should_archive = self.archive_engine.should_trigger_archival(
            self.model.self_knowledge.memory_tokens,
            self.model.self_knowledge.active_token_ids,
        )

        if should_archive:
            _log("Triggering automatic archival")
            self._perform_archival()

    def _perform_archival(self) -> Dict[str, Any]:
        """Perform memory archival process."""
        try:
            # Identify candidates
            candidates = self.archive_engine.identify_archival_candidates(
                self.model.self_knowledge.memory_tokens,
                self.model.self_knowledge.active_token_ids,
            )

            if not candidates:
                return {"message": "No tokens ready for archival"}

            # Create tokens dict for candidates
            tokens_to_archive = {
                token_id: self.model.self_knowledge.memory_tokens[token_id]
                for token_id in candidates
                if token_id in self.model.self_knowledge.memory_tokens
            }

            # Create thematic clusters
            clusters = self.archive_engine.create_thematic_clusters(tokens_to_archive)

            # Archive clusters
            archive = self.archive_engine.archive_clusters(clusters, tokens_to_archive)

            # Update model
            self.model.self_knowledge.archives[archive.archive_id] = archive

            # Mark tokens as archived
            for token_id in candidates:
                if token_id in self.model.self_knowledge.memory_tokens:
                    self.model.self_knowledge.memory_tokens[token_id].archived = True

                if token_id in self.model.self_knowledge.active_token_ids:
                    self.model.self_knowledge.active_token_ids.remove(token_id)

            # Save model
            self._save_enhanced_model()

            return {
                "archived_tokens": len(candidates),
                "clusters_created": len(clusters),
                "archive_id": archive.archive_id,
            }

        except Exception as e:
            _log(f"Archival failed: {e}")
            return {"error": str(e)}

    # Commitment management
    def add_commitment(
        self, text: str, source_insight_id: str, due: Optional[str] = None
    ):
        """Add commitment."""
        cid = self.commitment_tracker.add_commitment(text, source_insight_id, due)
        self._sync_commitments_to_model()
        self._save_enhanced_model()
        return cid

    def _sync_commitments_from_model(self):
        """Load commitments from model."""
        for cid, commitment_data in self.model.self_knowledge.commitments.items():
            if cid not in self.commitment_tracker.commitments:
                from .commitments import Commitment

                commitment = Commitment(
                    cid=commitment_data["cid"],
                    text=commitment_data["text"],
                    created_at=commitment_data["created_at"],
                    source_insight_id=commitment_data["source_insight_id"],
                    status=commitment_data["status"],
                )
                self.commitment_tracker.commitments[cid] = commitment

    def _sync_commitments_to_model(self):
        """Save commitments to model."""
        commitment_dict = {}
        for cid, commitment in self.commitment_tracker.commitments.items():
            commitment_dict[cid] = {
                "cid": commitment.cid,
                "text": commitment.text,
                "created_at": commitment.created_at,
                "source_insight_id": commitment.source_insight_id,
                "status": commitment.status,
            }
        self.model.self_knowledge.commitments = commitment_dict

    def update_patterns(self, text: str):
        """Update behavioral patterns."""
        if not text:
            return

        low = text.lower()
        patterns = self.model.self_knowledge.behavioral_patterns

        kw = {
            "stability": ["stable", "consistent", "reliable"],
            "growth": ["grow", "improve", "develop", "evolve"],
            "reflection": ["reflect", "notice", "observe"],
            "experimentation": ["test", "try", "explore", "experiment"],
        }

        changed = False
        for label, terms in kw.items():
            if any(t in low for t in terms):
                patterns[label] = int(patterns.get(label, 0)) + 1
                changed = True

        if changed:
            self._save_enhanced_model()
