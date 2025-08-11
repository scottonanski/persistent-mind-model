"""
Enhanced PMM Model with Memory Tokenization and Archive Support

Extends the existing PMM schema to support cryptographic integrity,
quantum-inspired memory states, and self-sovereign AI identity.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timezone

# Import existing model components
from .model import (
    CoreIdentity,
    Personality,
    NarrativeIdentity,
    Metrics,
    DriftConfig,
    MetaCognition,
    Event,
    Thought,
    Insight,
)
from .memory_token import MemoryToken, MemoryChain, IdentityLockpoint, MemoryArchive


@dataclass
class EnhancedSelfKnowledge:
    """
    Enhanced self-knowledge with memory tokenization support.
    Maintains backward compatibility while adding next-stage features.
    """

    # Legacy fields (preserved for compatibility)
    behavioral_patterns: Dict[str, int] = field(default_factory=dict)
    autobiographical_events: List[Event] = field(default_factory=list)
    thoughts: List[Thought] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)
    commitments: Dict[str, Dict] = field(default_factory=dict)

    # Next-stage memory tokenization
    memory_tokens: Dict[str, MemoryToken] = field(
        default_factory=dict
    )  # token_id -> token
    memory_chain: MemoryChain = field(default_factory=MemoryChain)

    # Archive system
    archives: Dict[str, MemoryArchive] = field(
        default_factory=dict
    )  # archive_id -> archive
    active_token_ids: List[str] = field(default_factory=list)  # Currently active tokens

    # Identity lockpoints
    lockpoints: List[IdentityLockpoint] = field(default_factory=list)
    last_lockpoint_at: Optional[str] = None
    lockpoint_interval_tokens: int = 100  # Create lockpoint every N tokens


@dataclass
class ProviderConfig:
    """
    Configuration for LLM provider abstraction supporting local and API inference.
    """

    # Provider selection
    primary_provider: str = "openai"  # openai, ollama, local, huggingface
    fallback_provider: str = "openai"  # Fallback if primary fails

    # OpenAI API config
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_base_url: Optional[str] = None

    # Local LLM config (Ollama, LM Studio, llama.cpp)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # LM Studio config
    lm_studio_base_url: str = "http://localhost:1234"
    lm_studio_model: str = "local-model"

    # llama.cpp config
    llamacpp_model_path: Optional[str] = None
    llamacpp_n_ctx: int = 4096
    llamacpp_n_threads: int = 4

    # HuggingFace config
    hf_model_name: str = "microsoft/DialoGPT-medium"
    hf_device: str = "auto"
    hf_max_length: int = 1024

    # Hybrid mode settings
    use_local_first: bool = True  # Try local before API
    local_context_limit: int = 2048  # Switch to API for larger contexts
    api_fallback_enabled: bool = True  # Fall back to API if local fails

    # Performance settings
    max_retries: int = 3
    timeout_seconds: int = 30
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class ArchiveConfig:
    """
    Configuration for memory archival and compression system.
    """

    # Archive triggers
    max_active_tokens: int = 500  # Max tokens in active memory
    amplitude_threshold: float = 0.1  # Archive tokens below this amplitude
    age_threshold_days: int = 30  # Archive tokens older than this

    # Compression settings
    enable_compression: bool = True
    compression_algorithm: str = "gzip"  # gzip, lzma, brotli
    target_compression_ratio: float = 0.3  # Target 70% size reduction

    # Thematic clustering
    enable_clustering: bool = True
    cluster_algorithm: str = "kmeans"  # kmeans, hierarchical, dbscan
    min_cluster_size: int = 5
    max_clusters: int = 20

    # Archive storage
    archive_base_path: str = "archives/"
    archive_format: str = "jsonl"  # jsonl, sqlite, parquet

    # Lockpoint settings
    lockpoint_interval: int = 100  # Create lockpoint every N tokens
    max_lockpoints: int = 50  # Keep only recent lockpoints
    lockpoint_compression: bool = True


@dataclass
class RecallConfig:
    """
    Configuration for cue-based memory recall system.
    """

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    similarity_threshold: float = 0.7

    # Recall parameters
    max_recall_tokens: int = 10  # Max tokens to recall per query
    phase_weight: float = 0.3  # Weight of phase similarity vs embedding
    amplitude_boost: float = 0.1  # Boost amplitude after recall

    # Cache settings
    enable_embedding_cache: bool = True
    cache_size: int = 1000
    cache_ttl_hours: int = 24


@dataclass
class IntegrityConfig:
    """
    Configuration for cryptographic integrity and portability.
    """

    # Hash verification
    verify_on_load: bool = True
    verify_chain_integrity: bool = True
    hash_algorithm: str = "sha256"

    # Export/import settings
    export_format: str = "json"  # json, msgpack, cbor
    include_archives: bool = True
    compress_exports: bool = True

    # Backup settings
    enable_distributed_backup: bool = False
    ipfs_gateway: Optional[str] = None
    backup_interval_hours: int = 24

    # Security
    enable_encryption: bool = False
    encryption_key_path: Optional[str] = None


@dataclass
class NextStageConfig:
    """
    Configuration container for all next-stage PMM features.
    """

    # Feature toggles
    enable_memory_tokens: bool = True
    enable_archival: bool = True
    enable_recall: bool = True
    enable_local_inference: bool = True
    enable_integrity_checks: bool = True

    # Component configs
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)


@dataclass
class EnhancedMetrics:
    """
    Enhanced metrics with next-stage memory and integrity tracking.
    """

    # Legacy metrics (preserved)
    identity_coherence: float = 0.5
    self_consistency: float = 0.5
    drift_velocity: Dict[str, float] = field(default_factory=dict)
    reflection_cadence_days: int = 7
    last_reflection_at: Optional[str] = None
    commitments_open: int = 0
    commitments_closed: int = 0

    # Memory tokenization metrics
    total_tokens: int = 0
    active_tokens: int = 0
    archived_tokens: int = 0
    chain_integrity_score: float = 1.0

    # Archive metrics
    total_archives: int = 0
    compression_ratio: float = 1.0
    archive_hit_rate: float = 0.0  # Successful recalls from archive

    # Recall metrics
    recall_accuracy: float = 0.0  # How often recalled memories are relevant
    recall_latency_ms: float = 0.0  # Average recall time
    embedding_cache_hit_rate: float = 0.0

    # Provider metrics
    local_inference_success_rate: float = 0.0
    api_fallback_rate: float = 0.0
    average_response_time_ms: float = 0.0

    # Integrity metrics
    chain_verification_passes: int = 0
    chain_verification_failures: int = 0
    lockpoint_count: int = 0
    last_integrity_check: Optional[str] = None


@dataclass
class EnhancedPersistentMindModel:
    """
    Enhanced PMM with next-stage architecture while maintaining full backward compatibility.
    """

    # Schema versioning
    schema_version: int = 2  # Increment from v1 to v2
    next_stage_enabled: bool = True

    # Legacy fields (preserved exactly)
    inception_moment: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )
    core_identity: CoreIdentity = field(default_factory=CoreIdentity)
    personality: Personality = field(default_factory=Personality)
    narrative_identity: NarrativeIdentity = field(default_factory=NarrativeIdentity)
    drift_config: DriftConfig = field(default_factory=DriftConfig)
    meta_cognition: MetaCognition = field(default_factory=MetaCognition)

    # Enhanced components
    self_knowledge: EnhancedSelfKnowledge = field(default_factory=EnhancedSelfKnowledge)
    metrics: EnhancedMetrics = field(default_factory=EnhancedMetrics)

    # Next-stage configuration
    next_stage_config: NextStageConfig = field(default_factory=NextStageConfig)

    def is_legacy_compatible(self) -> bool:
        """Check if model can be used with legacy PMM code."""
        return hasattr(self, "core_identity") and hasattr(self, "personality")

    def get_legacy_model(self):
        """Convert to legacy PMM model for backward compatibility."""
        from .model import PersistentMindModel, SelfKnowledge

        # Create legacy self_knowledge
        legacy_sk = SelfKnowledge(
            behavioral_patterns=self.self_knowledge.behavioral_patterns,
            autobiographical_events=self.self_knowledge.autobiographical_events,
            thoughts=self.self_knowledge.thoughts,
            insights=self.self_knowledge.insights,
            commitments=self.self_knowledge.commitments,
        )

        # Create legacy metrics

        legacy_metrics = Metrics(
            identity_coherence=self.metrics.identity_coherence,
            self_consistency=self.metrics.self_consistency,
            drift_velocity=self.metrics.drift_velocity,
            reflection_cadence_days=self.metrics.reflection_cadence_days,
            last_reflection_at=self.metrics.last_reflection_at,
            commitments_open=self.metrics.commitments_open,
            commitments_closed=self.metrics.commitments_closed,
        )

        return PersistentMindModel(
            schema_version=1,  # Legacy version
            inception_moment=self.inception_moment,
            core_identity=self.core_identity,
            personality=self.personality,
            narrative_identity=self.narrative_identity,
            self_knowledge=legacy_sk,
            metrics=legacy_metrics,
            drift_config=self.drift_config,
            meta_cognition=self.meta_cognition,
        )

    def upgrade_from_legacy(self, legacy_model) -> None:
        """Upgrade from legacy PMM model to enhanced version."""
        # Copy all legacy fields
        self.inception_moment = legacy_model.inception_moment
        self.core_identity = legacy_model.core_identity
        self.personality = legacy_model.personality
        self.narrative_identity = legacy_model.narrative_identity
        self.drift_config = legacy_model.drift_config
        self.meta_cognition = legacy_model.meta_cognition

        # Migrate self_knowledge
        self.self_knowledge.behavioral_patterns = (
            legacy_model.self_knowledge.behavioral_patterns
        )
        self.self_knowledge.autobiographical_events = (
            legacy_model.self_knowledge.autobiographical_events
        )
        self.self_knowledge.thoughts = legacy_model.self_knowledge.thoughts
        self.self_knowledge.insights = legacy_model.self_knowledge.insights
        self.self_knowledge.commitments = legacy_model.self_knowledge.commitments

        # Migrate metrics
        self.metrics.identity_coherence = legacy_model.metrics.identity_coherence
        self.metrics.self_consistency = legacy_model.metrics.self_consistency
        self.metrics.drift_velocity = legacy_model.metrics.drift_velocity
        self.metrics.reflection_cadence_days = (
            legacy_model.metrics.reflection_cadence_days
        )
        self.metrics.last_reflection_at = legacy_model.metrics.last_reflection_at
        self.metrics.commitments_open = legacy_model.metrics.commitments_open
        self.metrics.commitments_closed = legacy_model.metrics.commitments_closed

        # Initialize next-stage components
        self.schema_version = 2
        self.next_stage_enabled = True
