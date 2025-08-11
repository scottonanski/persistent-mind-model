"""
Memory Tokenization System for PMM Next-Stage Architecture

Implements cryptographic integrity and quantum-inspired memory states for
self-sovereign AI identity with tamper-evident history.
"""

from __future__ import annotations
import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class MemoryToken:
    """
    Cryptographically verifiable memory token with quantum-inspired state representation.
    
    Each token represents a discrete memory unit with:
    - SHA-256 hash for tamper detection
    - Blockchain-style linking to previous tokens
    - Amplitude/phase for activation probability and semantic context
    - Minimal metadata for efficient active storage
    """
    
    # Core identity
    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    
    # Cryptographic integrity
    content_hash: str = ""  # SHA-256 of full content
    prev_hash: str = ""     # Link to previous token (blockchain-style)
    chain_position: int = 0 # Position in the memory chain
    
    # Quantum-inspired state
    amplitude: float = 1.0   # Probability of being "active" (0.0-1.0)
    phase: float = 0.0       # Semantic/emotional orientation (radians)
    
    # Minimal metadata (stored in active memory)
    event_type: str = ""
    salience: float = 0.5
    valence: float = 0.5     # Emotional valence (-1 to 1, stored as 0-1)
    tags: List[str] = field(default_factory=list)
    
    # Archive references
    archived: bool = False
    archive_path: Optional[str] = None
    summary: str = ""        # Condensed version for active memory
    
    # Provenance
    source_event_id: Optional[str] = None
    source_insight_id: Optional[str] = None
    
    def compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content for integrity verification."""
        content_data = {
            "content": content,
            "timestamp": self.created_at,
            "token_id": self.token_id,
            "event_type": self.event_type
        }
        content_str = json.dumps(content_data, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def update_quantum_state(self, delta_amplitude: float = 0.0, delta_phase: float = 0.0):
        """Update amplitude and phase with bounds checking."""
        self.amplitude = max(0.0, min(1.0, self.amplitude + delta_amplitude))
        self.phase = (self.phase + delta_phase) % (2 * math.pi)
    
    def decay_amplitude(self, decay_rate: float = 0.01):
        """Apply time-based decay to amplitude (memories fade over time)."""
        self.amplitude *= (1.0 - decay_rate)
        self.amplitude = max(0.0, self.amplitude)
    
    def boost_amplitude(self, boost_factor: float = 0.1):
        """Boost amplitude when memory is recalled (memories strengthen with use)."""
        self.amplitude = min(1.0, self.amplitude + boost_factor)
    
    def get_activation_probability(self) -> float:
        """Get probability of this token being activated for recall."""
        return self.amplitude
    
    def get_semantic_vector(self) -> Tuple[float, float]:
        """Get 2D semantic position from phase angle."""
        return (math.cos(self.phase), math.sin(self.phase))
    
    def should_archive(self, threshold: float = 0.1) -> bool:
        """Determine if token should be moved to archive based on amplitude."""
        return self.amplitude < threshold and not self.archived


@dataclass 
class MemoryChain:
    """
    Blockchain-style chain of memory tokens for integrity verification.
    """
    
    chain_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    genesis_hash: str = ""
    current_position: int = 0
    tokens: List[MemoryToken] = field(default_factory=list)
    
    def add_token(self, token: MemoryToken, content: str) -> str:
        """Add token to chain with proper linking and hash computation."""
        # Set chain position
        token.chain_position = self.current_position
        
        # Link to previous token
        if self.tokens:
            token.prev_hash = self.tokens[-1].content_hash
        else:
            # Genesis token
            self.genesis_hash = token.token_id
            token.prev_hash = "genesis"
        
        # Compute content hash
        token.content_hash = token.compute_content_hash(content)
        
        # Add to chain
        self.tokens.append(token)
        self.current_position += 1
        
        return token.content_hash
    
    def verify_chain_integrity(self) -> Tuple[bool, List[str]]:
        """Verify the entire chain integrity. Returns (is_valid, error_messages)."""
        errors = []
        
        if not self.tokens:
            return True, []
        
        # Check genesis
        if self.tokens[0].prev_hash != "genesis":
            errors.append(f"Invalid genesis token: {self.tokens[0].token_id}")
        
        # Check chain links
        for i in range(1, len(self.tokens)):
            current = self.tokens[i]
            previous = self.tokens[i-1]
            
            if current.prev_hash != previous.content_hash:
                errors.append(f"Chain break at position {i}: {current.token_id}")
            
            if current.chain_position != i:
                errors.append(f"Invalid position at {i}: {current.token_id}")
        
        return len(errors) == 0, errors
    
    def get_chain_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the memory chain."""
        if not self.tokens:
            return {"length": 0, "active_tokens": 0, "archived_tokens": 0}
        
        active_count = sum(1 for t in self.tokens if not t.archived)
        archived_count = sum(1 for t in self.tokens if t.archived)
        avg_amplitude = sum(t.amplitude for t in self.tokens) / len(self.tokens)
        
        return {
            "chain_id": self.chain_id,
            "length": len(self.tokens),
            "active_tokens": active_count,
            "archived_tokens": archived_count,
            "genesis_hash": self.genesis_hash,
            "current_position": self.current_position,
            "avg_amplitude": avg_amplitude
        }


@dataclass
class IdentityLockpoint:
    """
    Periodic snapshot of complete PMM state for long-term coherence verification.
    """
    
    lockpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    chain_position: int = 0
    
    # Full state snapshot
    personality_snapshot: Dict[str, Any] = field(default_factory=dict)
    narrative_snapshot: Dict[str, Any] = field(default_factory=dict)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Chain state at lockpoint
    chain_length: int = 0
    chain_hash: str = ""
    
    # Verification
    integrity_hash: str = ""
    
    def create_snapshot(self, pmm_model: Any) -> str:
        """Create complete state snapshot and return integrity hash."""
        from dataclasses import asdict
        
        # Capture core state
        self.personality_snapshot = asdict(pmm_model.personality)
        self.narrative_snapshot = asdict(pmm_model.narrative_identity)  
        self.metrics_snapshot = asdict(pmm_model.metrics)
        
        # Create integrity hash
        snapshot_data = {
            "lockpoint_id": self.lockpoint_id,
            "created_at": self.created_at,
            "chain_position": self.chain_position,
            "personality": self.personality_snapshot,
            "narrative": self.narrative_snapshot,
            "metrics": self.metrics_snapshot,
            "chain_length": self.chain_length,
            "chain_hash": self.chain_hash
        }
        
        snapshot_str = json.dumps(snapshot_data, sort_keys=True)
        self.integrity_hash = hashlib.sha256(snapshot_str.encode()).hexdigest()
        
        return self.integrity_hash
    
    def verify_integrity(self) -> bool:
        """Verify lockpoint hasn't been tampered with."""
        snapshot_data = {
            "lockpoint_id": self.lockpoint_id,
            "created_at": self.created_at,
            "chain_position": self.chain_position,
            "personality": self.personality_snapshot,
            "narrative": self.narrative_snapshot,
            "metrics": self.metrics_snapshot,
            "chain_length": self.chain_length,
            "chain_hash": self.chain_hash
        }
        
        snapshot_str = json.dumps(snapshot_data, sort_keys=True)
        computed_hash = hashlib.sha256(snapshot_str.encode()).hexdigest()
        
        return computed_hash == self.integrity_hash


@dataclass
class MemoryArchive:
    """
    Compressed storage for low-activation memory tokens.
    """
    
    archive_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Archive metadata
    token_count: int = 0
    size_bytes: int = 0
    compression_ratio: float = 1.0
    
    # Thematic clustering
    themes: Dict[str, List[str]] = field(default_factory=dict)  # theme -> token_ids
    summaries: Dict[str, str] = field(default_factory=dict)     # theme -> summary
    
    # Storage path
    storage_path: str = ""
    
    def add_theme_cluster(self, theme: str, token_ids: List[str], summary: str):
        """Add a thematic cluster of archived tokens."""
        self.themes[theme] = token_ids
        self.summaries[theme] = summary
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """Get archive statistics."""
        return {
            "archive_id": self.archive_id,
            "created_at": self.created_at,
            "token_count": self.token_count,
            "size_bytes": self.size_bytes,
            "compression_ratio": self.compression_ratio,
            "theme_count": len(self.themes),
            "themes": list(self.themes.keys())
        }
