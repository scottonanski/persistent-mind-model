"""
Memory Archive & Compression Engine - Layer 4 Implementation

Implements thematic clustering, compression, and archival system for managing
memory growth while preserving complete historical integrity.
"""

from __future__ import annotations
import json
import gzip
import lzma
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .memory_token import MemoryToken, MemoryArchive, IdentityLockpoint
from .enhanced_model import ArchiveConfig


@dataclass
class ThemeCluster:
    """
    Thematic cluster of related memory tokens.
    """
    
    cluster_id: str
    theme_label: str
    token_ids: List[str]
    centroid_summary: str
    salience_score: float
    temporal_span: Tuple[str, str]  # (earliest, latest)
    representative_tokens: List[str]  # Most representative token IDs
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about this cluster."""
        return {
            "cluster_id": self.cluster_id,
            "theme": self.theme_label,
            "token_count": len(self.token_ids),
            "salience": self.salience_score,
            "temporal_span_days": self._calculate_span_days(),
            "representative_count": len(self.representative_tokens)
        }
    
    def _calculate_span_days(self) -> float:
        """Calculate temporal span in days."""
        try:
            start = datetime.fromisoformat(self.temporal_span[0].replace('Z', '+00:00'))
            end = datetime.fromisoformat(self.temporal_span[1].replace('Z', '+00:00'))
            return (end - start).total_seconds() / 86400
        except:
            return 0.0


class ArchiveEngine:
    """
    Core engine for memory archival, compression, and thematic clustering.
    
    Manages the transition of low-activation memories from active storage
    to compressed archives while maintaining full retrievability.
    """
    
    def __init__(self, config: ArchiveConfig):
        self.config = config
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Ensure archive directory exists
        os.makedirs(config.archive_base_path, exist_ok=True)
    
    def should_trigger_archival(self, 
                               active_tokens: Dict[str, MemoryToken],
                               active_token_ids: List[str]) -> bool:
        """
        Determine if archival should be triggered based on configured thresholds.
        """
        # Check active token count
        if len(active_token_ids) > self.config.max_active_tokens:
            return True
        
        # Check for old low-amplitude tokens
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.age_threshold_days)
        old_low_tokens = 0
        
        for token_id in active_token_ids:
            if token_id not in active_tokens:
                continue
            
            token = active_tokens[token_id]
            token_date = datetime.fromisoformat(token.created_at.replace('Z', '+00:00'))
            
            if (token_date < cutoff_date and 
                token.amplitude < self.config.amplitude_threshold):
                old_low_tokens += 1
        
        # Trigger if we have significant old low-amplitude tokens
        return old_low_tokens > len(active_token_ids) * 0.2
    
    def identify_archival_candidates(self, 
                                   active_tokens: Dict[str, MemoryToken],
                                   active_token_ids: List[str]) -> List[str]:
        """
        Identify tokens that should be archived based on amplitude and age.
        """
        candidates = []
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.age_threshold_days)
        
        for token_id in active_token_ids:
            if token_id not in active_tokens:
                continue
            
            token = active_tokens[token_id]
            token_date = datetime.fromisoformat(token.created_at.replace('Z', '+00:00'))
            
            # Archive if low amplitude OR old with medium-low amplitude
            should_archive = (
                token.amplitude < self.config.amplitude_threshold or
                (token_date < cutoff_date and token.amplitude < 0.3)
            )
            
            if should_archive:
                candidates.append(token_id)
        
        return candidates
    
    def create_thematic_clusters(self, 
                               tokens_to_archive: Dict[str, MemoryToken]) -> List[ThemeCluster]:
        """
        Create thematic clusters from tokens to be archived.
        
        Uses TF-IDF vectorization and K-means clustering to group
        semantically related memories.
        """
        if not tokens_to_archive:
            return []
        
        # Prepare text data for clustering
        token_ids = list(tokens_to_archive.keys())
        token_texts = []
        
        for token_id in token_ids:
            token = tokens_to_archive[token_id]
            # Combine summary and tags for clustering
            text = f"{token.summary} {' '.join(token.tags)}"
            token_texts.append(text)
        
        # Skip clustering if too few tokens
        if len(token_texts) < self.config.min_cluster_size:
            return self._create_single_cluster(tokens_to_archive)
        
        try:
            # Vectorize text
            tfidf_matrix = self.vectorizer.fit_transform(token_texts)
            
            # Determine optimal cluster count
            n_clusters = min(
                self.config.max_clusters,
                max(2, len(token_texts) // self.config.min_cluster_size)
            )
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Create theme clusters
            clusters = []
            for cluster_idx in range(n_clusters):
                cluster_token_ids = [
                    token_ids[i] for i, label in enumerate(cluster_labels) 
                    if label == cluster_idx
                ]
                
                if len(cluster_token_ids) >= self.config.min_cluster_size:
                    cluster = self._build_theme_cluster(
                        cluster_idx, cluster_token_ids, tokens_to_archive
                    )
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            # Fallback to single cluster if clustering fails
            print(f"Clustering failed: {e}, creating single cluster")
            return self._create_single_cluster(tokens_to_archive)
    
    def archive_clusters(self, 
                        clusters: List[ThemeCluster],
                        tokens_to_archive: Dict[str, MemoryToken]) -> MemoryArchive:
        """
        Archive thematic clusters to compressed storage.
        """
        archive = MemoryArchive(
            storage_path=os.path.join(
                self.config.archive_base_path,
                f"archive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )
        )
        
        # Prepare archive data
        archive_data = {
            "metadata": {
                "archive_id": archive.archive_id,
                "created_at": archive.created_at,
                "token_count": len(tokens_to_archive),
                "cluster_count": len(clusters)
            },
            "clusters": {},
            "tokens": {}
        }
        
        # Add clusters and tokens
        for cluster in clusters:
            archive.add_theme_cluster(
                cluster.theme_label,
                cluster.token_ids,
                cluster.centroid_summary
            )
            
            archive_data["clusters"][cluster.cluster_id] = asdict(cluster)
        
        # Add full token data
        for token_id, token in tokens_to_archive.items():
            archive_data["tokens"][token_id] = asdict(token)
        
        # Save archive
        self._save_archive_data(archive, archive_data)
        
        # Update archive statistics
        archive.token_count = len(tokens_to_archive)
        archive.size_bytes = self._calculate_archive_size(archive.storage_path)
        
        return archive
    
    def create_identity_lockpoint(self, 
                                 pmm_model: Any,
                                 chain_position: int) -> IdentityLockpoint:
        """
        Create identity lockpoint for long-term coherence verification.
        """
        lockpoint = IdentityLockpoint(
            chain_position=chain_position,
            chain_length=len(pmm_model.self_knowledge.memory_chain.tokens),
            chain_hash=(pmm_model.self_knowledge.memory_chain.tokens[-1].content_hash 
                       if pmm_model.self_knowledge.memory_chain.tokens else "empty")
        )
        
        # Create comprehensive snapshot
        integrity_hash = lockpoint.create_snapshot(pmm_model)
        
        # Save lockpoint to disk
        lockpoint_path = os.path.join(
            self.config.archive_base_path,
            f"lockpoint_{lockpoint.lockpoint_id}.json"
        )
        
        with open(lockpoint_path, 'w') as f:
            json.dump(asdict(lockpoint), f, indent=2)
        
        return lockpoint
    
    def _create_single_cluster(self, tokens_to_archive: Dict[str, MemoryToken]) -> List[ThemeCluster]:
        """Create a single cluster when clustering is not possible/needed."""
        token_ids = list(tokens_to_archive.keys())
        
        # Calculate temporal span
        timestamps = [tokens_to_archive[tid].created_at for tid in token_ids]
        temporal_span = (min(timestamps), max(timestamps))
        
        # Calculate average salience
        avg_salience = sum(tokens_to_archive[tid].salience for tid in token_ids) / len(token_ids)
        
        # Create summary
        summaries = [tokens_to_archive[tid].summary for tid in token_ids]
        centroid_summary = f"Mixed memories cluster ({len(token_ids)} items)"
        
        cluster = ThemeCluster(
            cluster_id="cluster_0",
            theme_label="mixed_memories",
            token_ids=token_ids,
            centroid_summary=centroid_summary,
            salience_score=avg_salience,
            temporal_span=temporal_span,
            representative_tokens=token_ids[:3]  # First 3 as representatives
        )
        
        return [cluster]
    
    def _build_theme_cluster(self, 
                           cluster_idx: int,
                           cluster_token_ids: List[str],
                           tokens_to_archive: Dict[str, MemoryToken]) -> ThemeCluster:
        """Build a ThemeCluster from clustered token IDs."""
        
        # Calculate cluster statistics
        cluster_tokens = [tokens_to_archive[tid] for tid in cluster_token_ids]
        
        # Temporal span
        timestamps = [token.created_at for token in cluster_tokens]
        temporal_span = (min(timestamps), max(timestamps))
        
        # Average salience
        avg_salience = sum(token.salience for token in cluster_tokens) / len(cluster_tokens)
        
        # Generate theme label from common tags and content
        theme_label = self._generate_theme_label(cluster_tokens)
        
        # Create centroid summary
        centroid_summary = self._create_centroid_summary(cluster_tokens)
        
        # Find representative tokens (highest salience)
        sorted_tokens = sorted(
            cluster_token_ids,
            key=lambda tid: tokens_to_archive[tid].salience,
            reverse=True
        )
        representative_tokens = sorted_tokens[:min(3, len(sorted_tokens))]
        
        return ThemeCluster(
            cluster_id=f"cluster_{cluster_idx}",
            theme_label=theme_label,
            token_ids=cluster_token_ids,
            centroid_summary=centroid_summary,
            salience_score=avg_salience,
            temporal_span=temporal_span,
            representative_tokens=representative_tokens
        )
    
    def _generate_theme_label(self, tokens: List[MemoryToken]) -> str:
        """Generate a theme label from cluster tokens."""
        # Collect all tags
        all_tags = []
        for token in tokens:
            all_tags.extend(token.tags)
        
        # Find most common tags
        tag_counts = defaultdict(int)
        for tag in all_tags:
            tag_counts[tag] += 1
        
        if tag_counts:
            # Use most common tag as theme
            most_common_tag = max(tag_counts.items(), key=lambda x: x[1])[0]
            return most_common_tag
        
        # Fallback to event type analysis
        event_types = [token.event_type for token in tokens]
        type_counts = defaultdict(int)
        for etype in event_types:
            type_counts[etype] += 1
        
        if type_counts:
            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
            return f"{most_common_type}_cluster"
        
        return "mixed_cluster"
    
    def _create_centroid_summary(self, tokens: List[MemoryToken]) -> str:
        """Create a summary representing the cluster centroid."""
        # Use the summary from the highest salience token as base
        highest_salience_token = max(tokens, key=lambda t: t.salience)
        base_summary = highest_salience_token.summary
        
        # Add cluster context
        cluster_summary = (
            f"Cluster of {len(tokens)} related memories. "
            f"Representative: {base_summary[:100]}..."
        )
        
        return cluster_summary
    
    def _save_archive_data(self, archive: MemoryArchive, data: Dict[str, Any]):
        """Save archive data to disk with compression."""
        base_path = archive.storage_path
        
        if self.config.archive_format == "jsonl":
            # Save as compressed JSONL
            if self.config.enable_compression:
                if self.config.compression_algorithm == "gzip":
                    with gzip.open(f"{base_path}.jsonl.gz", 'wt') as f:
                        json.dump(data, f)
                elif self.config.compression_algorithm == "lzma":
                    with lzma.open(f"{base_path}.jsonl.xz", 'wt') as f:
                        json.dump(data, f)
                else:
                    # Fallback to uncompressed
                    with open(f"{base_path}.jsonl", 'w') as f:
                        json.dump(data, f, indent=2)
            else:
                with open(f"{base_path}.jsonl", 'w') as f:
                    json.dump(data, f, indent=2)
        
        elif self.config.archive_format == "sqlite":
            # Save to SQLite database
            self._save_to_sqlite(f"{base_path}.db", data)
        
        else:
            # Default to JSON
            with open(f"{base_path}.json", 'w') as f:
                json.dump(data, f, indent=2)
    
    def _save_to_sqlite(self, db_path: str, data: Dict[str, Any]):
        """Save archive data to SQLite database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS archive_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id TEXT PRIMARY KEY,
                theme_label TEXT,
                centroid_summary TEXT,
                salience_score REAL,
                temporal_span_start TEXT,
                temporal_span_end TEXT,
                token_count INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tokens (
                token_id TEXT PRIMARY KEY,
                created_at TEXT,
                event_type TEXT,
                summary TEXT,
                salience REAL,
                valence REAL,
                amplitude REAL,
                phase REAL,
                tags TEXT,
                cluster_id TEXT
            )
        ''')
        
        # Insert metadata
        for key, value in data["metadata"].items():
            cursor.execute(
                "INSERT OR REPLACE INTO archive_metadata (key, value) VALUES (?, ?)",
                (key, json.dumps(value))
            )
        
        # Insert clusters
        for cluster_id, cluster_data in data["clusters"].items():
            cursor.execute('''
                INSERT OR REPLACE INTO clusters 
                (cluster_id, theme_label, centroid_summary, salience_score, 
                 temporal_span_start, temporal_span_end, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                cluster_id,
                cluster_data["theme_label"],
                cluster_data["centroid_summary"],
                cluster_data["salience_score"],
                cluster_data["temporal_span"][0],
                cluster_data["temporal_span"][1],
                len(cluster_data["token_ids"])
            ))
        
        # Insert tokens
        for token_id, token_data in data["tokens"].items():
            # Find which cluster this token belongs to
            cluster_id = None
            for cid, cluster_data in data["clusters"].items():
                if token_id in cluster_data["token_ids"]:
                    cluster_id = cid
                    break
            
            cursor.execute('''
                INSERT OR REPLACE INTO tokens 
                (token_id, created_at, event_type, summary, salience, valence,
                 amplitude, phase, tags, cluster_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                token_id,
                token_data["created_at"],
                token_data["event_type"],
                token_data["summary"],
                token_data["salience"],
                token_data["valence"],
                token_data["amplitude"],
                token_data["phase"],
                json.dumps(token_data["tags"]),
                cluster_id
            ))
        
        conn.commit()
        conn.close()
    
    def _calculate_archive_size(self, storage_path: str) -> int:
        """Calculate the size of archived data."""
        total_size = 0
        
        # Check for various file extensions
        extensions = [".json", ".jsonl", ".jsonl.gz", ".jsonl.xz", ".db"]
        
        for ext in extensions:
            file_path = storage_path + ext
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
        
        return total_size


class ArchiveRetriever:
    """
    Utility for retrieving data from archived memory clusters.
    """
    
    def __init__(self, config: ArchiveConfig):
        self.config = config
    
    def load_archive(self, archive: MemoryArchive) -> Dict[str, Any]:
        """Load archived data from disk."""
        base_path = archive.storage_path
        
        # Try different file formats
        for ext in [".jsonl.gz", ".jsonl.xz", ".jsonl", ".json", ".db"]:
            file_path = base_path + ext
            if os.path.exists(file_path):
                return self._load_from_file(file_path)
        
        raise FileNotFoundError(f"Archive not found: {base_path}")
    
    def retrieve_tokens_by_theme(self, 
                                archive: MemoryArchive, 
                                theme: str) -> List[Dict[str, Any]]:
        """Retrieve all tokens from a specific theme cluster."""
        archive_data = self.load_archive(archive)
        
        # Find cluster by theme
        target_cluster = None
        for cluster_id, cluster_data in archive_data["clusters"].items():
            if cluster_data["theme_label"] == theme:
                target_cluster = cluster_data
                break
        
        if not target_cluster:
            return []
        
        # Retrieve tokens from cluster
        tokens = []
        for token_id in target_cluster["token_ids"]:
            if token_id in archive_data["tokens"]:
                tokens.append(archive_data["tokens"][token_id])
        
        return tokens
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load archive data from various file formats."""
        if file_path.endswith(".gz"):
            with gzip.open(file_path, 'rt') as f:
                return json.load(f)
        elif file_path.endswith(".xz"):
            with lzma.open(file_path, 'rt') as f:
                return json.load(f)
        elif file_path.endswith(".db"):
            return self._load_from_sqlite(file_path)
        else:
            with open(file_path, 'r') as f:
                return json.load(f)
    
    def _load_from_sqlite(self, db_path: str) -> Dict[str, Any]:
        """Load archive data from SQLite database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Load metadata
        cursor.execute("SELECT key, value FROM archive_metadata")
        metadata = {key: json.loads(value) for key, value in cursor.fetchall()}
        
        # Load clusters
        cursor.execute("SELECT * FROM clusters")
        clusters = {}
        for row in cursor.fetchall():
            cluster_id = row[0]
            clusters[cluster_id] = {
                "cluster_id": row[0],
                "theme_label": row[1],
                "centroid_summary": row[2],
                "salience_score": row[3],
                "temporal_span": [row[4], row[5]],
                "token_count": row[6],
                "token_ids": []  # Will be populated from tokens
            }
        
        # Load tokens and populate cluster token_ids
        cursor.execute("SELECT * FROM tokens")
        tokens = {}
        for row in cursor.fetchall():
            token_id = row[0]
            cluster_id = row[9]
            
            tokens[token_id] = {
                "token_id": row[0],
                "created_at": row[1],
                "event_type": row[2],
                "summary": row[3],
                "salience": row[4],
                "valence": row[5],
                "amplitude": row[6],
                "phase": row[7],
                "tags": json.loads(row[8]) if row[8] else []
            }
            
            # Add to cluster
            if cluster_id and cluster_id in clusters:
                clusters[cluster_id]["token_ids"].append(token_id)
        
        conn.close()
        
        return {
            "metadata": metadata,
            "clusters": clusters,
            "tokens": tokens
        }
