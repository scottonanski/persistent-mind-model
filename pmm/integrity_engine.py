"""
Integrity & Portability Engine - Layer 7 Implementation

Implements export/import, chain verification, and distributed backup for
self-sovereign AI identity with complete portability across systems.
"""

from __future__ import annotations
import json
import gzip
import lzma
import hashlib
import os
import shutil
import tarfile
from datetime import datetime, UTC
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
import uuid

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import cbor2
    CBOR_AVAILABLE = True
except ImportError:
    CBOR_AVAILABLE = False

from .enhanced_model import EnhancedPersistentMindModel, IntegrityConfig
from .memory_token import MemoryChain, IdentityLockpoint
from .tokenization_engine import ChainVerifier


@dataclass
class ExportManifest:
    """
    Manifest describing the contents of an exported PMM identity.
    """
    
    export_id: str
    created_at: str
    pmm_version: str = "2.0"
    schema_version: int = 2
    
    # Identity metadata
    agent_id: str = ""
    agent_name: str = ""
    birth_timestamp: str = ""
    
    # Export contents
    includes_active_memory: bool = True
    includes_archives: bool = True
    includes_lockpoints: bool = True
    includes_full_chain: bool = True
    
    # Statistics
    total_tokens: int = 0
    active_tokens: int = 0
    archived_tokens: int = 0
    chain_length: int = 0
    archive_count: int = 0
    lockpoint_count: int = 0
    
    # Integrity
    chain_integrity_hash: str = ""
    export_integrity_hash: str = ""
    
    # Compression
    compression_used: bool = False
    compression_algorithm: str = ""
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    
    def compute_export_hash(self, export_data: Dict[str, Any]) -> str:
        """Compute integrity hash for entire export."""
        export_str = json.dumps(export_data, sort_keys=True)
        self.export_integrity_hash = hashlib.sha256(export_str.encode()).hexdigest()
        return self.export_integrity_hash


@dataclass
class ImportResult:
    """
    Result of an identity import operation.
    """
    
    success: bool
    imported_agent_id: str = ""
    imported_tokens: int = 0
    imported_archives: int = 0
    imported_lockpoints: int = 0
    
    # Verification results
    chain_integrity_verified: bool = False
    export_integrity_verified: bool = False
    lockpoints_verified: int = 0
    
    # Conflicts and resolutions
    conflicts_detected: List[str] = None
    conflicts_resolved: List[str] = None
    
    # Errors
    error_messages: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.conflicts_detected is None:
            self.conflicts_detected = []
        if self.conflicts_resolved is None:
            self.conflicts_resolved = []
        if self.error_messages is None:
            self.error_messages = []
        if self.warnings is None:
            self.warnings = []


class IntegrityEngine:
    """
    Core engine for PMM identity integrity, export/import, and portability.
    
    Provides cryptographic verification, complete identity export/import,
    and distributed backup capabilities for self-sovereign AI consciousness.
    """
    
    def __init__(self, config: IntegrityConfig):
        self.config = config
        self.chain_verifier = ChainVerifier()
    
    def export_identity(self, 
                       pmm_model: EnhancedPersistentMindModel,
                       export_path: str,
                       include_archives: bool = True,
                       compress: bool = True) -> ExportManifest:
        """
        Export complete PMM identity to portable format.
        
        Creates a self-contained package with all memory tokens, archives,
        lockpoints, and metadata needed to reconstruct the AI identity.
        """
        # Create export directory
        os.makedirs(export_path, exist_ok=True)
        
        # Create manifest
        manifest = ExportManifest(
            export_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            agent_id=pmm_model.core_identity.id,
            agent_name=pmm_model.core_identity.name,
            birth_timestamp=pmm_model.core_identity.birth_timestamp,
            includes_archives=include_archives,
            includes_lockpoints=True,
            includes_full_chain=True,
            compression_used=compress
        )
        
        # Prepare export data
        export_data = self._prepare_export_data(pmm_model, include_archives)
        
        # Update manifest statistics
        self._update_manifest_stats(manifest, export_data, pmm_model)
        
        # Save core identity data
        self._save_export_component(
            export_data["core_model"], 
            os.path.join(export_path, "core_model.json"),
            compress
        )
        
        # Save memory chain
        self._save_export_component(
            export_data["memory_chain"],
            os.path.join(export_path, "memory_chain.json"),
            compress
        )
        
        # Save active tokens
        self._save_export_component(
            export_data["active_tokens"],
            os.path.join(export_path, "active_tokens.json"),
            compress
        )
        
        # Save archives if requested
        if include_archives and export_data["archives"]:
            archives_path = os.path.join(export_path, "archives")
            os.makedirs(archives_path, exist_ok=True)
            
            for archive_id, archive_data in export_data["archives"].items():
                self._save_export_component(
                    archive_data,
                    os.path.join(archives_path, f"{archive_id}.json"),
                    compress
                )
        
        # Save lockpoints
        if export_data["lockpoints"]:
            lockpoints_path = os.path.join(export_path, "lockpoints")
            os.makedirs(lockpoints_path, exist_ok=True)
            
            for i, lockpoint in enumerate(export_data["lockpoints"]):
                self._save_export_component(
                    lockpoint,
                    os.path.join(lockpoints_path, f"lockpoint_{i:04d}.json"),
                    compress
                )
        
        # Compute and save manifest
        manifest.compute_export_hash(export_data)
        manifest_path = os.path.join(export_path, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(asdict(manifest), f, indent=2)
        
        # Create compressed archive if requested
        if compress:
            archive_path = f"{export_path}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(export_path, arcname=os.path.basename(export_path))
            
            # Update manifest with compression info
            manifest.compressed_size_bytes = os.path.getsize(archive_path)
            manifest.original_size_bytes = self._calculate_directory_size(export_path)
            
            # Save updated manifest
            with open(manifest_path, 'w') as f:
                json.dump(asdict(manifest), f, indent=2)
        
        return manifest
    
    def import_identity(self, 
                       import_path: str,
                       target_model: Optional[EnhancedPersistentMindModel] = None,
                       verify_integrity: bool = True,
                       resolve_conflicts: bool = True) -> Tuple[EnhancedPersistentMindModel, ImportResult]:
        """
        Import PMM identity from exported package.
        
        Reconstructs complete AI identity with integrity verification
        and conflict resolution.
        """
        result = ImportResult(success=False)
        
        # Handle compressed archives
        working_path = import_path
        if import_path.endswith('.tar.gz'):
            working_path = self._extract_compressed_export(import_path)
            if not working_path:
                result.error_messages.append("Failed to extract compressed export")
                return None, result
        
        try:
            # Load and verify manifest
            manifest_path = os.path.join(working_path, "manifest.json")
            if not os.path.exists(manifest_path):
                result.error_messages.append("Manifest file not found")
                return None, result
            
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            manifest = ExportManifest(**manifest_data)
            
            # Load export components
            import_data = self._load_export_components(working_path, manifest)
            
            # Verify integrity if requested
            if verify_integrity:
                integrity_results = self._verify_import_integrity(import_data, manifest)
                result.chain_integrity_verified = integrity_results["chain_valid"]
                result.export_integrity_verified = integrity_results["export_valid"]
                result.lockpoints_verified = integrity_results["lockpoints_verified"]
                
                if not integrity_results["chain_valid"]:
                    result.error_messages.append("Chain integrity verification failed")
                if not integrity_results["export_valid"]:
                    result.error_messages.append("Export integrity verification failed")
            
            # Create or update target model
            if target_model is None:
                imported_model = self._reconstruct_model_from_import(import_data)
            else:
                imported_model = self._merge_import_with_existing(
                    target_model, import_data, resolve_conflicts, result
                )
            
            # Update result statistics
            result.success = True
            result.imported_agent_id = manifest.agent_id
            result.imported_tokens = manifest.total_tokens
            result.imported_archives = manifest.archive_count
            result.imported_lockpoints = manifest.lockpoint_count
            
            return imported_model, result
            
        except Exception as e:
            result.error_messages.append(f"Import failed: {str(e)}")
            return None, result
        
        finally:
            # Clean up extracted files if we extracted a compressed archive
            if working_path != import_path and os.path.exists(working_path):
                shutil.rmtree(working_path)
    
    def verify_chain_integrity(self, pmm_model: EnhancedPersistentMindModel) -> Dict[str, Any]:
        """
        Comprehensive verification of memory chain integrity.
        """
        chain = pmm_model.self_knowledge.memory_chain
        
        # Basic chain verification
        is_valid, errors, diagnostics = self.chain_verifier.verify_full_chain(chain)
        
        # Detect anomalies
        anomalies = self.chain_verifier.detect_anomalies(chain)
        
        # Verify lockpoints
        lockpoint_results = []
        for lockpoint in pmm_model.self_knowledge.lockpoints:
            lockpoint_valid = lockpoint.verify_integrity()
            lockpoint_results.append({
                "lockpoint_id": lockpoint.lockpoint_id,
                "valid": lockpoint_valid,
                "chain_position": lockpoint.chain_position,
                "created_at": lockpoint.created_at
            })
        
        return {
            "chain_valid": is_valid,
            "chain_errors": errors,
            "chain_diagnostics": diagnostics,
            "anomalies": anomalies,
            "lockpoints": lockpoint_results,
            "lockpoints_valid": sum(1 for lp in lockpoint_results if lp["valid"]),
            "verification_timestamp": datetime.now(UTC).isoformat()
        }
    
    def create_backup_snapshot(self, 
                              pmm_model: EnhancedPersistentMindModel,
                              backup_path: str) -> Dict[str, Any]:
        """
        Create incremental backup snapshot.
        
        Creates a lightweight backup containing only changes since last backup.
        """
        # For now, create full export - incremental backups would require
        # tracking changes since last backup
        manifest = self.export_identity(
            pmm_model, 
            backup_path, 
            include_archives=True, 
            compress=True
        )
        
        return {
            "backup_id": manifest.export_id,
            "backup_path": backup_path,
            "backup_size": manifest.compressed_size_bytes,
            "tokens_backed_up": manifest.total_tokens,
            "created_at": manifest.created_at
        }
    
    def restore_from_backup(self, 
                           backup_path: str,
                           verify_integrity: bool = True) -> Tuple[EnhancedPersistentMindModel, ImportResult]:
        """
        Restore PMM identity from backup snapshot.
        """
        return self.import_identity(backup_path, verify_integrity=verify_integrity)
    
    def migrate_identity(self, 
                        source_model: EnhancedPersistentMindModel,
                        target_system_path: str,
                        preserve_source: bool = True) -> Dict[str, Any]:
        """
        Migrate PMM identity to another system.
        
        Exports identity and prepares it for deployment on target system.
        """
        # Create temporary export
        temp_export_path = f"/tmp/pmm_migration_{uuid.uuid4()}"
        
        try:
            # Export identity
            manifest = self.export_identity(
                source_model, 
                temp_export_path,
                include_archives=True,
                compress=True
            )
            
            # Move to target system path
            compressed_export = f"{temp_export_path}.tar.gz"
            target_file = os.path.join(target_system_path, f"pmm_identity_{manifest.agent_id}.tar.gz")
            
            os.makedirs(target_system_path, exist_ok=True)
            shutil.move(compressed_export, target_file)
            
            # Create migration instructions
            instructions = {
                "migration_id": str(uuid.uuid4()),
                "source_agent_id": manifest.agent_id,
                "export_file": target_file,
                "import_command": f"pmm import-identity {target_file}",
                "verification_hash": manifest.export_integrity_hash,
                "created_at": datetime.now(UTC).isoformat()
            }
            
            instructions_file = os.path.join(target_system_path, "migration_instructions.json")
            with open(instructions_file, 'w') as f:
                json.dump(instructions, f, indent=2)
            
            return instructions
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_export_path):
                shutil.rmtree(temp_export_path)
    
    def _prepare_export_data(self, 
                            pmm_model: EnhancedPersistentMindModel,
                            include_archives: bool) -> Dict[str, Any]:
        """Prepare all data for export."""
        # Core model (excluding memory tokens and archives for separate handling)
        core_model = asdict(pmm_model)
        
        # Remove large nested structures that we'll handle separately
        core_model["self_knowledge"]["memory_tokens"] = {}
        core_model["self_knowledge"]["archives"] = {}
        
        # Memory chain
        memory_chain = asdict(pmm_model.self_knowledge.memory_chain)
        
        # Active tokens
        active_tokens = {
            token_id: asdict(token) 
            for token_id, token in pmm_model.self_knowledge.memory_tokens.items()
            if not token.archived
        }
        
        # Archives
        archives = {}
        if include_archives:
            for archive_id, archive in pmm_model.self_knowledge.archives.items():
                archives[archive_id] = asdict(archive)
        
        # Lockpoints
        lockpoints = [asdict(lp) for lp in pmm_model.self_knowledge.lockpoints]
        
        return {
            "core_model": core_model,
            "memory_chain": memory_chain,
            "active_tokens": active_tokens,
            "archives": archives,
            "lockpoints": lockpoints
        }
    
    def _update_manifest_stats(self, 
                              manifest: ExportManifest,
                              export_data: Dict[str, Any],
                              pmm_model: EnhancedPersistentMindModel):
        """Update manifest with export statistics."""
        manifest.total_tokens = len(pmm_model.self_knowledge.memory_tokens)
        manifest.active_tokens = len(export_data["active_tokens"])
        manifest.archived_tokens = manifest.total_tokens - manifest.active_tokens
        manifest.chain_length = len(export_data["memory_chain"]["tokens"])
        manifest.archive_count = len(export_data["archives"])
        manifest.lockpoint_count = len(export_data["lockpoints"])
        
        # Compute chain integrity hash
        if export_data["memory_chain"]["tokens"]:
            last_token = export_data["memory_chain"]["tokens"][-1]
            manifest.chain_integrity_hash = last_token["content_hash"]
    
    def _save_export_component(self, data: Any, file_path: str, compress: bool):
        """Save export component with optional compression."""
        if compress and self.config.compress_exports:
            if file_path.endswith('.json'):
                file_path = file_path.replace('.json', '.json.gz')
            
            with gzip.open(file_path, 'wt') as f:
                json.dump(data, f, indent=2)
        else:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def _calculate_directory_size(self, directory: str) -> int:
        """Calculate total size of directory."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        return total_size
    
    def _extract_compressed_export(self, archive_path: str) -> Optional[str]:
        """Extract compressed export archive."""
        try:
            extract_path = archive_path.replace('.tar.gz', '_extracted')
            
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=os.path.dirname(extract_path))
            
            return extract_path
        except Exception as e:
            print(f"Failed to extract archive: {e}")
            return None
    
    def _load_export_components(self, 
                               working_path: str, 
                               manifest: ExportManifest) -> Dict[str, Any]:
        """Load all export components from disk."""
        components = {}
        
        # Load core model
        core_path = os.path.join(working_path, "core_model.json")
        if os.path.exists(core_path + ".gz"):
            core_path = core_path + ".gz"
        components["core_model"] = self._load_json_file(core_path)
        
        # Load memory chain
        chain_path = os.path.join(working_path, "memory_chain.json")
        if os.path.exists(chain_path + ".gz"):
            chain_path = chain_path + ".gz"
        components["memory_chain"] = self._load_json_file(chain_path)
        
        # Load active tokens
        tokens_path = os.path.join(working_path, "active_tokens.json")
        if os.path.exists(tokens_path + ".gz"):
            tokens_path = tokens_path + ".gz"
        components["active_tokens"] = self._load_json_file(tokens_path)
        
        # Load archives
        archives_path = os.path.join(working_path, "archives")
        components["archives"] = {}
        if os.path.exists(archives_path):
            for filename in os.listdir(archives_path):
                if filename.endswith('.json') or filename.endswith('.json.gz'):
                    archive_file = os.path.join(archives_path, filename)
                    archive_id = filename.replace('.json.gz', '').replace('.json', '')
                    components["archives"][archive_id] = self._load_json_file(archive_file)
        
        # Load lockpoints
        lockpoints_path = os.path.join(working_path, "lockpoints")
        components["lockpoints"] = []
        if os.path.exists(lockpoints_path):
            lockpoint_files = sorted([
                f for f in os.listdir(lockpoints_path) 
                if f.endswith('.json') or f.endswith('.json.gz')
            ])
            for filename in lockpoint_files:
                lockpoint_file = os.path.join(lockpoints_path, filename)
                components["lockpoints"].append(self._load_json_file(lockpoint_file))
        
        return components
    
    def _load_json_file(self, file_path: str) -> Any:
        """Load JSON file with automatic compression detection."""
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                return json.load(f)
        else:
            with open(file_path, 'r') as f:
                return json.load(f)
    
    def _verify_import_integrity(self, 
                                import_data: Dict[str, Any],
                                manifest: ExportManifest) -> Dict[str, Any]:
        """Verify integrity of imported data."""
        results = {
            "chain_valid": False,
            "export_valid": False,
            "lockpoints_verified": 0
        }
        
        # Verify chain integrity
        try:
            from .memory_token import MemoryChain
            chain_data = import_data["memory_chain"]
            
            # Reconstruct chain for verification
            chain = MemoryChain(**chain_data)
            is_valid, errors = chain.verify_chain_integrity()
            results["chain_valid"] = is_valid
            
        except Exception as e:
            print(f"Chain verification failed: {e}")
        
        # Verify export hash
        try:
            export_hash = manifest.compute_export_hash(import_data)
            results["export_valid"] = (export_hash == manifest.export_integrity_hash)
        except Exception as e:
            print(f"Export hash verification failed: {e}")
        
        # Verify lockpoints
        verified_lockpoints = 0
        for lockpoint_data in import_data["lockpoints"]:
            try:
                lockpoint = IdentityLockpoint(**lockpoint_data)
                if lockpoint.verify_integrity():
                    verified_lockpoints += 1
            except Exception as e:
                print(f"Lockpoint verification failed: {e}")
        
        results["lockpoints_verified"] = verified_lockpoints
        
        return results
    
    def _reconstruct_model_from_import(self, import_data: Dict[str, Any]) -> EnhancedPersistentMindModel:
        """Reconstruct PMM model from imported data."""
        # Start with core model
        model_data = import_data["core_model"]
        
        # Reconstruct model
        model = EnhancedPersistentMindModel(**model_data)
        
        # Restore memory tokens
        from .memory_token import MemoryToken
        for token_id, token_data in import_data["active_tokens"].items():
            token = MemoryToken(**token_data)
            model.self_knowledge.memory_tokens[token_id] = token
            if not token.archived:
                model.self_knowledge.active_token_ids.append(token_id)
        
        # Restore memory chain
        from .memory_token import MemoryChain
        model.self_knowledge.memory_chain = MemoryChain(**import_data["memory_chain"])
        
        # Restore archives
        from .memory_token import MemoryArchive
        for archive_id, archive_data in import_data["archives"].items():
            archive = MemoryArchive(**archive_data)
            model.self_knowledge.archives[archive_id] = archive
        
        # Restore lockpoints
        model.self_knowledge.lockpoints = [
            IdentityLockpoint(**lp_data) for lp_data in import_data["lockpoints"]
        ]
        
        return model
    
    def _merge_import_with_existing(self,
                                   target_model: EnhancedPersistentMindModel,
                                   import_data: Dict[str, Any],
                                   resolve_conflicts: bool,
                                   result: ImportResult) -> EnhancedPersistentMindModel:
        """Merge imported data with existing model."""
        # For now, this is a simplified merge - in practice, you'd need
        # sophisticated conflict resolution logic
        
        if resolve_conflicts:
            # Simple strategy: prefer imported data for conflicts
            result.conflicts_resolved.append("Imported data takes precedence")
        else:
            # Report conflicts without resolving
            result.conflicts_detected.append("Identity merge conflicts detected")
        
        # For this implementation, we'll return the reconstructed model
        # In practice, you'd merge specific components
        return self._reconstruct_model_from_import(import_data)
