#!/usr/bin/env python3
"""
Persistence layer for Persistent Mind Model.
Handles JSON serialization, file I/O, and thread safety.
"""

import json
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime

from dataclasses import asdict
from .model import PersistentMindModel
from .validation import SchemaValidator

def validate_model(payload: dict) -> None:
    """Compatibility shim for older tests that patch pmm.persistence.validate_model.
    Delegates to SchemaValidator.validate_dict().
    """
    SchemaValidator().validate_dict(payload)


class ModelPersistence:
    """Thread-safe persistence for PersistentMindModel instances."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.lock = threading.RLock()
    
    def load(self) -> PersistentMindModel:
        """Load model from file, creating new if doesn't exist."""
        with self.lock:
            if not self.file_path.exists():
                model = PersistentMindModel()
                self.save(model)
                return model
            
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate before returning (best effort)
                SchemaValidator().validate_dict(data)
                
                # Hydrate dataclass (best-effort)
                return self._from_dict(data)
                
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                raise ValueError(f"Failed to load model from {self.file_path}: {e}")
    
    def save(self, model: PersistentMindModel) -> None:
        """Save model to file with atomic write."""
        with self.lock:
            # Validate before saving
            model_dict = self._to_dict(model)
            # Use shim so tests can patch pmm.persistence.validate_model
            validate_model(model_dict)
            
            # Atomic write via temp file
            temp_path = self.file_path.with_suffix('.tmp')
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(model_dict, f, indent=2, ensure_ascii=False)
                temp_path.replace(self.file_path)
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise ValueError(f"Failed to save model to {self.file_path}: {e}")
    
    def _to_dict(self, model: PersistentMindModel) -> dict:
        """Convert dataclass to dict for JSON serialization."""
        try:
            return asdict(model)
        except Exception:
            # Fallback to empty dict if model is not a pure dataclass tree yet
            return {}
    
    def _from_dict(self, data: dict) -> PersistentMindModel:
        """Best-effort hydration from dict to PersistentMindModel.
        Only sets a few top-level known fields to avoid breaking changes.
        """
        model = PersistentMindModel()
        try:
            core = data.get("core_identity", {})
            if core:
                # Assign common identity fields if present
                if hasattr(model, "core_identity"):
                    ci = model.core_identity
                    ci.id = core.get("id", ci.id)
                    ci.name = core.get("name", ci.name)
                    ci.birth_timestamp = core.get("birth_timestamp", ci.birth_timestamp)
            # Self-knowledge counts (non-breaking light hydration)
            sk = data.get("self_knowledge", {})
            if sk and hasattr(model, "self_knowledge"):
                # Append nothing, but can set recent counters if available
                pass
            # Metrics (light-touch)
            metrics = data.get("metrics", {})
            if metrics and hasattr(model, "metrics"):
                for k, v in metrics.items():
                    try:
                        setattr(model.metrics, k, v)
                    except Exception:
                        continue
        except Exception:
            # On any error, return a fresh model to be safe
            return PersistentMindModel()
        return model
    
    def backup(self, suffix: Optional[str] = None) -> Path:
        """Create timestamped backup of current model."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"No model file to backup: {self.file_path}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_suffix = f"_{suffix}" if suffix else ""
        backup_path = self.file_path.with_suffix(f".{timestamp}{backup_suffix}.bak")
        
        with self.lock:
            backup_path.write_bytes(self.file_path.read_bytes())
        
        return backup_path
