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

from .model import PersistentMindModel
from .validation import validate_model


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
                
                # Validate before returning
                validate_model(data)
                
                # Convert to dataclass (simplified - would need full deserialization)
                model = PersistentMindModel()
                # TODO: Implement proper dataclass deserialization
                return model
                
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                raise ValueError(f"Failed to load model from {self.file_path}: {e}")
    
    def save(self, model: PersistentMindModel) -> None:
        """Save model to file with atomic write."""
        with self.lock:
            # Validate before saving
            model_dict = self._to_dict(model)
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
        # TODO: Implement proper dataclass serialization
        return {}
    
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
