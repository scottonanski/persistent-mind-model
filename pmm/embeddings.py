"""
Local-only embeddings utility for PMM.

- Lazy loads a sentence-transformers model
- Auto device selection (CUDA if available, else CPU)
- Offline-friendly: attempts local cache first; if unavailable, disables embeddings gracefully
- Returns float32 bytes suitable for SQLite BLOB storage
"""
from __future__ import annotations
import os
import threading
from typing import Optional


class Embedder:
    """Singleton-style embedder with lazy model loading and safety guards."""

    _instance_lock = threading.Lock()
    _instance: Optional["Embedder"] = None

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._enabled = None  # tri-state: None (unknown), True, False
        self._lock = threading.Lock()
        self._model_name: Optional[str] = None

    @property
    def enabled(self) -> bool:
        if self._enabled is None:
            # Default: enabled unless env explicitly disables
            val = os.getenv("PMM_ENABLE_EMBEDDINGS", "auto").strip().lower()
            if val in ("0", "false", "no", "off"):
                self._enabled = False
            else:
                self._enabled = True
        return bool(self._enabled)

    def _select_device(self) -> str:
        if self._device:
            return self._device
        prefer = os.getenv("PMM_DEVICE", "auto").strip().lower()
        try:
            import torch  # noqa
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        if prefer in ("cuda", "gpu") and cuda_ok:
            self._device = "cuda"
        elif prefer in ("cpu",):
            self._device = "cpu"
        else:
            self._device = "cuda" if cuda_ok else "cpu"
        return self._device

    def _load_model(self) -> None:
        if self._model is not None or not self.enabled:
            return
        with self._lock:
            if self._model is not None:
                return
            model_name = os.getenv("PMM_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
            # Try local-only first to honor offline/privacy requirements
            try:
                from sentence_transformers import SentenceTransformer
                device = self._select_device()
                self._model = SentenceTransformer(model_name, device=device, cache_folder=None, local_files_only=True)
                self._model_name = model_name
            except Exception:
                # If local cache missing, disable gracefully (no network by default)
                self._model = None
                self._enabled = False
                self._model_name = model_name

    def ensure_loaded(self) -> bool:
        """Public: attempt to load the model now. Returns True if ready."""
        self._load_model()
        return self._model is not None and self.enabled

    def encode_bytes(self, text: str) -> Optional[bytes]:
        """Return float32 bytes for the embedding or None if disabled/unavailable.

        Uses small safety guards to avoid crashing on OOM or long inputs.
        """
        if not text or not self.enabled:
            return None
        if self._model is None:
            self._load_model()
        if self._model is None:
            return None
        try:
            # Hard cap length to avoid pathological inputs
            raw = (text or "")[:4096]
            vec = self._model.encode(raw, normalize_embeddings=True)
            # Ensure numpy float32 bytes
            import numpy as np
            arr = np.asarray(vec, dtype=np.float32)
            return arr.tobytes(order="C")
        except RuntimeError:
            # e.g., CUDA OOM; disable for remainder of process
            self._enabled = False
            return None
        except Exception:
            return None

    def get_status(self) -> dict:
        """Return lightweight status for UI/diagnostics."""
        return {
            "enabled": bool(self.enabled),
            "loaded": self._model is not None,
            "device": self._device or self._select_device(),
            "model_name": self._model_name or os.getenv("PMM_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip(),
        }
