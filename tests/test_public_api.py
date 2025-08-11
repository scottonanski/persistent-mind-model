import tempfile
from pathlib import Path

from pmm.enhanced_manager import EnhancedSelfModelManager
from pmm.enhanced_model import NextStageConfig


def test_basic_event_recall_and_integrity():
    with tempfile.TemporaryDirectory() as tmp:
        model_path = str(Path(tmp) / "test_model.json")
        cfg = NextStageConfig(
            enable_memory_tokens=True,
            enable_archival=True,
            enable_recall=False,  # disable embeddings/recall engine for fast tests
            enable_local_inference=False,  # disable inference stack for fast tests
            enable_integrity_checks=True,
        )
        mgr = EnhancedSelfModelManager(
            model_path=model_path, enable_next_stage=True, config=cfg
        )

        # Add a few events
        mgr.add_event("Learned about integrity verification")
        mgr.add_event("Experimented with memory recall heuristics")
        mgr.add_thought("Consider archiving low-salience tokens")

        # Recall
        results = mgr.recall_memories("integrity", max_results=5)
        assert isinstance(results, list)

        # Verify integrity
        integrity = mgr.verify_integrity()
        # Integrity may return a dict from IntegrityEngine
        assert isinstance(integrity, dict)
        assert integrity.get("chain_valid") is True

        # Export identity to a temp dir
        export_dir = Path(tmp) / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        manifest = mgr.export_identity(str(export_dir))
        assert hasattr(manifest, "total_tokens")
        assert manifest.total_tokens >= 1
