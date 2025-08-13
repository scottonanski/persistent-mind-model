import tempfile
from pathlib import Path

from pmm.self_model_manager import SelfModelManager


def test_basic_event_and_thought_management():
    """Test core PMM functionality with SelfModelManager."""
    with tempfile.TemporaryDirectory() as tmp:
        model_path = str(Path(tmp) / "test_model.json")
        mgr = SelfModelManager(model_path=model_path)

        # Add a few events
        mgr.add_event("Learned about PMM system")
        mgr.add_event("Experimented with personality traits")
        mgr.add_thought("Consider improving trait evolution")

        # Verify events were added
        assert len(mgr.model.self_knowledge.autobiographical_events) >= 2
        assert len(mgr.model.self_knowledge.thoughts) >= 1

        # Test personality traits access
        big5 = mgr.get_big5()
        assert isinstance(big5, dict)
        assert "openness" in big5
        assert "conscientiousness" in big5
        assert "extraversion" in big5
        assert "agreeableness" in big5
        assert "neuroticism" in big5

        # Test model persistence
        mgr.save_model()
        
        # Load in new manager instance
        mgr2 = SelfModelManager(model_path=model_path)
        assert len(mgr2.model.self_knowledge.autobiographical_events) >= 2
        assert len(mgr2.model.self_knowledge.thoughts) >= 1
