from pmm.self_model_manager import SelfModelManager


def test_drift_applies_and_bounds(tmp_path):
    mgr = SelfModelManager(filepath=str(tmp_path / "m.json"))
    mgr.add_event(
        "test",
        effects=[
            {
                "target": "personality.traits.big5.conscientiousness.score",
                "delta": 0.5,
                "confidence": 0.7,
            }
        ],
    )
    net = mgr.apply_drift_and_save()
    assert "personality.traits.big5.conscientiousness" in net
    val = mgr.model.personality.traits.big5.conscientiousness.score
    assert 0.05 <= val <= 0.95
