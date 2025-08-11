from pmm.self_model_manager import SelfModelManager


def test_round_trip_ok(tmp_path):
    path = tmp_path / "m.json"
    mgr = SelfModelManager(filepath=str(path))
    mgr.save_model()
    mgr2 = SelfModelManager(filepath=str(path))
    assert mgr2.model.core_identity.id == mgr.model.core_identity.id
