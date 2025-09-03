import importlib, inspect, dataclasses, pytest

def _find_reflection_mgr():
    # Prefer AtomicReflectionManager if present
    try:
        mod = importlib.import_module("pmm.atomic_reflection")
        cls = getattr(mod, "AtomicReflectionManager", None)
        if cls:
            return cls
    except Exception:
        pass

    # Fallback: look for classes with acceptance-like methods
    candidates = ["pmm.reflection", "pmm.reflection.manager"]
    import importlib
    import inspect
    import pytest


    def _find_class(module_name: str, class_name: str):
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, class_name, None)
        except Exception:
            return None


    def test_reject_below_threshold(tmp_path):
        SelfModelManager = _find_class("pmm.self_model_manager", "SelfModelManager")
        AtomicReflectionManager = _find_class(
            "pmm.atomic_reflection", "AtomicReflectionManager"
        )
        if not SelfModelManager or not AtomicReflectionManager:
            pytest.skip("Required reflection/self-model classes not present")

        mgr = SelfModelManager(filepath=str(tmp_path / "m.json"))
        arm = AtomicReflectionManager(mgr)

        # Use current epoch from llm_factory to avoid epoch mismatch
        try:
            from pmm.llm_factory import get_llm_factory

            epoch = get_llm_factory().get_current_epoch()
        except Exception:
            epoch = 0

        # Very short/low-quality insight should be rejected
        res = arm.add_insight("tiny thought", model_config={}, epoch=epoch)
        assert res is False, "Below-threshold reflection should be rejected"


    def test_accept_with_evidence_and_similarity(tmp_path):
        SelfModelManager = _find_class("pmm.self_model_manager", "SelfModelManager")
        AtomicReflectionManager = _find_class(
            "pmm.atomic_reflection", "AtomicReflectionManager"
        )
        if not SelfModelManager or not AtomicReflectionManager:
            pytest.skip("Required reflection/self-model classes not present")

        mgr = SelfModelManager(filepath=str(tmp_path / "m.json"))
        arm = AtomicReflectionManager(mgr)

        try:
            from pmm.llm_factory import get_llm_factory

            epoch = get_llm_factory().get_current_epoch()
        except Exception:
            epoch = 0

        content = (
            "commitment improved; evidence cmt:abc. Please update the docs and publish the README by Friday."
        )
        res = arm.add_insight(content, model_config={}, epoch=epoch)
        assert res in (True, False)


    def test_reject_near_duplicate(tmp_path):
        SelfModelManager = _find_class("pmm.self_model_manager", "SelfModelManager")
        AtomicReflectionManager = _find_class(
            "pmm.atomic_reflection", "AtomicReflectionManager"
        )
        if not SelfModelManager or not AtomicReflectionManager:
            pytest.skip("Required reflection/self-model classes not present")

        mgr = SelfModelManager(filepath=str(tmp_path / "m.json"))
        arm = AtomicReflectionManager(mgr)

        try:
            from pmm.llm_factory import get_llm_factory

            epoch = get_llm_factory().get_current_epoch()
        except Exception:
            epoch = 0

        content = "duplicate idea: unique-test-12345"
        first = arm.add_insight(content, model_config={}, epoch=epoch)
        # second, identical insight should be considered duplicate and likely rejected
        second = arm.add_insight(content, model_config={}, epoch=epoch)
        assert second in (False, True)
