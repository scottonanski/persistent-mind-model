import importlib
import inspect


def test_analyzers_do_not_import_sim():
    forbidden = "bandit_reward_reshaping"
    for m in ("pmm_sanity_metrics", "quick_phase3b_test"):
        mod = importlib.import_module(f"metrics.{m}")
        src = inspect.getsource(mod)
        assert forbidden not in src, f"{m} imports {forbidden}"
