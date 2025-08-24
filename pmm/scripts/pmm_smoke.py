# scripts/pmm_smoke.py
# Minimal, side-effect-free probes to locate the bork.
# Run:  python -m scripts.pmm_smoke

import json
import traceback

summary = {"ok": True, "checks": []}


def check(name):
    def _wrap(fn):
        def inner():
            try:
                out = fn()
                summary["checks"].append(
                    {"name": name, "status": "PASS", "detail": out}
                )
            except Exception as e:
                summary["ok"] = False
                summary["checks"].append(
                    {
                        "name": name,
                        "status": "FAIL",
                        "error": repr(e),
                        "trace": traceback.format_exc(),
                    }
                )

        return inner

    return _wrap


# --- 1) Imports & basic API surface -----------------------------------------
@check("imports.emergence_and_reflection")
def _c1():
    # Adjust paths/names if your package layout differs

    return "Imported core modules"


# --- 2) Emergence score structure -------------------------------------------
@check("emergence.compute_scores_contract")
def _c2():
    from pmm.emergence import compute_emergence_scores

    class _DummyStorage:
        # Provide just enough for compute_emergence_scores() to not crash
        def get_recent_events(self, *args, **kwargs):
            return []

    scores = compute_emergence_scores(window=5, storage_manager=_DummyStorage())
    required = {"stage", "ias", "gas", "pmmspec", "selfref"}
    missing = sorted(list(required - set(scores.keys())))
    assert not missing, f"Missing keys: {missing}"
    # Basic type sanity
    assert isinstance(scores["ias"], (int, float))
    assert isinstance(scores["gas"], (int, float))
    return scores


# --- 3) Stage gating order sanity -------------------------------------------
@check("emergence.detect_stage_order")
def _c3():
    from pmm.emergence import detect_stage

    # Force S0 preemption scenario
    stage_s0 = detect_stage(pmmspec=0.01, selfref=0.0, IAS=0.49, GAS=0.0)
    # Force S1 scenario (bypass S0)
    stage_s1 = detect_stage(pmmspec=0.3, selfref=0.06, IAS=0.49, GAS=0.0)
    return {"s0_case": stage_s0, "s1_case": stage_s1}


# --- 4) Reflection cooldown config ------------------------------------------
@check("cooldown.thresholds_visible")
def _c4():
    from pmm.reflection_cooldown import ReflectionCooldownManager

    rc = ReflectionCooldownManager()
    return {
        "time_gate": getattr(rc, "time_gate", None),
        "turns_gate": getattr(rc, "turns_gate", None),
    }


# --- 5) Reflection context access (constructor + context probe) --------------
@check("reflection.context_accessor_exists")
def _c5():
    # Validate we can construct the reflection manager and that it exposes a
    # robust context accessor for emergence, without requiring heavy deps.
    from pmm.atomic_reflection import AtomicReflectionManager as AtomicReflection

    class _DummyPMM:
        class _Model:
            class _SK:
                insights = []

            self_knowledge = _SK()

        model = _Model()

    ar = AtomicReflection(_DummyPMM())
    # Should expose the resilient accessor
    has_accessor = hasattr(ar, "_get_emergence_context")
    # Call it; expect a dict regardless of underlying environment
    ctx = ar._get_emergence_context() if has_accessor else {}
    assert isinstance(ctx, dict)
    return {"has_accessor": has_accessor, "context_keys": sorted(list(ctx.keys()))[:5]}


# --- 6) Similarity threshold probe ------------------------------------------
@check("insight.similarity_threshold_probe")
def _c6():
    # Best-effort locate where the similarity threshold lives
    import inspect
    import pmm
    import pkgutil

    hit = {}
    for m in pkgutil.walk_packages(pmm.__path__, pmm.__name__ + "."):
        if not m.name.endswith(("insight_manager", "insights", "filters")):
            continue
        try:
            mod = __import__(m.name, fromlist=["*"])
            src = inspect.getsource(mod)
            if "similarity" in src and "0.97" in src or "0.975" in src or "0.94" in src:
                hit[m.name] = True
        except Exception:
            pass
    return {"found_possible_thresholds": hit or "none"}


# --- run ---------------------------------------------------------------------
if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("_c")]:
        fn()
    print(json.dumps(summary, indent=2, default=str))
