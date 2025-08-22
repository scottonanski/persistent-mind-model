import io
import sys
import types
from contextlib import redirect_stdout

import pytest

from pmm.atomic_reflection import AtomicReflectionManager
from pmm.reflection_cooldown import ReflectionCooldownManager


class DummyInsight:
    def __init__(self, content: str):
        self.content = content


class DummyPMM:
    class Model:
        class SelfKnowledge:
            def __init__(self):
                self.insights = []

        def __init__(self):
            self.self_knowledge = DummyPMM.Model.SelfKnowledge()

    def __init__(self):
        self.model = DummyPMM.Model()

    # Required by AtomicReflectionManager._persist_insight path if used in future
    def save_model(self):
        pass


@pytest.fixture(autouse=True)
def env_isolation(monkeypatch):
    # Ensure telemetry is on for tests that assert logs
    monkeypatch.setenv("PMM_TELEMETRY", "1")
    # Avoid real OpenAI network calls
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Reset adaptive flags in each test
    monkeypatch.delenv("PMM_ADAPTIVE_DEDUP", raising=False)
    monkeypatch.delenv("PMM_EMBEDDING_THRESHOLD", raising=False)
    yield


# --- Helper to mock OpenAI embeddings ---
class _MockEmbeddingsResp:
    def __init__(self, vec):
        self.data = [types.SimpleNamespace(embedding=vec)]


class _MockOpenAI:
    def __init__(self, vectors_by_text):
        self._vectors_by_text = vectors_by_text
        self.embeddings = self

    def create(self, input, model):
        # Return deterministic vectors from mapping, default to unit vector
        text = input.strip()
        vec = self._vectors_by_text.get(text)
        if vec is None:
            # Simple fallback: hash to two-dim vector that's normalized
            import math

            h = abs(hash(text)) % 1000 + 1
            x = (h % 10) / 10.0
            y = ((h // 10) % 10) / 10.0
            norm = math.sqrt(x * x + y * y) or 1.0
            vec = [x / norm, y / norm]
        return _MockEmbeddingsResp(vec)


def test_dedup_parity_with_memory_path(monkeypatch):
    """`PersistentMindMemory` path should match AtomicReflectionManager decisions.

    We construct recent insights, mock embeddings to make the new content a duplicate,
    then assert both ARM and the memory path agree on duplication.
    """
    from pmm.langchain_memory import PersistentMindMemory

    pmm_mgr = DummyPMM()
    # Two similar insights in cache
    pmm_mgr.model.self_knowledge.insights = [
        DummyInsight("I like pizza and coding."),
        DummyInsight("I enjoy pizza and programming."),
    ]

    # Mock embeddings to make any of these texts map to identical vector
    vectors = {
        "I like pizza and coding.": [1.0, 0.0],
        "I enjoy pizza and programming.": [1.0, 0.0],
        "I also like pizza and coding!": [1.0, 0.0],
    }

    # Monkeypatch OpenAI client used inside AtomicReflectionManager
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PMM_EMBEDDING_THRESHOLD", "0.90")
    monkeypatch.setenv("PMM_ADAPTIVE_DEDUP", "0")

    def _mock_openai_constructor():
        return _MockOpenAI(vectors)

    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=_mock_openai_constructor)
    )

    # Build memory wrapper but inject our DummyPMM
    mem = PersistentMindMemory(agent_path=":memory:")
    mem.pmm = pmm_mgr

    arm = AtomicReflectionManager(pmm_mgr, embedding_threshold=0.90)

    candidate = "I also like pizza and coding!"

    arm_decision = arm._is_duplicate_embedding(candidate)
    mem_decision = mem._is_similar_to_recent_insights(candidate)

    assert arm_decision is True
    assert mem_decision is True


def test_adaptive_threshold_adjustment_and_clamping(monkeypatch):
    monkeypatch.setenv("PMM_ADAPTIVE_DEDUP", "1")
    pmm_mgr = DummyPMM()
    arm = AtomicReflectionManager(pmm_mgr, embedding_threshold=0.90)

    # Start at configured effective threshold
    start = arm.get_stats()["embedding_threshold_effective"]
    assert start == pytest.approx(0.90, 1e-6)

    # Two accepts -> stricter (threshold decreases but not below min)
    arm._on_decision(accepted=True)
    arm._on_decision(accepted=True)
    after_accept = arm.get_stats()["embedding_threshold_effective"]
    assert after_accept <= start

    # Two rejects -> relax (threshold increases but not above max)
    arm._on_decision(accepted=False)
    arm._on_decision(accepted=False)
    after_reject = arm.get_stats()["embedding_threshold_effective"]
    assert after_reject >= after_accept

    # Clamp checks
    for _ in range(50):
        arm._on_decision(accepted=False)
    maxed = arm.get_stats()["embedding_threshold_effective"]
    assert maxed <= 0.95

    for _ in range(50):
        arm._on_decision(accepted=True)
    mined = arm.get_stats()["embedding_threshold_effective"]
    assert mined >= 0.80


def test_structured_override_with_new_references(monkeypatch):
    """High similarity but new referenced IDs should override dedup and allow acceptance."""
    pmm_mgr = DummyPMM()
    # Prior insight contains no refs
    pmm_mgr.model.self_knowledge.insights = [
        DummyInsight("We discussed the roadmap and milestones.")
    ]

    # Force both texts to identical vectors, sim=1.0
    vectors = {
        "We discussed the roadmap and milestones.": [0.0, 1.0],
        "We discussed the roadmap and milestones. ev999 new evidence": [0.0, 1.0],
    }

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _mock_openai_constructor():
        return _MockOpenAI(vectors)

    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=_mock_openai_constructor)
    )

    # Ensure low threshold so sim triggers duplicate branch
    arm = AtomicReflectionManager(pmm_mgr, embedding_threshold=0.80)

    buf = io.StringIO()
    with redirect_stdout(buf):
        is_dup = arm._is_duplicate_embedding(
            "We discussed the roadmap and milestones. ev999 new evidence"
        )
    out = buf.getvalue()

    assert is_dup is False  # override allowed
    assert "dedup_override" in out
    assert "added_refs" in out


def test_cooldown_telemetry_logging(monkeypatch):
    """Cooldown manager should emit structured telemetry with explicit reasons."""
    monkeypatch.setenv("PMM_TELEMETRY", "1")
    # Make turns requirement high to force block on a fresh manager
    monkeypatch.setenv("PMM_REFLECTION_MIN_TURNS", "5")

    rcm = ReflectionCooldownManager()

    buf = io.StringIO()
    with redirect_stdout(buf):
        allowed, reason = rcm.should_reflect("some context here")
    out = buf.getvalue()

    assert allowed is False
    assert isinstance(reason, str) and reason
    assert "[PMM_TELEMETRY]" in out
    assert "decision=blocked" in out or "cooldown" in out
