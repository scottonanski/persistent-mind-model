import os
import types
import io
import sys
from contextlib import redirect_stdout

import pytest

from pmm.atomic_reflection import AtomicReflectionManager
from pmm.llm_factory import get_llm_factory


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

    def save_model(self):
        # Persist no-op for tests
        pass


@pytest.fixture(autouse=True)
def env_isolation(monkeypatch):
    # Ensure clean env for each test
    monkeypatch.delenv("PMM_FORCE_ACCEPT_NEXT_INSIGHT", raising=False)
    monkeypatch.delenv("PMM_DISABLE_EMBEDDING_DEDUP", raising=False)
    monkeypatch.delenv("PMM_ADAPTIVE_DEDUP", raising=False)
    monkeypatch.delenv("PMM_EMBEDDING_THRESHOLD", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PMM_TELEMETRY", "1")
    yield


# --- Helper to mock OpenAI embeddings (mirrors pattern in other tests) ---
class _MockEmbeddingsResp:
    def __init__(self, vec):
        self.data = [types.SimpleNamespace(embedding=vec)]


class _MockOpenAI:
    def __init__(self, vectors_by_text):
        self._vectors_by_text = vectors_by_text
        self.embeddings = self

    def create(self, input, model):
        text = (input or "").strip()
        vec = self._vectors_by_text.get(text)
        if vec is None:
            vec = [1.0, 0.0]
        return _MockEmbeddingsResp(vec)


def test_force_accept_next_insight_is_one_shot(monkeypatch):
    pmm_mgr = DummyPMM()
    arm = AtomicReflectionManager(pmm_mgr)

    # Arrange: set one-shot flag
    monkeypatch.setenv("PMM_FORCE_ACCEPT_NEXT_INSIGHT", "1")

    epoch = get_llm_factory().get_current_epoch()

    # Act: first add should be accepted and persisted
    ok1 = arm.add_insight("A novel magical spider concept", {}, epoch)

    # Assert: persisted once
    assert ok1 is True
    assert len(pmm_mgr.model.self_knowledge.insights) == 1

    # The flag should be cleared after use
    assert os.getenv("PMM_FORCE_ACCEPT_NEXT_INSIGHT") is None

    # Act: second add of identical content should be rejected by fast text-dup gate
    ok2 = arm.add_insight("A novel magical spider concept", {}, epoch)
    assert ok2 is False


def test_disable_embedding_dedup_bypasses_duplicate_rejection(monkeypatch):
    pmm_mgr = DummyPMM()
    # Existing prior insight
    prior = "We discussed the roadmap and milestones."
    pmm_mgr.model.self_knowledge.insights = [DummyInsight(prior)]

    epoch = get_llm_factory().get_current_epoch()

    # Mock embeddings to make candidate and prior identical (high similarity)
    vectors = {
        prior: [0.0, 1.0],
        "We discussed the roadmap and milestones with more detail.": [0.0, 1.0],
    }

    def _mock_openai_constructor():
        return _MockOpenAI(vectors)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=_mock_openai_constructor)
    )

    # 1) Without override -> should be rejected due to high similarity
    arm_no_override = AtomicReflectionManager(pmm_mgr, embedding_threshold=0.80)
    cand = "We discussed the roadmap and milestones with more detail."

    buf = io.StringIO()
    with redirect_stdout(buf):
        ok_no = arm_no_override.add_insight(cand, {}, epoch)
    assert ok_no is False

    # 2) With PMM_DISABLE_EMBEDDING_DEDUP -> should be accepted (embedding check skipped)
    monkeypatch.setenv("PMM_DISABLE_EMBEDDING_DEDUP", "1")
    arm_override = AtomicReflectionManager(pmm_mgr, embedding_threshold=0.80)

    ok_yes = arm_override.add_insight(
        "We discussed the roadmap and milestones with more detail.", {}, epoch
    )
    assert ok_yes is True
    assert len(pmm_mgr.model.self_knowledge.insights) >= 2
