import os
from types import SimpleNamespace

import pytest

from pmm.bridges import BridgeManager


class FakeFactory:
    class Cfg(SimpleNamespace):
        pass

    def __init__(self):
        self._cfg = self.Cfg(
            provider="ollama", name="gemma3:4b", version="3.3", family="gemma", epoch=1
        )

    def get_active_config(self):
        return self._cfg


class FakeStages:
    def __init__(self):
        pass

    def reset_on_model_switch(self, name: str):
        return None

    def calculate_emergence_profile(self, model_name: str, ias: float, gas: float):
        # Map env PMM_HARD_STAGE to human label to mimic real manager
        hard = str(os.getenv("PMM_HARD_STAGE", "")).strip().upper()
        label = None
        if hard == "SS4":
            label = "SS4"
        elif hard in {"S0", "S1", "S2", "S3", "S4"}:
            label = {
                "S0": "S0: Substrate",
                "S1": "S1: Resistance",
                "S2": "S2: Adoption",
                "S3": "S3: Self-Model",
                "S4": "S4: Growth-Seeking",
            }[hard]
        else:
            label = "S2: Adoption"
        return SimpleNamespace(stage=SimpleNamespace(value=label))


class FakeNGram:
    def __init__(self):
        self.last = None

    def postprocess_style(self, text: str, model_name: str, stage=None):
        self.last = stage
        return text, []


class FakeStanceFilter:
    def __init__(self):
        self.last = None

    def filter_response(self, text: str, stage=None):
        self.last = stage
        return text, []


@pytest.mark.parametrize(
    "hard,expected",
    [
        ("S1", "S1: Resistance"),
        ("S4", "S4: Growth-Seeking"),
        ("SS4", "SS4"),
    ],
)
def test_bridge_threads_stage_to_ngram_and_stance(monkeypatch, hard, expected):
    monkeypatch.setenv("PMM_HARD_STAGE", hard)

    fake_ngram = FakeNGram()
    fake_factory = FakeFactory()
    fake_stages = FakeStages()

    # Build BridgeManager with our fake ngram system
    bm = BridgeManager(
        factory=fake_factory,
        storage=None,
        cooldown=SimpleNamespace(),
        ngram_ban=fake_ngram,
        stages=fake_stages,
    )

    # Inject fake stance filter so we can capture stage
    bm.stance_filter = FakeStanceFilter()

    out = bm.speak("Hello world.")
    assert isinstance(out, str)

    # Stage is passed to n-gram banning in adapters
    assert fake_ngram.last == expected

    # Stage is passed to stance filtering in bridge
    assert bm.stance_filter.last == expected
