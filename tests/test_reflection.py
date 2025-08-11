from pmm.self_model_manager import SelfModelManager
from pmm.reflection import reflect_once


class FakeLLM:
    def chat(self, system: str, user: str):
        return "I notice stable patterns; confidence is steady."


def test_reflect_appends_insight(tmp_path):
    mgr = SelfModelManager(filepath=str(tmp_path / "m.json"))
    ins = reflect_once(mgr, FakeLLM())
    assert ins and ins.content
    assert mgr.model.self_knowledge.insights[-1].content == ins.content
