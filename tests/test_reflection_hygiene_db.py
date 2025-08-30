import types

import pytest

from pmm.reflection import _validate_insight_references


class FakeEvent:
    def __init__(self, eid, content=""):
        self.id = eid
        self.content = content


class FakeSelfKnowledge:
    def __init__(self, events):
        self.autobiographical_events = events


class FakeModel:
    def __init__(self, events):
        self.self_knowledge = types.SimpleNamespace(autobiographical_events=events)


class FakeMgr:
    def __init__(self, events, open_commitments):
        self.model = FakeModel(events)
        self._open_commitments = open_commitments

    def get_open_commitments(self):
        return self._open_commitments


@pytest.mark.usefixtures("reset_analyzer")
def test_validate_insight_references_event_and_commit_matches():
    # Recent events with IDs
    events = [FakeEvent("ev101"), FakeEvent("ev202"), FakeEvent("ev303")]
    # Open commitments with hashes
    open_commits = [
        {"hash": "abcdef1234567890", "title": "Commit A"},
        {"hash": "1122334455667788", "title": "Commit B"},
    ]
    mgr = FakeMgr(events, open_commits)

    # Content references both an event id and a short commit hash
    content = "I noticed from event ev202 that my memory improved. Also see abcdef12."

    accepted, refs = _validate_insight_references(content, mgr)

    assert accepted is True
    # Should capture ev202 and commit short hash expanded to 16 chars
    assert any(r.startswith("ev202") for r in refs)
    assert any(r.startswith("abcdef12") for r in refs)


@pytest.mark.usefixtures("reset_analyzer")
def test_validate_insight_references_soft_accept_self_anchor():
    events = []
    open_commits = []
    mgr = FakeMgr(events, open_commits)

    # No explicit IDs, but first-person + PMM anchor should soft-accept
    content = (
        "I think my identity is shifting towards better commitment follow-through."
    )

    accepted, refs = _validate_insight_references(content, mgr)

    assert accepted is True
    assert any(str(r).startswith("unverified:") for r in refs)


@pytest.mark.usefixtures("reset_analyzer")
def test_validate_insight_references_reject_plaintext():
    events = []
    open_commits = []
    mgr = FakeMgr(events, open_commits)

    # No IDs and no anchors -> reject
    content = "This is a generic statement without references."

    accepted, refs = _validate_insight_references(content, mgr)

    assert accepted is False
    assert refs == []
