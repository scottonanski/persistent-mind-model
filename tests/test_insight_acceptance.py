# tests/test_insight_acceptance.py
from pmm.atomic_reflection import AtomicReflectionManager


def _arm(threshold=0.94):
    # Bypass __init__ so we don't need a full PMM manager
    arm = object.__new__(AtomicReflectionManager)
    # Minimal attributes used by _should_accept_insight
    arm._effective_threshold = threshold
    return arm


def _decide(sim, content, refs=None, threshold=0.94):
    arm = _arm(threshold)
    return arm._should_accept_insight(
        content=content, best_sim=sim, candidate_refs=refs
    )


def test_reject_under_threshold_band():
    # best_sim below threshold -> override path should not accept (handled by main pipeline)
    assert _decide(0.93, "some reflection", refs=set(), threshold=0.94) is False


def test_reject_over_threshold_no_refs_or_anchors():
    # Within narrow band but without anchors or evidence -> reject
    assert _decide(0.96, "some reflection", refs=set(), threshold=0.94) is False


def test_accept_over_threshold_with_refs():
    # Within narrow band; sufficient evidence refs -> accept
    assert (
        _decide(
            0.96, "some reflection", refs={"ev12", "abcd1234efgh5678"}, threshold=0.94
        )
        is True
    )


def test_accept_over_threshold_with_anchor():
    # Within narrow band; anchor keyword present -> accept
    assert (
        _decide(
            0.95,
            "This references commitments and PMM drift.",
            refs=set(),
            threshold=0.94,
        )
        is True
    )


def test_reject_near_exact_dup():
    # Near-exact duplicates are hard rejected in this path
    assert (
        _decide(0.999, "PMM mention but too similar", refs=set(), threshold=0.94)
        is False
    )
