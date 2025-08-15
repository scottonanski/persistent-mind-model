#!/usr/bin/env python3
"""
Phase 3C Foundation Tests

Tests the core Phase 3C components:
- EmergenceAnalyzer.commitment_close_rate with real SQLite integration
- AdaptiveTrigger decision logic with time/event/emergence factors
- SemanticAnalyzer novelty detection and similarity scoring
- MetaReflectionAnalyzer pattern analysis
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add PMM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pmm.emergence import EmergenceAnalyzer
from pmm.adaptive_triggers import AdaptiveTrigger, TriggerConfig, TriggerState
from pmm.semantic_analysis import SemanticAnalyzer
from pmm.meta_reflection import MetaReflectionAnalyzer
from pmm.storage.sqlite_store import SQLiteStore


def test_emergence_analyzer_commitment_close_rate():
    """Test that EmergenceAnalyzer.commitment_close_rate works with real SQLite data."""
    print("🧪 Testing EmergenceAnalyzer.commitment_close_rate...")

    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path = tmp_db.name

    try:
        # Initialize SQLite store and create schema
        store = SQLiteStore(db_path)

        # Insert test commitment
        commitment_hash = "abc123def456"
        store.append_event(
            kind="commitment",
            content="Next, I will complete the Phase 3C implementation",
            meta={"commitment_id": "test_commit_1"},
            hsh=commitment_hash,
            prev="prev123",
        )

        # Insert evidence referencing the commitment
        store.append_event(
            kind="evidence",
            content="Done: Phase 3C implementation completed successfully",
            meta={"commit_ref": commitment_hash, "evidence_type": "done"},
            hsh="evidence123",
            prev=commitment_hash,
        )

        # Test analyzer
        analyzer = EmergenceAnalyzer(storage_manager=store)
        close_rate = analyzer.commitment_close_rate(window=10)

        print(f"✅ Commitment close rate: {close_rate}")
        assert close_rate == 1.0, f"Expected 1.0, got {close_rate}"

        # Test with no evidence
        store.append_event(
            kind="commitment",
            content="Next, I will test without evidence",
            meta={"commitment_id": "test_commit_2"},
            hsh="unclosed456",
            prev="evidence123",
        )

        close_rate_partial = analyzer.commitment_close_rate(window=10)
        print(f"✅ Partial close rate: {close_rate_partial}")
        assert close_rate_partial == 0.5, f"Expected 0.5, got {close_rate_partial}"

    finally:
        os.unlink(db_path)

    print("✅ EmergenceAnalyzer.commitment_close_rate test passed")


def test_adaptive_trigger_logic():
    """Test AdaptiveTrigger decision logic with various scenarios."""
    print("🧪 Testing AdaptiveTrigger decision logic...")

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Test 1: Event accumulation trigger with time gate passed
    config = TriggerConfig(cadence_days=1.0, events_min_gap=4)
    state = TriggerState(
        last_reflection_at=now - timedelta(days=2), events_since_reflection=4
    )
    trigger = AdaptiveTrigger(config, state)

    should_reflect, reason = trigger.decide(
        now, ias=0.5, gas=0.5, events_since_reflection=4
    )
    print(f"✅ Event accumulation: {should_reflect} ({reason})")
    assert should_reflect, "Should reflect due to event accumulation"

    # Test 2: Time gate trigger
    state = TriggerState(
        last_reflection_at=now - timedelta(days=2), events_since_reflection=2
    )
    trigger = AdaptiveTrigger(config, state)

    should_reflect, reason = trigger.decide(
        now, ias=0.5, gas=0.5, events_since_reflection=2
    )
    print(f"✅ Time gate: {should_reflect} ({reason})")
    assert should_reflect, "Should reflect due to time gate"

    # Test 3: Low emergence trigger (sooner reflection)
    state = TriggerState(
        last_reflection_at=now - timedelta(hours=6), events_since_reflection=4
    )
    trigger = AdaptiveTrigger(config, state)

    should_reflect, reason = trigger.decide(
        now, ias=0.2, gas=0.2, events_since_reflection=4
    )
    print(f"✅ Low emergence: {should_reflect} ({reason})")
    assert should_reflect, "Should reflect sooner due to low emergence"

    # Test 4: High emergence skip
    state = TriggerState(
        last_reflection_at=now - timedelta(days=1.5), events_since_reflection=2
    )
    trigger = AdaptiveTrigger(config, state)

    should_reflect, reason = trigger.decide(
        now, ias=0.8, gas=0.8, events_since_reflection=2
    )
    print(f"✅ High emergence skip: {should_reflect} ({reason})")
    assert not should_reflect, "Should skip reflection due to high emergence"

    print("✅ AdaptiveTrigger decision logic test passed")


def test_semantic_analyzer_novelty():
    """Test SemanticAnalyzer novelty detection and similarity scoring."""
    print("🧪 Testing SemanticAnalyzer novelty detection...")

    # Note: This test uses a mock provider since we don't want to require OpenAI API
    class MockEmbeddingProvider:
        def embed_text(self, text):
            # Simple mock: return character frequency as embedding
            chars = {}
            for char in text.lower():
                if char.isalpha():
                    chars[char] = chars.get(char, 0) + 1
            # Convert to fixed-size vector
            embedding = [0.0] * 26
            for char, count in chars.items():
                embedding[ord(char) - ord("a")] = float(count)
            return embedding

    analyzer = SemanticAnalyzer(embedding_provider=MockEmbeddingProvider())

    # Test novelty scoring
    reference_texts = [
        "I will focus on improving my reflection quality",
        "My goal is to enhance commitment tracking",
    ]

    # Similar text (should have low novelty)
    similar_text = "I will work on improving my reflection quality"
    novelty_similar = analyzer.semantic_novelty_score(similar_text, reference_texts)
    print(f"✅ Similar text novelty: {novelty_similar}")

    # Novel text (should have high novelty)
    novel_text = "Today I learned about quantum computing applications"
    novelty_novel = analyzer.semantic_novelty_score(novel_text, reference_texts)
    print(f"✅ Novel text novelty: {novelty_novel}")

    assert (
        novelty_novel > novelty_similar
    ), "Novel text should have higher novelty score"

    # Test duplicate detection
    is_duplicate = analyzer.is_semantic_duplicate(
        similar_text, reference_texts, threshold=0.8
    )
    print(f"✅ Duplicate detection: {is_duplicate}")

    print("✅ SemanticAnalyzer novelty detection test passed")


def test_meta_reflection_analyzer():
    """Test MetaReflectionAnalyzer pattern analysis."""
    print("🧪 Testing MetaReflectionAnalyzer pattern analysis...")

    # Mock reflections data
    reflections = [
        {
            "content": "I noticed my commitment completion rate is improving",
            "timestamp": (
                datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
            ).isoformat(),
            "meta": {},
        },
        {
            "content": "My reflection quality seems to be getting better with practice",
            "timestamp": (
                datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=12)
            ).isoformat(),
            "meta": {},
        },
        {
            "content": "I should focus more on specific, actionable insights",
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "meta": {},
        },
    ]

    # Use mock semantic analyzer to avoid API calls
    class MockSemanticAnalyzer:
        def semantic_novelty_score(self, text, references):
            return 0.7  # Mock novelty score

        def cluster_similar_texts(self, texts, similarity_threshold=0.8):
            return [[0], [1], [2]]  # Each text in its own cluster

    # Mock the embedding provider to avoid OpenAI API calls
    class MockEmbeddingProvider:
        def embed_text(self, text):
            return [0.1] * 10  # Mock embedding

    # Create analyzer with mocked semantic analyzer to avoid OpenAI API calls
    analyzer = MetaReflectionAnalyzer()
    # Override the semantic analyzer that was created during init
    analyzer.semantic_analyzer = MockSemanticAnalyzer()

    # Analyze patterns
    analysis = analyzer.analyze_reflection_patterns(reflections, window_days=7)

    print(f"✅ Total reflections: {analysis['total_reflections']}")
    print(f"✅ Average quality: {analysis['avg_quality']:.2f}")
    print(f"✅ Novelty trend: {analysis['novelty_trend']:.2f}")
    print(f"✅ Patterns found: {len(analysis['patterns'])}")
    print(f"✅ Recommendations: {len(analysis['recommendations'])}")

    assert analysis["total_reflections"] == 3, "Should analyze all 3 reflections"
    assert analysis["avg_quality"] > 0, "Should have positive quality score"
    assert len(analysis["recommendations"]) > 0, "Should generate recommendations"

    # Test meta-insight generation
    meta_insight = analyzer.generate_meta_insight(analysis)
    print(f"✅ Meta-insight: {meta_insight}")

    print("✅ MetaReflectionAnalyzer pattern analysis test passed")


def run_all_tests():
    """Run all Phase 3C foundation tests."""
    print("🚀 Phase 3C Foundation Tests")
    print("=" * 50)

    tests = [
        test_emergence_analyzer_commitment_close_rate,
        test_adaptive_trigger_logic,
        test_semantic_analyzer_novelty,
        test_meta_reflection_analyzer,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")

    print("=" * 50)
    print(f"🏁 Phase 3C Foundation Tests: {passed}/{len(tests)} passed")

    if passed == len(tests):
        print("🎉 All Phase 3C foundation components working correctly!")
        return True
    else:
        print("⚠️  Some Phase 3C components need attention")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
