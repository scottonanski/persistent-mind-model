#!/usr/bin/env python3
"""
Test script for S0→S1 transition parameter tuning validation.

This script validates that the PMM parameter adjustments are working correctly
to facilitate transition from S0 (Substrate) to S1 (Pattern Formation).
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pmm.s0_s1_tuning import s0_s1_config
from pmm.reflection_cooldown import ReflectionCooldownManager
from pmm.emergence_stages import EmergenceStageManager
from pmm.pattern_continuity import PatternContinuityManager
from pmm.model_baselines import ModelBaselineManager


def test_reflection_cooldown_tuning():
    """Test that reflection cooldown parameters are properly tuned."""
    print("Testing reflection cooldown tuning...")

    # Create reflection manager with default parameters
    manager = ReflectionCooldownManager()

    # Verify tuned parameters
    assert (
        manager.min_wall_time_seconds == 180
    ), f"Expected 180s cooldown, got {manager.min_wall_time_seconds}s"
    assert (
        manager.novelty_threshold == 0.85
    ), f"Expected 0.85 novelty threshold, got {manager.novelty_threshold}"

    print(f"✅ Reflection cooldown: {manager.min_wall_time_seconds}s (was 30s)")
    print(f"✅ Novelty threshold: {manager.novelty_threshold} (was 0.78)")


def test_stage_transition_thresholds():
    """Test that stage transition thresholds are relaxed for easier S0→S1 transition."""
    print("\nTesting stage transition thresholds...")

    # Create stage manager
    baselines = ModelBaselineManager()
    stage_manager = EmergenceStageManager(baselines)

    # Verify relaxed thresholds
    thresholds = stage_manager.thresholds
    assert (
        thresholds.dormant_max == -0.8
    ), f"Expected -0.8 dormant exit, got {thresholds.dormant_max}"
    assert (
        thresholds.awakening_max == -0.3
    ), f"Expected -0.3 awakening entry, got {thresholds.awakening_max}"

    print(f"✅ Dormant exit threshold: {thresholds.dormant_max} (was -1.0)")
    print(f"✅ Awakening entry threshold: {thresholds.awakening_max} (was -0.5)")


def test_novelty_thresholds():
    """Test that novelty thresholds are adjusted for better pattern recognition."""
    print("\nTesting novelty threshold adjustments...")

    baselines = ModelBaselineManager()
    stage_manager = EmergenceStageManager(baselines)

    # Check stage-specific novelty thresholds
    from pmm.emergence_stages import EmergenceStage

    dormant_novelty = stage_manager.stage_behaviors[EmergenceStage.DORMANT][
        "novelty_threshold"
    ]
    awakening_novelty = stage_manager.stage_behaviors[EmergenceStage.AWAKENING][
        "novelty_threshold"
    ]

    assert (
        dormant_novelty == 0.8
    ), f"Expected 0.8 dormant novelty, got {dormant_novelty}"
    assert (
        awakening_novelty == 0.7
    ), f"Expected 0.7 awakening novelty, got {awakening_novelty}"

    print(f"✅ Dormant novelty threshold: {dormant_novelty} (was 0.9)")
    print(f"✅ Awakening novelty threshold: {awakening_novelty} (was 0.8)")


def test_pattern_continuity():
    """Test pattern continuity enhancement functionality."""
    print("\nTesting pattern continuity enhancement...")

    # Mock SQLite store for testing
    class MockSQLiteStore:
        def recent_events(self, limit=20):
            return [
                {
                    "id": "c123",
                    "kind": "commitment",
                    "content": "I will focus on deeper analysis",
                    "summary": "Commitment to analytical depth",
                    "ts": "2024-08-24",
                },
                {
                    "id": "r456",
                    "kind": "reflection",
                    "content": "I notice patterns in my responses",
                    "summary": "Pattern recognition insight",
                    "ts": "2024-08-24",
                },
                {
                    "id": "e789",
                    "kind": "evidence",
                    "content": "Completed analysis task",
                    "summary": "Evidence of task completion",
                    "ts": "2024-08-24",
                },
            ]

    mock_store = MockSQLiteStore()
    continuity_manager = PatternContinuityManager(mock_store, min_references=3)

    # Test continuity enhancement
    original_context = "What should I focus on next?"
    enhanced_context = continuity_manager.enhance_context_with_continuity(
        original_context
    )

    assert "CONTINUITY CONTEXT" in enhanced_context, "Continuity context not added"
    assert "Commitment c123" in enhanced_context, "Commitment reference not found"
    assert "Insight r456" in enhanced_context, "Insight reference not found"
    assert "Evidence e789" in enhanced_context, "Evidence reference not found"

    print("✅ Pattern continuity enhancement working")

    # Test pattern reuse scoring
    current_text = "I will focus on deeper analysis and pattern recognition"
    historical_events = mock_store.recent_events()

    reuse_score = continuity_manager.calculate_pattern_reuse_score(
        current_text, historical_events
    )
    assert reuse_score > 0, f"Expected positive pattern reuse score, got {reuse_score}"

    print(f"✅ Pattern reuse scoring: {reuse_score:.3f}")

    # Test novelty decay
    base_novelty = 1.0
    adjusted_novelty = continuity_manager.apply_novelty_decay(base_novelty, reuse_score)
    assert (
        adjusted_novelty < base_novelty
    ), f"Expected novelty decay, got {adjusted_novelty}"

    print(f"✅ Novelty decay: {base_novelty} → {adjusted_novelty:.3f}")


def test_s0_s1_config():
    """Test S0→S1 configuration system."""
    print("\nTesting S0→S1 configuration system...")

    # Test configuration values
    config = s0_s1_config

    assert config.reflection_cooldown_seconds == 180, "Incorrect cooldown setting"
    assert config.reflection_novelty_threshold == 0.85, "Incorrect novelty threshold"
    assert config.semantic_context_results == 8, "Incorrect semantic context limit"
    assert config.recent_events_limit == 45, "Incorrect recent events limit"

    print("✅ S0→S1 configuration values correct")

    # Test configuration methods
    reflection_config = config.get_reflection_config()
    memory_config = config.get_memory_config()
    stage_config = config.get_stage_config()
    pattern_config = config.get_pattern_config()

    assert "min_wall_time_seconds" in reflection_config, "Missing reflection config"
    assert "semantic_max_results" in memory_config, "Missing memory config"
    assert "dormant_max" in stage_config, "Missing stage config"
    assert "min_event_references" in pattern_config, "Missing pattern config"

    print("✅ Configuration methods working")


def main():
    """Run all S0→S1 tuning validation tests."""
    print("PMM S0→S1 Transition Parameter Tuning Validation")
    print("=" * 50)

    try:
        test_reflection_cooldown_tuning()
        test_stage_transition_thresholds()
        test_novelty_thresholds()
        test_pattern_continuity()
        test_s0_s1_config()

        print("\n" + "=" * 50)
        print("🎉 ALL S0→S1 TUNING TESTS PASSED!")
        print("\nParameter Summary:")
        print(s0_s1_config.get_summary())

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
